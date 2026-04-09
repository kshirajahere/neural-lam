# Third-party
import torch

# Local
from .. import metrics
from .. import utils
from .graph_lam import GraphLAM


class NoiseEnsembleGraphLAM(GraphLAM):
    """
    Prototype ensemble model that wraps the GraphLAM backbone with per-step
    noise injection and empirical CRPS training.

    This is intentionally simple: it keeps the underlying graph model
    single-trajectory and creates the ensemble dimension through repeated
    autoregressive rollouts.
    """

    def __init__(self, args, config, datastore):
        super().__init__(args, config=config, datastore=datastore)

        if self.output_std:
            raise ValueError(
                "NoiseEnsembleGraphLAM uses sample-based uncertainty and does "
                "not support --output_std."
            )
        if args.loss.lower() != "crps_ensemble":
            raise ValueError(
                "NoiseEnsembleGraphLAM is intended to be trained with "
                "--loss crps_ensemble."
            )
        if args.num_pred_samples < 2:
            raise ValueError(
                "NoiseEnsembleGraphLAM requires num_pred_samples >= 2."
            )
        if args.noise_dim < 1:
            raise ValueError("NoiseEnsembleGraphLAM requires noise_dim >= 1.")
        if args.noise_scale <= 0.0:
            raise ValueError(
                "NoiseEnsembleGraphLAM requires noise_scale > 0."
            )

        self.num_pred_samples = args.num_pred_samples
        self.noise_dim = args.noise_dim
        self.noise_scale = args.noise_scale
        self.noise_embedder = utils.make_mlp(
            [self.noise_dim] + self.mlp_blueprint_end
        )

        self.val_metrics["crps"] = []
        self.test_metrics["crps"] = []

    def sample_step_noise(self, batch_size, device):
        """
        Draw one latent noise vector per sample in the batch for the current
        autoregressive step.
        """
        return torch.randn(
            batch_size, self.noise_dim, device=device
        ) * self.noise_scale

    def predict_step_with_noise(
        self, prev_state, prev_prev_state, forcing, step_noise
    ):
        """
        Reuse the GraphLAM encode-process-decode path, but perturb the encoded
        grid representation with a per-sample noise vector before graph
        message passing.
        """
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )

        grid_emb = self.grid_embedder(grid_features)
        noise_emb = self.noise_embedder(step_noise).unsqueeze(1)
        grid_emb = grid_emb + noise_emb

        g2m_emb = self.g2m_embedder(self.g2m_features)
        m2g_emb = self.m2g_embedder(self.m2g_features)
        mesh_emb = self.embedd_mesh_nodes()

        mesh_emb_expanded = self.expand_to_batch(mesh_emb, batch_size)
        g2m_emb_expanded = self.expand_to_batch(g2m_emb, batch_size)
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )
        grid_rep = grid_emb + self.encoding_grid_mlp(grid_emb)

        mesh_rep = self.process_step(mesh_rep)

        m2g_emb_expanded = self.expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(mesh_rep, grid_rep, m2g_emb_expanded)

        pred_delta_mean = self.output_map(grid_rep)
        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean
        return prev_state + rescaled_delta_mean

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out multiple stochastic trajectories and stack them into an
        ensemble dimension.
        """
        batch_size = init_states.shape[0]
        pred_steps = forcing_features.shape[1]
        ensemble_predictions = []

        for _ in range(self.num_pred_samples):
            prev_prev_state = init_states[:, 0]
            prev_state = init_states[:, 1]
            prediction_list = []

            for step_i in range(pred_steps):
                forcing = forcing_features[:, step_i]
                border_state = true_states[:, step_i]
                step_noise = self.sample_step_noise(
                    batch_size=batch_size, device=prev_state.device
                )

                pred_state = self.predict_step_with_noise(
                    prev_state, prev_prev_state, forcing, step_noise
                )
                new_state = (
                    self.boundary_mask * border_state
                    + self.interior_mask * pred_state
                )
                prediction_list.append(new_state)

                prev_prev_state = prev_state
                prev_state = new_state

            ensemble_predictions.append(torch.stack(prediction_list, dim=1))

        prediction = torch.stack(ensemble_predictions, dim=1)
        return prediction, None

    def common_step(self, batch):
        init_states, target_states, forcing_features, batch_times = batch
        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )
        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        prediction, target, _, _ = self.common_step(batch)
        batch_loss = torch.mean(
            self.loss(prediction, target, None, mask=self.interior_mask_bool)
        )
        self.log_dict(
            {"train_loss": batch_loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def validation_step(self, batch, batch_idx):
        prediction, target, _, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(prediction, target, None, mask=self.interior_mask_bool),
            dim=0,
        )
        mean_loss = torch.mean(time_step_loss)
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        ensemble_mean_prediction = torch.mean(prediction, dim=1)
        self.val_metrics["mse"].append(
            metrics.mse(
                ensemble_mean_prediction,
                target,
                self.per_var_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
        )
        self.val_metrics["crps"].append(
            self.loss(
                prediction,
                target,
                None,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
        )

    def test_step(self, batch, batch_idx):
        prediction, target, _, _ = self.common_step(batch)
        time_step_loss = torch.mean(
            self.loss(prediction, target, None, mask=self.interior_mask_bool),
            dim=0,
        )
        mean_loss = torch.mean(time_step_loss)
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        test_log_dict["test_mean_loss"] = mean_loss
        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        ensemble_mean_prediction = torch.mean(prediction, dim=1)
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            self.test_metrics[metric_name].append(
                metric_func(
                    ensemble_mean_prediction,
                    target,
                    self.per_var_std,
                    mask=self.interior_mask_bool,
                    sum_vars=False,
                )
            )

        self.test_metrics["crps"].append(
            self.loss(
                prediction,
                target,
                None,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
        )

        spatial_loss = self.loss(
            prediction,
            target,
            None,
            average_grid=False,
        )
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)

        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            n_additional_examples = min(
                ensemble_mean_prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )
            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=ensemble_mean_prediction,
                split="test",
            )
