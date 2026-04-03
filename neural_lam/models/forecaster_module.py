# Standard library
import os
import warnings
from typing import Any, Dict, List, Optional

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam.utils import get_integer_time

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset
from .forecaster import ForecastResult, Forecaster


class ForecasterModule(pl.LightningModule):
    """
    Lightning module handling training, validation and testing loops.
    Wraps a Forecaster instance which performs the actual prediction.
    """

    # pylint: disable=arguments-differ

    def __init__(
        self,
        forecaster: Forecaster,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        loss: str = "wmse",
        lr: float = 1e-3,
        restore_opt: bool = False,
        n_example_pred: int = 1,
        val_steps_to_log: Optional[List[int]] = None,
        metrics_watch: Optional[List[str]] = None,
        var_leads_metrics_watch: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        # Resolve mutable defaults
        if val_steps_to_log is None:
            val_steps_to_log = [
                1,
            ]
        if metrics_watch is None:
            metrics_watch = []
        if var_leads_metrics_watch is None:
            var_leads_metrics_watch = {}

        # Note: datastore is excluded from saved hparams and must be provided
        # explicitly when calling load_from_checkpoint(path,
        # datastore=datastore)
        self.save_hyperparameters(ignore=["datastore", "forecaster"])
        self.datastore = datastore
        self.forecaster = forecaster

        # Compute interior_mask_bool directly from datastore
        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )  # (1, num_grid_nodes, 1)
        interior_mask = 1.0 - boundary_mask
        self.register_buffer(
            "interior_mask_bool",
            interior_mask[0, :, 0].to(torch.bool),
            persistent=False,
        )

        # Store per_var_std here if predictor does not output std
        if not self.forecaster.predicts_std:
            da_state_stats = datastore.get_standardization_dataarray(
                category="state"
            )
            state_feature_weights = get_state_feature_weighting(
                config=config, datastore=datastore
            )
            diff_std = torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            )
            feature_weights_t = torch.tensor(
                state_feature_weights, dtype=torch.float32
            )
            self.register_buffer(
                "per_var_std",
                diff_std / torch.sqrt(feature_weights_t),
                persistent=False,
            )
        else:
            self.per_var_std = None

        # Instantiate loss function
        self.loss = metrics.get_metric(loss)

        self.val_metrics: Dict[str, List] = {
            "mse": [],
        }
        self.test_metrics: Dict[str, List] = {
            "mse": [],
            "mae": [],
        }
        if self.forecaster.predicts_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = restore_opt

        # For example plotting
        self.n_example_pred = n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps: List[Any] = []

        self.time_step_int, self.time_step_unit = get_integer_time(
            self.datastore.step_length
        )

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: torch.Tensor,
        split: str,
        category: str,
    ) -> xr.DataArray:
        weather_dataset = WeatherDataset(datastore=self.datastore, split=split)
        time = np.array(time.cpu(), dtype="datetime64[ns]")
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.95)
        )
        return opt

    def forecast_result_for_batch(
        self, batch
    ) -> tuple[ForecastResult, torch.Tensor]:
        """Run the forecaster on a batch and return a structured result."""
        init_states, target_states, forcing_features, _batch_times = batch

        forecast_result = self._normalize_forecaster_output(
            self.forecaster(init_states, forcing_features, target_states)
        )
        self._validate_forecast_result(forecast_result, target_states)
        return forecast_result, target_states

    def forecast_for_batch(self, batch):
        """Run the forecaster on a batch and return predictions.

        Unpacks the batch, runs the forecaster, and returns the raw pred_std
        (None if the forecaster does not output uncertainty estimates).

        Parameters
        ----------
        batch : tuple
            Tuple of (init_states, target_states, forcing_features,
            batch_times) tensors from the dataloader.

        Returns
        -------
        prediction : torch.Tensor
            Predicted states.
        target_states : torch.Tensor
            Ground-truth target states.
        pred_std : torch.Tensor or None
            Predicted standard deviations, or None if not output by model.
        """
        forecast_result, target_states = self.forecast_result_for_batch(batch)

        return (
            forecast_result.prediction,
            target_states,
            forecast_result.pred_std,
        )

    @staticmethod
    def _normalize_forecaster_output(
        output: ForecastResult | tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> ForecastResult:
        """Normalize legacy and structured forecaster outputs."""
        if isinstance(output, ForecastResult):
            return output

        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "Forecaster must return either ForecastResult or "
                "(prediction, pred_std)."
            )

        prediction, pred_std = output
        return ForecastResult(prediction=prediction, pred_std=pred_std)

    @staticmethod
    def _validate_forecast_result(
        forecast_result: ForecastResult, target_states: torch.Tensor
    ) -> None:
        """Validate deterministic and ensemble forecast tensor contracts."""
        prediction = forecast_result.prediction
        pred_std = forecast_result.pred_std

        deterministic_shape = target_states.shape
        ensemble_shape = (
            target_states.shape[0],
            None,
            *target_states.shape[1:],
        )

        if prediction.ndim == target_states.ndim:
            if prediction.shape != deterministic_shape:
                raise ValueError(
                    "Deterministic predictions must have shape "
                    f"{deterministic_shape}, got {tuple(prediction.shape)}."
                )
        elif prediction.ndim == target_states.ndim + 1:
            if (
                prediction.shape[0] != ensemble_shape[0]
                or prediction.shape[2:] != ensemble_shape[2:]
            ):
                raise ValueError(
                    "Ensemble predictions must have shape "
                    f"(B, S, T, N, d_f) matching target {deterministic_shape}, "
                    f"got {tuple(prediction.shape)}."
                )
        else:
            raise ValueError(
                "Forecaster predictions must follow either the deterministic "
                "(B, T, N, d_f) or ensemble (B, S, T, N, d_f) contract."
            )

        if pred_std is None:
            return
        if pred_std.ndim == 1:
            if pred_std.shape[0] != target_states.shape[-1]:
                raise ValueError(
                    "One-dimensional pred_std must match the state feature "
                    f"dimension {target_states.shape[-1]}, got "
                    f"{pred_std.shape[0]}."
                )
            return
        if pred_std.shape != prediction.shape:
            raise ValueError(
                "pred_std must either match prediction shape or be shaped "
                f"(d_f,). Got prediction {tuple(prediction.shape)} and "
                f"pred_std {tuple(pred_std.shape)}."
            )

    @staticmethod
    def _reduce_ensemble_prediction(
        forecast_result: ForecastResult,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reduce ensemble predictions to the deterministic contract expected by
        the existing loss/metric code paths.
        """
        prediction = forecast_result.prediction
        pred_std = forecast_result.pred_std

        if not forecast_result.is_ensemble_prediction:
            return prediction, pred_std

        prediction = torch.mean(prediction, dim=1)
        if pred_std is not None and pred_std.ndim == 5:
            pred_std = torch.mean(pred_std, dim=1)

        return prediction, pred_std

    def _compute_ensemble_spread_metric(
        self, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Compute entrywise ensemble spread reduced to (B, T, d_f)."""
        entry_spread = torch.std(prediction, dim=1, correction=0)
        return metrics.mask_and_reduce_metric(
            entry_spread,
            mask=self.interior_mask_bool,
            average_grid=True,
            sum_vars=False,
        )

    def training_step(self, batch):
        forecast_result, target_states = self.forecast_result_for_batch(batch)
        prediction, pred_std = self._reduce_ensemble_prediction(forecast_result)
        if pred_std is None:
            pred_std = self.per_var_std

        batch_loss = torch.mean(
            self.loss(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            )
        )

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        gathered = self.all_gather(tensor_to_gather)
        # all_gather adds dim 0 only on multi-device; on single
        # device it returns the same tensor unchanged.
        if gathered.dim() > tensor_to_gather.dim():
            gathered = gathered.flatten(0, 1)
        return gathered

    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        forecast_result, target_states = self.forecast_result_for_batch(batch)
        prediction, pred_std = self._reduce_ensemble_prediction(forecast_result)
        if pred_std is None:
            pred_std = self.per_var_std

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            ),
            dim=0,
        )
        mean_loss = torch.mean(time_step_loss)

        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.hparams.val_steps_to_log
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

        entry_mses = metrics.mse(
            prediction,
            target_states,
            pred_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        metric_name = (
            "ensemble_mse" if forecast_result.is_ensemble_prediction else "mse"
        )
        self.val_metrics.setdefault(metric_name, []).append(entry_mses)

        if forecast_result.is_ensemble_prediction:
            self.val_metrics.setdefault("ensemble_spread", []).append(
                self._compute_ensemble_spread_metric(forecast_result.prediction)
            )

    def on_validation_epoch_end(self):
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        forecast_result, target_states = self.forecast_result_for_batch(batch)
        prediction, pred_std = self._reduce_ensemble_prediction(forecast_result)

        if pred_std is not None:
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )
            self.test_metrics["output_std"].append(mean_pred_std)

        if pred_std is None:
            pred_std = self.per_var_std

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            ),
            dim=0,
        )
        mean_loss = torch.mean(time_step_loss)

        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.hparams.val_steps_to_log
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

        metric_name_map = (
            {"ensemble_mse": "mse", "ensemble_mae": "mae"}
            if forecast_result.is_ensemble_prediction
            else {"mse": "mse", "mae": "mae"}
        )
        for metric_name, base_metric_name in metric_name_map.items():
            metric_func = metrics.get_metric(base_metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
            self.test_metrics.setdefault(metric_name, []).append(
                batch_metric_vals
            )

        if forecast_result.is_ensemble_prediction:
            self.test_metrics.setdefault("ensemble_spread", []).append(
                self._compute_ensemble_spread_metric(forecast_result.prediction)
            )

        spatial_loss = self.loss(
            prediction, target_states, pred_std, average_grid=False
        )
        log_spatial_losses = spatial_loss[
            :,
            [
                step - 1
                for step in self.hparams.val_steps_to_log
                if step <= spatial_loss.shape[1]
            ],
        ]
        self.spatial_loss_maps.append(log_spatial_losses)

        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def plot_examples(self, batch, n_examples, split, prediction):

        target = batch[1]
        time = batch[3]

        da_state_stats = self.datastore.get_standardization_dataarray("state")
        state_std = torch.tensor(
            da_state_stats.state_std.values,
            dtype=torch.float32,
            device=prediction.device,
        )
        state_mean = torch.tensor(
            da_state_stats.state_mean.values,
            dtype=torch.float32,
            device=prediction.device,
        )

        prediction_rescaled = prediction * state_std + state_mean
        target_rescaled = target * state_std + state_mean

        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            self.plotted_examples += 1

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vranges = list(zip(var_vmin, var_vmax))

            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                var_figs = [
                    vis.plot_prediction(
                        datastore=self.datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({(self.time_step_int * t_i)}"
                        f"{self.time_step_unit})",
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self.datastore.get_vars_names("state"),
                            self.datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_i = self.plotted_examples

                for var_name, fig in zip(
                    self.datastore.get_vars_names("state"), var_figs
                ):
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        key = f"{var_name}_example_{example_i}"
                    else:
                        key = f"{var_name}_example"

                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(key=key, images=[fig], step=t_i)
                    else:
                        warnings.warn(
                            f"{self.logger} does not support image logging."
                        )

                plt.close("all")

            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_pred_{self.plotted_examples}.pt",
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_target_{self.plotted_examples}.pt",
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self.datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            metric_fig.savefig(
                os.path.join(self.logger.save_dir, f"{full_log_name}.pdf")
            )
            np.savetxt(
                os.path.join(self.logger.save_dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        var_names = self.datastore.get_vars_names(category="state")
        if full_log_name in self.hparams.metrics_watch:
            for (
                var_i,
                timesteps,
            ) in self.hparams.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)

                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                da_state_stats = self.datastore.get_standardization_dataarray(
                    "state"
                )
                state_std = torch.tensor(
                    da_state_stats.state_std.values,
                    dtype=torch.float32,
                    device=metric_tensor_averaged.device,
                )
                metric_rescaled = metric_tensor_averaged * state_std

                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        figure_dict = {
            k: v for k, v in log_dict.items() if isinstance(v, plt.Figure)
        }
        scalar_dict = {
            k: v for k, v in log_dict.items() if not isinstance(v, plt.Figure)
        }

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            # Log scalars via Lightning's built-in mechanism
            if scalar_dict:
                self.log_dict(scalar_dict, sync_dist=True)

            current_epoch = self.trainer.current_epoch

            for key, figure in figure_dict.items():
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[figure])

            plt.close("all")

    def on_test_epoch_end(self):
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(spatial_loss_tensor, dim=0)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self.datastore,
                    title=f"Test loss, t={t_i} "
                    f"({(self.time_step_int * t_i)} {self.time_step_unit})",
                )
                for t_i, loss_map in zip(
                    self.hparams.val_steps_to_log, mean_spatial_loss
                )
            ]

            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            pdf_loss_map_figs = [
                vis.plot_spatial_error(error=loss_map, datastore=self.datastore)
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(
                self.hparams.val_steps_to_log, pdf_loss_map_figs
            ):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))

            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        loaded_state_dict = checkpoint["state_dict"]

        # 1. Broad namespace remap: for pre-refactor checkpoints
        # The old ARModel was a flat LightningModule. Everything that belonged
        # to the predictor needs to be moved to 'forecaster.predictor.'
        old_keys = list(loaded_state_dict.keys())
        for key in old_keys:
            if not key.startswith("forecaster.") and key not in (
                "interior_mask_bool",
                "per_var_std",
            ):
                new_key = f"forecaster.predictor.{key}"
                loaded_state_dict[new_key] = loaded_state_dict.pop(key)

        # 2. Specific rename from g2m_gnn.grid_mlp -> encoding_grid_mlp
        # Will be under forecaster.predictor due to the remap above, or
        # already there if from a recent checkpoint before this rename.
        if (
            "forecaster.predictor.g2m_gnn.grid_mlp.0.weight"
            in loaded_state_dict
        ):
            replace_keys = list(
                filter(
                    lambda key: key.startswith(
                        "forecaster.predictor.g2m_gnn.grid_mlp"
                    ),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "forecaster.predictor.g2m_gnn.grid_mlp",
                    "forecaster.predictor.encoding_grid_mlp",
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]

        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt.state_dict()]
