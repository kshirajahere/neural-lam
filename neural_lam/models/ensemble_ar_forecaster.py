# Third-party
import torch

# Local
from ..datastore import BaseDatastore
from .forecaster import ForecastResult, Forecaster
from .step_predictor import StepPredictor


class EnsembleARForecaster(Forecaster):
    """
    Batched ensemble forecaster that samples autoregressive trajectories from a
    probabilistic StepPredictor while keeping the single-trajectory
    StepPredictor/ARForecaster semantics unchanged.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
        num_pred_samples: int,
    ):
        super().__init__()
        if num_pred_samples < 1:
            raise ValueError(
                f"num_pred_samples must be >= 1, got {num_pred_samples}"
            )
        if num_pred_samples > 1 and not predictor.predicts_std:
            raise ValueError(
                "EnsembleARForecaster requires a predictor with output_std "
                "enabled when num_pred_samples > 1."
            )

        self.predictor = predictor
        self.num_pred_samples = int(num_pred_samples)

        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

    @property
    def predicts_std(self) -> bool:
        return self.predictor.predicts_std

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> ForecastResult:
        if self.num_pred_samples == 1:
            prediction, pred_std = self.predictor_rollout(
                init_states, forcing_features, boundary_states
            )
            return ForecastResult(prediction=prediction, pred_std=pred_std)

        batch_size, pred_steps = init_states.shape[0], forcing_features.shape[1]

        prev_prev_state = (
            init_states[:, 0]
            .unsqueeze(1)
            .expand(-1, self.num_pred_samples, -1, -1)
        )
        prev_state = (
            init_states[:, 1]
            .unsqueeze(1)
            .expand(-1, self.num_pred_samples, -1, -1)
        )
        forcing_features = forcing_features.unsqueeze(1).expand(
            -1, self.num_pred_samples, -1, -1, -1
        )
        boundary_states = boundary_states.unsqueeze(1).expand(
            -1, self.num_pred_samples, -1, -1, -1
        )

        prediction_list = []
        pred_std_list = []

        for i in range(pred_steps):
            forcing = forcing_features[:, :, i].flatten(0, 1)
            boundary_state = boundary_states[:, :, i]

            pred_state_mean, pred_std = self.predictor(
                prev_state.flatten(0, 1),
                prev_prev_state.flatten(0, 1),
                forcing,
            )
            pred_state_mean = pred_state_mean.unflatten(
                0, (batch_size, self.num_pred_samples)
            )
            pred_std = pred_std.unflatten(
                0, (batch_size, self.num_pred_samples)
            )

            sampled_state = (
                pred_state_mean + torch.randn_like(pred_std) * pred_std
            )
            new_state = (
                self.boundary_mask.unsqueeze(1) * boundary_state
                + self.interior_mask.unsqueeze(1) * sampled_state
            )

            prediction_list.append(new_state)
            pred_std_list.append(pred_std)
            prev_prev_state = prev_state
            prev_state = new_state

        return ForecastResult(
            prediction=torch.stack(prediction_list, dim=2),
            pred_std=torch.stack(pred_std_list, dim=2),
        )

    def predictor_rollout(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            boundary_state = boundary_states[:, i]

            pred_state, pred_std = self.predictor(
                prev_state, prev_prev_state, forcing
            )

            new_state = (
                self.boundary_mask * boundary_state
                + self.interior_mask * pred_state
            )
            prediction_list.append(new_state)
            if pred_std is not None:
                pred_std_list.append(pred_std)

            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(prediction_list, dim=1)
        pred_std = torch.stack(pred_std_list, dim=1) if pred_std_list else None
        return prediction, pred_std
