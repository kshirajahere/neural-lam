# Third-party
import pytest
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.ar_forecaster import ARForecaster
from neural_lam.models.ensemble_ar_forecaster import EnsembleARForecaster
from neural_lam.models.forecaster import ForecastResult, Forecaster
from neural_lam.models.forecaster_module import ForecasterModule
from neural_lam.models.step_predictor import StepPredictor
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


class NoStaticDummyDatastore(DummyDatastore):
    """DummyDatastore variant that returns None for static features."""

    def get_dataarray(self, category, split, standardize=False):
        if category == "static":
            return None
        return super().get_dataarray(category, split, standardize=standardize)


class MockStepPredictor(StepPredictor):
    def __init__(self, config, datastore, **kwargs):
        super().__init__(config, datastore, **kwargs)

    def forward(self, prev_state, prev_prev_state, forcing):
        # Return zeros for state
        # The true state will be mixed in at boundaries
        pred_state = torch.zeros_like(prev_state)
        pred_std = torch.zeros_like(prev_state) if self.output_std else None
        return pred_state, pred_std


class MockStructuredForecaster(Forecaster):
    def __init__(self, predicts_std: bool):
        super().__init__()
        self._predicts_std = predicts_std

    @property
    def predicts_std(self) -> bool:
        return self._predicts_std

    def forward(self, init_states, forcing_features, boundary_states):
        pred_steps = forcing_features.shape[1]
        prediction = boundary_states.clone()
        pred_std = (
            torch.full_like(boundary_states, 0.5)
            if self._predicts_std
            else None
        )
        return ForecastResult(
            prediction=prediction[:, :pred_steps],
            pred_std=pred_std[:, :pred_steps] if pred_std is not None else None,
            aux_data={"posterior_loc": torch.zeros(())},
        )


class MockStructuredEnsembleForecaster(Forecaster):
    def __init__(self, ensemble_size: int = 4):
        super().__init__()
        self.ensemble_size = ensemble_size

    @property
    def predicts_std(self) -> bool:
        return True

    def forward(self, init_states, forcing_features, boundary_states):
        prediction = boundary_states.unsqueeze(1).repeat(
            1, self.ensemble_size, 1, 1, 1
        )
        pred_std = torch.full_like(prediction, 0.25)
        return ForecastResult(prediction=prediction, pred_std=pred_std)


def test_ar_forecaster_unroll():
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=False,
    )

    forecaster = ARForecaster(predictor, datastore)

    # Override masks to test boundary masking behaviour
    forecaster.interior_mask = torch.zeros_like(forecaster.interior_mask)
    forecaster.interior_mask[0, 0] = 1  # One node is interior
    forecaster.boundary_mask = 1 - forecaster.interior_mask

    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    pred_steps = 3
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, pred_steps, num_grid_nodes, d_forcing)
    true_states = torch.ones(B, pred_steps, num_grid_nodes, d_state) * 5.0

    prediction, pred_std = forecaster(
        init_states, forcing_features, true_states
    )

    assert prediction.shape == (B, pred_steps, num_grid_nodes, d_state)

    # Boundary (where interior_mask == 0) should equal true_state (5.0)
    # Interior (where interior_mask == 1) should equal predictor output (0.0)
    assert torch.all(prediction[:, :, 0, :] == 0.0)
    assert torch.all(prediction[:, :, 1:, :] == 5.0)


def test_forecaster_module_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    ckpt_path = tmp_path / "test.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    # Build a fresh forecaster structure for loading weights into
    load_predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    load_forecaster = ARForecaster(load_predictor, datastore)

    # Load from checkpoint
    loaded_model = ForecasterModule.load_from_checkpoint(
        ckpt_path,
        datastore=datastore,
        forecaster=load_forecaster,
        weights_only=False,
    )

    # Validate the correct internal hierarchy has been constructed
    assert loaded_model.forecaster.predictor.__class__.__name__ == "GraphLAM"

    # Verify that outputs match (checkpoint successfully restored weights)
    B, num_grid_nodes = 2, model.forecaster.predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, 1, num_grid_nodes, d_state) * 5.0

    with torch.no_grad():
        out_before = model.forecaster(
            init_states, forcing_features, boundary_states
        )
        out_after = loaded_model.forecaster(
            init_states, forcing_features, boundary_states
        )

    assert torch.allclose(out_before[0], out_after[0])


def test_forecaster_module_old_checkpoint(tmp_path):
    datastore = init_datastore_example("mdp")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # First-party
    from neural_lam.models import MODELS

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    ckpt_path = tmp_path / "test_old.ckpt"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(ckpt_path)

    # Manually hack the checkpoint to emulate a pre-refactor state dict
    ckpt = torch.load(ckpt_path, weights_only=False)
    old_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("forecaster.predictor."):
            # Revert structural rename to emulate old flat keys
            new_k = k.replace("forecaster.predictor.", "")
            if "encoding_grid_mlp" in new_k:
                new_k = new_k.replace("encoding_grid_mlp", "g2m_gnn.grid_mlp")
            old_state_dict[new_k] = v
        else:
            old_state_dict[k] = v

    ckpt["state_dict"] = old_state_dict
    torch.save(ckpt, ckpt_path)

    # Build a fresh forecaster structure for loading weights into
    load_predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        processor_layers=1,
        mesh_aggr="sum",
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        output_std=False,
    )
    load_forecaster = ARForecaster(load_predictor, datastore)

    # Load from hacked old checkpoint
    loaded_model = ForecasterModule.load_from_checkpoint(
        ckpt_path,
        datastore=datastore,
        forecaster=load_forecaster,
        weights_only=False,
    )

    # Validate the correct internal hierarchy has been constructed
    assert loaded_model.forecaster.predictor.__class__.__name__ == "GraphLAM"

    # Verify that outputs match (checkpoint successfully restored weights)
    B, num_grid_nodes = 2, model.forecaster.predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    num_past_forcing_steps = 1
    num_future_forcing_steps = 1
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        num_past_forcing_steps + num_future_forcing_steps + 1
    )
    init_states = torch.ones(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.ones(B, 1, num_grid_nodes, d_state) * 5.0

    with torch.no_grad():
        out_before = model.forecaster(
            init_states, forcing_features, boundary_states
        )
        out_after = loaded_model.forecaster(
            init_states, forcing_features, boundary_states
        )

    assert torch.allclose(out_before[0], out_after[0])


def test_step_predictor_no_static_features():
    """Model should run correctly when the datastore has no static features,
    using an empty (N, 0) tensor in place of static features."""
    datastore = NoStaticDummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=False,
    )

    # Static features buffer should exist but be empty (zero width)
    assert predictor.grid_static_features.shape == (
        datastore.num_grid_points,
        0,
    )

    # Verify a forward pass works end-to-end via ARForecaster
    forecaster = ARForecaster(predictor, datastore)
    B, num_grid_nodes = 2, predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")
    init_states = torch.zeros(B, 2, num_grid_nodes, d_state)
    forcing_features = torch.zeros(B, 1, num_grid_nodes, d_forcing)
    boundary_states = torch.zeros(B, 1, num_grid_nodes, d_state)

    prediction, pred_std = forecaster(
        init_states, forcing_features, boundary_states
    )
    assert prediction.shape == (B, 1, num_grid_nodes, d_state)
    assert pred_std is None


def test_forecaster_module_accepts_structured_forecast_output():
    datastore = DummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    forecaster = MockStructuredForecaster(predicts_std=True)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="nll",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    batch_size = 2
    pred_steps = 3
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    batch = (
        torch.zeros(batch_size, 2, num_grid_nodes, d_state),
        torch.ones(batch_size, pred_steps, num_grid_nodes, d_state),
        torch.zeros(batch_size, pred_steps, num_grid_nodes, d_forcing),
        torch.zeros(batch_size, pred_steps, dtype=torch.int64),
    )

    forecast_result, forecast_target = model.forecast_result_for_batch(batch)
    assert isinstance(forecast_result, ForecastResult)
    assert forecast_target.shape == batch[1].shape
    assert "posterior_loc" in forecast_result.aux_data

    prediction, target, pred_std = model.forecast_for_batch(batch)
    assert torch.allclose(prediction, target)
    assert pred_std is not None
    assert torch.all(pred_std == 0.5)

    loss = model.training_step(batch)
    assert torch.isfinite(loss)


def test_forecaster_module_normalizes_legacy_tuple_output():
    prediction = torch.zeros(2, 3, 4, 5)
    pred_std = torch.ones_like(prediction)
    result = ForecasterModule._normalize_forecaster_output(
        (prediction, pred_std)
    )

    assert isinstance(result, ForecastResult)
    assert result.prediction is prediction
    assert result.pred_std is pred_std


def test_forecaster_module_accepts_ensemble_shape_contract():
    datastore = DummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    model = ForecasterModule(
        forecaster=MockStructuredEnsembleForecaster(ensemble_size=4),
        config=config,
        datastore=datastore,
        loss="nll",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    batch_size = 2
    pred_steps = 3
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    batch = (
        torch.zeros(batch_size, 2, num_grid_nodes, d_state),
        torch.ones(batch_size, pred_steps, num_grid_nodes, d_state),
        torch.zeros(batch_size, pred_steps, num_grid_nodes, d_forcing),
        torch.zeros(batch_size, pred_steps, dtype=torch.int64),
    )

    forecast_result, forecast_target = model.forecast_result_for_batch(batch)
    assert forecast_result.is_ensemble_prediction
    assert forecast_result.prediction.shape == (
        batch_size,
        4,
        pred_steps,
        num_grid_nodes,
        d_state,
    )
    assert forecast_target.shape == batch[1].shape

    prediction, _, pred_std = model.forecast_for_batch(batch)
    assert prediction.shape == forecast_result.prediction.shape
    assert pred_std is not None
    assert pred_std.shape == prediction.shape


def test_forecaster_module_rejects_invalid_prediction_shape():
    target_states = torch.zeros(2, 3, 4, 5)
    invalid_result = ForecastResult(prediction=torch.zeros(2, 4, 5))

    with pytest.raises(
        ValueError, match="Forecaster predictions must follow either"
    ):
        ForecasterModule._validate_forecast_result(
            invalid_result, target_states
        )


def test_forecaster_module_rejects_invalid_pred_std_shape():
    target_states = torch.zeros(2, 3, 4, 5)
    invalid_result = ForecastResult(
        prediction=torch.zeros_like(target_states),
        pred_std=torch.zeros(2, 3, 4, 6),
    )

    with pytest.raises(ValueError, match="pred_std must either match"):
        ForecasterModule._validate_forecast_result(
            invalid_result, target_states
        )


def test_ensemble_ar_forecaster_returns_sample_dimension():
    datastore = DummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=True,
    )
    forecaster = EnsembleARForecaster(
        predictor=predictor,
        datastore=datastore,
        num_pred_samples=4,
    )

    batch_size = 2
    pred_steps = 3
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    init_states = torch.ones(batch_size, 2, num_grid_nodes, d_state)
    forcing_features = torch.ones(
        batch_size, pred_steps, num_grid_nodes, d_forcing
    )
    boundary_states = (
        torch.ones(batch_size, pred_steps, num_grid_nodes, d_state) * 5.0
    )

    forecast_result = forecaster(init_states, forcing_features, boundary_states)
    assert isinstance(forecast_result, ForecastResult)
    assert forecast_result.is_ensemble_prediction
    assert forecast_result.prediction.shape == (
        batch_size,
        4,
        pred_steps,
        num_grid_nodes,
        d_state,
    )
    assert forecast_result.pred_std is not None
    assert forecast_result.pred_std.shape == forecast_result.prediction.shape


def test_ensemble_ar_forecaster_requires_output_std_for_sampling():
    datastore = DummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    predictor = MockStepPredictor(
        config=config,
        datastore=datastore,
        output_std=False,
    )

    with pytest.raises(
        ValueError, match="requires a predictor with output_std"
    ):
        EnsembleARForecaster(
            predictor=predictor,
            datastore=datastore,
            num_pred_samples=4,
        )


def test_forecaster_module_routes_ensemble_predictions_to_mean_path():
    datastore = DummyDatastore()
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    model = ForecasterModule(
        forecaster=MockStructuredEnsembleForecaster(ensemble_size=4),
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1],
        metrics_watch=[],
    )

    batch_size = 2
    pred_steps = 3
    num_grid_nodes = datastore.num_grid_points
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing")

    batch = (
        torch.zeros(batch_size, 2, num_grid_nodes, d_state),
        torch.ones(batch_size, pred_steps, num_grid_nodes, d_state),
        torch.zeros(batch_size, pred_steps, num_grid_nodes, d_forcing),
        torch.zeros(batch_size, pred_steps, dtype=torch.int64),
    )

    forecast_result, target_states = model.forecast_result_for_batch(batch)
    reduced_prediction, reduced_pred_std = model._reduce_ensemble_prediction(
        forecast_result
    )
    assert reduced_prediction.shape == target_states.shape
    assert reduced_pred_std is not None
    assert reduced_pred_std.shape == target_states.shape

    loss = model.training_step(batch)
    assert torch.isfinite(loss)

    model.validation_step(batch, batch_idx=0)
    assert "ensemble_mse" in model.val_metrics
    assert "ensemble_spread" in model.val_metrics

    model._trainer = type(
        "DummyTrainer",
        (),
        {
            "is_global_zero": False,
            "sanity_checking": False,
            "current_epoch": 0,
        },
    )()
    model.test_step(batch, batch_idx=0)
    assert "ensemble_mse" in model.test_metrics
    assert "ensemble_mae" in model.test_metrics
    assert "ensemble_spread" in model.test_metrics
