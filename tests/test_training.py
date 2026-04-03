# Standard library
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models.forecaster_module import ForecasterModule
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example

# Model architecture defaults for tests
GRAPH = "1level"
HIDDEN_DIM = 4
HIDDEN_LAYERS = 1
PROCESSOR_LAYERS = 2
MESH_AGGR = "sum"
NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def run_simple_training(datastore, set_output_std, metrics_watch=None):
    """
    Run one epoch of a simple model training setup using the given datastore.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore to load data from for training
    set_output_std : bool
        If --output_std should be set during training
    """

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s
    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        # Keep two devices here since aggregate_and_plot_metrics expects
        # to exercise the multi-device all_gather_cat path.
        devices=2,
        log_every_n_steps=1,
        detect_anomaly=True,
    )

    graph_name = GRAPH
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        standardize=True,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
    )

    _mw = metrics_watch or []
    _vlmw = {0: [1]} if _mw else {}

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    from neural_lam.models import MODELS
    from neural_lam.models.ar_forecaster import ARForecaster

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=HIDDEN_DIM,
        hidden_layers=HIDDEN_LAYERS,
        processor_layers=PROCESSOR_LAYERS,
        mesh_aggr=MESH_AGGR,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=set_output_std,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1, 3],
        metrics_watch=_mw,
        var_leads_metrics_watch=_vlmw,
    )

    wandb.init(mode="disabled")
    trainer.fit(model=model, datamodule=data_module)


@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as it is not a regular "
            "grid datastore."
        )

    run_simple_training(datastore, set_output_std=False)


def test_training_output_std():
    datastore = init_datastore_example("mdp")
    run_simple_training(datastore, set_output_std=True)


def test_all_gather_cat_single_device():
    """
    Test that all_gather_cat preserves tensor shape on single-device runs.
    On a single device, all_gather returns the tensor unchanged (no new
    leading dim), so all_gather_cat should not flatten any existing dims.
    """

    class MockModule:
        """Minimal object with mocked single-device all_gather."""

        def all_gather(self, tensor_to_gather, sync_grads=False):
            return tensor_to_gather

    module = MockModule()
    module.all_gather_cat = ForecasterModule.all_gather_cat.__get__(
        module, MockModule
    )

    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)

    assert result.shape == tensor.shape, (
        f"all_gather_cat changed shape on single device: "
        f"{tensor.shape} -> {result.shape}"
    )
    assert torch.equal(result, tensor)


def test_all_gather_cat_multi_device_simulation():
    """
    Test that all_gather_cat correctly flattens when all_gather adds a
    leading dimension (simulating multi-device behavior).
    """

    class MockModule:
        """Object with mocked multi-device all_gather."""

        def all_gather(self, tensor, sync_grads=False):
            return torch.stack([tensor, tensor], dim=0)

    module = MockModule()
    module.all_gather_cat = ForecasterModule.all_gather_cat.__get__(
        module, MockModule
    )

    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)

    assert result.shape == (
        8,
        3,
        5,
    ), f"all_gather_cat wrong shape on multi-device: {result.shape}"
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(result, expected), (
        "all_gather_cat produced incorrectly ordered/combined values "
        "on multi-device simulation"
    )
