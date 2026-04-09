# Standard library
from pathlib import Path
from types import SimpleNamespace

# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import metrics
from neural_lam.models.noise_ensemble_graph_lam import NoiseEnsembleGraphLAM
from neural_lam.weather_dataset import WeatherDataset
from tests.dummy_datastore import DummyDatastore


def write_minimal_graph(graph_dir_path: Path, num_grid_nodes: int):
    graph_dir_path.mkdir(parents=True, exist_ok=True)
    num_mesh_nodes = 4

    grid_indices = torch.arange(num_grid_nodes, dtype=torch.long)
    mesh_indices = torch.div(
        grid_indices,
        num_grid_nodes // num_mesh_nodes,
        rounding_mode="floor",
    )
    g2m_edge_index = torch.stack((grid_indices, mesh_indices), dim=0)
    m2g_edge_index = torch.stack((mesh_indices, grid_indices), dim=0)
    ring_edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
    )

    torch.save([ring_edge_index], graph_dir_path / "m2m_edge_index.pt")
    torch.save(g2m_edge_index, graph_dir_path / "g2m_edge_index.pt")
    torch.save(m2g_edge_index, graph_dir_path / "m2g_edge_index.pt")
    torch.save(
        [torch.ones(ring_edge_index.shape[1], 1)],
        graph_dir_path / "m2m_features.pt",
    )
    torch.save(
        torch.ones(g2m_edge_index.shape[1], 1),
        graph_dir_path / "g2m_features.pt",
    )
    torch.save(
        torch.ones(m2g_edge_index.shape[1], 1),
        graph_dir_path / "m2g_features.pt",
    )
    torch.save(
        [torch.ones(num_mesh_nodes, 1)],
        graph_dir_path / "mesh_features.pt",
    )


def build_noise_ensemble_model_and_batch(ar_steps=3, num_pred_samples=4):
    datastore = DummyDatastore(n_grid_points=16, n_timesteps=12)
    graph_name = "1level"
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        write_minimal_graph(
            graph_dir_path=graph_dir_path,
            num_grid_nodes=datastore.boundary_mask.size,
        )

    dataset = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=ar_steps,
        standardize=True,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )
    samples = [dataset[0], dataset[1]]
    batch = tuple(torch.stack(parts, dim=0) for parts in zip(*samples))

    class ModelArgs:
        pass

    model_args = ModelArgs()
    model_args.output_std = False
    model_args.loss = "crps_ensemble"
    model_args.restore_opt = False
    model_args.n_example_pred = 1
    model_args.graph = graph_name
    model_args.hidden_dim = 4
    model_args.hidden_layers = 1
    model_args.processor_layers = 2
    model_args.mesh_aggr = "sum"
    model_args.lr = 1.0e-3
    model_args.val_steps_to_log = [1, 2, 3]
    model_args.metrics_watch = []
    model_args.var_leads_metrics_watch = {}
    model_args.num_past_forcing_steps = 1
    model_args.num_future_forcing_steps = 1
    model_args.num_pred_samples = num_pred_samples
    model_args.noise_dim = 3
    model_args.noise_scale = 0.5

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )
    model = NoiseEnsembleGraphLAM(
        args=model_args,
        datastore=datastore,
        config=config,
    )
    return model, batch, datastore


def test_crps_ensemble_is_zero_for_perfect_ensemble():
    prediction = torch.ones(1, 3, 2, 1, 1)
    target = torch.ones(1, 2, 1, 1)

    loss = metrics.crps_ensemble(prediction, target, pred_std=None)

    assert torch.allclose(loss, torch.zeros_like(loss))


def test_noise_ensemble_graph_lam_common_step_shape():
    model, batch, datastore = build_noise_ensemble_model_and_batch()

    prediction, target, pred_std, _ = model.common_step(batch)

    assert pred_std is None
    assert prediction.shape[0] == batch[0].shape[0]
    assert prediction.shape[1] == model.num_pred_samples
    assert prediction.shape[2] == target.shape[1]
    assert prediction.shape[3] == target.shape[2]
    assert prediction.shape[4] == target.shape[3]
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])


def test_noise_ensemble_graph_lam_training_step_returns_finite_loss():
    model, batch, _ = build_noise_ensemble_model_and_batch()
    model.log_dict = lambda *args, **kwargs: None

    loss = model.training_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_noise_ensemble_graph_lam_test_step_keeps_full_grid_spatial_losses():
    model, batch, datastore = build_noise_ensemble_model_and_batch()
    model.log_dict = lambda *args, **kwargs: None
    model.plot_examples = lambda *args, **kwargs: None
    model._trainer = SimpleNamespace(is_global_zero=False)

    model.test_step(batch, batch_idx=0)

    assert len(model.spatial_loss_maps) == 1
    assert model.spatial_loss_maps[0].shape[-1] == datastore.boundary_mask.size
