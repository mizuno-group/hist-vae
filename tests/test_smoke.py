import os
from pathlib import Path
import copy

import numpy as np
import pytest
import torch

from histvae import HistVAE
from histvae.utils import get_default_config_path, load_config

RUN_SLOW = os.environ.get("RUN_SLOW_HISTVAE_TESTS") == "1"

def make_config(tmp_path):
    return {
        "device": "cpu",
        "num_points": 4,
        "in_channels": 1,
        "in_dims": 2,
        "max_vals": [1.0, 1.0],
        "bins": 4,
        "num_workers": 0,
        "pin_memory": False,
        "latent_dim": 2,
        "hidden_dims": [1],
        "dropout_conv": 0.0,
        "beta": 1.0,
        "num_classes": 2,
        "num_layers": 1,
        "hidden_head": 2,
        "dropout_head": 0.0,
        "frozen": False,
        "use_pretrain_loss": False,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "weight_decay": 0.0,
        "transform": False,
        "accum_grad": 1,
        "clip_grad": 1.0,
        "save_model_every": 0,
        "patience": 0,
        "log_every": 1,
    }


def make_toy_data(with_labels=False):
    train_data = np.array([
        [0.1, 0.2], [0.15, 0.25], [0.2, 0.3],
        [0.7, 0.8], [0.75, 0.85], [0.8, 0.9],
    ], dtype=np.float32)
    train_group = np.array([0, 0, 0, 1, 1, 1])
    test_data = np.array([
        [0.12, 0.18], [0.18, 0.28], [0.22, 0.32],
        [0.68, 0.82], [0.74, 0.88], [0.78, 0.92],
    ], dtype=np.float32)
    test_group = np.array([10, 10, 10, 11, 11, 11])

    if not with_labels:
        return train_data, train_group, test_data, test_group, None, None

    train_label = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    test_label = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return train_data, train_group, test_data, test_group, train_label, test_label


@pytest.mark.smoke
def test_histvae_prep_data_without_labels(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, _, _ = make_toy_data(with_labels=False)

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )

    assert len(model.train_dataset) == 2
    assert len(model.test_dataset) == 2
    (hist0, hist1), label = model.train_dataset[0]
    assert hist0.shape == (1, 4, 4)
    assert hist1.shape == (1, 4, 4)
    assert label is None



@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_HISTVAE_TESTS=1 to run model-preparation and end-to-end training tests.")
def test_prep_model_pretrain_smoke(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy-pretrain")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    model.prep_model("pretrain")

    assert model.model is not None
    assert model.optimizer is not None
    assert model.trainer is not None
    assert len(model.train_loader) == 1
    batch = next(iter(model.train_loader))
    (hist0, hist1), label = batch
    assert hist0.shape == (2, 1, 4, 4)
    assert hist1.shape == (2, 1, 4, 4)
    assert label.shape == (2,)



@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_HISTVAE_TESTS=1 to run model-preparation and end-to-end training tests.")
def test_prep_model_finetune_smoke(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    pretrain = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-pretrain-ckpt")
    pretrain.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    pretrain.prep_model("pretrain")
    ckpt_path = Path(tmp_path) / "pretrained_state.pt"
    torch.save(pretrain.model.state_dict(), ckpt_path)

    finetune = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-finetune")
    finetune.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    finetune.prep_model("finetune", model_path=str(ckpt_path))

    assert finetune.model is not None
    assert finetune.optimizer is not None
    assert finetune.trainer is not None
    assert finetune.loss_fn is not None
    batch = next(iter(finetune.train_loader))
    (hist0, hist1), label = batch
    assert hist0.shape == (2, 1, 4, 4)
    assert hist1.shape == (2, 1, 4, 4)
    assert label.shape == (2,)



@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_HISTVAE_TESTS=1 to run prediction-path tests.")
def test_predict_requires_finetune_and_returns_arrays(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    pretrain = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-pretrain-predict")
    pretrain.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    pretrain.prep_model("pretrain")
    with pytest.raises(RuntimeError):
        pretrain.predict(pretrain.train_loader)

    ckpt_path = Path(tmp_path) / "predict_state.pt"
    torch.save(pretrain.model.state_dict(), ckpt_path)

    finetune = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-finetune-predict")
    finetune.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    finetune.prep_model("finetune", model_path=str(ckpt_path))

    preds, probs, labels = finetune.predict(finetune.test_loader)
    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
    assert labels.shape == (2,)


@pytest.mark.smoke
def test_installed_distribution_metadata():
    import importlib.metadata as metadata

    assert metadata.version("histvae") == "0.1.0"



@pytest.mark.smoke
def test_src_layout_is_active_package():
    import histvae

    root = Path(__file__).resolve().parents[1]
    src_pkg = root / "src" / "histvae"
    root_pkg = root / "histvae"

    assert src_pkg.exists()
    assert not root_pkg.exists()
    assert Path(histvae.__file__).resolve() == (src_pkg / "__init__.py").resolve()



@pytest.mark.smoke
def test_obsolete_packaging_files_removed():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "setup.py").exists()
    assert not (root / "MANIFEST.in").exists()
    assert not (root / "requirements.txt").exists()



@pytest.mark.smoke
def test_top_level_modules_are_real_files():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "src/histvae/models.py",
        "src/histvae/trainer.py",
        "src/histvae/data_handler.py",
        "src/histvae/utils.py",
    ]:
        path = root / rel
        assert path.exists()
        text = path.read_text()
        assert "from .." not in text



@pytest.mark.smoke
def test_inner_src_directory_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src" / "histvae" / "src").exists()



@pytest.mark.smoke
def test_load_config_uses_packaged_default_when_not_given():
    config, meta = load_config()

    assert isinstance(config, dict)
    assert meta["default_config_path"] == get_default_config_path()
    assert meta["user_config_path"] is None
    assert config["num_points"] == 2048
    assert config["batch_size"] == 32



@pytest.mark.smoke
def test_load_config_merges_user_config_and_runtime_overrides(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        "batch_size: 8\n"
        "nested:\n"
        "  alpha: 2\n"
        "  beta: 3\n"
    )

    config, meta = load_config(
        config_path=str(override_path),
        overrides={
            "device": "cpu",
            "nested": {"beta": 99, "gamma": 7},
        },
    )

    assert meta["user_config_path"] == str(override_path)
    assert config["batch_size"] == 8
    assert config["device"] == "cpu"
    assert config["nested"] == {"alpha": 2, "beta": 99, "gamma": 7}


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_HISTVAE_TESTS=1 to run end-to-end pretrain/finetune tests.")
def test_pretrain_train_end_to_end_toy(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    model = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-pretrain-e2e")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    model.prep_model("pretrain")
    model.train(verbose=False)

    resdir = Path(tmp_path) / "toy-pretrain-e2e"
    assert (resdir / "config.yaml").exists()
    assert (resdir / "history.json").exists()
    assert (resdir / "model_best.pt").exists()
    assert (resdir / "progress_loss.tif").exists()


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_HISTVAE_TESTS=1 to run end-to-end pretrain/finetune tests.")
def test_finetune_train_end_to_end_toy(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    pretrain = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-pretrain-for-finetune")
    pretrain.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    pretrain.prep_model("pretrain")
    pretrain.train(verbose=False)

    ckpt_path = Path(tmp_path) / "toy-pretrain-for-finetune" / "model_best.pt"
    assert ckpt_path.exists()

    finetune = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-finetune-e2e")
    finetune.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    finetune.prep_model("finetune", model_path=str(ckpt_path))
    finetune.train(verbose=False)

    resdir = Path(tmp_path) / "toy-finetune-e2e"
    assert (resdir / "config.yaml").exists()
    assert (resdir / "history.json").exists()
    assert (resdir / "model_best.pt").exists()
    assert (resdir / "progress_loss.tif").exists()
