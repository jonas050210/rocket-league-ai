"""Shared fixtures. Every test uses an isolated data directory."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("RLAI_DATA_DIR", str(root))
    return root


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """A tiny but real checkpoint saved through the production code path."""
    import torch

    from rlai import checkpoints
    from rlai.policy import PolicyConfig

    root = tmp_path_factory.mktemp("models")
    config = PolicyConfig(hidden=(32, 32))
    model = config.build()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    manifest = checkpoints.save_checkpoint(
        root,
        model_state=model.state_dict(),
        policy_config=config.to_dict(),
        step=128,
        mode="soccer-1v1",
        label="test",
    )
    manifest["_models_dir"] = str(root)
    return manifest


@pytest.fixture()
def models_dir(trained_model, data_root):
    import shutil

    target = data_root / "models"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(trained_model["_models_dir"]) / trained_model["id"],
        target / trained_model["id"],
    )
    return target


@pytest.fixture()
def model_path(models_dir, trained_model):
    return models_dir / trained_model["path"]
