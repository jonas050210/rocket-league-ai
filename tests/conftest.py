import os

import pytest
from fastapi.testclient import TestClient

from rocket_league_ai.api import create_app
from rocket_league_ai.checkpoints import REQUIRED
from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.metadata import package_versions

# Integration tests that launch the real worker/evaluator need the locked
# training extra. Pure unit tests never skip: CI installs every extra, while a
# control-plane-only checkout still verifies validation, storage and API logic.
requires_training = pytest.mark.skipif(
    any(version is None for version in package_versions().values()),
    reason="training extra is not installed (uv sync --frozen --all-extras)",
)

# Protect test runner and spawned simulations from BLAS oversubscription before any NumPy import.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


@pytest.fixture
def client(tmp_path):
    application = create_app(tmp_path / "workspace", shutdown_timeout=5)
    with TestClient(application) as client:
        client.headers["Authorization"] = "Bearer " + application.state.manager.token
        yield client


@pytest.fixture
def manager(client):
    return client.app.state.manager


@pytest.fixture
def small_config():
    return TrainingConfig(
        name="Isolated CPU lifecycle test", device="cpu", hidden_sizes=(32, 32),
        total_timesteps=32768, timesteps_per_iteration=128, batch_size=128, buffer_size=384,
        n_minibatches=2, checkpoint_every_iterations=2,
        episode_timeout_seconds=1, no_touch_timeout_seconds=1,
    )


def unit_checkpoint_writer(path):
    """Byte fixtures for storage tests ONLY. Never passed to a training worker."""
    for name in REQUIRED:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"unit-test-checkpoint-bytes")
