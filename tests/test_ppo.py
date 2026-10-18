"""PPO maths and one optimisation step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlai.config import TrainingConfig
from rlai.ppo import Batch, compute_gae, flatten_rollout, ppo_update
from rlai.obs import OBS_SIZE
from rlai.policy import ActorCritic


def test_gae_matches_manual_computation():
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    dones = np.zeros((3, 1), dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, np.array([2.0]), 0.9, 0.95)
    # last step: delta = 1 + 0.9*2 - 0 = 2.8, gae = 2.8
    assert advantages[-1, 0] == pytest.approx(2.8, abs=1e-5)
    assert advantages[1, 0] == pytest.approx(3.394, abs=1e-5)
    # first step: delta 1, gae = 1 + 0.9*0.95*3.394
    assert advantages[0, 0] == pytest.approx(3.9019, abs=1e-4)
    assert returns[1, 0] == pytest.approx(advantages[1, 0] + values[1, 0], abs=1e-5)


def test_dones_cut_the_bootstrap():
    rewards = np.ones((2, 1), dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    dones = np.array([[0.0], [1.0]], dtype=np.float32)
    advantages, _ = compute_gae(rewards, values, dones, np.array([100.0]), 0.9, 0.95)
    assert advantages[1, 0] == pytest.approx(1.0, abs=1e-5)
    assert advantages[0, 0] == pytest.approx(1.855, abs=1e-5)


def test_ppo_update_produces_finite_losses():
    config = TrainingConfig(
        total_steps=2048, rollout_steps=256, n_workers=1, minibatch_size=64, epochs=1
    )
    model = ActorCritic(hidden=(32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(0)
    n = 256
    batch = Batch(
        obs=rng.normal(size=(n, OBS_SIZE)).astype(np.float32),
        actions=rng.integers(0, 90, size=n),
        logprobs=np.zeros(n, dtype=np.float32),
        advantages=rng.normal(size=n).astype(np.float32),
        returns=rng.normal(size=n).astype(np.float32),
        values=np.zeros(n, dtype=np.float32),
    )
    losses = ppo_update(model, optimizer, batch, config, device=torch.device("cpu"))
    assert {"losses/policy", "losses/value", "losses/entropy"}.issubset(losses)
    for value in losses.values():
        assert np.isfinite(value)


def test_gae_truncation_uses_bootstrap():
    rewards = np.ones((2, 1), dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    dones = np.ones((2, 1), dtype=np.float32)
    truncated = np.array([[0.0], [1.0]], dtype=np.float32)
    bootstrap = np.array([[0.0], [5.0]], dtype=np.float32)
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        np.zeros(1, dtype=np.float32),
        0.9,
        0.95,
        truncated=truncated,
        bootstrap=bootstrap,
    )
    # step 1 is a timeout: 1 + 0.9 * 5 = 5.5 instead of 1.0
    assert advantages[1, 0] == pytest.approx(5.5, abs=1e-5)
    # step 0 is a real termination: 1.0, no propagation
    assert advantages[0, 0] == pytest.approx(1.0, abs=1e-5)
    # returns equal advantages because values are zero
    assert returns[1, 0] == pytest.approx(5.5, abs=1e-5)


def test_ppo_early_stop_reports_reason():
    config = TrainingConfig(
        total_steps=2048,
        rollout_steps=256,
        n_workers=1,
        minibatch_size=64,
        epochs=10,
        target_kl=1e-6,
    )
    model = ActorCritic(hidden=(32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(1)
    n = 256
    batch = Batch(
        obs=rng.normal(size=(n, OBS_SIZE)).astype(np.float32),
        actions=rng.integers(0, 90, size=n),
        logprobs=rng.normal(size=n).astype(np.float32),
        advantages=rng.normal(size=n).astype(np.float32),
        returns=rng.normal(size=n).astype(np.float32),
        values=np.zeros(n, dtype=np.float32),
    )
    stats = ppo_update(model, optimizer, batch, config, device=torch.device("cpu"))
    assert stats["losses/early_stopped"] == 1.0
    assert stats["losses/epochs_run"] <= 10
    assert stats["losses/grad_norm"] >= 0.0


def test_flatten_rollout_shapes():
    config = TrainingConfig(total_steps=2048, rollout_steps=64, n_workers=2, minibatch_size=64)
    rollout = {
        "obs": np.zeros((64, 2, OBS_SIZE), dtype=np.float32),
        "actions": np.zeros((64, 2), dtype=np.int64),
        "logprobs": np.zeros((64, 2), dtype=np.float32),
        "rewards": np.zeros((64, 2), dtype=np.float32),
        "values": np.zeros((64, 2), dtype=np.float32),
        "dones": np.zeros((64, 2), dtype=np.float32),
    }
    batch = flatten_rollout(rollout, np.zeros(2), config)
    assert batch.obs.shape == (128, OBS_SIZE)
    assert batch.actions.shape == (128,)
