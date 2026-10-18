"""Regression tests for GAE across time-limit truncations.

The rollout buffer concatenates many episodes.  Two separate masks matter at
an episode boundary:

* a *truncation* (clock ran out) must bootstrap the value of the final state,
  because the game would have continued;
* no advantage may ever propagate *backwards across* an episode boundary,
  because the following step belongs to a different episode.

The second rule was violated by the original implementation, which reused the
``terminal`` mask for the recursion.  These tests pin both rules down.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlai.ppo import compute_gae


def reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
    truncated: np.ndarray,
    bootstrap: np.ndarray,
) -> np.ndarray:
    """Independent per-episode implementation used as the oracle.

    Every agent column is treated as its own sequence of episodes: a column is
    cut wherever ``dones`` is set, and each segment is evaluated on its own.
    """
    T, K = rewards.shape
    advantages = np.zeros((T, K), dtype=np.float64)
    for k in range(K):
        # boundaries: every step with dones set ends an episode; the tail
        # after the last boundary is its own (unfinished) episode.
        ends = [t for t in range(T) if dones[t, k] > 0.5]
        if not ends or ends[-1] != T - 1:
            ends.append(T - 1)
        start = 0
        for end in ends:
            running = 0.0
            for t in range(end, start - 1, -1):
                if t == end:
                    if truncated[t, k] > 0.5:
                        # time limit: the game continues, bootstrap the value
                        next_value = float(bootstrap[t, k])
                        bootstrapped = True
                    elif dones[t, k] > 0.5:
                        # real termination (goal): the trajectory ends at zero
                        next_value = 0.0
                        bootstrapped = False
                    else:
                        # unfinished rollout tail: bootstrap from last_value
                        next_value = float(last_value[k])
                        bootstrapped = True
                else:
                    next_value = float(values[t + 1, k])
                    bootstrapped = True
                delta = rewards[t, k] + gamma * next_value * float(bootstrapped) - values[t, k]
                running = delta if t == end else delta + gamma * lam * running
                advantages[t, k] = running
            start = end + 1
    return advantages


def test_truncation_does_not_leak_the_next_episode():
    """The bug: a timeout copied the following episode's advantage backwards."""
    gamma, lam = 0.99, 0.95
    rewards = np.array([[0.0], [0.0], [100.0], [0.0]], dtype=np.float32)
    values = np.ones((4, 1), dtype=np.float32)
    dones = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    truncated = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    bootstrap = np.array([[0.0], [5.0], [0.0], [0.0]], dtype=np.float32)

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        np.array([2.0], dtype=np.float32),
        gamma,
        lam,
        truncated=truncated,
        bootstrap=bootstrap,
    )

    # step 1 is a timeout: it bootstraps from 5.0 and nothing else
    assert advantages[1, 0] == pytest.approx(gamma * 5.0 - 1.0, abs=1e-5)
    # step 0 belongs to the same episode and sees the truncated step 1
    expected_0 = (0.0 + gamma * values[1, 0] - values[0, 0]) + gamma * lam * advantages[1, 0]
    assert advantages[0, 0] == pytest.approx(expected_0, abs=1e-5)
    # step 2 starts a new episode: the huge reward stays inside it
    assert advantages[3, 0] == pytest.approx(0.0 + gamma * 2.0 - 1.0, abs=1e-5)
    assert advantages[2, 0] == pytest.approx(
        100.0 + gamma * values[3, 0] - values[2, 0] + gamma * lam * advantages[3, 0], abs=1e-4
    )
    assert np.all(np.isfinite(returns))
    # the leaked value was ~98.9 before the fix
    assert abs(advantages[1, 0]) < 10.0


def test_matches_per_episode_reference_on_random_rollouts():
    rng = np.random.default_rng(11)
    for trial in range(12):
        T, K = int(rng.integers(6, 40)), int(rng.integers(1, 4))
        rewards = rng.normal(size=(T, K)).astype(np.float32)
        values = rng.normal(size=(T, K)).astype(np.float32)
        # random episode boundaries, roughly half of them truncations
        dones = (rng.random((T, K)) < 0.25).astype(np.float32)
        truncated = dones * (rng.random((T, K)) < 0.5).astype(np.float32)
        bootstrap = rng.normal(size=(T, K)).astype(np.float32)
        last_value = rng.normal(size=K).astype(np.float32)
        gamma, lam = 0.99, 0.95

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value,
            gamma,
            lam,
            truncated=truncated,
            bootstrap=bootstrap,
        )
        expected = reference_gae(
            rewards, values, dones, last_value, gamma, lam, truncated, bootstrap
        )
        assert advantages == pytest.approx(expected.astype(np.float32), abs=1e-3), trial
        assert returns == pytest.approx(advantages + values, abs=1e-5)


def test_real_termination_still_cuts_both_masks():
    """A goal ends the episode: no bootstrap, no propagation."""
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    truncated = np.zeros((3, 1), dtype=np.float32)
    bootstrap = np.full((3, 1), 50.0, dtype=np.float32)

    advantages, _ = compute_gae(
        rewards,
        values,
        dones,
        np.array([9.0], dtype=np.float32),
        0.9,
        0.95,
        truncated=truncated,
        bootstrap=bootstrap,
    )
    assert advantages[1, 0] == pytest.approx(1.0, abs=1e-6)  # terminal: no bootstrap
    assert advantages[0, 0] == pytest.approx(1.0 + 0.9 * 0.95 * 1.0, abs=1e-6)
    assert advantages[2, 0] == pytest.approx(1.0 + 0.9 * 9.0, abs=1e-6)


def test_truncation_at_the_buffer_end_uses_last_value_only_when_not_truncated():
    rewards = np.ones((2, 1), dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    dones = np.ones((2, 1), dtype=np.float32)
    truncated = np.array([[0.0], [1.0]], dtype=np.float32)
    bootstrap = np.array([[0.0], [5.0]], dtype=np.float32)
    advantages, _ = compute_gae(
        rewards,
        values,
        dones,
        np.array([100.0], dtype=np.float32),
        0.9,
        0.95,
        truncated=truncated,
        bootstrap=bootstrap,
    )
    # the truncation bootstraps from 5.0, not from last_value=100
    assert advantages[1, 0] == pytest.approx(1.0 + 0.9 * 5.0, abs=1e-6)
    assert advantages[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_legacy_path_without_truncation_arrays_is_unchanged():
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.zeros((3, 1), dtype=np.float32)
    dones = np.zeros((3, 1), dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, np.array([2.0]), 0.9, 0.95)
    assert advantages[-1, 0] == pytest.approx(2.8, abs=1e-5)
    assert advantages[1, 0] == pytest.approx(3.394, abs=1e-5)
    assert advantages[0, 0] == pytest.approx(3.9019, abs=1e-4)
    assert returns[0, 0] == pytest.approx(advantages[0, 0], abs=1e-5)
