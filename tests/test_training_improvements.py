"""Reward shaping, scenario and PPO-knob behaviour (P2 package)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rlgym", reason="training extra is not installed")

from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.training.environment import build_environment, shaping_rewards  # noqa: E402


def fake_state(car_velocity, ball_velocity=(0.0, 0.0, 0.0)):
    car = SimpleNamespace(
        physics=SimpleNamespace(position=np.array([0.0, -1000.0, 17.0]),
                                linear_velocity=np.array(car_velocity, dtype=float)),
        is_blue=True)
    orange = SimpleNamespace(
        physics=SimpleNamespace(position=np.array([0.0, 1000.0, 17.0]),
                                linear_velocity=np.zeros(3)),
        is_blue=False)
    ball = SimpleNamespace(position=np.array([0.0, 0.0, 93.0]),
                           linear_velocity=np.array(ball_velocity, dtype=float))
    return SimpleNamespace(cars={"blue-0": car, "orange-0": orange}, ball=ball)


def test_shaping_rewards_direction_and_normalization():
    moving_to_ball = shaping_rewards(["blue-0"], fake_state((0.0, 2300.0, 0.0)),
                                     velocity_to_ball_weight=1.0, ball_to_goal_weight=0.0)
    assert moving_to_ball["blue-0"] == pytest.approx(1.0, rel=0.01)  # ball floats above car height
    moving_away = shaping_rewards(["blue-0"], fake_state((0.0, -2300.0, 0.0)),
                                  velocity_to_ball_weight=1.0, ball_to_goal_weight=0.0)
    assert moving_away["blue-0"] == pytest.approx(-1.0, rel=0.01)
    ball_to_goal = shaping_rewards(["blue-0"], fake_state((0.0, 0.0, 0.0), (0.0, 2300.0, 0.0)),
                                   velocity_to_ball_weight=0.0, ball_to_goal_weight=0.5)
    assert ball_to_goal["blue-0"] == pytest.approx(0.5, rel=0.01)
    zero = shaping_rewards(["blue-0"], fake_state((0.0, 2300.0, 0.0)),
                           velocity_to_ball_weight=0.0, ball_to_goal_weight=0.0)
    assert zero["blue-0"] == 0.0  # foundation reward unchanged by default


@pytest.mark.integration
def test_mixed_scenarios_produce_varied_genuine_resets():
    config = TrainingConfig(device="cpu", environment_version="mixed-scenarios-v1", seed=5,
                            episode_timeout_seconds=5, no_touch_timeout_seconds=5)
    env = build_environment(config)
    ball_positions = set()
    for _ in range(8):
        env.reset()
        ball_positions.add(tuple(np.round(env.state.ball.position, 1)))
        env.step({agent: np.array([8]) for agent in env.agents})
    # Not every reset is the centre kickoff any more; all remain valid states.
    assert len(ball_positions) > 1


@pytest.mark.integration
def test_shaped_config_trains_and_saves_checkpoint(manager):
    config = TrainingConfig(
        name="Shaped smoke", device="cpu", hidden_sizes=(32, 32), total_timesteps=256,
        timesteps_per_iteration=128, batch_size=128, buffer_size=384, n_minibatches=2,
        checkpoint_every_iterations=2, episode_timeout_seconds=5, no_touch_timeout_seconds=5,
        reward_version="goal10-touch0.1-shaped-v2", reward_velocity_to_ball_weight=0.3,
        reward_ball_to_goal_weight=0.2, clip_range=0.25, max_grad_norm=1.0,
        environment_version="mixed-scenarios-v1",
    )
    run_id = manager._launch(config)["run_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        if manager.store.run(run_id)["status"] == "completed" and manager.active is None:
            break
        time.sleep(0.5)
    assert manager.store.run(run_id)["status"] == "completed"
    assert manager.checkpoints(run_id).latest_valid(config.digest()) is not None
