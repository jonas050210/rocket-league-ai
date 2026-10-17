"""Reward shaping, scenario and PPO-knob behaviour (P2 package)."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rlgym", reason="training extra is not installed")

from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.training.environment import build_environment, shaping_rewards  # noqa: E402


def fake_state(car_velocity, ball_velocity=(0.0, 0.0, 0.0), boost_amount=50.0, ball_touches=0, ball_pos=(0.0, 0.0, 93.0), car_pos=(0.0, -1000.0, 17.0)):
    car = SimpleNamespace(
        physics=SimpleNamespace(position=np.array(car_pos, dtype=float),
                                linear_velocity=np.array(car_velocity, dtype=float)),
        is_blue=True,
        boost_amount=boost_amount,
        ball_touches=ball_touches,
    )
    orange = SimpleNamespace(
        physics=SimpleNamespace(position=np.array([0.0, 1000.0, 17.0]),
                                linear_velocity=np.zeros(3)),
        is_blue=False,
        boost_amount=33.3,
        ball_touches=0,
    )
    ball = SimpleNamespace(position=np.array(ball_pos, dtype=float),
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


def test_pro_v3_shaping_rewards_boost_defense_and_aerial():
    # Boost reward scales with boost reserve
    with_boost = shaping_rewards(
        ["blue-0"],
        fake_state((0.0, 0.0, 0.0), boost_amount=100.0),
        boost_weight=0.4,
    )
    no_boost = shaping_rewards(
        ["blue-0"],
        fake_state((0.0, 0.0, 0.0), boost_amount=0.0),
        boost_weight=0.4,
    )
    assert with_boost["blue-0"] > no_boost["blue-0"] == 0.0

    # Defensive positioning between own goal (0, -5120) and ball (0, -2000)
    defensive = shaping_rewards(
        ["blue-0"],
        fake_state((0.0, 0.0, 0.0), ball_pos=(0.0, -2000.0, 93.0), car_pos=(0.0, -3500.0, 17.0)),
        defensive_weight=0.5,
    )
    assert defensive["blue-0"] > 0.0

    # Aerial touch + fast shot toward orange goal
    aerial_touch = shaping_rewards(
        ["blue-0"],
        fake_state((0.0, 0.0, 0.0), ball_velocity=(0.0, 2000.0, 0.0), ball_touches=1, ball_pos=(0.0, 0.0, 500.0)),
        aerial_shot_weight=0.6,
    )
    assert aerial_touch["blue-0"] > 0.2

    # Speedflip / supersonic mechanics reward
    speedflip_state = fake_state(
        (0.0, 2200.0, 0.0),
        ball_pos=(0.0, 1500.0, 93.0),
        car_pos=(0.0, 0.0, 30.0),
    )
    speedflip_state.cars["blue-0"].is_flipping = True
    speedflip_state.cars["blue-0"].is_supersonic = True
    speedflip_state.cars["blue-0"].physics.forward = np.array([0.0, 1.0, 0.05], dtype=np.float32)
    speedflip_rew = shaping_rewards(["blue-0"], speedflip_state, mechanics_weight=0.8)
    assert speedflip_rew["blue-0"] > 0.3


@pytest.mark.integration
def test_2v2_and_unlimited_boost_environment_produces_92_float_obs():
    config = TrainingConfig(
        device="cpu",
        team_size=2,
        boost_mode="unlimited",
        arena_map="ChampionsField",
        environment_version="mixed-scenarios-v1",
        seed=9,
        episode_timeout_seconds=5,
        no_touch_timeout_seconds=5,
    )
    env = build_environment(config)
    obs = env.reset()
    assert len(obs) == 4  # blue-0, blue-1, orange-0, orange-1
    assert all(arr.shape == (92,) and arr.dtype == np.float32 for arr in obs.values())
    obs, _, _, _ = env.step({agent: np.array([8], dtype=np.int64) for agent in env.agents})
    assert all(car.boost_amount == 100.0 for car in env.state.cars.values())
    env.close()


@pytest.mark.integration
def test_mixed_scenarios_produce_varied_genuine_resets():
    config = TrainingConfig(device="cpu", environment_version="mixed-scenarios-v1", seed=5,
                            episode_timeout_seconds=5, no_touch_timeout_seconds=5)
    env = build_environment(config)
    ball_positions = set()
    for _ in range(12):
        env.reset()
        ball_positions.add(tuple(np.round(env.state.ball.position, 1)))
        env.step({agent: np.array([8]) for agent in env.agents})
    # Not every reset is the centre kickoff any more; all remain valid states.
    assert len(ball_positions) > 1


@pytest.mark.integration
def test_shaped_config_trains_and_saves_checkpoint(manager):
    pytest.importorskip("torch", reason="training extra is not installed (uv sync --frozen --all-extras)")
    pytest.importorskip("rlgym_learn", reason="training extra is not installed (uv sync --frozen --all-extras)")
    config = TrainingConfig(
        name="Shaped smoke", device="cpu", hidden_sizes=(32, 32), total_timesteps=256,
        timesteps_per_iteration=128, batch_size=128, buffer_size=384, n_minibatches=2,
        checkpoint_every_iterations=2, episode_timeout_seconds=5, no_touch_timeout_seconds=5,
        reward_version="goal10-touch0.1-pro-v3", reward_velocity_to_ball_weight=0.3,
        reward_ball_to_goal_weight=0.2, reward_boost_weight=0.1,
        reward_defensive_weight=0.15, reward_aerial_shot_weight=0.25,
        clip_range=0.25, max_grad_norm=1.0,
        environment_version="mixed-scenarios-v1",
    )
    run_id = manager._launch(config)["run_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        status = manager.store.run(run_id)["status"]
        if status in ("completed", "failed", "interrupted") and manager.active is None:
            break
        time.sleep(0.5)
    assert manager.store.run(run_id)["status"] == "completed"
    assert manager.checkpoints(run_id).latest_valid(config.digest()) is not None
