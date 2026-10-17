"""Published RLGym/RocketSim contracts and real upstream GAE regression checks."""

import random

import pytest
from conftest import requires_training

pytestmark = [pytest.mark.integration, requires_training]


@pytest.fixture
def env(small_config):
    from rocket_league_ai.training.environment import build_environment

    environment = build_environment(small_config)
    yield environment
    environment.close()


def noops(env):
    import numpy as np

    # LookupTableAction's neutral ground control is index 8, not a guessed raw action vector.
    table = env.action_parser.parser._lookup_table
    neutral = int(np.where((table == 0).all(axis=1))[0][0])
    return {agent: np.array([neutral], dtype=np.int64) for agent in env.agents}


def test_reset_step_shape_dtype_actions_and_seeded_kickoff(env):
    import numpy as np

    with pytest.raises(RuntimeError, match="Reset"):
        env.step({})
    random.seed(123)
    np.random.seed(123)
    first = env.reset()
    assert len(first) == 2 and env.renderer is None
    assert all(obs.shape == (92,) and obs.dtype == np.float32 and obs.flags.c_contiguous
               and np.isfinite(obs).all() for obs in first.values())
    assert all(env.action_space(agent) == ("discrete", 90) for agent in env.agents)
    for _ in range(3):
        env.step(noops(env))
    assert env.state.tick_count == 24
    random.seed(123)
    np.random.seed(123)
    second = env.reset()
    assert first.keys() == second.keys()
    for agent in first:
        np.testing.assert_array_equal(first[agent], second[agent])


def test_no_touch_timeout_is_truncation_and_requires_reset(env):
    env.reset()
    for _ in range(14):
        _, _, terminated, truncated = env.step(noops(env))
        assert not any(terminated.values()) and not any(truncated.values())
    _, _, terminated, truncated = env.step(noops(env))
    assert not any(terminated.values()) and all(truncated.values())
    assert env.state.tick_count == 120
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(noops(env))
    assert len(env.reset()) == 2 and env.needs_reset is False


def test_real_goal_terminates_before_simultaneous_timeout_and_audits_reward(env):
    import numpy as np

    env.reset()
    state = env.state
    state.tick_count = 120  # At the configured time limit as well as a goal.
    state.ball.position = np.array([0, 5500, 94], dtype=np.float32)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    env.set_state(state)
    _, rewards, terminated, truncated = env.step(noops(env))
    assert env.state.goal_scored and all(terminated.values())
    assert not any(truncated.values())
    assert sorted(rewards.values()) == [-10, 10]
    assert env.shared_info["reward_goal_abs"] == 20
    assert env.shared_info["reward_touch"] == 0
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(noops(env))


def test_released_engine_touch_callback_is_per_transition_not_cumulative(env):
    from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition

    env.reset()
    engine = env.transition_engine
    agent, car = next(iter(engine._cars.items()))
    # Exercise the actual release's callback/state extraction, rather than a fabricated Car.
    # This is a contract regression, not a claim that a policy learned to contact the ball.
    for _ in range(2):
        engine._ball_touch_callback(engine._arena, car, None)
        state = engine._get_state()
        assert state.cars[agent].ball_touches == 1
        rewards = env.reward_fn.get_rewards(env.agents, state, {}, {}, env.shared_info)
        assert rewards[agent] == pytest.approx(0.1)
        assert env.shared_info["reward_touch"] == pytest.approx(0.1)
    timeout = NoTouchTimeoutCondition(timeout_seconds=1)
    timeout.reset(env.agents, state, {})
    state.tick_count = 120
    assert not any(timeout.is_done(env.agents, state, {}).values())
    idle = engine._get_state()
    assert idle.cars[agent].ball_touches == 0
    idle.tick_count = 239
    assert not any(timeout.is_done(env.agents, idle, {}).values())
    idle.tick_count = 240
    assert all(timeout.is_done(env.agents, idle, {}).values())


@pytest.mark.parametrize(("truncated", "expected"), [(True, 5.5), (False, 1.0)])
def test_upstream_rust_gae_bootstraps_only_truncations(truncated, expected):
    import numpy as np
    import torch
    from rlgym_learn_algos.ppo import GAETrajectoryProcessor, GAETrajectoryProcessorConfigModel
    from rlgym_learn_algos.ppo.trajectory import Trajectory
    from rlgym_learn_algos.ppo.trajectory_processor import DerivedTrajectoryProcessorConfig

    processor = GAETrajectoryProcessor()
    processor.load(DerivedTrajectoryProcessorConfig(
        trajectory_processor_config=GAETrajectoryProcessorConfigModel(
            gamma=0.9, lmbda=0.95, standardize_rewards=False, reward_clip=None,
        ),
        agent_controller_name="contract-regression", dtype=torch.float32,
        device=torch.device("cpu"), seed=123,
    ))
    trajectory = Trajectory(
        agent_id="blue-0", obs_list=[np.zeros(92, dtype=np.float32)],
        action_list=[np.zeros(1, dtype=np.int64)], log_probs=torch.zeros(1),
        reward_list=[1.0], val_preds=torch.zeros(1), final_obs=np.zeros(92, dtype=np.float32),
        final_val_pred=torch.tensor(5.0), truncated=truncated,
    )
    result, _ = processor.process_trajectories([trajectory])
    assert result[5].item() == pytest.approx(expected)


def test_terminal_environment_resets_even_when_update_cleared_controller_trajectories():
    from rlgym_learn import EnvActionType

    from rocket_league_ai.training.adapter import FoundationController

    # A narrowly isolated controller regression: no learner/update is simulated here.
    controller = FoundationController.__new__(FoundationController)
    controller.current_env_trajectories = {}
    controller.current_env_controlled_agent_ids = {}
    controller.iteration_trajectories = []
    _, actions = controller.get_env_actions(
        {"env": (["blue-0", "orange-0"], [])},
        {"env": (None, None, {"blue-0": True, "orange-0": True},
                 {"blue-0": False, "orange-0": False})},
    )
    assert actions["env"].enum_type == EnvActionType.RESET


def test_response_fence_blocks_only_post_startup_publication(small_config):
    import threading

    from rocket_league_ai.training.environment import build_environment

    fence = threading.Semaphore(0)
    env = build_environment(small_config, response_fence=fence)
    try:
        assert len(env.reset()) == 2  # Initial collector handshake must not wait for an action.
        finished = threading.Event()
        errors = []

        def step():
            try:
                env.step(noops(env))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=step, daemon=True)
        thread.start()
        from test_training_integration import wait_for

        wait_for(lambda: env.state.tick_count == 8, timeout=3, description="real physics transition")
        assert not finished.is_set()  # Real result computed, but not published into Rust yet.
        fence.release()  # What the coordinator does AFTER send_env_actions/drain has returned.
        assert finished.wait(3) and errors == []
        thread.join(timeout=1)
    finally:
        fence.release()
        env.close()
