"""Regression tests for the learning overhaul (docs/LEARNING_OVERHAUL.md).

Each test pins one root cause of the 2M-step stall: farmable proxy rewards,
a win-rate-only curriculum gate, kickoff-only state coverage, silent
checkpoint config inheritance, unobservable own velocity, team-mixed reward
accounting and tick/decision unit confusion.  Nothing here trains.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlai.config import TrainingConfig
from rlai.curriculum import Curriculum, Stage, stages_for_mode
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.opponents import IdleOpponent
from rlai.resume import INHERITED_FIELDS, merge_config
from rlai.rewards import RewardWeights, SoccerReward


def _env(**kwargs):
    env = MatchEnv(get_mode("soccer-1v1"), IdleOpponent(), seed=2, episode_seconds=10, **kwargs)
    env.reset()
    return env


def _reward_setup(weights=None):
    env = _env()
    reward = SoccerReward(env.spec, weights or RewardWeights())
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared: dict = {}
    reward.reset(agents, state, shared)
    blue = next(a for a in agents if state.cars[a].team_num == 0)
    return reward, state, agents, shared, blue


# ---------------------------------------------------------------------------
# Reward: proxies must not be farmable, the task must be shaped
# ---------------------------------------------------------------------------

def test_parked_car_facing_ball_earns_nothing():
    """The old optimum (park & face the ball, ~4.5/episode) now pays zero."""
    reward, state, agents, shared, blue = _reward_setup()
    total = 0.0
    for _ in range(225):  # one old 15 s episode in decision steps
        values = reward.get_rewards(agents, state, {}, {}, shared)
        total += values[blue]
    assert abs(total) < 1e-6


def test_ball_oscillation_nets_zero_ball_progress():
    reward, state, agents, shared, blue = _reward_setup(
        RewardWeights(progress=0.0, touch=0.0)
    )
    start = state.ball.position.copy()
    total = 0.0
    for offset in (500.0, 1000.0, 500.0, 0.0):  # toward orange goal and back
        state.ball.position = start + np.array([0.0, offset, 0.0], dtype=np.float32)
        total += reward.get_rewards(agents, state, {}, {}, shared)[blue]
    assert abs(total) < 1e-5


def test_ball_moving_toward_goal_is_the_dense_signal():
    reward, state, agents, shared, blue = _reward_setup(
        RewardWeights(progress=0.0, touch=0.0, ball_progress=1.0)
    )
    state.ball.position = state.ball.position + np.array([0.0, 1000.0, 0.0], dtype=np.float32)
    value = reward.get_rewards(agents, state, {}, {}, shared)[blue]
    assert value == pytest.approx(1.0, abs=0.05)


def test_touch_pays_for_direction_and_counts_shots():
    from rlai.rewards import TOUCH_COOLDOWN_STEPS

    reward, state, agents, shared, blue = _reward_setup()
    # hard touch straight at the orange goal from midfield.  ``ball_touches``
    # is a per-step counter that the engine zeroes after every state read.
    state.ball.linear_velocity = np.array([0.0, 2000.0, 0.0], dtype=np.float32)
    state.cars[blue].ball_touches = 1
    good = reward.get_rewards(agents, state, {}, {}, shared)[blue]
    state.cars[blue].ball_touches = 0
    # let the anti-tap cooldown elapse without a contact in between
    for _ in range(TOUCH_COOLDOWN_STEPS):
        reward.get_rewards(agents, state, {}, {}, shared)
    # a second contact, ball fired at the own goal instead
    state.ball.linear_velocity = np.array([0.0, -2000.0, 0.0], dtype=np.float32)
    state.cars[blue].ball_touches = 1
    bad = reward.get_rewards(agents, state, {}, {}, shared)[blue]
    assert good > 0.5 > bad
    stats = shared["reward_stats_by_agent"][blue]
    assert stats["touches"] == 2
    assert stats["shots"] == 1


def test_reward_stats_are_per_agent_not_team_mixed():
    reward, state, agents, shared, blue = _reward_setup()
    orange = next(a for a in agents if a != blue)
    state.cars[blue].ball_touches += 1
    reward.get_rewards(agents, state, {}, {}, shared)
    assert shared["reward_stats_by_agent"][blue]["touch"] > 0
    assert shared["reward_stats_by_agent"][orange]["touch"] == 0.0


def test_legacy_state_bonuses_default_to_zero_but_still_parse():
    weights = RewardWeights.from_dict(
        {"alignment": 0.02, "approach": 0.03, "progress": 0.9, "demo": 0.3}
    )
    assert weights.alignment == 0.02 and weights.bump == 0.3  # exact-resume compat
    assert RewardWeights().alignment == 0.0 and RewardWeights().approach == 0.0


# ---------------------------------------------------------------------------
# Curriculum: stages test the skill they teach and explain themselves
# ---------------------------------------------------------------------------

def _draws_with_touches(n, touches=3):
    return [
        {"result": "draw", "goals_for": 0, "goals_against": 0, "touches": touches}
        for _ in range(n)
    ]


def test_first_stage_promotes_on_touches_without_any_win():
    curriculum = Curriculum(stages_for_mode("soccer-1v1"))
    assert curriculum.stage.name == "first-touch"
    assert "touches_per_episode" in (curriculum.blocked_reason() or "")
    promoted = None
    for _ in range(3):
        promoted = curriculum.observe(_draws_with_touches(20)) or promoted
    assert promoted == "shooting"
    assert curriculum.last_event == "promotion"


def test_blocked_reason_is_exact_and_criteria_are_reported():
    curriculum = Curriculum(stages_for_mode("soccer-1v1"))
    curriculum.observe(_draws_with_touches(40, touches=0))
    status = curriculum.status()
    names = {row["name"]: row for row in status["criteria"]}
    assert names["touches_per_episode"]["required"] == 1.5
    assert names["touches_per_episode"]["observed"] == 0.0
    assert names["touches_per_episode"]["ok"] is False
    assert "touches_per_episode 0 < required 1.5" in status["blocked_reason"]


def test_curriculum_window_survives_a_checkpoint_round_trip():
    stages = stages_for_mode("soccer-1v1")
    curriculum = Curriculum(stages)
    curriculum.observe(_draws_with_touches(25))
    described = curriculum.describe()
    restored = Curriculum(
        stages,
        start_index=described["index"],
        stage_iterations=described["iterations"],
        total_iterations=described["total_iterations"],
        window_episodes=described["window"],
    )
    assert restored.metrics()["curriculum/window_episodes"] == 25.0
    assert restored.metrics()["curriculum/window_touches"] == pytest.approx(3.0)


def test_demotion_and_stagnation_are_reported_not_silent():
    stages = (
        Stage("easy", "idle", min_iterations=1, win_rate=0.5, min_episodes=4, window=10),
        Stage(
            "hard", "chase", min_iterations=1, win_rate=0.5, min_episodes=4, window=10,
            demote_win_rate=0.2, stagnation_iterations=3,
        ),
        Stage("final", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
    )
    curriculum = Curriculum(stages, start_index=1)
    losses = [{"result": "loss", "goals_for": 0, "goals_against": 1} for _ in range(6)]
    assert curriculum.observe(losses) == "easy"
    assert curriculum.last_event == "demotion"

    stalled = Curriculum(stages, start_index=1)
    # 33 % wins: above the demotion floor, below the promotion bar - the
    # stage neither moves up nor down, so it must report stagnation
    mixed = [{"result": "win"}, {"result": "loss"}, {"result": "loss"}]
    for _ in range(3):
        stalled.observe(mixed)
    assert stalled.last_event == "stagnation"
    assert stalled.blocked_reason()


def test_stage_without_any_criteria_is_a_config_error():
    stages = (
        Stage("stuck", "idle", min_iterations=1, win_rate=0.0),
        Stage("final", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
    )
    with pytest.raises(ValueError, match="no promotion criteria"):
        Curriculum(stages)


# ---------------------------------------------------------------------------
# Config / resume precedence
# ---------------------------------------------------------------------------

def _config(**overrides):
    base = dict(
        run_name="unit", total_steps=4096, rollout_steps=64, n_workers=1,
        minibatch_size=64, policy={"hidden": [16]}, device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def test_episode_horizon_follows_the_request_not_the_checkpoint():
    """The reported bug: requested 60 s was silently restored as 15 s."""
    assert "episode_seconds" not in INHERITED_FIELDS
    checkpoint = _config(episode_seconds=15.0).to_dict()
    merged, overrides, _inherited = merge_config(checkpoint, _config(episode_seconds=60.0))
    assert merged.episode_seconds == 60.0
    entry = next(item for item in overrides if item["field"] == "episode_seconds")
    assert entry["used"] == 60.0 and entry["source"] == "request"


def test_reward_is_inherited_by_default_and_explicitly_overridable():
    checkpoint = _config().to_dict()
    checkpoint["reward"]["touch"] = 0.35
    requested = _config()
    requested.reward.touch = 0.4

    merged, overrides, _ = merge_config(checkpoint, requested)
    assert merged.reward.touch == 0.35
    assert next(i for i in overrides if i["field"] == "reward")["source"] == "inherited"

    merged, overrides, inherited = merge_config(
        checkpoint, requested, explicit_overrides=("reward",)
    )
    assert merged.reward.touch == 0.4
    assert next(i for i in overrides if i["field"] == "reward")["source"] == "explicit"
    assert "reward" not in inherited


def test_unknown_explicit_override_is_refused_before_training():
    with pytest.raises(ValueError, match="cannot explicitly override"):
        merge_config(_config().to_dict(), _config(), explicit_overrides=("nonsense",))


def test_resume_plan_shows_overrides_and_effective_config(data_root):
    from rlai import checkpoints
    from rlai.resume import plan_resume

    models = data_root / "models"
    old = _config(episode_seconds=15.0)
    old.reward.touch = 0.35
    policy_config = old.policy_config()
    model = policy_config.build()
    checkpoints.save_checkpoint(
        models,
        model_state=model.state_dict(),
        policy_config=policy_config.to_dict(),
        step=512,
        run_id="run-a",
        mode=old.mode,
        training_config=old.to_dict(),
        extra={"iteration": 8, "obs_version": old.obs_version},
    )
    requested = _config(episode_seconds=60.0, total_steps=8192)
    plan = plan_resume(models, requested, mode="continue", run_id="run-b")
    assert plan.will_resume
    assert plan.effective_config is not None
    assert plan.effective_config["episode_seconds"] == 60.0
    assert plan.effective_config["reward"]["touch"] == 0.35  # inherited, shown
    assert any(item["field"] == "reward" for item in plan.overrides)
    assert any("replaced by the checkpoint" in w for w in plan.warnings)

    plan2 = plan_resume(
        models, requested, mode="continue", run_id="run-b",
        explicit_overrides=("reward",),
    )
    assert plan2.phase_change == ["reward"]
    assert plan2.effective_config["reward"]["touch"] == requested.reward.touch


def test_reset_mode_is_validated():
    assert _config(reset_mode="drill").reset_mode == "drill"
    with pytest.raises(Exception):
        _config(reset_mode="chaos")


# ---------------------------------------------------------------------------
# Observation v3: own velocity becomes observable, v2 stays intact
# ---------------------------------------------------------------------------

def test_obs_v3_extends_v2_and_exposes_own_velocity():
    from rlai.obs import (
        ACTION_START_BY_VERSION,
        VELOCITY_START,
        build_observation,
        mirror_observation,
    )

    env = _env()
    agent = env.policy_agents[0]
    state = env.state
    car = state.cars[agent]
    car.physics.linear_velocity = car.physics.forward * 1150.0  # half supersonic
    v2 = build_observation(state, agent, env.spec, 3, obs_version=2)
    v3 = build_observation(state, agent, env.spec, 3, obs_version=3)
    assert v2.shape == (198,) and v3.shape == (204,)
    # everything before the v2 action block is identical; the action one-hot
    # simply moved behind the new velocity block
    assert np.allclose(v3[: ACTION_START_BY_VERSION[2]], v2[: ACTION_START_BY_VERSION[2]])
    assert v3[ACTION_START_BY_VERSION[3] + 3] == 1.0
    forward_speed = v3[VELOCITY_START]
    assert forward_speed == pytest.approx(0.5, abs=0.05)
    # car-frame velocities are mirror invariant
    mirrored = mirror_observation(v3, obs_version=3)
    assert np.allclose(mirrored[VELOCITY_START : VELOCITY_START + 6], v3[VELOCITY_START : VELOCITY_START + 6])
    env.close()


# ---------------------------------------------------------------------------
# Env/workers: drill starts, decision-step lengths, honest components
# ---------------------------------------------------------------------------

def test_drill_reset_starts_cars_behind_the_ball():
    env = MatchEnv(
        get_mode("soccer-1v1"), IdleOpponent(), seed=5, episode_seconds=10, reset_mode="drill"
    )
    for _ in range(5):
        env.reset()
        ball_y = float(env.state.ball.position[1])
        for agent in env.policy_agents:  # blue attacks +y
            assert float(env.state.cars[agent].physics.position[1]) <= ball_y + 1.0
        for agent in env.opponent_agents:  # orange attacks -y
            assert float(env.state.cars[agent].physics.position[1]) >= ball_y - 1.0
    env.close()


def test_episode_length_is_decision_steps_and_components_sum_to_return():
    from rlai.policy import PolicyConfig, TorchPolicy
    from rlai.workers import collect_rollout

    env = MatchEnv(
        get_mode("soccer-1v1"), IdleOpponent(), seed=1, episode_seconds=2, reset_mode="drill"
    )
    policy = TorchPolicy.from_config(PolicyConfig(obs_size=198, hidden=(32, 32)), device="cpu")
    out = collect_rollout(env, policy, 90, seed=11, fresh=True)
    out = collect_rollout(env, policy, 90, seed=12, fresh=False) | out
    episodes = out["episodes"]
    assert episodes, "a 2 s horizon must finish episodes within 180 decisions"
    for episode in episodes:
        assert episode["length"] <= 31  # 2 s at 15 Hz, not 240 ticks
        assert episode["length_ticks"] == episode["length"] * env.tick_skip
        assert "touches" in episode and "shots" in episode
        # learning-team components now add up to the reported return
        assert sum(episode["reward_components"].values()) == pytest.approx(
            episode["reward"], abs=1e-4
        )
    env.close()


# ---------------------------------------------------------------------------
# Store/report evidence
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Round 2: shot accuracy, touch anti-farming, discounted potentials
# ---------------------------------------------------------------------------

def test_shot_on_target_projection():
    from rlai.rewards import shot_on_target

    goal = [0, 5120, 321]
    assert shot_on_target([0, 0, 93], [0, 2000, 0], goal) is True
    assert shot_on_target([0, 0, 93], [1200, 2000, 0], goal) is False  # wide
    assert shot_on_target([0, 4000, 93], [0, 1000, 900], goal) is False  # over
    assert shot_on_target([0, 0, 93], [0, -2000, 0], goal) is False  # backward


def test_off_target_power_is_counted_but_unpaid():
    reward, state, agents, shared, blue = _reward_setup()
    # hard drive toward the goal's half but metres wide of the post
    state.ball.linear_velocity = np.array([1600.0, 1600.0, 0.0], dtype=np.float32)
    state.cars[blue].ball_touches += 1
    reward.get_rewards(agents, state, {}, {}, shared)
    stats = shared["reward_stats_by_agent"][blue]
    assert stats["shots"] == 0
    assert stats["shots_off_target"] == 1
    assert stats["shot"] == 0.0  # no shooting bonus for missing the goal


def test_touch_cooldown_stops_tap_farming_but_not_counting():
    from rlai.rewards import TOUCH_COOLDOWN_STEPS

    reward, state, agents, shared, blue = _reward_setup(RewardWeights(touch=1.0))
    paid_steps = 0
    for _ in range(TOUCH_COOLDOWN_STEPS * 3):
        # one contact during this decision step; the engine zeroes the
        # counter after every state read, so each step starts from zero
        state.cars[blue].ball_touches = 1
        before = shared["reward_stats_by_agent"][blue]["touch"]
        reward.get_rewards(agents, state, {}, {}, shared)
        if shared["reward_stats_by_agent"][blue]["touch"] > before:
            paid_steps += 1
        state.cars[blue].ball_touches = 0
    stats = shared["reward_stats_by_agent"][blue]
    assert stats["touches"] == TOUCH_COOLDOWN_STEPS * 3  # every contact counted
    assert paid_steps == 3  # ...but paid at most once per cooldown


def test_discounted_potentials_stay_bounded_in_discounted_sum():
    env = _env()
    reward = SoccerReward(env.spec, RewardWeights(progress=0.0, touch=0.0), potential_gamma=0.995)
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared: dict = {}
    reward.reset(agents, state, shared)
    blue = next(a for a in agents if state.cars[a].team_num == 0)
    goal = reward._opponent_goal(0)
    from rlai.rewards import _distance

    d0 = _distance(state.ball.position, goal)
    discounted = 0.0
    for t in range(400):  # nothing moves: discounted total must telescope
        discounted += (0.995**t) * reward.get_rewards(agents, state, {}, {}, shared)[blue]
    # sum_t gamma^t (d - gamma*d)/1000 = d * (1 - gamma^400) / 1000 exactly
    assert discounted == pytest.approx(d0 * (1 - 0.995**400) / 1000.0, abs=1e-3)
    env.close()


# ---------------------------------------------------------------------------
# Round 2: drill variety, obs v4, fast inference, per-worker opponents
# ---------------------------------------------------------------------------

def test_drill_scenarios_cover_attack_defense_and_race():
    from collections import Counter

    env = MatchEnv(
        get_mode("soccer-1v1"), IdleOpponent(), seed=9, episode_seconds=10, reset_mode="drill"
    )
    seen = Counter()
    for _ in range(60):
        env.reset()
        scenario = env.env.shared_info.get("drill_scenario")
        seen[scenario] += 1
        if scenario == "defense":
            # ball travels at the learning (blue) goal, defender goal-side
            assert float(env.state.ball.linear_velocity[1]) < -700.0
            agent = env.policy_agents[0]
            assert float(env.state.cars[agent].physics.position[1]) <= float(
                env.state.ball.position[1]
            ) + 1.0
        if scenario == "race":
            assert abs(float(env.state.ball.position[1])) <= 1200.0
    assert seen["attack"] and seen["defense"] and seen["race"]
    env.close()


def test_obs_v4_orientation_block_and_mirror():
    from rlai.obs import (
        ACTION_START_BY_VERSION,
        ORIENTATION_START,
        build_observation,
        mirror_observation,
        obs_size,
    )

    env = _env()
    agent = env.policy_agents[0]
    v3 = build_observation(env.state, agent, env.spec, 5, obs_version=3)
    v4 = build_observation(env.state, agent, env.spec, 5, obs_version=4)
    assert obs_size(4) == 222 and v4.shape == (222,)
    assert np.allclose(v4[: ACTION_START_BY_VERSION[3]], v3[: ACTION_START_BY_VERSION[3]])
    assert v4[ACTION_START_BY_VERSION[4] + 5] == 1.0
    up = v4[ORIENTATION_START : ORIENTATION_START + 3]
    assert up[2] == pytest.approx(1.0, abs=1e-4)  # upright on the ground
    opp_fwd = v4[ORIENTATION_START + 3 + 2 * 3 : ORIENTATION_START + 3 + 3 * 3]
    assert np.linalg.norm(opp_fwd) == pytest.approx(1.0, abs=1e-3)  # unit forward
    mirrored = mirror_observation(v4, obs_version=4)
    assert mirrored[ORIENTATION_START] == -v4[ORIENTATION_START]
    assert mirrored[ORIENTATION_START + 1] == -v4[ORIENTATION_START + 1]
    assert mirrored[ORIENTATION_START + 2] == v4[ORIENTATION_START + 2]
    # car-frame forwards survive the side swap unchanged
    assert np.allclose(
        mirrored[ORIENTATION_START + 3 : ORIENTATION_START + 18],
        v4[ORIENTATION_START + 3 : ORIENTATION_START + 18],
    )
    env.close()


def test_act_batch_matches_act_and_true_logprobs():
    import torch

    from rlai.policy import PolicyConfig, TorchPolicy

    torch.manual_seed(3)
    policy = TorchPolicy.from_config(PolicyConfig(obs_size=198, hidden=(32, 32)), device="cpu")
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(4, 198)).astype(np.float32)
    actions, logprobs, values = policy.act_batch(batch, deterministic=True)
    via_dict = policy.act({str(i): batch[i] for i in range(4)}, deterministic=True)
    assert [via_dict[0][str(i)] for i in range(4)] == actions.tolist()
    with torch.no_grad():
        logits, vals = policy.model(torch.as_tensor(batch))
    expected_lp = torch.log_softmax(logits, dim=-1).gather(
        1, torch.as_tensor(actions).unsqueeze(-1)
    ).squeeze(-1).numpy()
    assert np.allclose(logprobs, expected_lp, atol=1e-5)
    assert np.allclose(values, vals.numpy(), atol=1e-6)


def test_mixed_stage_draws_opponents_per_worker(data_root):
    from rlai.store import Store
    from rlai.trainer import Trainer

    config = TrainingConfig(
        run_name="unit", total_steps=8192, rollout_steps=64, n_workers=8,
        minibatch_size=64, policy={"hidden": [16]}, device="cpu", curriculum=True,
    )
    trainer = Trainer(config, "unit-run", Store(data_root / "rlai.sqlite3"))
    trainer.curriculum.index = 3  # the "mixed" stage
    names, states, ids, stage_name, reset_mode, seconds = trainer._select_opponents()
    assert stage_name == "mixed"
    assert len(names) == 8 and set(names) <= {"chase", "random"}
    assert states == [None] * 8 and ids == [None] * 8
    assert reset_mode == "mixed" and seconds == 45.0


def test_detect_eval_stall():
    from rlai.trainer import detect_eval_stall

    flat = [{"goal_diff": -5, "wins": 0, "touches": 10} for _ in range(4)]
    assert detect_eval_stall(flat) is True
    improving_touches = flat[:3] + [{"goal_diff": -5, "wins": 0, "touches": 40}]
    assert detect_eval_stall(improving_touches) is False
    a_win = flat[:3] + [{"goal_diff": -2, "wins": 1, "touches": 10}]
    assert detect_eval_stall(a_win) is False
    assert detect_eval_stall(flat[:3]) is False  # not enough history


def test_best_pointers_are_per_protocol(data_root):
    from rlai import checkpoints
    from rlai.policy import PolicyConfig

    models = data_root / "models"
    policy = PolicyConfig(hidden=(16,))
    manifest = checkpoints.save_checkpoint(
        models, model_state=policy.build().state_dict(),
        policy_config=policy.to_dict(), step=64, mode="soccer-1v1",
    )
    proto_a = {"opponent": "idle", "games": 8, "seed": 1}
    proto_b = {"opponent": "chase", "games": 8, "seed": 1}
    checkpoints.set_best(models, manifest["id"], 0.9, protocol=proto_a)
    # the easy-protocol score must not act as the incumbent for the hard one
    assert checkpoints.best_record(models, protocol=proto_b) == {}
    assert checkpoints.best_record(models, protocol=proto_a)["score"] == pytest.approx(0.9)


def test_episode_window_stats_compare_first_and_last(data_root):
    from rlai.store import Store

    store = Store(data_root / "test.sqlite3")
    run_id = store.create_run("unit", "soccer-1v1", {}, 1000)
    for index in range(10):
        store.log_episode(run_id, index, 1.0, 300, 0, 0, "draw", step=index, touches=0)
    for index in range(10, 20):
        store.log_episode(
            run_id, index, 3.0, 300, 1, 0, "win", step=index, touches=4, shots=2
        )
    first = store.episode_window_stats(run_id, which="first", window=10)
    last = store.episode_window_stats(run_id, which="last", window=10)
    assert first["touches_per_episode"] == 0.0 and first["goals_per_episode"] == 0.0
    assert last["touches_per_episode"] == 4.0 and last["win_rate"] == 1.0
    assert last["shot_conversion"] == pytest.approx(0.5)
