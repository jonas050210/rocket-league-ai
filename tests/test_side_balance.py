"""Side-balanced evaluation and the side-swapped environment.

A checkpoint that only ever played blue must still be measurable on orange,
otherwise every "win rate" is really "win rate on the convenient side".  These
tests check the environment side swap, the observation mirror and the
evaluation protocol that reports both.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlai.envs import MatchEnv
from rlai.evaluation import classify_opponent, evaluate_policy, evaluate_suite
from rlai.modes import get_mode
from rlai.opponents import ChaseOpponent, IdleOpponent
from rlai.policy import MirroredPolicy, PolicyConfig, TorchPolicy


@pytest.fixture(scope="module")
def fixed_policy() -> TorchPolicy:
    torch.manual_seed(5)
    return TorchPolicy.from_config(PolicyConfig(hidden=(32,)))


def _invert(vector) -> np.ndarray:
    """The 180 deg rotation about z used by RLGym's own inversion helpers."""
    return np.array([-vector[0], -vector[1], vector[2]], dtype=np.float32)


def test_env_side_swap_mirrors_the_whole_trajectory(fixed_policy):
    mirrored = MirroredPolicy(fixed_policy)
    blue_env = MatchEnv(get_mode("soccer-1v1"), IdleOpponent(), seed=11, episode_seconds=20)
    orange_env = MatchEnv(
        get_mode("soccer-1v1"), IdleOpponent(), seed=11, episode_seconds=20, policy_team=1
    )
    blue_obs = blue_env.reset(episode_seed=77)
    orange_obs = orange_env.reset(episode_seed=77)
    blue_agent = blue_env.policy_agents[0]
    orange_agent = orange_env.policy_agents[0]

    assert blue_env.policy_agents == blue_env.blue_agents
    assert orange_env.policy_agents == orange_env.orange_agents
    assert orange_env.opponent_agents == orange_env.blue_agents

    max_position_error = 0.0
    max_observation_error = 0.0
    for _ in range(60):
        blue_actions, _, _ = fixed_policy.act({blue_agent: blue_obs[blue_agent]}, deterministic=True)
        orange_actions, _, _ = mirrored.act(
            {orange_agent: orange_obs[orange_agent]}, deterministic=True
        )
        # the mirrored side must decide exactly the same thing
        assert blue_actions[blue_agent] == orange_actions[orange_agent]

        blue_result = blue_env.step({blue_agent: blue_actions[blue_agent]})
        orange_result = orange_env.step({orange_agent: orange_actions[orange_agent]})
        blue_obs, orange_obs = blue_result.obs, orange_result.obs

        max_observation_error = max(
            max_observation_error,
            float(
                np.abs(
                    mirrored._mirror(orange_obs[orange_agent]) - blue_obs[blue_agent]
                ).max()
            ),
        )
        blue_position = blue_env.state.cars[blue_agent].physics.position
        orange_position = orange_env.state.cars[orange_agent].physics.position
        max_position_error = max(
            max_position_error, float(np.abs(blue_position - _invert(orange_position)).max())
        )
        assert blue_env.score_for == orange_env.score_for
        assert blue_env.score_against == orange_env.score_against

    # float32 physics noise only; a broken mirror is off by thousands of units
    assert max_position_error < 1.0
    assert max_observation_error < 1e-3


def test_side_balanced_series_is_reproducible_and_balanced(fixed_policy):
    common = dict(
        mode="soccer-1v1",
        opponent="idle",
        games=4,
        seed=2024,
        match_seconds=6,
        episode_seconds=6,
        sides=("blue", "orange"),
    )
    first = evaluate_policy(fixed_policy, **common)
    second = evaluate_policy(fixed_policy, **common)
    assert first["details"]["games"] == second["details"]["games"]
    assert first["win_rate"] == second["win_rate"]
    sides = [game["side"] for game in first["details"]["games"]]
    assert sides.count("blue") == sides.count("orange") == 2
    assert set(first["side_results"]) == {"blue", "orange"}
    assert first["side_results"]["blue"]["games"] == 2
    assert first["protocol"]["sides"] == ["blue", "orange"]
    assert first["protocol"]["mirror"] is True
    # an odd number of games still stays as balanced as possible
    odd = evaluate_policy(fixed_policy, **{**common, "games": 3})
    odd_sides = [game["side"] for game in odd["details"]["games"]]
    assert odd_sides.count("blue") == 2 and odd_sides.count("orange") == 1


def test_single_side_stays_the_default(fixed_policy):
    result = evaluate_policy(
        fixed_policy,
        mode="soccer-1v1",
        opponent="idle",
        games=1,
        seed=5,
        match_seconds=4,
        episode_seconds=4,
    )
    assert result["sides"] == ["blue"]
    assert all(game["side"] == "blue" for game in result["details"]["games"])
    assert all(game["mirrored"] is False for game in result["details"]["games"])


def test_unknown_side_is_rejected(fixed_policy):
    with pytest.raises(ValueError, match="unknown side"):
        evaluate_policy(
            fixed_policy,
            mode="soccer-1v1",
            opponent="idle",
            games=1,
            match_seconds=4,
            sides=("sideways",),
        )


def test_suite_separates_diagnostics_from_strength(fixed_policy):
    suite = evaluate_suite(
        fixed_policy,
        mode="soccer-1v1",
        opponents=("idle", "chase"),
        games=1,
        seed=31,
        match_seconds=5,
        episode_seconds=5,
        sides=("blue",),
    )
    assert suite["kind"] == "suite"
    assert set(suite["results"]) == {"idle", "chase"}
    assert suite["diagnostic"]["opponents"] == ["idle"]
    assert suite["strength"]["opponents"] == ["chase"]
    # the headline score never mixes a do-nothing baseline into real strength
    assert suite["score"] == suite["strength"]["win_rate"]
    assert suite["protocol"]["opponents"] == ["idle", "chase"]
    assert suite["protocol"]["games_per_opponent"] == 1
    assert classify_opponent("idle") == "diagnostic"
    assert classify_opponent("random") == "diagnostic"
    assert classify_opponent("chase") == "strength"
    assert classify_opponent("self") == "strength"
    assert classify_opponent("model:data/models/x/model.pt") == "strength"


def test_suite_without_strength_opponents_reports_no_score(fixed_policy):
    suite = evaluate_suite(
        fixed_policy,
        mode="soccer-1v1",
        opponents=("idle",),
        games=1,
        seed=3,
        match_seconds=4,
        episode_seconds=4,
        sides=("blue",),
    )
    assert suite["strength"]["measured"] is False
    assert suite["score"] is None
    assert suite["diagnostic"]["measured"] is True


def test_mirror_matches_a_chase_opponent_on_both_sides(fixed_policy):
    """Against a symmetric opponent, both sides must be playable."""
    result = evaluate_policy(
        fixed_policy,
        mode="soccer-1v1",
        opponent="chase",
        games=2,
        seed=909,
        match_seconds=6,
        episode_seconds=6,
        sides=("blue", "orange"),
    )
    assert result["side_results"]["blue"]["games"] == 1
    assert result["side_results"]["orange"]["games"] == 1
    assert result["details"]["games"][1]["mirrored"] is True
    assert result["duration_s"] > 0


def test_evaluation_reports_the_opponent_it_actually_used():
    env = MatchEnv(get_mode("soccer-1v1"), ChaseOpponent(), seed=1, episode_seconds=10)
    env.reset(episode_seed=1)
    assert env.opponent.name == "chase"
