"""Evaluation is deterministic for a seed and reports honest numbers."""

from __future__ import annotations

from rlai.evaluation import evaluate_model


def test_evaluation_reports_complete_record(model_path):
    result = evaluate_model(
        model_path,
        mode="soccer-1v1",
        opponent="idle",
        games=2,
        seed=11,
        match_seconds=6,
        episode_seconds=6,
        device="cpu",
    )
    assert result["games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert len(result["details"]["games"]) == 2
    assert result["goals_for"] >= 0 and result["goals_against"] >= 0
    for game in result["details"]["games"]:
        assert game["outcome"] in {"win", "loss", "draw"}
        assert game["ticks"] >= 0


def test_same_seed_same_result(model_path):
    kwargs = dict(
        mode="soccer-1v1", opponent="idle", games=1, seed=99, match_seconds=5, episode_seconds=5
    )
    first = evaluate_model(model_path, **kwargs)
    second = evaluate_model(model_path, **kwargs)
    assert first["wins"] == second["wins"]
    assert first["goals_for"] == second["goals_for"]
    assert first["goals_against"] == second["goals_against"]
    assert first["details"]["games"][0]["ticks"] == second["details"]["games"][0]["ticks"]


def test_evaluate_policy_reports_confidence_interval(model_path):
    from rlai.evaluation import evaluate_policy, load_policy, policy_obs_version

    policy = load_policy(model_path)
    assert policy_obs_version(policy) == 2
    result = evaluate_policy(
        policy,
        mode="soccer-1v1",
        opponent="idle",
        games=3,
        seed=5,
        match_seconds=5,
        episode_seconds=5,
        device="cpu",
    )
    low, high = result["win_rate_ci95"]
    assert 0.0 <= low <= result["win_rate"] <= high <= 1.0
    decisive = result["wins"] + result["losses"]
    assert result["win_rate_decisive"] == (
        result["wins"] / decisive if decisive else None
    )
    assert result["games"] == 3
    assert len(result["details"]["games"]) == 3
    assert result["touches"] >= 0


def test_legacy_v1_policy_is_still_evaluable(tmp_path):
    """Old 190-dim checkpoints must keep working after the v2 obs change."""
    from rlai import checkpoints
    from rlai.evaluation import evaluate_policy, load_policy, policy_obs_version
    from rlai.policy import PolicyConfig

    config = PolicyConfig(obs_size=190, hidden=(16,))
    model = config.build()
    manifest = checkpoints.save_checkpoint(
        tmp_path,
        model_state=model.state_dict(),
        policy_config=config.to_dict(),
        step=1,
        mode="soccer-1v1",
    )
    policy = load_policy(tmp_path / manifest["path"])
    assert policy_obs_version(policy) == 1
    result = evaluate_policy(
        policy,
        mode="soccer-1v1",
        opponent="idle",
        games=1,
        seed=3,
        match_seconds=4,
        episode_seconds=4,
        device="cpu",
    )
    assert result["games"] == 1
    assert result["win_rate_ci95"][0] <= result["win_rate"] <= result["win_rate_ci95"][1]


def test_best_candidate_requires_a_win():
    from rlai.trainer import is_best_candidate

    assert is_best_candidate({"wins": 1, "win_rate": 0.25}, -1.0)
    assert is_best_candidate({"wins": 2, "win_rate": 0.5}, 0.4)
    assert not is_best_candidate({"wins": 0, "win_rate": 0.0}, -1.0)
    assert not is_best_candidate({"wins": 1, "win_rate": 0.25}, 0.25)


def test_wilson_interval_is_sane():
    from rlai.evaluation import wilson_interval

    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(10, 10)
    assert low > 0.6 and high == 1.0
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    # more games -> tighter interval
    assert (wilson_interval(50, 100)[1] - wilson_interval(50, 100)[0]) < (
        wilson_interval(5, 10)[1] - wilson_interval(5, 10)[0]
    )
