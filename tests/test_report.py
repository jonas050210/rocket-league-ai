"""Post-run statistics report: honest labelling, pairing and persistence.

Everything here is synthetic (no simulator, no training) so the assertions are
about *what the report claims*, which is the part that must never be invented.
"""

from __future__ import annotations

import json

import pytest
import torch

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.policy import PolicyConfig
from rlai.report import build_run_report, compare_runs, persist_report
from rlai.store import Store

SCHEMA = "rlai-run-report-v1"


def _store(data_root) -> Store:
    return Store(data_root / "rlai.sqlite3")


def _config(**overrides) -> TrainingConfig:
    base = {
        "run_name": "report unit",
        "total_steps": 4000,
        "rollout_steps": 256,
        "n_workers": 1,
        "minibatch_size": 256,
        "eval_interval": 0,
        "policy": {"hidden": [16]},
        "device": "cpu",
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _protocol(opponent: str, seed: int = 1000, **overrides) -> dict:
    protocol = {
        "mode": "soccer-1v1",
        "opponent": opponent,
        "games": 4,
        "seed": seed,
        "match_seconds": 30.0,
        "episode_seconds": 30.0,
        "sides": ["blue", "orange"],
        "mirror": True,
        "stochastic": False,
        "obs_version": 2,
        "opponent_class": "strength" if opponent not in ("idle", "random") else "diagnostic",
    }
    protocol.update(overrides)
    return protocol


def _evaluation(
    run_id: str,
    opponent: str,
    phase: str,
    *,
    wins: int = 1,
    losses: int = 2,
    draws: int = 1,
    games: int = 4,
    model_id: str = "m-1",
    protocol: dict | None = None,
    **extra,
) -> dict:
    protocol = protocol or _protocol(opponent)
    result = {
        "run_id": run_id,
        "model_id": model_id,
        "mode": "soccer-1v1",
        "opponent": opponent,
        "phase": phase,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "goals_for": wins * 2,
        "goals_against": losses * 2,
        "win_rate": wins / games if games else None,
        "win_rate_ci95": [0.1, 0.6],
        "win_rate_decisive": wins / max(wins + losses, 1),
        "goal_diff": (wins - losses) * 2,
        "touches": 12,
        "sides": protocol["sides"],
        "mirror": protocol["mirror"],
        "stochastic": protocol["stochastic"],
        "obs_version": protocol["obs_version"],
        "seed": protocol["seed"],
        "match_seconds": protocol["match_seconds"],
        "opponent_class": protocol["opponent_class"],
        "protocol": protocol,
        "side_results": {side: {"games": games // 2, "wins": 0} for side in protocol["sides"]},
        "duration_s": 1.5,
        "details": {"games": [{"side": side, "outcome": "draw"} for side in protocol["sides"]]},
    }
    result.update(extra)
    return result


# --------------------------------------------------------------------- basics


def test_report_labels_everything_it_could_not_measure(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("empty", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="failed", message="worker died")

    report = build_run_report(store, run_id)
    assert report["schema"] == SCHEMA
    assert report["status"] == "failed" and report["message"] == "worker died"
    assert report["steps"] == {
        "start": 0,
        "end": 0,
        "completed": 0,
        "target": config.total_steps,
        "progress": 0.0,
        # start of the current learning-rate anneal window (0 = whole run;
        # a "--steps N" continuation restarts it at the restored step)
        "lr_anneal_start": 0,
    }
    # nothing was measured: every section says so instead of reporting zeros
    assert report["duration_s"] is None
    assert report["steps_per_second"] is None
    assert report["throughput"]["iteration_steps_per_second"]["available"] is False
    assert report["optimisation"]["losses/policy"]["available"] is False
    assert "metric was not recorded" in report["optimisation"]["losses/policy"]["reason"]
    assert report["reward_components"]["available"] is False
    assert report["episodes"]["this_run"]["available"] is False
    assert report["final_checkpoint"]["available"] is False
    assert report["evaluation"]["before"]["available"] is False
    assert report["evaluation"]["after"]["available"] is False
    assert report["evaluation"]["delta"]["available"] is False
    assert report["curves"] == {}
    assert any("duration could not be computed" in warning for warning in report["warnings"])
    assert report["notes"], "the report must state what the numbers do and do not prove"


def test_report_counts_only_the_steps_this_run_did(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("child", config.mode, config.to_dict(), config.total_steps)
    store.update_run(
        run_id,
        status="completed",
        step=3000,
        resumed_step=1000,
        parent_run="parent",
        started_at="2026-10-18T10:00:00.000+00:00",
        ended_at="2026-10-18T10:00:10.000+00:00",
    )
    # a metric from the parent's step range must not be attributed to this run
    store.log_metric(run_id, 500, "losses/policy", -9.0)
    store.log_metric(run_id, 1500, "losses/policy", -0.5)
    store.log_metric(run_id, 2500, "losses/policy", -0.25)
    store.log_metric(run_id, 1500, "time/sps", 900.0)
    store.log_episode(run_id, 1, reward=1.5, length=600, goals_for=1, goals_against=0,
                      result="win", step=1500, truncated=False, opponent="chase")
    store.log_episode(run_id, 2, reward=-0.5, length=600, goals_for=0, goals_against=1,
                      result="loss", step=500, truncated=False, opponent="chase")

    report = build_run_report(store, run_id, wall_seconds=10.0)
    assert report["steps"]["start"] == 1000
    assert report["steps"]["end"] == 3000
    assert report["steps"]["completed"] == 2000
    assert report["duration_s"] == pytest.approx(10.0)
    assert report["steps_per_second"] == pytest.approx(200.0)
    assert report["throughput"]["wall_clock_steps_per_second"] == pytest.approx(200.0)
    assert report["throughput"]["iteration_steps_per_second"]["mean"] == pytest.approx(900.0)
    assert "wall-clock" in report["steps_per_second_basis"]
    policy = report["optimisation"]["losses/policy"]
    assert policy["count"] == 2, "the parent's step must be excluded"
    assert policy["mean"] == pytest.approx(-0.375)
    assert report["episodes"]["this_run"]["count"] == 1
    assert report["episodes"]["all_time"]["count"] == 2
    assert report["resume"]["parent_run"] == "parent"


def test_reward_components_and_timings_are_reported_separately(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("timings", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=512)
    store.log_metrics(
        run_id,
        256,
        {
            "reward/goal": 1.0,
            "reward/touch": 0.5,
            "time/collect_seconds": 0.2,
            "time/sim_seconds": 0.1,
            "time/infer_seconds": 0.05,
            "time/update_seconds": 0.02,
            "time/eval_seconds": 0.0,
            "time/checkpoint_seconds": 0.01,
            "time/iteration_seconds": 0.3,
        },
    )
    store.log_metrics(run_id, 512, {"reward/goal": 2.0, "time/iteration_seconds": 0.5})

    report = build_run_report(store, run_id)
    assert report["reward_components"]["reward/goal"]["mean"] == pytest.approx(1.5)
    assert report["reward_components"]["reward/goal"]["last"] == pytest.approx(2.0)
    assert report["reward_components"]["reward/touch"]["count"] == 1
    assert report["timings"]["time/iteration_seconds"]["mean"] == pytest.approx(0.4)
    assert report["timings"]["time/sim_seconds"]["mean"] == pytest.approx(0.1)
    assert report["timings"]["time/update_seconds"]["available"] is True
    # a phase that never ran is labelled, not zero
    assert report["timings"]["time/eval_seconds"]["mean"] == pytest.approx(0.0)


# ------------------------------------------------------------- before / after


def test_before_and_after_must_share_opponent_and_protocol(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("mismatch", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    store.add_evaluation(_evaluation(run_id, "chase", "before", wins=0, losses=0, draws=4))
    store.add_evaluation(_evaluation(run_id, "idle", "after", wins=4, losses=0, draws=0))

    report = build_run_report(store, run_id)
    evaluation = report["evaluation"]
    assert evaluation["before"]["opponent"] == "chase"
    assert evaluation["after"]["opponent"] == "idle"
    assert evaluation["delta"]["available"] is False
    assert "different opponents" in evaluation["delta"]["reason"]
    assert evaluation["pairs"] == []
    assert evaluation["pair_count"] == 0
    assert any("no before/after" in warning or "different opponents" in warning
               for warning in report["warnings"])

    # adding the matching after row makes the change reportable
    store.add_evaluation(_evaluation(run_id, "chase", "after", wins=2, losses=1, draws=1))
    report = build_run_report(store, run_id)
    evaluation = report["evaluation"]
    assert evaluation["pair_count"] == 1
    assert evaluation["before"]["opponent"] == evaluation["after"]["opponent"] == "chase"
    delta = evaluation["delta"]
    assert delta["available"] is True and delta["comparable_protocol"] is True
    assert delta["win_rate_before"] == pytest.approx(0.0)
    assert delta["win_rate_after"] == pytest.approx(0.5)
    assert delta["win_rate_change"] == pytest.approx(0.5)


def test_headline_pair_prefers_a_strength_opponent(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("suite", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    # a suite stores one row per opponent for each phase
    store.add_evaluation(_evaluation(run_id, "idle", "before", wins=2, losses=0, draws=2))
    store.add_evaluation(_evaluation(run_id, "chase", "before", wins=0, losses=2, draws=2))
    store.add_evaluation(_evaluation(run_id, "idle", "after", wins=4, losses=0, draws=0))
    store.add_evaluation(_evaluation(run_id, "chase", "after", wins=1, losses=3, draws=0))

    evaluation = build_run_report(store, run_id)["evaluation"]
    assert evaluation["pair_count"] == 2
    assert evaluation["before"]["opponent"] == "chase"
    assert evaluation["after"]["opponent"] == "chase"
    assert evaluation["before"]["opponent_class"] == "strength"
    assert evaluation["delta"]["opponent"] == "chase"
    assert {pair["delta"]["opponent"] for pair in evaluation["pairs"]} == {"idle", "chase"}
    assert evaluation["opponents"] == ["chase", "idle"]
    diagnostic = next(pair for pair in evaluation["pairs"] if pair["delta"]["opponent"] == "idle")
    assert diagnostic["delta"]["opponent_class"] == "diagnostic"
    assert diagnostic["delta"]["win_rate_change"] == pytest.approx(0.5)


def test_protocol_survives_the_database_round_trip(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("protocol", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    store.add_evaluation(_evaluation(run_id, "chase", "after", wins=3, losses=1, draws=0))

    row = store.list_evaluations(run_id=run_id)[0]
    details = row["details"]
    assert details["protocol"]["sides"] == ["blue", "orange"]
    assert details["protocol"]["mirror"] is True
    assert details["sides"] == ["blue", "orange"]
    assert details["mirror"] is True
    assert details["opponent_class"] == "strength"
    assert details["side_results"]["blue"]["games"] == 2

    evaluation = build_run_report(store, run_id)["evaluation"]
    after = evaluation["after"]
    assert after["sides"] == ["blue", "orange"]
    assert after["mirror"] is True
    assert after["stochastic"] is False
    assert after["protocol"]["seed"] == 1000
    assert after["protocol"]["match_seconds"] == 30.0
    assert after["win_rate_ci95"] == [0.1, 0.6]
    assert after["complete"] is True


def test_incomplete_evaluations_are_listed(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("incomplete", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="stopped", step=1000)
    store.add_evaluation(
        _evaluation(run_id, "chase", "periodic", wins=0, losses=0, draws=0, games=0)
    )
    report = build_run_report(store, run_id)
    assert len(report["evaluation"]["incomplete"]) == 1
    assert report["evaluation"]["incomplete"][0]["complete"] is False
    assert any("incomplete" in warning for warning in report["warnings"])


# ---------------------------------------------------------------- checkpoint


def test_disabled_evaluation_is_not_reported_as_a_failure(data_root):
    store = _store(data_root)
    config = _config()  # eval_interval=0: no measurement was ever requested
    run_id = store.create_run("no eval", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=config.total_steps)

    report = build_run_report(store, run_id, models_dir=data_root / "models")
    assert "disabled" in report["evaluation"]["after"]["reason"]
    assert "disabled" in report["evaluation"]["before"]["reason"]
    assert not any("final evaluation" in warning for warning in report["warnings"])


def test_final_checkpoint_is_revalidated(data_root, tmp_path):
    store = _store(data_root)
    # evaluation was requested, so this completed run *should* have a final
    # measurement; its absence is a warning
    config = _config(eval_interval=25, eval_before=True)
    run_id = store.create_run("ckpt", config.mode, config.to_dict(), config.total_steps)
    store.update_run(
        run_id,
        status="completed",
        step=2000,
        started_at="2026-10-18T10:00:00.000+00:00",
        ended_at="2026-10-18T10:00:08.000+00:00",
    )
    policy = config.policy_config()
    manifest = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=policy.build().state_dict(),
        policy_config=policy.to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=2000,
        run_id=run_id,
        mode=config.mode,
        training_config=config.to_dict(),
        extra={"obs_version": config.obs_version},
        label="final",
    )
    path = data_root / "models" / manifest["path"]
    report = build_run_report(
        store,
        run_id,
        models_dir=data_root / "models",
        final_checkpoint={
            "path": str(path),
            "model_id": manifest["id"],
            "step": 2000,
            "label": "final",
            "size_bytes": path.stat().st_size,
        },
    )
    section = report["final_checkpoint"]
    assert section["available"] is True and section["valid"] is True
    assert section["validation"]["reasons"] == []
    assert not any("checkpoint" in warning for warning in report["warnings"]), report["warnings"]
    # the missing evaluations are still reported as warnings, not hidden
    assert any("final evaluation" in warning for warning in report["warnings"])

    # a truncated file is reported as invalid, with the reason and a warning
    broken = tmp_path / "broken.pt"
    broken.write_bytes(path.read_bytes()[:200])
    report = build_run_report(
        store,
        run_id,
        models_dir=data_root / "models",
        final_checkpoint={"path": str(broken), "model_id": "broken", "step": 2000},
    )
    assert report["final_checkpoint"]["valid"] is False
    assert report["final_checkpoint"]["validation"]["reasons"]
    assert any("final checkpoint failed validation" in warning for warning in report["warnings"])


# ------------------------------------------------------------- comparisons


def test_comparisons_only_include_the_same_setup(data_root):
    store = _store(data_root)
    config = _config()
    same = store.create_run("same", config.mode, config.to_dict(), config.total_steps)
    other_mode = store.create_run("other", "soccer-2v2", _config(mode="soccer-2v2").to_dict(), 4000)
    other_obs = store.create_run(
        "obs", config.mode, _config(obs_version=1).to_dict(), config.total_steps
    )
    other_policy = store.create_run(
        "policy", config.mode, _config(policy={"hidden": [32]}).to_dict(), config.total_steps
    )
    for run_id in (same, other_mode, other_obs, other_policy):
        store.update_run(run_id, status="completed", step=2000)
    run_id = store.create_run("subject", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)

    report = build_run_report(store, run_id)
    assert [item["run_id"] for item in report["comparisons"]] == [same]

    comparison = compare_runs(store, [run_id, same])
    assert comparison["comparable"] is True and comparison["modes"] == ["soccer-1v1"]
    mixed = compare_runs(store, [run_id, other_mode])
    assert mixed["comparable"] is False
    assert "not comparable" in mixed["warning"]
    assert sorted(mixed["modes"]) == ["soccer-1v1", "soccer-2v2"]


# ---------------------------------------------------------------- persistence


def test_report_persists_to_disk_and_database(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("persist", config.mode, config.to_dict(), config.total_steps)
    store.update_run(
        run_id,
        status="completed",
        step=2000,
        started_at="2026-10-18T10:00:00.000+00:00",
        ended_at="2026-10-18T10:00:05.000+00:00",
    )
    report = build_run_report(store, run_id)
    run_dir = data_root / "runs" / run_id
    path = persist_report(report, store, run_dir=run_dir)
    assert path is not None and path.exists()
    assert json.loads(path.read_text())["run_id"] == run_id

    stored = store.get_report(run_id)
    assert stored["status"] == "completed"
    assert stored["steps_completed"] == 2000
    assert stored["duration_s"] == pytest.approx(5.0)
    assert stored["report"]["schema"] == SCHEMA
    assert stored["path"] == str(path)
    assert store.get_run(run_id)["report_path"] == str(path)

    listed = store.list_reports()
    assert [item["run_id"] for item in listed] == [run_id]
    assert listed[0]["summary"]["status"] == "completed"

    # rebuilding produces the same headline numbers
    rebuilt = build_run_report(store, run_id)
    assert rebuilt["steps"] == report["steps"]
    assert rebuilt["duration_s"] == report["duration_s"]


def test_report_survives_a_missing_run_directory(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("no-dir", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    report = build_run_report(store, run_id)
    # a read-only/absent run directory must not lose the database copy
    path = persist_report(report, store, run_dir=data_root / "runs" / run_id)
    assert path is not None
    assert store.get_report(run_id)["report"]["run_id"] == run_id


def test_report_config_comes_from_the_run_row(data_root):
    store = _store(data_root)
    config = _config(curriculum=True, self_play=True, league_states=False)
    run_id = store.create_run("cfg", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    report = build_run_report(store, run_id)
    assert report["configuration"]["curriculum"] is True
    assert report["configuration"]["self_play"] is True
    assert report["configuration"]["league_states"] is False
    assert report["configuration"]["policy"]["hidden"] == [16]
    device = report["hardware"]["device"]
    assert isinstance(device, dict) or device in {"cpu", "cuda", "auto", "mps"}


def test_hardware_section_is_measured_not_assumed(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("hw", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    report = build_run_report(store, run_id, device="cpu", workers=3)
    assert report["hardware"]["device"] == "cpu"
    assert report["hardware"]["workers"] == 3
    assert report["hardware"]["cpu_logical"] >= 1
    assert "torch" in report["hardware"]

    # a run that never recorded its device says so
    unknown = build_run_report(store, run_id, device=None, workers=None)
    assert unknown["hardware"]["device"]["available"] is False
    assert unknown["hardware"]["workers"]["available"] is False


def test_checkpoint_payload_round_trips_the_pool_states(data_root):
    """The report is only useful if the checkpoint it points at can be resumed."""
    config = _config(self_play=True)
    policy_config = config.policy_config()
    model = PolicyConfig.from_dict(policy_config.to_dict()).build()
    pool_state = {key: value.clone() for key, value in model.state_dict().items()}
    manifest = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=policy_config.to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=2048,
        run_id="pool-run",
        mode=config.mode,
        training_config=config.to_dict(),
        extra={
            "obs_version": config.obs_version,
            "league": {
                "capacity": 8,
                "agent_elo": 1042.0,
                "states_included": True,
                "entries": [
                    {"id": "snap-1", "step": 2048, "elo": 958.0, "games": 4, "wins": 1,
                     "losses": 3, "draws": 0, "has_state": True, "state": pool_state}
                ],
            },
        },
    )
    path = data_root / "models" / manifest["path"]
    payload = checkpoints.load_checkpoint(path)
    league = payload["extra"]["league"]
    assert league["states_included"] is True
    restored = league["entries"][0]["state"]
    assert torch.equal(next(iter(restored.values())), next(iter(pool_state.values())))


def test_different_seeds_still_pair_but_are_labelled(data_root):
    """The trainer derives a fresh kickoff seed per phase; that is not a new protocol."""
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("seeds", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    store.add_evaluation(
        _evaluation(run_id, "chase", "before", wins=1, losses=3, draws=0,
                    protocol=_protocol("chase", seed=900043))
    )
    store.add_evaluation(
        _evaluation(run_id, "chase", "after", wins=3, losses=1, draws=0,
                    protocol=_protocol("chase", seed=900059))
    )

    evaluation = build_run_report(store, run_id)["evaluation"]
    assert evaluation["pair_count"] == 1
    delta = evaluation["delta"]
    assert delta["available"] is True and delta["comparable_protocol"] is True
    assert delta["same_seed"] is False
    assert delta["seed_before"] == 900043 and delta["seed_after"] == 900059
    assert "different kickoff seeds" in delta["note"]
    assert delta["win_rate_change"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "change",
    [
        {"match_seconds": 60.0},
        {"sides": ["blue"]},
        {"mirror": False},
        {"stochastic": True},
        {"games": 8},
        {"obs_version": 1},
    ],
)
def test_a_changed_protocol_never_pairs(data_root, change):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("protocol-change", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    store.add_evaluation(_evaluation(run_id, "chase", "before"))
    store.add_evaluation(
        _evaluation(run_id, "chase", "after", protocol=_protocol("chase", **change))
    )

    evaluation = build_run_report(store, run_id)["evaluation"]
    assert evaluation["pair_count"] == 0
    assert evaluation["delta"]["available"] is False
    assert "different protocols" in evaluation["delta"]["reason"]
    assert any("not comparable" in warning for warning in build_run_report(store, run_id)["warnings"])


def test_suite_rows_pair_per_opponent(data_root):
    """A suite writes one row per opponent; each one pairs with its own baseline."""
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("suite-pairs", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    suite_before = {"opponents": ["idle", "chase"], "games_per_opponent": 4, "kind": "suite"}
    suite_after = dict(suite_before)
    for opponent, phase, wins in (("idle", "before", 2), ("chase", "before", 0),
                                  ("idle", "after", 4), ("chase", "after", 1)):
        protocol = _protocol(opponent, seed=900043 if phase == "before" else 900059,
                             **(suite_before if phase == "before" else suite_after))
        store.add_evaluation(
            _evaluation(run_id, opponent, phase, wins=wins, losses=4 - wins, draws=0,
                        protocol=protocol)
        )

    evaluation = build_run_report(store, run_id)["evaluation"]
    assert evaluation["pair_count"] == 2
    assert evaluation["delta"]["opponent"] == "chase"
    assert evaluation["delta"]["opponent_class"] == "strength"
    assert evaluation["delta"]["win_rate_change"] == pytest.approx(0.25)
    by_opponent = {pair["delta"]["opponent"]: pair["delta"] for pair in evaluation["pairs"]}
    assert by_opponent["idle"]["win_rate_change"] == pytest.approx(0.5)
    assert by_opponent["idle"]["opponent_class"] == "diagnostic"
    assert evaluation["before"]["opponent"] == "chase"


def test_rebuilt_report_recovers_resume_mode_and_checkpoint(data_root):
    """A rebuilt report must not downgrade a continuation to 'fresh'."""
    store = _store(data_root)
    config = _config()
    parent = store.create_run("parent", config.mode, config.to_dict(), config.total_steps)
    store.update_run(parent, status="completed", step=2000)
    child = store.create_run("child", config.mode, config.to_dict(), config.total_steps)
    store.update_run(child, status="completed", step=4000, parent_run=parent, resumed_step=2000)

    manifest = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=config.policy_config().build().state_dict(),
        policy_config=config.policy_config().to_dict(),
        optimizer_state={"state": {}, "param_groups": []},
        step=4000,
        run_id=child,
        mode=config.mode,
        training_config=config.to_dict(),
        extra={"obs_version": config.obs_version, "resume_mode": "continue", "parent_run": parent},
        label="final",
    )

    report = build_run_report(store, child, models_dir=data_root / "models")
    assert report["resume"]["mode"] == "continue"
    assert report["resume"]["parent_run"] == parent
    assert report["resume"]["restored_step"] == 2000
    assert report["resume"]["source"] == "reconstructed"
    assert report["steps"]["completed"] == 2000

    section = report["final_checkpoint"]
    assert section["available"] is True and section["valid"] is True
    assert section["model_id"] == manifest["id"]
    assert section["step"] == 4000


def test_rebuilt_report_of_a_fresh_run_stays_fresh(data_root):
    store = _store(data_root)
    config = _config()
    run_id = store.create_run("plain", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    report = build_run_report(store, run_id, models_dir=data_root / "models")
    assert report["resume"]["mode"] == "fresh"
    assert report["resume"]["parent_run"] is None
    assert report["final_checkpoint"]["available"] is False
    assert "no checkpoint" in report["final_checkpoint"]["reason"]
