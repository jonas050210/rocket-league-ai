"""Post-run training statistics.

Every finished *or interrupted* run gets one report.  It is written to
``data/runs/<run_id>/report.json``, stored in the database and rendered by the
GUI.

Two rules shape this module:

* **Nothing is invented.**  A value that was never measured is reported as
  ``{"available": false, "reason": ...}`` instead of a plausible-looking zero.
* **Training reward is not playing strength.**  Reward, episode length and
  PPO losses describe the optimisation; only the ``evaluation`` section -
  measured against fixed opponents under a recorded protocol - says anything
  about how the policy plays.  The report keeps the two apart and says so.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from . import checkpoints
from .store import Store

#: metrics shown as training curves in the GUI
CURVE_METRICS = (
    "charts/episodic_return",
    "charts/episode_length",
    "charts/win_rate",
    "charts/draw_rate",
    "charts/goals_for",
    "charts/goals_against",
    "losses/policy",
    "losses/value",
    "losses/entropy",
    "losses/approx_kl",
    "losses/clipfrac",
    "losses/explained_variance",
    "losses/grad_norm",
    "losses/learning_rate",
    "time/sps",
    "eval/win_rate",
    "eval/goal_diff",
    "curriculum/stage",
    "league/agent_elo",
)

#: PPO / optimisation metrics summarised in the report
OPTIMISATION_METRICS = (
    "losses/policy",
    "losses/value",
    "losses/entropy",
    "losses/approx_kl",
    "losses/clipfrac",
    "losses/explained_variance",
    "losses/grad_norm",
    "losses/learning_rate",
    "losses/epochs_run",
    "losses/early_stopped",
    "losses/old_values_mean",
)

#: where the time went, measured separately
TIMING_METRICS = (
    "time/sps",
    "time/collect_seconds",
    "time/update_seconds",
    "time/sim_seconds",
    "time/infer_seconds",
    "time/eval_seconds",
    "time/checkpoint_seconds",
    "time/iteration_seconds",
)

MAX_CURVE_POINTS = 400

#: Protocol keys that define *what was measured*.  The seed is deliberately not
#: part of the identity: the trainer derives a new kickoff seed per evaluation
#: phase, so before/after are independent samples of the same protocol rather
#: than a bit-identical replay.  Everything that changes the meaning of the
#: number (opponent, match length, sides, mirroring, sampling, obs layout) is.
PROTOCOL_IDENTITY_KEYS = (
    "mode",
    "opponent",
    "opponents",
    "games",
    "games_per_opponent",
    "match_seconds",
    "episode_seconds",
    "sides",
    "mirror",
    "stochastic",
    "obs_version",
    "kind",
)


def unavailable(reason: str) -> dict:
    """Explicit marker for a value that was never measured."""
    return {"available": False, "reason": reason}


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(started: Any, ended: Any) -> float | None:
    start = _parse_timestamp(started)
    end = _parse_timestamp(ended)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _downsample(points: list[dict], limit: int = MAX_CURVE_POINTS) -> list[dict]:
    if len(points) <= limit:
        return points
    stride = len(points) / float(limit)
    sampled = []
    index = 0.0
    while int(index) < len(points) and len(sampled) < limit:
        sampled.append(points[int(index)])
        index += stride
    if sampled and sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    return sampled


def _timing_section(store: Store, run: dict, wall_duration: float | None) -> dict:
    """The training-time block of the report.

    ``accumulated_s`` is the sum of the measured sessions of the run - the
    time the trainer actually worked.  ``wall_clock_s`` is the difference
    between the run's start and end timestamps; it includes the time the
    process was stopped or the machine was busy, so it is reported
    separately and never used as the training time.
    """
    from .timing import format_duration

    totals = store.session_totals(run.get("id", "")) if run.get("id") else {}
    sessions = totals.get("sessions") or []
    accumulated = totals.get("accumulated_s")
    session_count = int(totals.get("finished_count") or 0)
    has_sessions = bool(sessions)

    if not has_sessions:
        # No session rows: an old run written before the session tracking.
        accumulated = None
        note = "this run predates session tracking; the training time is unknown"
    else:
        note = None

    current = None
    for row in sessions:
        if row.get("duration_s") is None and row.get("status") == "running":
            current = row
            break

    interrupted = [row for row in sessions if row.get("status") == "interrupted"]
    return {
        "available": has_sessions,
        "accumulated_s": accumulated,
        "accumulated_human": format_duration(accumulated),
        "session_count": session_count,
        "sessions": [
            {
                "started_at": row.get("started_at"),
                "ended_at": row.get("ended_at"),
                "duration_s": row.get("duration_s"),
                "duration_human": format_duration(row.get("duration_s")),
                "status": row.get("status"),
                "start_step": row.get("start_step"),
                "end_step": row.get("end_step"),
                "note": row.get("note") or "",
            }
            for row in sessions
        ],
        "last_session_s": (
            sessions[0].get("duration_s") if sessions else None
        ),
        # the session that is running right now: its duration is not
        # measured yet, so it stays unknown until the session is closed
        "current_session_s": current.get("duration_s") if current else None,
        "current_session_human": format_duration(
            current.get("duration_s") if current else None
        ),
        "current_session_started_at": current.get("started_at") if current else None,
        "interrupted_sessions": len(interrupted),
        "unfinished_sessions": int(totals.get("unfinished_count") or 0),
        "wall_clock_s": wall_duration,
        "wall_clock_human": format_duration(wall_duration),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "last_checkpoint_at": run.get("last_checkpoint_at"),
        "last_checkpoint_id": run.get("last_checkpoint_id"),
        "note": note,
        "basis": "sum of the sessions measured with a monotonic clock",
    }


def hardware_section(device: str | None, workers: int | None) -> dict:
    """Device, worker count and machine facts, measured - never assumed."""
    import os

    section: dict[str, Any] = {
        "device": device if device is not None else unavailable("device not recorded"),
        "workers": workers if workers is not None else unavailable("worker count not recorded"),
        "cpu_logical": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch

        section["torch"] = torch.__version__
        section["gpu_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            section["gpu_name"] = torch.cuda.get_device_name(0)
            section["gpu_count"] = int(torch.cuda.device_count())
        else:
            section["gpu_name"] = None
            section["gpu_count"] = 0
        section["torch_threads"] = int(torch.get_num_threads())
    except Exception as exc:  # pragma: no cover - torch always present here
        section["torch"] = unavailable(f"torch introspection failed: {exc}")
    return section


def evaluation_section(store: Store, run_id: str, models_dir: Path | None) -> dict:
    """Before/after and periodic evaluations, with the protocol they used.

    A before/after *change* is only reported for a matched pair: same opponent,
    same protocol.  A suite stores one row per opponent, so picking "the first
    before row" and "the last after row" would silently compare two different
    opponents; those rows are paired explicitly here and anything unmatched is
    labelled unavailable instead of being turned into a number.
    """
    from .evaluation import classify_opponent

    rows = store.list_evaluations(limit=500, run_id=run_id)
    # list_evaluations returns newest first
    ordered = list(reversed(rows))

    def protocol_of(row: dict) -> dict:
        details = row.get("details") or {}
        return details.get("protocol") or {
            key: details.get(key)
            for key in ("seed", "match_seconds", "obs_version", "sides", "mirror")
            if key in details
        }

    def identity_of(row: dict) -> dict:
        protocol = protocol_of(row)
        return {key: protocol[key] for key in PROTOCOL_IDENTITY_KEYS if key in protocol}

    def class_of(row: dict) -> str:
        details = row.get("details") or {}
        return details.get("opponent_class") or classify_opponent(str(row.get("opponent") or ""))

    def slim(row: dict | None) -> dict | None:
        if row is None:
            return None
        details = row.get("details") or {}
        protocol = protocol_of(row)
        return {
            "id": row.get("id"),
            "phase": row.get("phase"),
            "model_id": row.get("model_id"),
            "opponent": row.get("opponent"),
            "opponent_class": class_of(row),
            "mode": row.get("mode"),
            "games": row.get("games"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "draws": row.get("draws"),
            "goals_for": row.get("goals_for"),
            "goals_against": row.get("goals_against"),
            "win_rate": details.get("win_rate"),
            "win_rate_ci95": details.get("win_rate_ci95"),
            "win_rate_decisive": details.get("win_rate_decisive"),
            "goal_diff": details.get("goal_diff"),
            "touches": details.get("touches"),
            "sides": details.get("sides") or protocol.get("sides"),
            "mirror": details.get("mirror") if details.get("mirror") is not None else protocol.get("mirror"),
            "stochastic": details.get("stochastic") if details.get("stochastic") is not None else protocol.get("stochastic"),
            "side_results": details.get("side_results"),
            "protocol": protocol,
            "created_at": row.get("created_at"),
            "duration_s": row.get("duration_s"),
            "complete": bool(int(row.get("games") or 0)),
        }

    before_rows = [row for row in ordered if row.get("phase") == "before"]
    after_rows = [row for row in ordered if row.get("phase") == "after"]
    periodic = [row for row in ordered if row.get("phase") in (None, "periodic")]
    incomplete = [row for row in ordered if not int(row.get("games") or 0)]

    # -- pair before/after rows by opponent *and* protocol -----------------
    pairs: list[dict] = []
    for after_row in after_rows:
        for before_row in before_rows:
            if before_row.get("opponent") != after_row.get("opponent"):
                continue
            if identity_of(before_row) != identity_of(after_row):
                continue
            before_slim, after_slim = slim(before_row), slim(after_row)
            seed_before = protocol_of(before_row).get("seed")
            seed_after = protocol_of(after_row).get("seed")
            delta: dict[str, Any] = {
                "available": True,
                "comparable_protocol": True,
                "opponent": after_row.get("opponent"),
                "opponent_class": class_of(after_row),
                "win_rate_before": before_slim.get("win_rate"),
                "win_rate_after": after_slim.get("win_rate"),
                "win_rate_change": (
                    None
                    if before_slim.get("win_rate") is None or after_slim.get("win_rate") is None
                    else round(after_slim["win_rate"] - before_slim["win_rate"], 4)
                ),
                "goal_diff_before": before_slim.get("goal_diff"),
                "goal_diff_after": after_slim.get("goal_diff"),
                "wins_before": before_slim.get("wins"),
                "wins_after": after_slim.get("wins"),
                "games_before": before_slim.get("games"),
                "games_after": after_slim.get("games"),
                "opponent_before": before_slim.get("opponent"),
                "opponent_after": after_slim.get("opponent"),
                "protocol": after_slim.get("protocol"),
                "seed_before": seed_before,
                "seed_after": seed_after,
                "same_seed": seed_before == seed_after,
            }
            if not delta["same_seed"]:
                delta["note"] = (
                    "the two series used different kickoff seeds "
                    f"({seed_before} vs {seed_after}), so they are independent samples of the "
                    "same protocol, not a replay of identical matches"
                )
            pairs.append({"before": before_slim, "after": after_slim, "delta": delta})
            break
    # strength opponents first, then the series with the most games
    pairs.sort(
        key=lambda pair: (
            pair["delta"].get("opponent_class") != "strength",
            -int(pair["after"].get("games") or 0),
        )
    )

    headline = pairs[0] if pairs else None
    if headline is not None:
        before_slim = headline["before"]
        after_slim = headline["after"]
        delta = headline["delta"]
    else:
        before_slim = slim(before_rows[0]) if before_rows else None
        after_slim = slim(after_rows[-1]) if after_rows else None
        if before_slim and after_slim:
            delta = unavailable(
                "before and after were measured against different opponents or with "
                f"different protocols (before: {before_slim.get('opponent')}, "
                f"after: {after_slim.get('opponent')}); no change is reported"
            )
        else:
            delta = unavailable("no before/after evaluation pair for this run")

    best = checkpoints.best_record(models_dir) if models_dir else {}
    return {
        "before": before_slim or unavailable("no baseline evaluation was run"),
        "after": after_slim or unavailable("run ended before the final evaluation completed"),
        "delta": delta,
        "pairs": pairs,
        "pair_count": len(pairs),
        "opponents": sorted({str(row.get("opponent")) for row in ordered if row.get("opponent")}),
        "periodic": [slim(row) for row in periodic[-50:]],
        "periodic_count": len(periodic),
        "incomplete": [slim(row) for row in incomplete],
        "best": best or unavailable("no best-model pointer has been set"),
    }


def _newest_checkpoint_of_run(models_dir: Path, run_id: str) -> dict | None:
    """Newest checkpoint the registry attributes to ``run_id`` (rebuilt reports)."""
    try:
        manifests = checkpoints.list_models(models_dir)
    except Exception:  # pragma: no cover - an unreadable registry must not kill a report
        return None
    own = [item for item in manifests if item.get("run_id") == run_id]
    if not own:
        return None
    own.sort(key=lambda item: (int(item.get("step") or 0), str(item.get("created_at") or "")))
    newest = own[-1]
    return {
        "path": str(models_dir / newest["path"]),
        "model_id": newest.get("id"),
        "step": newest.get("step"),
        "label": newest.get("label"),
        "size_bytes": newest.get("size_bytes"),
    }


def comparison_section(store: Store, run: dict) -> list[dict]:
    """Other runs that can meaningfully be compared with this one."""
    config = run.get("config") or {}
    mode = run.get("mode")
    obs_version = config.get("obs_version")
    policy = config.get("policy")
    comparisons = []
    for other in store.list_runs(limit=200):
        if other["id"] == run["id"]:
            continue
        other_config = other.get("config") or {}
        if other.get("mode") != mode:
            continue
        if obs_version is not None and other_config.get("obs_version") != obs_version:
            continue
        if policy is not None and other_config.get("policy") != policy:
            continue
        steps_start = int(other.get("resumed_step") or 0)
        steps_end = int(other.get("step") or 0)
        duration = _duration_seconds(other.get("started_at"), other.get("ended_at"))
        comparisons.append(
            {
                "run_id": other["id"],
                "name": other.get("name"),
                "status": other.get("status"),
                "mode": other.get("mode"),
                "obs_version": other_config.get("obs_version"),
                "steps_start": steps_start,
                "steps_end": steps_end,
                "steps_completed": max(0, steps_end - steps_start),
                "duration_s": duration,
                "steps_per_second": (
                    (steps_end - steps_start) / duration if duration else None
                ),
                "final_win_rate": store.last_metric(other["id"], "eval/win_rate"),
                "mean_return": (store.metric_stats(other["id"], "charts/episodic_return") or {}).get(
                    "mean"
                ),
                "parent_run": other.get("parent_run"),
                "created_at": other.get("created_at"),
            }
        )
    comparisons.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return comparisons[:20]


def build_run_report(
    store: Store,
    run_id: str,
    *,
    models_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    config: dict | None = None,
    resume: dict | None = None,
    curriculum: dict | None = None,
    device: str | None = None,
    workers: int | None = None,
    wall_seconds: float | None = None,
    final_checkpoint: dict | None = None,
    extra_warnings: Iterable[str] = (),
    hardware: dict | None = None,
) -> dict:
    """Assemble the full statistics report for one run."""
    run = store.get_run(run_id) or {}
    run_config = dict(config or run.get("config") or {})
    models_dir = Path(models_dir) if models_dir else None
    warnings: list[str] = list(extra_warnings)

    resumed_step = int(run.get("resumed_step") or 0)
    end_step = int(run.get("step") or 0)
    completed = max(0, end_step - resumed_step)
    total_steps = run.get("total_steps") or run_config.get("total_steps")

    duration = _duration_seconds(run.get("started_at"), run.get("ended_at"))
    if duration is None and wall_seconds is not None:
        duration = float(wall_seconds)
    if duration is None:
        warnings.append(
            "duration could not be computed: the run has no started_at/ended_at pair"
        )

    # The measured training time (monotonic clock, summed over the
    # sessions) is the authoritative duration.  The wall-clock difference
    # above is only a fallback for runs that predate the session tracking.
    timing_section = _timing_section(store, run, duration)

    steps_per_second = completed / duration if duration else None

    optimisation = {}
    for name in OPTIMISATION_METRICS:
        stats = store.metric_stats(run_id, name, since_step=resumed_step)
        if not stats.get("available"):
            optimisation[name] = unavailable("metric was not recorded for this run")
        else:
            stats = dict(stats)
            stats["first"] = _round(stats.get("first"))
            stats["last"] = _round(stats.get("last"))
            stats["mean"] = _round(stats.get("mean"))
            stats["min"] = _round(stats.get("min"))
            stats["max"] = _round(stats.get("max"))
            optimisation[name] = stats

    timings = {}
    for name in TIMING_METRICS:
        stats = store.metric_stats(run_id, name, since_step=resumed_step)
        timings[name] = (
            unavailable("metric was not recorded for this run")
            if not stats.get("available")
            else {
                "available": True,
                "count": stats["count"],
                "mean": _round(stats.get("mean")),
                "min": _round(stats.get("min")),
                "max": _round(stats.get("max")),
                "last": _round(stats.get("last")),
            }
        )

    iteration_sps = store.metric_stats(run_id, "time/sps", since_step=resumed_step)
    throughput = {
        "wall_clock_steps_per_second": steps_per_second,
        "iteration_steps_per_second": (
            unavailable("no iteration was measured")
            if not iteration_sps.get("available")
            else {
                "available": True,
                "mean": _round(iteration_sps.get("mean")),
                "min": _round(iteration_sps.get("min")),
                "max": _round(iteration_sps.get("max")),
                "last": _round(iteration_sps.get("last")),
                "basis": "environment steps divided by collect+update time per iteration",
            }
        ),
    }

    reward_components = {}
    for name in store.metric_names(run_id):
        if not name.startswith("reward/"):
            continue
        stats = store.metric_stats(run_id, name, since_step=resumed_step)
        if stats.get("available"):
            reward_components[name] = {
                "mean": _round(stats.get("mean")),
                "last": _round(stats.get("last")),
                "count": stats["count"],
            }
    if not reward_components:
        reward_components = unavailable(  # type: ignore[assignment]
            "no reward components were recorded (episodes did not finish or the "
            "worker build predates component tracking)"
        )

    episodes_all = store.episode_stats(run_id)
    episodes_this_run = (
        store.episode_stats(run_id, since_step=resumed_step) if resumed_step else episodes_all
    )
    episodes_window = store.episode_summary(run_id, window=100)

    # Learning evidence independent of the shaped reward: what did the first
    # episodes of this run look like vs the last ones, in outcomes and ball
    # interaction?  A rising reward with flat touches/goals is proxy farming,
    # and this block makes that visible at a glance.
    first_window = store.episode_window_stats(
        run_id, which="first", window=200, since_step=resumed_step
    )
    last_window = store.episode_window_stats(
        run_id, which="last", window=200, since_step=resumed_step
    )
    learning_evidence: dict[str, Any] = {
        "first_200_episodes": first_window,
        "last_200_episodes": last_window,
        "basis": (
            "training-episode outcomes of this run (not an independent "
            "evaluation); compare with the evaluation section for strength"
        ),
    }
    if first_window.get("available") and last_window.get("available"):
        learning_evidence["delta"] = {
            key: (
                round(float(last_window[key]) - float(first_window[key]), 4)
                if first_window.get(key) is not None and last_window.get(key) is not None
                else None
            )
            for key in (
                "mean_reward", "win_rate", "goals_per_episode",
                "conceded_per_episode", "touches_per_episode", "shots_per_episode",
            )
        }
        reward_up = (learning_evidence["delta"].get("mean_reward") or 0) > 0
        play_flat = (
            abs(learning_evidence["delta"].get("goals_per_episode") or 0) < 1e-9
            and abs(learning_evidence["delta"].get("touches_per_episode") or 0) < 1e-9
        )
        if reward_up and play_flat:
            warnings.append(
                "training reward rose while goals and touches per episode did not "
                "move: the policy is likely optimising shaping proxies, not play"
            )

    curves = {
        name: _downsample(points)
        for name, points in store.metric_series(run_id, CURVE_METRICS, limit=20000).items()
        if points
    }

    evaluation = evaluation_section(store, run_id, models_dir)

    if not final_checkpoint and models_dir:
        # A rebuilt report has no in-memory manifest, but the registry knows
        # which checkpoints this run wrote: use the newest one instead of
        # claiming the run never produced a checkpoint.
        final_checkpoint = _newest_checkpoint_of_run(models_dir, run_id)

    checkpoint_section: dict[str, Any]
    if final_checkpoint:
        path = final_checkpoint.get("path")
        validation = None
        if path:
            validation = checkpoints.validate_checkpoint(path, load=True)
        checkpoint_section = {
            "available": True,
            "model_id": final_checkpoint.get("model_id"),
            "path": path,
            "step": final_checkpoint.get("step"),
            "label": final_checkpoint.get("label"),
            "size_bytes": final_checkpoint.get("size_bytes"),
            "valid": bool(validation and validation.ok),
            "validation": validation.to_dict() if validation else None,
        }
        if validation and not validation.ok:
            warnings.append(
                f"final checkpoint failed validation: {'; '.join(validation.reasons)}"
            )
    else:
        checkpoint_section = unavailable("no checkpoint was written by this run")

    for item in evaluation.get("incomplete") or []:
        if isinstance(item, dict):
            warnings.append(
                f"evaluation {item.get('id')} did not finish any game and is incomplete"
            )
    if isinstance(evaluation.get("after"), dict) and not evaluation["after"].get("available", True):
        warnings.append(str(evaluation["after"].get("reason")))
    delta = evaluation.get("delta")
    if (
        isinstance(delta, dict)
        and not delta.get("available", True)
        and isinstance(evaluation.get("before"), dict)
        and evaluation["before"].get("available", True)
        and isinstance(evaluation.get("after"), dict)
        and evaluation["after"].get("available", True)
    ):
        # both series exist but cannot be compared: say so loudly
        warnings.append(f"before/after evaluation not comparable: {delta.get('reason')}")
    elif isinstance(delta, dict) and delta.get("available") and delta.get("warning"):
        warnings.append(str(delta["warning"]))
    if isinstance(evaluation.get("before"), dict) and not evaluation["before"].get("available", True):
        warnings.append(str(evaluation["before"].get("reason")))

    resume_section = dict(resume or {})
    if not resume_section:
        # Rebuilt report: the original plan object is gone, but the run row and
        # the final checkpoint still say whether this was a continuation. Say
        # where the answer comes from instead of defaulting to "fresh".
        checkpoint_extra = (checkpoint_section.get("validation") or {}).get("extra") or {}
        mode = (
            run_config.get("resume_mode")
            or checkpoint_extra.get("resume_mode")
            or ("continue" if (run.get("parent_run") or resumed_step) else "fresh")
        )
        resume_section = {
            "mode": mode,
            "parent_run": run.get("parent_run") or checkpoint_extra.get("parent_run"),
            "restored_step": resumed_step,
            "checkpoint": checkpoint_section.get("path"),
            "model_id": checkpoint_section.get("model_id"),
            "warnings": [],
            "source": "reconstructed",
            "note": (
                "reconstructed from the run row and the final checkpoint; the plan the "
                "trainer built at start-up is not stored verbatim"
            ),
        }

    report = {
        "schema": "rlai-run-report-v1",
        "run_id": run_id,
        "name": run.get("name"),
        "mode": run.get("mode"),
        "status": run.get("status"),
        "message": run.get("message"),
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "ended_at": run.get("ended_at"),
        "duration_s": duration,
        "timing": timing_section,
        "steps": {
            "start": resumed_step,
            "end": end_step,
            "completed": completed,
            "target": total_steps,
            "progress": (
                round(end_step / float(total_steps), 4) if total_steps else None
            ),
        },
        "steps_per_second": steps_per_second,
        "steps_per_second_basis": (
            "steps completed in this run divided by wall-clock time between started_at and "
            "ended_at; includes worker start-up, evaluation and checkpoint writing"
        ),
        "iterations": (store.metric_stats(run_id, "time/iteration_seconds") or {}).get("count", 0),
        "hardware": hardware or hardware_section(device, workers),
        "configuration": run_config,
        "resume": resume_section,
        "optimisation": optimisation,
        "throughput": throughput,
        "timings": timings,
        "reward_components": reward_components,
        "episodes": {
            "this_run": episodes_this_run,
            "all_time": episodes_all,
            "last_100": episodes_window,
        },
        "learning_evidence": learning_evidence,
        "curriculum": curriculum
        if curriculum is not None
        else unavailable("no curriculum was active for this run"),
        "evaluation": evaluation,
        "final_checkpoint": checkpoint_section,
        "curves": curves,
        "comparisons": comparison_section(store, run),
        "warnings": warnings,
        "notes": [
            "Reward, episode length and PPO losses describe the optimisation only.",
            "Playing strength comes exclusively from the evaluation section, which is "
            "measured against fixed opponents under a recorded protocol.",
            "Values marked available=false were never measured; they are not zero.",
        ],
    }
    return report


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def persist_report(
    report: dict,
    store: Store,
    *,
    run_dir: str | Path | None = None,
) -> Path | None:
    """Write the report to disk and to the database.  Returns the path."""
    run_id = report.get("run_id")
    path: Path | None = None
    if run_dir:
        run_dir = Path(run_dir)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "report.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            report.setdefault("warnings", []).append(f"report file could not be written: {exc}")
            path = None
    try:
        store.save_report(run_id, report, path=path)
        if path is not None:
            store.update_run(run_id, report_path=str(path))
    except Exception as exc:  # pragma: no cover - database problems must not kill a run
        report.setdefault("warnings", []).append(f"report could not be stored: {exc}")
    return path


def compare_runs(store: Store, run_ids: Iterable[str]) -> dict:
    """Side-by-side comparison of several runs (GUI: history view)."""
    reports = []
    for run_id in run_ids:
        stored = store.get_report(run_id)
        if stored and stored.get("report"):
            reports.append(stored["report"])
            continue
        run = store.get_run(run_id)
        if run:
            reports.append(
                {
                    "run_id": run_id,
                    "name": run.get("name"),
                    "mode": run.get("mode"),
                    "status": run.get("status"),
                    "report_available": False,
                    "reason": "run has no stored report (finished before reporting existed)",
                }
            )
    comparable = [item for item in reports if item.get("mode")]
    modes = {item.get("mode") for item in comparable}
    return {
        "runs": reports,
        "comparable": len(modes) <= 1,
        "modes": sorted(mode for mode in modes if mode),
        "warning": (
            None
            if len(modes) <= 1
            else "these runs use different modes; their numbers are not comparable"
        ),
    }
