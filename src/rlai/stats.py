"""Statistics for the GUI's Stats tab.

One place computes every number the statistics view shows, so the GUI never
has to derive a value itself and the headless report and the GUI can never
disagree.

The rules are the ones the rest of the project follows:

* a value that was never measured is reported as unavailable, with the
  reason - never as a zero;
* the training time comes from the measured sessions (monotonic clock), not
  from the difference of two wall-clock timestamps;
* the learning metrics and the playing strength stay apart: reward and PPO
  losses describe the optimisation, only a measured evaluation says
  anything about how the policy plays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from . import checkpoints
from .store import Store
from .timing import format_duration, parse_utc

#: metrics worth keeping a chart for
DEFAULT_CURVE_METRICS = (
    "charts/episodic_return",
    "charts/win_rate",
    "charts/episode_length",
    "losses/policy",
    "losses/value",
    "losses/approx_kl",
    "losses/explained_variance",
    "time/sps",
    "eval/win_rate",
)

#: metrics shown as compact summary numbers
SUMMARY_METRICS = (
    "charts/episodic_return",
    "charts/win_rate",
    "charts/draw_rate",
    "charts/goals_for",
    "charts/goals_against",
    "charts/episode_length",
    "losses/policy",
    "losses/value",
    "losses/entropy",
    "losses/approx_kl",
    "losses/clipfrac",
    "losses/explained_variance",
    "losses/grad_norm",
    "losses/learning_rate",
    "time/sps",
    "time/collect_seconds",
    "time/update_seconds",
    "time/sim_seconds",
    "time/infer_seconds",
    "time/eval_seconds",
    "time/checkpoint_seconds",
    "time/iteration_seconds",
    "eval/win_rate",
    "eval/goal_diff",
    "eval/games",
    "curriculum/stage",
    "curriculum/window_win_rate",
    "league/agent_elo",
    "league/pool_size",
)

#: how many points a chart keeps
MAX_POINTS = 400


def _unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def _round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _downsample(points: list[dict], limit: int = MAX_POINTS) -> list[dict]:
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


def _local_time(value: str | None) -> str | None:
    """UTC timestamp -> local ISO string (the GUI shows local time)."""
    parsed = parse_utc(value)
    if parsed is None:
        return None
    return parsed.astimezone().isoformat(timespec="seconds")


def timing_block(store: Store, run: dict) -> dict:
    """Session-based training time of one run."""
    run_id = run.get("id")
    totals = store.session_totals(run_id) if run_id else {}
    sessions = totals.get("sessions") or []
    accumulated = totals.get("accumulated_s")
    if not sessions:
        accumulated = None
    current = next(
        (row for row in sessions if row.get("duration_s") is None and row.get("status") == "running"),
        None,
    )
    return {
        "available": bool(sessions),
        "accumulated_s": accumulated,
        "accumulated_human": format_duration(accumulated),
        "current_session_s": current.get("duration_s") if current else None,
        "current_session_human": format_duration(current.get("duration_s") if current else None),
        "session_count": int(totals.get("finished_count") or 0),
        "interrupted_count": sum(1 for row in sessions if row.get("status") == "interrupted"),
        "unfinished_count": int(totals.get("unfinished_count") or 0),
        "started_at": run.get("started_at"),
        "started_at_local": _local_time(run.get("started_at")),
        "ended_at": run.get("ended_at"),
        "ended_at_local": _local_time(run.get("ended_at")),
        "last_checkpoint_at": run.get("last_checkpoint_at"),
        "last_checkpoint_at_local": _local_time(run.get("last_checkpoint_at")),
        "last_checkpoint_id": run.get("last_checkpoint_id"),
        "sessions": [
            {
                "started_at": row.get("started_at"),
                "started_at_local": _local_time(row.get("started_at")),
                "ended_at": row.get("ended_at"),
                "ended_at_local": _local_time(row.get("ended_at")),
                "duration_s": row.get("duration_s"),
                "duration_human": format_duration(row.get("duration_s")),
                "status": row.get("status"),
                "start_step": row.get("start_step"),
                "end_step": row.get("end_step"),
                "note": row.get("note") or "",
            }
            for row in sessions
        ],
    }


def progress_block(store: Store, run: dict) -> dict:
    """Steps, iteration, episodes and status."""
    run_id = run.get("id", "")
    episodes = store.episode_stats(run_id)
    metrics = store.metric_stats(run_id, "time/total_steps")
    iteration = store.metric_stats(run_id, "curriculum/stage_iterations")
    total_steps = run.get("total_steps")
    if not total_steps and metrics.get("available"):
        # the run row can be older than the metric; the metric is the value
        # the trainer itself reported
        total_steps = int(metrics.get("last") or 0) or None
    return {
        "status": run.get("status"),
        "step": int(run.get("step") or 0),
        "total_steps": total_steps,
        "resumed_step": int(run.get("resumed_step") or 0),
        "progress": (
            round(int(run.get("step") or 0) / float(run["total_steps"]), 4)
            if run.get("total_steps")
            else None
        ),
        "episodes": episodes,
        "episode_count": episodes.get("count", 0) if episodes.get("available") else 0,
        "iterations": (store.metric_stats(run_id, "time/iteration_seconds") or {}).get("count", 0),
        "stage_iterations": (iteration or {}).get("last"),
        "mode": run.get("mode"),
        "name": run.get("name"),
    }


def learning_block(store: Store, run: dict, *, since_step: int = 0) -> dict:
    """Summary numbers of the optimisation."""
    run_id = run.get("id", "")
    summary: dict[str, Any] = {}
    for name in SUMMARY_METRICS:
        stats = store.metric_stats(run_id, name, since_step=since_step)
        if not stats.get("available"):
            continue
        summary[name] = {
            "last": _round(stats.get("last")),
            "mean": _round(stats.get("mean")),
            "min": _round(stats.get("min")),
            "max": _round(stats.get("max")),
            "count": stats.get("count"),
        }
    return summary or _unavailable("no metrics were recorded for this run")


def curves_block(
    store: Store,
    run: dict,
    *,
    metrics: Iterable[str] = DEFAULT_CURVE_METRICS,
    since_step: int = 0,
) -> dict:
    """Chart series, downsampled to a sensible number of points."""
    run_id = run.get("id", "")
    names = list(metrics)
    series = store.metric_series(run_id, names, limit=50_000)
    result: dict[str, list[dict]] = {}
    for name, points in series.items():
        filtered = [point for point in points if point["step"] >= since_step]
        if filtered:
            result[name] = _downsample(filtered)
    return result


def performance_block(
    run: dict,
    *,
    resources: dict | None = None,
    store: Store | None = None,
) -> dict:
    """Throughput and (when available) the machine's load.

    Two *measured* numbers are reported and clearly separated:

    * the throughput of the last iteration (``time/sps``) - what the
      pipeline actually achieved, including rollout, inference and the PPO
      update;
    * the average over the whole run - the environment steps of the run
      divided by the *measured* training time, so the time the process
      spent stopped is not part of it.

    A run that never finished an iteration has no throughput; the value is
    then reported as unavailable instead of zero.
    """
    block: dict[str, Any] = {
        "steps_per_second": _round(run.get("steps_per_second")),
        "basis": "environment steps divided by the measured wall-clock time of the run",
        "available": run.get("steps_per_second") is not None,
    }
    if store is not None:
        sps = store.metric_stats(run.get("id", ""), "time/sps")
        if sps.get("available"):
            block.update(
                {
                    "available": True,
                    "steps_per_second": _round(sps.get("last")),
                    "mean_steps_per_second": _round(sps.get("mean")),
                    "min_steps_per_second": _round(sps.get("min")),
                    "max_steps_per_second": _round(sps.get("max")),
                    "samples": sps.get("count"),
                    "source": "time/sps metric of the last finished iteration",
                }
            )
        else:
            block["source"] = "no finished iteration was measured yet"

    duration = run.get("duration_s")
    steps = int(run.get("step") or 0)
    if duration and steps:
        block["run_steps_per_training_second"] = _round(steps / float(duration))
        block["run_steps_per_training_second_basis"] = (
            "environment steps of the run divided by the accumulated measured "
            "training time (the time spent stopped is not part of it)"
        )
    if resources:
        block["resources"] = resources
    return block


def checkpoint_block(
    store: Store,
    models_dir: str | Path | None,
    run: dict,
) -> list[dict]:
    """Checkpoints of the run with their duration metadata."""
    run_id = run.get("id")
    if not models_dir:
        return []
    rows = []
    for manifest in checkpoints.list_models(models_dir):
        if manifest.get("run_id") != run_id:
            continue
        score = store.get_model_eval_score(manifest["id"])
        rows.append(
            {
                "id": manifest.get("id"),
                "step": manifest.get("step"),
                "created_at": manifest.get("created_at"),
                "created_at_local": _local_time(manifest.get("created_at")),
                "duration_s": manifest.get("duration_s"),
                "duration_human": format_duration(manifest.get("duration_s")),
                "duration_available": manifest.get("duration_s") is not None,
                "session_duration_s": manifest.get("session_duration_s"),
                "label": manifest.get("label") or "",
                "state": manifest.get("state"),
                "stable": bool(manifest.get("stable")),
                "best": bool(manifest.get("best")),
                "size_bytes": manifest.get("size_bytes"),
                "eval_score": score,
            }
        )
    rows.sort(key=lambda item: (item.get("created_at") or "", item.get("step") or 0), reverse=True)
    return rows


def evaluation_block(store: Store, run: dict) -> dict:
    """Measured win rates, per opponent, and the checkpoint comparison."""
    run_id = run.get("id")
    rows = store.list_evaluations(run_id=run_id, limit=200)
    per_opponent: dict[str, dict] = {}
    for row in rows:
        name = row.get("opponent") or "unknown"
        games = int(row.get("games") or 0)
        wins = int(row.get("wins") or 0)
        entry = per_opponent.setdefault(
            name,
            {"opponent": name, "games": 0, "wins": 0, "losses": 0, "draws": 0, "evaluations": 0},
        )
        entry["games"] += games
        entry["wins"] += wins
        entry["losses"] += int(row.get("losses") or 0)
        entry["draws"] += int(row.get("draws") or 0)
        entry["evaluations"] += 1
    for entry in per_opponent.values():
        entry["win_rate"] = entry["wins"] / entry["games"] if entry["games"] else None
    latest = rows[0] if rows else None
    return {
        "available": bool(rows),
        "count": len(rows),
        "per_opponent": list(per_opponent.values()),
        "latest": latest,
        "reason": None if rows else "no evaluation was measured for this run",
        "evaluations": [
            {
                "id": row.get("id"),
                "model_id": row.get("model_id"),
                "opponent": row.get("opponent"),
                "games": row.get("games"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "draws": row.get("draws"),
                "win_rate": (row.get("details") or {}).get("win_rate"),
                "win_rate_ci95": (row.get("details") or {}).get("win_rate_ci95"),
                "goal_diff": (row.get("details") or {}).get("goal_diff"),
                "phase": row.get("phase"),
                "created_at": row.get("created_at"),
                "created_at_local": _local_time(row.get("created_at")),
            }
            for row in rows[:50]
        ],
    }


def checkpoint_comparison(
    store: Store,
    models_dir: str | Path | None,
    run: dict,
) -> dict:
    """Compare the checkpoints of a run by their measured evaluation."""
    rows = checkpoint_block(store, models_dir, run)
    scored = [row for row in rows if row.get("eval_score") is not None]
    scored.sort(key=lambda item: float(item["eval_score"]), reverse=True)
    return {
        "available": bool(scored),
        "count": len(scored),
        "best": scored[0] if scored else None,
        "ranking": [
            {
                "id": row["id"],
                "step": row["step"],
                "eval_score": row["eval_score"],
                "duration_s": row["duration_s"],
                "duration_human": row["duration_human"],
                "created_at_local": row["created_at_local"],
            }
            for row in scored
        ],
        "reason": None if scored else "no checkpoint of this run has a measured score",
    }


def build_stats(
    store: Store,
    run_id: str,
    *,
    models_dir: str | Path | None = None,
    resources: dict | None = None,
    since_step: int = 0,
) -> dict:
    """Everything the Stats tab shows for one run."""
    run = store.get_run(run_id)
    if not run:
        return {"available": False, "reason": f"unknown run {run_id!r}"}
    resumed_step = int(run.get("resumed_step") or 0)
    return {
        "available": True,
        "run_id": run_id,
        "status": run.get("status"),
        "name": run.get("name"),
        "mode": run.get("mode"),
        "timing": timing_block(store, run),
        "progress": progress_block(store, run),
        "learning": learning_block(store, run, since_step=resumed_step),
        "curves": curves_block(store, run, since_step=resumed_step),
        "performance": performance_block(run, resources=resources, store=store),
        "checkpoints": checkpoint_block(store, models_dir, run),
        "comparison": checkpoint_comparison(store, models_dir, run),
        "evaluation": evaluation_block(store, run),
        "history": timing_block(store, run).get("sessions", []),
        "retention": store.metric_retention_plan(run_id),
    }
