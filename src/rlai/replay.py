"""Match and replay analysis.

The project records what it can actually measure: every finished episode and
every evaluated game leaves a row with its goals, touches, outcome and
duration.  This module analyses *that* data.

Rocket League replays (``.replay`` files) are a different matter: they are
produced by the game, they are not part of the training pipeline and no
parser is available in this environment.  The module therefore says
explicitly that replay parsing is unavailable instead of pretending to
analyse a file it cannot read.

Everything the GUI shows in the "match analysis" view therefore comes from
real measurements, and every section states its data source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .store import Store

#: how many games an analysis may look at
MAX_GAMES = 200


def replay_support() -> dict:
    """What the environment can do with replays."""
    return {
        "available": False,
        "reason": (
            "no replay parser is available: Rocket League replays are produced by "
            "the game, are not part of the training pipeline and cannot be read "
            "in this environment.  The analysis below works on the match data the "
            "simulator and the evaluations actually recorded."
        ),
        "supported_inputs": ["evaluation game rows", "episode rows"],
    }


def analyze_evaluations(store: Store, run_id: str | None = None) -> dict:
    """Analyse the games of the recorded evaluations.

    The per-game rows are stored inside the evaluation's ``details``; they
    contain the goals, the touches, the side the policy played and the
    duration of every single game, which is enough for a real analysis.
    """
    rows = store.list_evaluations(run_id=run_id, limit=MAX_GAMES)
    games: list[dict] = []
    for row in rows:
        details = row.get("details") or {}
        for game in details.get("games") or []:
            games.append(
                {
                    "evaluation_id": row.get("id"),
                    "model_id": row.get("model_id"),
                    "opponent": row.get("opponent"),
                    "created_at": row.get("created_at"),
                    **{
                        key: game.get(key)
                        for key in (
                            "game",
                            "seed",
                            "side",
                            "mirrored",
                            "goals_for",
                            "goals_against",
                            "outcome",
                            "ticks",
                            "touches",
                        )
                    },
                }
            )
    if not games:
        return {
            "available": False,
            "reason": "no evaluated game with per-game details was recorded yet",
            "games": 0,
            "source": "evaluation game rows",
        }
    return {
        "available": True,
        "source": "evaluation game rows",
        "games": len(games),
        "summary": _summarise_games(games),
        "by_opponent": _group_by_opponent(games),
        "by_side": _group_by_side(games),
        "timeline": _timeline(games),
    }


def _summarise_games(games: list[dict]) -> dict:
    wins = sum(1 for game in games if game.get("outcome") == "win")
    losses = sum(1 for game in games if game.get("outcome") == "loss")
    draws = sum(1 for game in games if game.get("outcome") == "draw")
    goals_for = sum(int(game.get("goals_for") or 0) for game in games)
    goals_against = sum(int(game.get("goals_against") or 0) for game in games)
    touches = [int(game.get("touches") or 0) for game in games]
    durations = [float(game.get("ticks") or 0) / 120.0 for game in games]
    decisive = wins + losses
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(games) if games else None,
        "win_rate_decisive": (wins / decisive) if decisive else None,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "mean_touches": (sum(touches) / len(touches)) if touches else None,
        "mean_match_seconds": (sum(durations) / len(durations)) if durations else None,
    }


def _group_by_opponent(games: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for game in games:
        groups.setdefault(str(game.get("opponent") or "unknown"), []).append(game)
    return [
        {"opponent": name, **_summarise_games(items), "games": len(items)}
        for name, items in sorted(groups.items())
    ]


def _group_by_side(games: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for game in games:
        groups.setdefault(str(game.get("side") or "unknown"), []).append(game)
    return [
        {"side": name, **_summarise_games(items), "games": len(items)}
        for name, items in sorted(groups.items())
    ]


def _timeline(games: list[dict]) -> list[dict]:
    """The games in the order they were played, with their result."""
    ordered = sorted(
        games,
        key=lambda game: (str(game.get("created_at") or ""), int(game.get("game") or 0)),
    )
    return [
        {
            "created_at": game.get("created_at"),
            "opponent": game.get("opponent"),
            "side": game.get("side"),
            "outcome": game.get("outcome"),
            "goals_for": game.get("goals_for"),
            "goals_against": game.get("goals_against"),
            "touches": game.get("touches"),
        }
        for game in ordered[-100:]
    ]


def analyze_episodes(store: Store, run_id: str) -> dict:
    """Analyse the training episodes of a run.

    The episodes describe the *optimisation* (reward, length, goals), not
    the playing strength - the report keeps the two apart.
    """
    stats = store.episode_stats(run_id)
    if not stats.get("available"):
        return {
            "available": False,
            "reason": "no episode finished in this run yet",
            "source": "episode rows",
        }
    return {
        "available": True,
        "source": "episode rows",
        "summary": stats,
        "opponents": stats.get("opponents", {}),
        "note": (
            "episodes describe the optimisation; the playing strength comes from "
            "the evaluation section"
        ),
    }


def analyze_matches(
    store: Store,
    run_id: str | None = None,
    *,
    models_dir: str | Path | None = None,
) -> dict:
    """The full match analysis view."""
    evaluations = analyze_evaluations(store, run_id)
    episodes = analyze_episodes(store, run_id) if run_id else {
        "available": False,
        "reason": "no run selected",
    }
    return {
        "schema": "rlai-match-analysis-v1",
        "replays": replay_support(),
        "evaluations": evaluations,
        "episodes": episodes,
        "generated_from": "stored evaluations and episodes",
    }


def load_replay(path: str | Path) -> dict:
    """Attempt to read a replay file.

    The environment has no replay parser, so this reports the limitation
    instead of failing with a traceback.
    """
    target = Path(path)
    if not target.exists():
        return {
            "available": False,
            "reason": f"the file {target} does not exist",
        }
    return {
        "available": False,
        "reason": (
            "replay parsing is not available in this environment; the analysis "
            "works on the match data recorded by the simulator and the evaluations"
        ),
        "path": str(target),
        "size_bytes": target.stat().st_size,
    }


def export_analysis(analysis: dict, path: str | Path) -> Path:
    """Write an analysis to a JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    return target


def merge_analyses(analyses: Iterable[dict]) -> dict:
    """Merge several analyses (used when a workspace view spans runs)."""
    merged: list[dict] = []
    for analysis in analyses:
        summary = (analysis.get("evaluations") or {}).get("summary")
        if summary:
            merged.append(summary)
    if not merged:
        return {"available": False, "reason": "no analysis with measured games"}
    games = sum(item.get("games", 0) for item in merged)
    wins = sum(item.get("wins", 0) for item in merged)
    return {
        "available": True,
        "games": games,
        "wins": wins,
        "losses": sum(item.get("losses", 0) for item in merged),
        "draws": sum(item.get("draws", 0) for item in merged),
        "win_rate": (wins / games) if games else None,
        "sources": len(merged),
    }


@dataclass
class AnalysisRequest:
    """The parameters of an analysis request."""

    run_id: str | None = None
    include_episodes: bool = True
    include_timeline: bool = True

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "include_episodes": self.include_episodes,
            "include_timeline": self.include_timeline,
        }


def describe_sources() -> dict:
    """What the analysis is built from - shown in the GUI."""
    return {
        "evaluation_rows": {
            "available": True,
            "description": "per-game results of every evaluation the project ran",
        },
        "episode_rows": {
            "available": True,
            "description": "finished training episodes (optimisation, not strength)",
        },
        "replay_files": replay_support(),
    }


def any_analysis(rows: list[dict]) -> dict[str, Any]:
    """Small helper for the API: does any row carry per-game details?"""
    for row in rows:
        details = row.get("details") or {}
        if details.get("games"):
            return {"available": True, "count": len(details["games"])}
    return {"available": False, "reason": "no per-game details stored"}
