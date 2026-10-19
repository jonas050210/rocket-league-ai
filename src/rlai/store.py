"""SQLite persistence for runs, metrics, logs, models and evaluations.

The database is the single source of truth the GUI reads from.  Nothing in
the dashboard is synthesised: if a table is empty, the corresponding view is
empty too.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    pid INTEGER,
    step INTEGER DEFAULT 0,
    total_steps INTEGER,
    message TEXT DEFAULT '',
    metrics_json TEXT,
    parent_run TEXT,
    resumed_step INTEGER DEFAULT 0,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    ts REAL NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, name, step);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode INTEGER NOT NULL,
    ts REAL NOT NULL,
    reward REAL,
    length INTEGER,
    goals_for INTEGER,
    goals_against INTEGER,
    result TEXT,
    step INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    opponent TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ts REAL NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_run ON logs(run_id, id);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    step INTEGER,
    path TEXT,
    state TEXT,
    mode TEXT,
    created_at TEXT,
    eval_score REAL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    model_id TEXT,
    run_id TEXT,
    mode TEXT NOT NULL,
    opponent TEXT NOT NULL,
    games INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    duration_s REAL,
    details_json TEXT,
    phase TEXT
);
CREATE INDEX IF NOT EXISTS idx_evals_model ON evaluations(model_id, created_at);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    path TEXT,
    status TEXT,
    steps_completed INTEGER,
    duration_s REAL,
    summary_json TEXT,
    report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_s REAL,
    status TEXT NOT NULL,
    start_step INTEGER DEFAULT 0,
    end_step INTEGER DEFAULT 0,
    pid INTEGER,
    note TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_run ON sessions(run_id, id);

CREATE TABLE IF NOT EXISTS player_memory (
    name TEXT PRIMARY KEY,
    games INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    first_seen REAL,
    last_seen REAL,
    data_json TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    mode TEXT,
    description TEXT,
    data_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rivals (
    name TEXT PRIMARY KEY,
    kind TEXT,
    reference TEXT,
    description TEXT,
    evaluations INTEGER DEFAULT 0,
    games INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    data_json TEXT
);

"""

#: columns added after the first release; existing databases are migrated in
#: place so no user data is ever rewritten or dropped
_RUN_MIGRATIONS = {
    "parent_run": "TEXT",
    "resumed_step": "INTEGER DEFAULT 0",
    "report_path": "TEXT",
    # accumulated training time of the run across all its sessions
    "duration_s": "REAL",
    # duration of the *last* session only
    "session_duration_s": "REAL",
    "last_checkpoint_at": "TEXT",
    "last_checkpoint_id": "TEXT",
}
_EPISODE_MIGRATIONS = {
    "step": "INTEGER DEFAULT 0",
    "truncated": "INTEGER DEFAULT 0",
    "opponent": "TEXT",
}
_EVALUATION_MIGRATIONS = {"phase": "TEXT"}


_EVAL_DETAIL_KEYS = (
    "win_rate",
    "win_rate_ci95",
    "win_rate_decisive",
    "goal_diff",
    "touches",
    "obs_version",
    "seed",
    "match_seconds",
    # the protocol has to survive: without it two series cannot be compared
    "protocol",
    "sides",
    "mirror",
    "stochastic",
    "side_results",
    "opponent_class",
    "opponents",
    "kind",
    "per_opponent",
    "diagnostic",
    "strength",
    "score",
    "score_metric",
)


def _evaluation_details(result: dict) -> dict:
    """Per-game rows plus the summary statistics the GUI needs.

    The summary lives at the top level of an evaluation result, but the table
    only has one JSON column.  Merging it into ``details`` keeps the schema
    stable and makes the confidence interval visible in the dashboard.
    """
    details = dict(result.get("details") or {})
    for key in _EVAL_DETAIL_KEYS:
        if key in result and key not in details:
            details[key] = result[key]
    return details


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns that older databases do not have yet.

        ``ALTER TABLE ... ADD COLUMN`` is additive and keeps every existing
        row, so an upgrade never loses run history.
        """
        for table, migrations in (
            ("runs", _RUN_MIGRATIONS),
            ("episodes", _EPISODE_MIGRATIONS),
            ("evaluations", _EVALUATION_MIGRATIONS),
        ):
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, declaration in migrations.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- runs --------------------------------------------------------------
    def create_run(
        self, name: str, mode: str, config: dict, total_steps: int, run_id: str | None = None
    ) -> str:
        run_id = run_id or uuid.uuid4().hex[:12]
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id,name,mode,status,config_json,created_at,total_steps)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, name, mode, "queued", json.dumps(config), utc_now(), int(total_steps)),
            )
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "status",
            "pid",
            "step",
            "message",
            "started_at",
            "ended_at",
            "metrics_json",
            "total_steps",
            "parent_run",
            "resumed_step",
            "report_path",
            "duration_s",
            "session_duration_s",
            "last_checkpoint_at",
            "last_checkpoint_id",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise KeyError(f"cannot update run fields {sorted(unknown)}")
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run_row(row) if row else None

    def list_runs(self, limit: int = 50, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM runs"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._run_row(row) for row in rows]

    def latest_active_run(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE status IN ('starting','running','paused','stopping')"
                " ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._run_row(row) if row else None

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["config"] = json.loads(data.pop("config_json"))
        if data.get("metrics_json"):
            try:
                data["last_metrics"] = json.loads(data["metrics_json"])
            except json.JSONDecodeError:
                data["last_metrics"] = {}
        return data

    # -- metrics -----------------------------------------------------------
    def log_metric(self, run_id: str, step: int, name: str, value: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, step, ts, name, value) VALUES (?,?,?,?,?)",
                (run_id, int(step), time.time(), name, float(value)),
            )

    def log_metrics(self, run_id: str, step: int, values: dict[str, float]) -> None:
        ts = time.time()
        rows = [(run_id, int(step), ts, name, float(value)) for name, value in values.items()]
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO metrics (run_id, step, ts, name, value) VALUES (?,?,?,?,?)", rows
            )

    def get_metrics(
        self,
        run_id: str,
        names: Iterable[str] | None = None,
        since_step: int = 0,
        limit: int = 5000,
    ) -> list[dict]:
        query = "SELECT step, ts, name, value FROM metrics WHERE run_id=? AND step>=?"
        params: list[Any] = [run_id, int(since_step)]
        names = list(names or [])
        if names:
            query += f" AND name IN ({','.join('?' * len(names))})"
            params.extend(names)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def metric_names(self, run_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT name FROM metrics WHERE run_id=? ORDER BY name", (run_id,)
            ).fetchall()
        return [row["name"] for row in rows]

    # -- episodes ----------------------------------------------------------
    def log_episode(
        self,
        run_id: str,
        episode: int,
        reward: float,
        length: int,
        goals_for: int,
        goals_against: int,
        result: str,
        step: int = 0,
        truncated: bool = False,
        opponent: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO episodes"
                " (run_id,episode,ts,reward,length,goals_for,goals_against,result,step,"
                " truncated,opponent)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    episode,
                    time.time(),
                    reward,
                    length,
                    goals_for,
                    goals_against,
                    result,
                    int(step),
                    1 if truncated else 0,
                    opponent,
                ),
            )

    def episode_summary(self, run_id: str, window: int = 100) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT reward,length,goals_for,goals_against,result FROM episodes"
                " WHERE run_id=? ORDER BY id DESC LIMIT ?",
                (run_id, window),
            ).fetchall()
        if not rows:
            return {"count": 0}
        rewards = [row["reward"] for row in rows]
        lengths = [row["length"] for row in rows]
        return {
            "count": len(rows),
            "mean_reward": sum(rewards) / len(rewards),
            "mean_length": sum(lengths) / len(lengths),
            "wins": sum(1 for row in rows if row["result"] == "win"),
            "losses": sum(1 for row in rows if row["result"] == "loss"),
            "draws": sum(1 for row in rows if row["result"] == "draw"),
            "goals_for": sum(row["goals_for"] for row in rows),
            "goals_against": sum(row["goals_against"] for row in rows),
        }

    def episode_stats(self, run_id: str, since_step: int | None = None) -> dict:
        """Whole-run (or since a step) episode aggregates for the report."""
        query = (
            "SELECT reward,length,goals_for,goals_against,result,truncated,opponent"
            " FROM episodes WHERE run_id=?"
        )
        params: list[Any] = [run_id]
        if since_step is not None:
            query += " AND step>=?"
            params.append(int(since_step))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return {"count": 0, "available": False}
        rewards = [float(row["reward"]) for row in rows if row["reward"] is not None]
        lengths = [float(row["length"]) for row in rows if row["length"] is not None]
        wins = sum(1 for row in rows if row["result"] == "win")
        losses = sum(1 for row in rows if row["result"] == "loss")
        draws = sum(1 for row in rows if row["result"] == "draw")
        opponents: dict[str, int] = {}
        for row in rows:
            name = row["opponent"] or "unknown"
            opponents[name] = opponents.get(name, 0) + 1
        return {
            "available": True,
            "count": len(rows),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "min_reward": min(rewards) if rewards else None,
            "max_reward": max(rewards) if rewards else None,
            "mean_length": sum(lengths) / len(lengths) if lengths else None,
            "total_length": sum(lengths) if lengths else 0.0,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / len(rows),
            "truncations": sum(1 for row in rows if row["truncated"]),
            "goals_for": sum(int(row["goals_for"] or 0) for row in rows),
            "goals_against": sum(int(row["goals_against"] or 0) for row in rows),
            "opponents": opponents,
        }

    # -- logs --------------------------------------------------------------
    def log(self, run_id: str | None, level: str, message: str) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO logs (run_id, ts, level, message) VALUES (?,?,?,?)",
                (run_id, time.time(), level.upper(), str(message)),
            )
            return int(cursor.lastrowid)

    def tail_logs(
        self, run_id: str | None = None, after_id: int = 0, limit: int = 500
    ) -> list[dict]:
        query = "SELECT * FROM logs WHERE id>?"
        params: list[Any] = [int(after_id)]
        if run_id:
            query += " AND run_id=?"
            params.append(run_id)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # -- models ------------------------------------------------------------
    def register_model(self, manifest: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO models (id,run_id,step,path,state,mode,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    manifest["id"],
                    manifest.get("run_id"),
                    manifest.get("step"),
                    manifest.get("path"),
                    manifest.get("state", "candidate"),
                    manifest.get("mode"),
                    manifest.get("created_at"),
                ),
            )

    def sync_models(self, manifests: list[dict]) -> None:
        with self._lock, self._connect() as conn:
            for manifest in manifests:
                conn.execute(
                    "INSERT INTO models (id,run_id,step,path,state,mode,created_at)"
                    " VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
                    " state=excluded.state, path=excluded.path, eval_score=models.eval_score",
                    (
                        manifest["id"],
                        manifest.get("run_id"),
                        manifest.get("step"),
                        manifest.get("path"),
                        manifest.get("state", "candidate"),
                        manifest.get("mode"),
                        manifest.get("created_at"),
                    ),
                )

    def list_models(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def set_model_eval_score(self, model_id: str, score: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE models SET eval_score=? WHERE id=?", (score, model_id))

    def get_model_eval_score(self, model_id: str) -> float | None:
        """The measured evaluation score of a checkpoint (``None`` if none).

        The score comes from the ``models`` table when the checkpoint was
        registered there.  Otherwise it is taken from the stored evaluations
        that reference the checkpoint - a checkpoint that was written before
        the database knew about it still has a measured score, and the
        comparison view must not lose it.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT eval_score FROM models WHERE id=?", (model_id,)
            ).fetchone()
            if row and row["eval_score"] is not None:
                return float(row["eval_score"])
            row = conn.execute(
                "SELECT wins, games, details_json FROM evaluations WHERE model_id=?"
                " ORDER BY created_at DESC, id DESC LIMIT 1",
                (model_id,),
            ).fetchone()
        if not row:
            return None
        try:
            details = json.loads(row["details_json"] or "{}")
        except json.JSONDecodeError:
            details = {}
        if details.get("win_rate") is not None:
            return float(details["win_rate"])
        games = int(row["games"] or 0)
        if games:
            return float(row["wins"] or 0) / games
        return None

    # -- evaluations -------------------------------------------------------
    def add_evaluation(self, result: dict) -> str:
        eval_id = result.get("id") or uuid.uuid4().hex[:12]
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluations (id,model_id,run_id,mode,opponent,games,wins,losses,draws,"
                "goals_for,goals_against,created_at,duration_s,details_json,phase)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    eval_id,
                    result.get("model_id"),
                    result.get("run_id"),
                    result["mode"],
                    result["opponent"],
                    result["games"],
                    result["wins"],
                    result["losses"],
                    result["draws"],
                    result["goals_for"],
                    result["goals_against"],
                    result.get("created_at", utc_now()),
                    result.get("duration_s"),
                    json.dumps(_evaluation_details(result)),
                    result.get("phase"),
                ),
            )
        return eval_id

    def list_evaluations(
        self,
        limit: int = 100,
        run_id: str | None = None,
        model_id: str | None = None,
        phase: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM evaluations"
        params: list[Any] = []
        conditions = []
        if run_id:
            conditions.append("run_id=?")
            params.append(run_id)
        if model_id:
            conditions.append("model_id=?")
            params.append(model_id)
        if phase:
            conditions.append("phase=?")
            params.append(phase)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json") or "{}")
            results.append(item)
        return results

    def metric_series(
        self, run_id: str, names: Iterable[str], limit: int = 5000
    ) -> dict[str, list[dict]]:
        """Ordered ``(step, value)`` points per metric name, for charts."""
        wanted = [name for name in names]
        series: dict[str, list[dict]] = {name: [] for name in wanted}
        if not wanted:
            return series
        query = (
            "SELECT step, ts, name, value FROM metrics WHERE run_id=? AND name IN "
            f"({','.join('?' * len(wanted))}) ORDER BY id ASC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, [run_id, *wanted, int(limit)]).fetchall()
        for row in rows:
            series[row["name"]].append(
                {"step": row["step"], "ts": row["ts"], "value": row["value"]}
            )
        return series

    def metric_stats(self, run_id: str, name: str, since_step: int = 0) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, MIN(value) AS min, MAX(value) AS max,"
                " AVG(value) AS avg FROM metrics WHERE run_id=? AND name=? AND step>=?",
                (run_id, name, int(since_step)),
            ).fetchone()
            first = conn.execute(
                "SELECT value FROM metrics WHERE run_id=? AND name=? AND step>=?"
                " ORDER BY id ASC LIMIT 1",
                (run_id, name, int(since_step)),
            ).fetchone()
            last = conn.execute(
                "SELECT value FROM metrics WHERE run_id=? AND name=? AND step>=?"
                " ORDER BY id DESC LIMIT 1",
                (run_id, name, int(since_step)),
            ).fetchone()
        count = int(row["n"] or 0)
        if not count:
            return {"available": False, "count": 0, "name": name}
        return {
            "available": True,
            "name": name,
            "count": count,
            "first": first["value"] if first else None,
            "last": last["value"] if last else None,
            "mean": row["avg"],
            "min": row["min"],
            "max": row["max"],
        }

    def last_metric(self, run_id: str, name: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM metrics WHERE run_id=? AND name=? ORDER BY id DESC LIMIT 1",
                (run_id, name),
            ).fetchone()
        return float(row["value"]) if row else None

    # -- reports -----------------------------------------------------------
    def save_report(
        self, run_id: str, report: dict, path: str | Path | None = None
    ) -> None:
        """Persist the post-run statistics report (one per run, overwritten)."""
        summary = {
            key: report.get(key)
            for key in (
                "run_id",
                "status",
                "mode",
                "steps",
                "duration_s",
                "steps_per_second",
                "device",
                "workers",
                "parent_checkpoint",
                "final_checkpoint",
                "evaluation",
                "warnings",
            )
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO reports (run_id,created_at,path,status,steps_completed,duration_s,"
                "summary_json,report_json) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(run_id) DO UPDATE SET created_at=excluded.created_at,"
                " path=excluded.path, status=excluded.status,"
                " steps_completed=excluded.steps_completed, duration_s=excluded.duration_s,"
                " summary_json=excluded.summary_json, report_json=excluded.report_json",
                (
                    run_id,
                    utc_now(),
                    str(path) if path else None,
                    report.get("status"),
                    (report.get("steps") or {}).get("completed"),
                    report.get("duration_s"),
                    json.dumps(summary, default=str),
                    json.dumps(report, default=str),
                ),
            )

    def get_report(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE run_id=?", (run_id,)
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["report"] = json.loads(data.pop("report_json") or "{}")
        except json.JSONDecodeError:
            data["report"] = {}
        try:
            data["summary"] = json.loads(data.pop("summary_json") or "{}")
        except json.JSONDecodeError:
            data["summary"] = {}
        return data

    def list_reports(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, path, status, steps_completed, duration_s,"
                " summary_json FROM reports ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            try:
                item["summary"] = json.loads(item.pop("summary_json") or "{}")
            except json.JSONDecodeError:
                item["summary"] = {}
            results.append(item)
        return results

    # -- player memory -----------------------------------------------------
    def list_player_memory(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM player_memory ORDER BY last_seen DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def replace_player_memory(self, rows: list[dict]) -> int:
        """Replace the whole memory with ``rows`` (atomic)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM player_memory")
            conn.executemany(
                "INSERT OR REPLACE INTO player_memory (name,games,wins,losses,draws,"
                "goals_for,goals_against,first_seen,last_seen,data_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        row["name"],
                        int(row.get("games") or 0),
                        int(row.get("wins") or 0),
                        int(row.get("losses") or 0),
                        int(row.get("draws") or 0),
                        int(row.get("goals_for") or 0),
                        int(row.get("goals_against") or 0),
                        float(row.get("first_seen") or 0.0),
                        float(row.get("last_seen") or 0.0),
                        row.get("data_json"),
                    )
                    for row in rows
                ],
            )
        return len(rows)

    def delete_player_memory(self, name: str | None = None) -> int:
        """Delete one player, or the whole memory.  Returns the row count."""
        with self._lock, self._connect() as conn:
            if name:
                cursor = conn.execute("DELETE FROM player_memory WHERE name=?", (name,))
            else:
                cursor = conn.execute("DELETE FROM player_memory")
            return cursor.rowcount or 0

    # -- teams -------------------------------------------------------------
    def save_team(self, name: str, data: dict, *, mode: str | None = None) -> None:
        now = utc_now()
        payload = json.dumps(data, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO teams (name,created_at,updated_at,mode,description,data_json)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET"
                " updated_at=excluded.updated_at, mode=excluded.mode,"
                " description=excluded.description, data_json=excluded.data_json",
                (
                    name,
                    now,
                    now,
                    mode or data.get("mode"),
                    data.get("description") or "",
                    payload,
                ),
            )

    def list_teams(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM teams ORDER BY updated_at DESC"
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            try:
                item["team"] = json.loads(item.pop("data_json") or "{}")
            except json.JSONDecodeError:
                item["team"] = {}
            results.append(item)
        return results

    def get_team(self, name: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM teams WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["team"] = json.loads(item.pop("data_json") or "{}")
        except json.JSONDecodeError:
            item["team"] = {}
        return item

    def delete_team(self, name: str) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM teams WHERE name=?", (name,))
            return cursor.rowcount or 0

    # -- rivals ------------------------------------------------------------
    def save_rival(self, name: str, data: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO rivals (name,kind,reference,description,evaluations,games,"
                "wins,losses,draws,goals_for,goals_against,data_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET"
                " kind=excluded.kind, reference=excluded.reference,"
                " description=excluded.description, evaluations=excluded.evaluations,"
                " games=excluded.games, wins=excluded.wins, losses=excluded.losses,"
                " draws=excluded.draws, goals_for=excluded.goals_for,"
                " goals_against=excluded.goals_against, data_json=excluded.data_json",
                (
                    name,
                    data.get("kind"),
                    data.get("reference"),
                    data.get("description") or "",
                    int(data.get("evaluations") or 0),
                    int(data.get("games") or 0),
                    int(data.get("wins") or 0),
                    int(data.get("losses") or 0),
                    int(data.get("draws") or 0),
                    int(data.get("goals_for") or 0),
                    int(data.get("goals_against") or 0),
                    json.dumps(data, default=str),
                ),
            )

    def list_rivals(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM rivals ORDER BY name").fetchall()
        results = []
        for row in rows:
            item = dict(row)
            try:
                item["rival"] = json.loads(item.pop("data_json") or "{}")
            except json.JSONDecodeError:
                item["rival"] = {}
            results.append(item)
        return results

    def delete_rival(self, name: str) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM rivals WHERE name=?", (name,))
            return cursor.rowcount or 0

    # -- retention ---------------------------------------------------------
    def prune_metrics(self, run_id: str, keep_per_name: int = 20_000) -> int:
        """Keep only the newest ``keep_per_name`` rows of every metric.

        A long run writes a few dozen metric names per iteration.  Without a
        bound the table grows without limit and the queries behind the
        charts get slower.  The *newest* rows are kept, so the charts and
        the report still describe the end of the run, where the interesting
        behaviour is.
        """
        keep = max(1, int(keep_per_name))
        with self._lock, self._connect() as conn:
            names = [
                row["name"]
                for row in conn.execute(
                    "SELECT DISTINCT name FROM metrics WHERE run_id=?", (run_id,)
                )
            ]
            deleted = 0
            for name in names:
                cursor = conn.execute(
                    "DELETE FROM metrics WHERE run_id=? AND name=? AND id NOT IN ("
                    "SELECT id FROM metrics WHERE run_id=? AND name=? ORDER BY id DESC"
                    " LIMIT ?)",
                    (run_id, name, run_id, name, keep),
                )
                deleted += cursor.rowcount or 0
            return deleted

    def prune_logs(self, run_id: str | None = None, keep: int = 5_000) -> int:
        """Keep only the newest ``keep`` log rows (of one run, or of all)."""
        keep = max(1, int(keep))
        with self._lock, self._connect() as conn:
            if run_id:
                cursor = conn.execute(
                    "DELETE FROM logs WHERE run_id=? AND id NOT IN ("
                    "SELECT id FROM logs WHERE run_id=? ORDER BY id DESC LIMIT ?)",
                    (run_id, run_id, keep),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM logs WHERE id NOT IN ("
                    "SELECT id FROM logs ORDER BY id DESC LIMIT ?)",
                    (keep,),
                )
            return cursor.rowcount or 0

    def prune_episodes(self, run_id: str, keep: int = 20_000) -> int:
        """Keep only the newest ``keep`` episode rows of a run."""
        keep = max(1, int(keep))
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM episodes WHERE run_id=? AND id NOT IN ("
                "SELECT id FROM episodes WHERE run_id=? ORDER BY id DESC LIMIT ?)",
                (run_id, run_id, keep),
            )
            return cursor.rowcount or 0

    def metric_retention_plan(self, run_id: str) -> dict:
        """Size of the metric table, for the report and the stats view."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COUNT(DISTINCT name) AS names FROM metrics"
                " WHERE run_id=?",
                (run_id,),
            ).fetchone()
            oldest = conn.execute(
                "SELECT MIN(ts) AS ts FROM metrics WHERE run_id=?", (run_id,)
            ).fetchone()
            newest = conn.execute(
                "SELECT MAX(ts) AS ts FROM metrics WHERE run_id=?", (run_id,)
            ).fetchone()
        return {
            "rows": int(row["n"] or 0),
            "names": int(row["names"] or 0),
            "oldest_ts": oldest["ts"] if oldest else None,
            "newest_ts": newest["ts"] if newest else None,
        }

    # -- sessions ----------------------------------------------------------
    def start_session(
        self,
        run_id: str,
        *,
        started_at: str,
        start_step: int = 0,
        pid: int | None = None,
    ) -> int:
        """Record the start of a training session and return its id."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (run_id,started_at,status,start_step,pid)"
                " VALUES (?,?,?,?,?)",
                (run_id, started_at, "running", int(start_step), pid),
            )
            return int(cursor.lastrowid)

    def finish_session(
        self,
        session_id: int,
        *,
        ended_at: str,
        duration_s: float | None,
        status: str,
        end_step: int = 0,
        note: str = "",
    ) -> None:
        """Close a session with its measured duration."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, duration_s=?, status=?, end_step=?,"
                " note=? WHERE id=?",
                (ended_at, duration_s, status, int(end_step), note, int(session_id)),
            )

    def list_sessions(self, run_id: str | None = None, limit: int = 200) -> list[dict]:
        """Sessions of a run (newest first), or of every run."""
        query = "SELECT * FROM sessions"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id=?"
            params.append(run_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def session_totals(self, run_id: str) -> dict:
        """Aggregated session information for one run.

        ``accumulated_s`` is the sum of the *finished* sessions only; a
        session that is still marked running (a crashed process) is reported
        separately, because its duration was never measured.  A session
        with the status ``inherited`` carries the training time a
        continued run took over from its parent, so the total of a resumed
        run is complete.
        """
        rows = self.list_sessions(run_id)
        finished = [row for row in rows if row.get("duration_s") is not None]
        # A session is "unfinished" while it is still marked running - a
        # crashed session is closed as *interrupted* and keeps its unknown
        # duration, but it is not running any more.
        unfinished = [row for row in rows if row.get("status") == "running"]
        inherited = [row for row in finished if row.get("status") == "inherited"]
        return {
            "sessions": rows,
            "count": len(rows),
            "finished_count": len(finished),
            "inherited_count": len(inherited),
            "accumulated_s": (
                sum(float(row["duration_s"]) for row in finished) if finished else 0.0
            ),
            "own_s": (
                sum(
                    float(row["duration_s"])
                    for row in finished
                    if row.get("status") != "inherited"
                )
                if finished
                else 0.0
            ),
            "has_unfinished": bool(unfinished),
            "unfinished_count": len(unfinished),
            "last_started_at": rows[0]["started_at"] if rows else None,
            "last_ended_at": next(
                (row["ended_at"] for row in rows if row.get("ended_at")), None
            ),
        }

    def recover_interrupted_sessions(self, run_id: str, *, note: str = "") -> int:
        """Mark sessions that never ended as interrupted.

        A process that died leaves a session row without an end.  The
        duration was never measured, so it stays ``None`` (unknown) instead
        of being guessed from the wall clock.
        """
        rows = [
            row
            for row in self.list_sessions(run_id)
            if row.get("duration_s") is None and row.get("status") == "running"
        ]
        for row in rows:
            self.finish_session(
                int(row["id"]),
                ended_at=utc_now(),
                duration_s=None,
                status="interrupted",
                end_step=int(row.get("end_step") or 0),
                note=note or "the process ended without closing the session",
            )
        return len(rows)

    # -- settings ----------------------------------------------------------
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO settings (key,value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
