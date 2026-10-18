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
    metrics_json TEXT
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
    result TEXT
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
    details_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_evals_model ON evaluations(model_id, created_at);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

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
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO episodes (run_id,episode,ts,reward,length,goals_for,goals_against,result)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (run_id, episode, time.time(), reward, length, goals_for, goals_against, result),
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

    # -- evaluations -------------------------------------------------------
    def add_evaluation(self, result: dict) -> str:
        eval_id = result.get("id") or uuid.uuid4().hex[:12]
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluations (id,model_id,run_id,mode,opponent,games,wins,losses,draws,"
                "goals_for,goals_against,created_at,duration_s,details_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    json.dumps(result.get("details", {})),
                ),
            )
        return eval_id

    def list_evaluations(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json") or "{}")
            results.append(item)
        return results

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
