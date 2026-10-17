"""Durable experiments, structured event cursors, settings and confirmation audit trail."""

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .config import TrainingConfig, canonical
from .domain import ACTIVE, AppError, RunStatus
from .persistence import atomic_json

ID = re.compile(r"^[a-f0-9]{32}$")


def new_id() -> str:
    return uuid.uuid4().hex


def checked_id(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise AppError("Invalid identifier; select an item from the inventory.", 422)
    return value


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "control.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                config TEXT NOT NULL, config_hash TEXT NOT NULL, metadata TEXT NOT NULL,
                metrics TEXT NOT NULL DEFAULT '{}', error TEXT, parent TEXT,
                created REAL NOT NULL, updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                run_id TEXT, level TEXT NOT NULL, kind TEXT NOT NULL,
                message TEXT NOT NULL, data TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_run_seq ON events(run_id, seq);
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                rationale TEXT NOT NULL, impact TEXT NOT NULL, digest TEXT NOT NULL,
                plan_revision INTEGER NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, expires REAL NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS benchmarks (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                recommendation TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, candidate TEXT NOT NULL,
                status TEXT NOT NULL, aggregate TEXT, resources TEXT, error TEXT,
                run_id TEXT, created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS benchmark_results_session ON benchmark_results(benchmark_id);
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                result TEXT, trajectory TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
            );
        """)
        if self.setting("plan") is None:
            self.set_settings({"plan": TrainingConfig().model_dump(mode="json"), "plan_revision": 1})
        self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()

    def setting(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_setting(self, key: str, value) -> None:
        self.set_settings({key: value})

    def set_settings(self, values: dict) -> None:
        """Publish related settings in one SQLite transaction, including plan + revision."""
        with self.lock, self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                [(key, canonical(value)) for key, value in values.items()],
            )

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / checked_id(run_id)

    @staticmethod
    def decode(row) -> dict | None:
        if row is None:
            return None
        value = dict(row)
        for field in ("config", "metadata", "metrics", "payload", "result", "data",
                      "spec", "candidate", "aggregate", "resources", "recommendation"):
            if field in value and value[field] is not None:
                value[field] = json.loads(value[field])
        return value

    def create_run(self, config: TrainingConfig, metadata: dict, parent: dict | None = None) -> dict:
        run_id, now = new_id(), time.time()
        path = self.run_dir(run_id)
        path.mkdir(parents=True, mode=0o700)
        atomic_json(path / "config.json", config.model_dump(mode="json"))
        atomic_json(path / "metadata.json", {**metadata, "parent": parent})
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO runs(id,name,status,config,config_hash,metadata,parent,created,updated) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, config.name, RunStatus.STARTING, config.model_dump_json(), config.digest(),
                 canonical(metadata), canonical(parent) if parent else None, now, now),
            )
        return self.run(run_id)

    def run(self, run_id: str) -> dict:
        with self.lock:
            row = self.db.execute("SELECT * FROM runs WHERE id=?", (checked_id(run_id),)).fetchone()
            result = self.decode(row)
            if result is None:
                raise AppError("Run not found. Refresh the experiment inventory.", 404)
            if result["parent"]:
                result["parent"] = json.loads(result["parent"])
            return result

    def runs(self, limit: int = 100) -> list[dict]:
        with self.lock:
            ids = self.db.execute("SELECT id FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
            return [self.run(row[0]) for row in ids]

    def update_run(self, run_id: str, **fields) -> dict:
        if not fields.keys() <= {"status", "metrics", "error"}:
            raise ValueError("Only mutable lifecycle fields may be updated")
        fields["updated"] = time.time()
        for key in ("metrics",):
            if key in fields:
                fields[key] = canonical(fields[key])
        with self.lock, self.db:
            self.db.execute(
                f"UPDATE runs SET {', '.join(key + '=?' for key in fields)} WHERE id=?",
                (*fields.values(), checked_id(run_id)),
            )
        return self.run(run_id)

    def interrupt_unfinished(self) -> None:
        # A review is valid only in the backend session in which it was presented.
        with self.lock, self.db:
            self.db.execute(
                "UPDATE proposals SET status='expired', result=? WHERE status='pending'",
                (canonical({"error": "Backend restarted. Review a new proposal before confirming."}),),
            )
        for run in self.runs(limit=100000):
            if run["status"] in ACTIVE:
                message = (
                    "Backend disconnected before a terminal acknowledgement. No automatic restart. "
                    "Inspect checkpoints and explicitly confirm recovery; environments will reset."
                )
                self.update_run(run["id"], status=RunStatus.INTERRUPTED, error=message)
                self.event("warning", "recovery_available", message, run["id"])
        with self.lock, self.db:
            self.db.execute(
                "UPDATE proposals SET status='failed', result=? WHERE status='executing'",
                (canonical({"error": "Backend interrupted during action; inspect inventory before retrying."}),),
            )
            self.db.execute(
                "UPDATE benchmarks SET status='interrupted', error='Backend restarted during the benchmark. "
                "Completed candidate results are retained; start a new benchmark to re-measure the rest.' "
                "WHERE status='running'"
            )
            self.db.execute(
                "UPDATE evaluations SET status='interrupted', error='Backend restarted during evaluation. "
                "No partial result is reported; start a new evaluation.' WHERE status='running'"
            )

    def event(self, level: str, kind: str, message: str, run_id: str | None = None, data=None) -> int:
        with self.lock, self.db:
            cursor = self.db.execute(
                "INSERT INTO events(ts,run_id,level,kind,message,data) VALUES (?,?,?,?,?,?)",
                (time.time(), run_id, level, kind, message[:16000], canonical(data or {})),
            )
            return cursor.lastrowid

    def events(self, after: int = 0, limit: int = 200) -> dict:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, min(limit, 1000)),
            ).fetchall()
            events = [self.decode(row) for row in rows]
            return {"events": events, "cursor": events[-1]["seq"] if events else after}

    def events_for_run(self, run_id: str, kind: str, limit: int = 200) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM events WHERE run_id=? AND kind=? ORDER BY seq DESC LIMIT ?",
                (checked_id(run_id), kind, min(limit, 1000)),
            ).fetchall()
            return [self.decode(row) for row in reversed(rows)]

    def save_proposal(self, value: dict) -> dict:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                tuple(canonical(value[key]) if key in {"payload", "result"} and value[key] is not None else value[key]
                      for key in ("id", "kind", "payload", "rationale", "impact", "digest", "plan_revision",
                                  "status", "created", "expires", "result")),
            )
        return self.proposal(value["id"])

    def proposal(self, proposal_id: str) -> dict:
        with self.lock:
            row = self.db.execute("SELECT * FROM proposals WHERE id=?", (checked_id(proposal_id),)).fetchone()
            if not row:
                raise AppError("Proposal not found.", 404)
            return self.decode(row)

    def finish_proposal(self, proposal_id: str, status: str, result: dict | None = None) -> None:
        with self.lock, self.db:
            self.db.execute(
                "UPDATE proposals SET status=?,result=? WHERE id=?",
                (status, canonical(result) if result is not None else None, checked_id(proposal_id)),
            )

    # -- Benchmark sessions and genuine measurement results -----------------
    def create_benchmark(self, spec: dict) -> dict:
        benchmark_id, now = new_id(), time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO benchmarks(id,spec,status,created,updated) VALUES (?,?,?,?,?)",
                (benchmark_id, canonical(spec), "running", now, now),
            )
        return self.benchmark(benchmark_id)

    def benchmark(self, benchmark_id: str) -> dict:
        with self.lock:
            row = self.db.execute("SELECT * FROM benchmarks WHERE id=?", (checked_id(benchmark_id),)).fetchone()
            result = self.decode(row)
            if result is None:
                raise AppError("Benchmark not found. Refresh the benchmark history.", 404)
            return result

    def benchmarks(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM benchmarks ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
            return [self.decode(row) for row in rows]

    def update_benchmark(self, benchmark_id: str, **fields) -> dict:
        if not fields.keys() <= {"status", "recommendation", "error"}:
            raise ValueError("Only benchmark lifecycle fields may be updated")
        fields["updated"] = time.time()
        for key in ("recommendation",):
            if key in fields and fields[key] is not None:
                fields[key] = canonical(fields[key])
        with self.lock, self.db:
            self.db.execute(
                f"UPDATE benchmarks SET {', '.join(key + '=?' for key in fields)} WHERE id=?",
                (*fields.values(), checked_id(benchmark_id)),
            )
        return self.benchmark(benchmark_id)

    def save_benchmark_result(self, benchmark_id: str, result: dict) -> dict:
        result_id, now = new_id(), time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO benchmark_results VALUES (?,?,?,?,?,?,?,?,?)",
                (result_id, checked_id(benchmark_id), canonical(result["candidate"]), result["status"],
                 canonical(result["aggregate"]) if result.get("aggregate") else None,
                 canonical(result["resources"]) if result.get("resources") else None,
                 result.get("error"), result.get("run_id"), now),
            )
        with self.lock:
            row = self.db.execute("SELECT * FROM benchmark_results WHERE id=?", (result_id,)).fetchone()
            return self.decode(row)

    def benchmark_results(self, benchmark_id: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM benchmark_results WHERE benchmark_id=? ORDER BY created", (checked_id(benchmark_id),),
            ).fetchall()
            return [self.decode(row) for row in rows]

    # -- Checkpoint evaluations ----------------------------------------------
    def create_evaluation(self, spec: dict) -> dict:
        evaluation_id, now = new_id(), time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO evaluations(id,spec,status,created,updated) VALUES (?,?,?,?,?)",
                (evaluation_id, canonical(spec), "running", now, now),
            )
        return self.evaluation(evaluation_id)

    def evaluation(self, evaluation_id: str) -> dict:
        with self.lock:
            row = self.db.execute("SELECT * FROM evaluations WHERE id=?", (checked_id(evaluation_id),)).fetchone()
            result = self.decode(row)
            if result is None:
                raise AppError("Evaluation not found. Refresh the evaluation history.", 404)
            return result

    def evaluations(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM evaluations ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
            return [self.decode(row) for row in rows]

    def update_evaluation(self, evaluation_id: str, **fields) -> dict:
        if not fields.keys() <= {"status", "result", "trajectory", "error"}:
            raise ValueError("Only evaluation lifecycle fields may be updated")
        fields["updated"] = time.time()
        for key in ("result",):
            if key in fields and fields[key] is not None:
                fields[key] = canonical(fields[key])
        with self.lock, self.db:
            self.db.execute(
                f"UPDATE evaluations SET {', '.join(key + '=?' for key in fields)} WHERE id=?",
                (*fields.values(), checked_id(evaluation_id)),
            )
        return self.evaluation(evaluation_id)
