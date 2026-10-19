"""Reliable training-time accounting.

Training time has to survive a crash, a restart and a resume, and it must
never claim that training continued while the process was stopped.  Three
different clocks are therefore involved:

* **UTC wall clock** - tells *when* something happened.  It can jump (NTP,
  manual clock changes), so it is never used to measure a duration.
* **Monotonic clock** - measures how long a running process actually worked.
  It cannot jump, so it is the only clock allowed to produce a duration.
* **Accumulated total** - the sum of the durations of all sessions that
  belong to one training run.  It is persisted, so a resume continues the
  total instead of restarting it.

The central object is :class:`SessionTimer`.  It owns one session: it
records the UTC start, measures the elapsed time with the monotonic clock
and hands the result to the store when the session ends.  The
:class:`TrainingClock` on top of it adds the accumulated total and knows how
to merge a duration recovered from a checkpoint.

Two rules keep the numbers honest:

* A checkpoint never stores more training time than was actually measured.
  The duration written into a checkpoint is the accumulated total at the
  moment the checkpoint was written.
* Time spent between sessions (the process was stopped, the machine was
  off) is not training time and is never counted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

#: status of a finished session
COMPLETED = "completed"
STOPPED = "stopped"
INTERRUPTED = "interrupted"
FAILED = "failed"
RUNNING = "running"


def utc_now_iso() -> str:
    """UTC timestamp with millisecond resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_utc(value: str | None) -> datetime | None:
    """Parse an ISO timestamp; ``None`` when it cannot be read."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_duration(seconds: float | None) -> str:
    """Human readable duration (``1h 02m 03s``); ``"unknown"`` when absent."""
    if seconds is None:
        return "unknown"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if total < 0:
        return "unknown"
    if total < 60:
        return f"{total:.0f} s"
    minutes, secs = divmod(int(total), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


@dataclass
class SessionRecord:
    """One training session: a period in which the process actually ran."""

    run_id: str
    started_at: str
    ended_at: str | None = None
    duration_s: float | None = None
    status: str = RUNNING
    start_step: int = 0
    end_step: int = 0
    pid: int | None = None
    session_id: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "status": self.status,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "pid": self.pid,
            "session_id": self.session_id,
            "note": self.note,
        }


class SessionTimer:
    """Measures the duration of one running training session.

    The elapsed time is read from a monotonic clock, so a system clock
    adjustment during the session cannot change the result.  The wall-clock
    timestamps are only used to say *when* the session started and ended.
    """

    def __init__(
        self,
        run_id: str,
        *,
        started_at: str | None = None,
        start_step: int = 0,
        pid: int | None = None,
        session_id: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.run_id = run_id
        self.started_at = started_at or utc_now_iso()
        self.start_step = int(start_step)
        self.pid = pid
        #: the id of the database row that holds this session
        self.session_id = session_id
        self._monotonic = monotonic
        self._origin = monotonic()
        self._ended_at: str | None = None
        self._duration: float | None = None
        self._status = RUNNING
        self._end_step: int | None = None
        #: free text describing how the session ended
        self.note = ""

    # -- state -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._status == RUNNING

    @property
    def status(self) -> str:
        return self._status

    @property
    def duration_s(self) -> float | None:
        """Measured duration; ``None`` while the session is still running."""
        if self._duration is not None:
            return self._duration
        if self._status == RUNNING:
            return max(0.0, self._monotonic() - self._origin)
        return None

    def elapsed(self) -> float:
        """Seconds since the session started (live)."""
        return max(0.0, self._monotonic() - self._origin)

    # -- lifecycle ---------------------------------------------------------
    def stop(
        self,
        status: str = COMPLETED,
        *,
        end_step: int | None = None,
        note: str = "",
    ) -> SessionRecord:
        """End the session and return the record to persist."""
        if self._status != RUNNING:
            # stopping twice must not change the measured duration
            return self.record(note=note)
        self._duration = max(0.0, self._monotonic() - self._origin)
        self._ended_at = utc_now_iso()
        self._status = status
        self._end_step = end_step
        if note:
            self.note = note
        return self.record()

    def mark_interrupted(self, note: str = "") -> SessionRecord:
        """Close the session as interrupted (used when recovering a crash)."""
        return self.stop(INTERRUPTED, note=note)

    def record(self, *, note: str = "") -> SessionRecord:
        return SessionRecord(
            run_id=self.run_id,
            started_at=self.started_at,
            ended_at=self._ended_at,
            duration_s=self._duration if self._status != RUNNING else None,
            status=self._status,
            start_step=self.start_step,
            end_step=self._end_step if self._end_step is not None else self.start_step,
            pid=self.pid,
            session_id=self.session_id,
            note=note or self.note,
        )


class TrainingClock:
    """Accumulated training time of one run, across all its sessions.

    ``accumulated_s`` is the sum of every finished session.  It is restored
    from a checkpoint when a run continues, so the total keeps growing
    instead of restarting at zero.  The value is written into every
    checkpoint, which makes it survive a crash.
    """

    def __init__(
        self,
        run_id: str,
        *,
        accumulated_s: float | None = None,
        start_step: int = 0,
        pid: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.run_id = run_id
        self._accumulated = max(0.0, float(accumulated_s or 0.0))
        self._sessions: list[SessionRecord] = []
        self._monotonic = monotonic
        self._timer: SessionTimer | None = None
        self._start_step = int(start_step)
        self._pid = pid

    # -- accumulated total --------------------------------------------------
    @property
    def accumulated_s(self) -> float:
        """Total training time of all finished sessions."""
        return self._accumulated

    @property
    def current_session_s(self) -> float | None:
        """Duration of the session that is running right now."""
        return self._timer.duration_s if self._timer else None

    @property
    def total_s(self) -> float:
        """Accumulated time including the running session."""
        current = self.current_session_s or 0.0
        return self._accumulated + current

    @property
    def sessions(self) -> list[SessionRecord]:
        return list(self._sessions)

    @property
    def timer(self) -> SessionTimer | None:
        return self._timer

    # -- lifecycle ---------------------------------------------------------
    def start_session(
        self, *, start_step: int | None = None, session_id: int | None = None
    ) -> SessionTimer:
        """Begin a new session.  The previous one must have been stopped."""
        if self._timer is not None and self._timer.running:
            raise RuntimeError(
                "a session is already running; stop it before starting the next one"
            )
        self._start_step = int(start_step if start_step is not None else self._start_step)
        self._timer = SessionTimer(
            self.run_id,
            start_step=self._start_step,
            pid=self._pid,
            session_id=session_id,
            monotonic=self._monotonic,
        )
        return self._timer

    def stop_session(
        self,
        status: str = COMPLETED,
        *,
        end_step: int | None = None,
        note: str = "",
    ) -> SessionRecord:
        """End the running session and add its duration to the total."""
        if self._timer is None:
            raise RuntimeError("no session is running")
        record = self._timer.stop(status, end_step=end_step, note=note)
        self._accumulated += float(record.duration_s or 0.0)
        self._sessions.append(record)
        self._timer = None
        return record

    def adopt(self, record: SessionRecord) -> None:
        """Add an already finished session to the total."""
        self._sessions.append(record)
        self._accumulated += float(record.duration_s or 0.0)

    def restore(self, accumulated_s: float | None) -> None:
        """Set the accumulated total (e.g. from a checkpoint).

        ``None`` means "unknown" - for example an old checkpoint written
        before durations were tracked.  The total then starts at zero and
        the report says the historical duration is unknown instead of
        inventing a number.
        """
        if accumulated_s is None:
            self._accumulated = 0.0
        else:
            self._accumulated = max(0.0, float(accumulated_s))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "accumulated_s": self._accumulated,
            "current_session_s": self.current_session_s,
            "total_s": self.total_s,
            "sessions": [record.to_dict() for record in self._sessions],
        }


def merge_durations(*values: float | None) -> float | None:
    """Sum durations, treating ``None`` as "not measured".

    Returns ``None`` when *every* value is unknown, so the caller can show
    "unknown" rather than a misleading zero.
    """
    known = [float(value) for value in values if value is not None]
    if not known:
        return None
    return sum(known)


def duration_summary(
    *,
    accumulated_s: float | None,
    current_session_s: float | None,
) -> dict:
    """The duration block used by the report, the API and the GUI."""
    total = merge_durations(accumulated_s, current_session_s)
    return {
        "accumulated_s": accumulated_s,
        "current_session_s": current_session_s,
        "total_s": total,
        "accumulated_available": accumulated_s is not None,
        "current_available": current_session_s is not None,
        "accumulated_human": format_duration(accumulated_s),
        "current_human": format_duration(current_session_s),
        "total_human": format_duration(total),
    }
