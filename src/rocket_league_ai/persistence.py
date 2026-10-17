"""Atomic files and advisory process leases, all on the Linux filesystem."""

import fcntl
import json
import os
import tempfile
from pathlib import Path

from .domain import AppError


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Lease:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise AppError(
                f"{path.name} is held by another process. Reconnect to the existing backend "
                "or wait for its worker to finish; do not remove a live lock file."
            ) from None

    def close(self) -> None:
        if not self.stream.closed:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def lease_available(path: Path) -> bool:
    try:
        with Lease(path):
            return True
    except AppError:
        return False
