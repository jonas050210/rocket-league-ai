"""Versioned JSON-line control protocol; stdout library chatter is kept separate."""

import json
import math
import os
import queue
import signal
import sys
import threading
import time

PREFIX = "RLAI_EVENT "


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("PPO produced non-finite metrics; inspect rewards and learning rate before recovery.")
    if hasattr(value, "item"):
        return finite(value.item())
    return value


class WorkerChannel:
    def __init__(self):
        self.lock = threading.Lock()
        self.commands = queue.Queue()
        self.done = threading.Event()
        self.request = None
        self.disconnected = False
        self.last_progress = time.monotonic()
        self.receiver = threading.Thread(target=self._receive, daemon=True, name="worker-controls")
        self.heartbeat = threading.Thread(target=self._heartbeat, daemon=True, name="worker-heartbeat")
        self.receiver.start()
        self.heartbeat.start()
        signal.signal(signal.SIGTERM, lambda *_: self.commands.put("stop"))
        signal.signal(signal.SIGINT, lambda *_: self.commands.put("stop"))

    def emit(self, kind: str, **data):
        message = {"version": 1, "kind": kind, **finite(data)}
        with self.lock:
            if self.disconnected:
                return
            try:
                print(PREFIX + json.dumps(message, allow_nan=False, separators=(",", ":")), flush=True)
            except BrokenPipeError:
                # A killed backend is not a training failure: finish the current update,
                # commit once, and release simulator descendants/leases on stdin EOF.
                self.disconnected = True
                self.commands.put("stop")
                descriptor = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(descriptor, sys.stdout.fileno())
                    os.dup2(descriptor, sys.stderr.fileno())
                finally:
                    os.close(descriptor)

    def _receive(self):
        for line in sys.stdin:
            try:
                command = json.loads(line)["command"]
                if command in {"pause", "resume", "stop"}:
                    self.commands.put(command)
            except (ValueError, KeyError, TypeError):
                self.emit("log", level="warning", message="Invalid worker control ignored")
        # Losing the backend is not permission to continue an unmonitored orphan indefinitely.
        self.commands.put("stop")

    def _heartbeat(self):
        while not self.done.wait(1):
            self.emit("heartbeat", progress_age_seconds=time.monotonic() - self.last_progress)

    def poll(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return self.request
            if self.request == "stop":
                continue
            if command == "stop":
                self.request = "stop"
            elif command == "pause":
                self.request = "pause"
            elif command == "resume" and self.request == "pause":
                self.request = None
