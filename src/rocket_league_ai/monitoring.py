"""Best-effort measured host telemetry. Unavailable GPU readings are never fabricated."""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil


class SystemMonitor:
    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sample = {"cpu_percent": None, "gpu_percent": None, "gpu_reason": "Not sampled yet"}
        self.thread = threading.Thread(target=self._run, daemon=True, name="system-monitor")
        self.thread.start()

    def _run(self):
        psutil.cpu_percent()
        executable = shutil.which("nvidia-smi")
        if not executable and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
            executable = "/usr/lib/wsl/lib/nvidia-smi"
        while not self.stop_event.is_set():
            memory = psutil.virtual_memory()
            sample = {
                "cpu_percent": psutil.cpu_percent(), "cpu_count": psutil.cpu_count(),
                "memory_used_bytes": memory.used, "memory_total_bytes": memory.total,
                "memory_percent": memory.percent, "gpu_percent": None, "gpu_memory_used_mib": None,
                "gpu_memory_total_mib": None, "gpu_name": None,
                "gpu_reason": "nvidia-smi unavailable; GPU utilization cannot be measured on this host",
                "sampled_at": time.time(),
            }
            if executable:
                try:
                    result = subprocess.run(
                        [executable, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=1,
                        check=True,
                    )
                    name, utilization, used, total = result.stdout.splitlines()[0].split(",")
                    sample.update(gpu_name=name.strip(), gpu_percent=float(utilization),
                                  gpu_memory_used_mib=float(used), gpu_memory_total_mib=float(total),
                                  gpu_reason=None)
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    sample["gpu_reason"] = "GPU query unsupported or timed out (some WSL NVML queries are unavailable)"
            with self.lock:
                self.sample = sample
            self.stop_event.wait(2)

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.sample, "backend_uptime_seconds": time.monotonic() - self.started}

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)


def recommendations(runs: list[dict], system: dict) -> list[dict]:
    """Read-only recommendations, intentionally no strategy-change or promotion executor."""
    result = []
    if not system.get("gpu_name"):
        result.append({"severity": "info", "title": "Verify CUDA on your training host",
                       "rationale": system.get("gpu_reason") or "No measured accelerator available",
                       "impact": "Run doctor --probe before CUDA training. CPU correctness checks are not a throughput benchmark."})
    if runs and runs[0]["status"] in {"failed", "interrupted"}:
        result.append({"severity": "warning", "title": "Review recovery before restarting",
                       "rationale": runs[0].get("error") or "The previous run did not finish cleanly",
                       "impact": "Recovery requires confirmation and starts new environments from a verified checkpoint."})
    if (system.get("memory_percent") or 0) > 90:
        result.append({"severity": "warning", "title": "Host memory pressure",
                       "rationale": "Measured memory utilization exceeds 90%",
                       "impact": "Consider a smaller process/buffer budget for a new experiment. No changes applied."})
    return result
