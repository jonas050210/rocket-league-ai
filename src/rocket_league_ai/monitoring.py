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


def recommendations(runs: list[dict], system: dict, *, evaluations: list[dict] | None = None,
                    metric_history: list[dict] | None = None, benchmark: dict | None = None) -> list[dict]:
    """Read-only insights from genuine measurements; no executor ever acts on them."""
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
    result.extend(_extra_insights(runs, evaluations, metric_history, benchmark))
    return result


def _extra_insights(runs, evaluations, metric_history, benchmark) -> list[dict]:
    """Bottleneck, stagnation and next-experiment insights from real records only."""
    result = []
    latest = runs[0] if runs else None
    metrics = (latest or {}).get("metrics") or {}
    collection, learning = metrics.get("collection_seconds"), metrics.get("learning_seconds")
    if collection and learning and collection + learning > 0:
        share = collection / (collection + learning)
        if share > 0.75:
            result.append({"severity": "info", "title": "Simulation collection is the bottleneck",
                           "rationale": f"Collection takes {share:.0%} of measured update time versus PPO learning.",
                           "impact": "A benchmark sweep of parallel environment counts is the evidence-based next step; raw speed still says nothing about learning quality."})
        elif share < 0.25:
            result.append({"severity": "info", "title": "PPO optimization is the bottleneck",
                           "rationale": f"PPO learning takes {1 - share:.0%} of measured update time versus simulation collection.",
                           "impact": "Consider a benchmark comparing batch/minibatch settings before changing strategy fields."})
    rewards = [item.get("reward") for item in (metric_history or []) if item.get("reward") is not None]
    if len(rewards) >= 10:
        half = len(rewards) // 2
        early = sum(rewards[:half]) / half
        late = sum(rewards[half:]) / (len(rewards) - half)
        if abs(late - early) < 0.01:
            result.append({"severity": "info", "title": "Reward has plateaued in recent updates",
                           "rationale": f"Mean transition reward moved from {early:.3f} to {late:.3f} across the last {len(rewards)} measured updates.",
                           "impact": "Do not react to reward alone: evaluate the latest checkpoint against a frozen baseline, then change one experiment variable at a time."})
    completed = [item for item in (evaluations or []) if item.get("status") == "completed" and item.get("result")]
    if latest and latest["status"] in {"completed", "stopped"} and not completed:
        result.append({"severity": "info", "title": "No gameplay evaluation yet",
                       "rationale": "Training reward and entropy do not measure playing strength.",
                       "impact": "Evaluate the newest checkpoint against the random or idle baseline in Models to establish a baseline before claiming improvement."})
    if completed:
        newest = completed[0]["result"]
        ci = newest.get("win_rate_ci95") or [0, 1]
        if ci[1] < 0.5 and newest.get("opponent_id") == "random-v1":
            result.append({"severity": "warning", "title": "Checkpoint does not yet beat random play",
                           "rationale": f"Win rate {newest.get('win_rate', 0):.0%} with 95% CI up to {ci[1]:.0%} over {newest.get('games', 0)} games.",
                           "impact": "Keep training or revisit reward/curriculum in a new, separately approved experiment; compare checkpoints under identical evaluation conditions."})
    session = (benchmark or {}).get("session") or {}
    recommendation = session.get("recommendation") or {}
    if session.get("status") == "completed" and recommendation.get("candidate"):
        result.append({"severity": "info", "title": "Benchmark recommendation available",
                       "rationale": recommendation["explanation"],
                       "impact": "Review it on the Benchmark page; adoption requires a confirmed configuration proposal and is never automatic."})
    return result


def next_experiment(runs: list[dict], evaluations: list[dict] | None, benchmark: dict | None,
                    plan: dict | None) -> dict | None:
    """One concrete, evidence-linked next experiment — or None. Never executed automatically."""
    plan = plan or {}
    completed = [item for item in (evaluations or []) if item.get("status") == "completed" and item.get("result")]
    if not completed:
        return {"kind": "evaluation", "title": "Establish a gameplay baseline first",
                "rationale": "No completed checkpoint evaluation exists, so no training change can be judged.",
                "action": "In Models, evaluate the newest valid checkpoint against random-v1 over at least 100 side-balanced games.",
                "evidence": "evaluation history is empty"}
    newest = completed[0]["result"]
    ci = newest.get("win_rate_ci95") or [0, 1]
    if newest.get("opponent_id") == "random-v1" and ci[1] < 0.5:
        if plan.get("reward_version") == "goal10-touch0.1-transition-v1":
            return {"kind": "reward_shaping", "title": "Try one shaped-reward experiment",
                    "rationale": f"The newest checkpoint does not beat random play (win rate {newest.get('win_rate', 0):.0%}, 95% CI up to {ci[1]:.0%}). The foundation reward is almost purely sparse goal/touch.",
                    "action": "Create a separately approved plan with reward_version goal10-touch0.1-shaped-v2 and only reward_velocity_to_ball_weight 0.3 enabled; train with the same step budget and evaluate identically.",
                    "evidence": f"evaluation {newest.get('candidate_checkpoint_id')} vs random-v1, {newest.get('games', 0)} games"}
        return {"kind": "training_budget", "title": "Extend training before changing strategy",
                "rationale": f"Even with shaping, the checkpoint does not beat random play (win rate {newest.get('win_rate', 0):.0%}).",
                "action": "Approve a longer run from the latest checkpoint (recovery) and re-evaluate under identical conditions.",
                "evidence": f"evaluation {newest.get('candidate_checkpoint_id')} vs random-v1, {newest.get('games', 0)} games"}
    session = (benchmark or {}).get("session") or {}
    candidate = (session.get("recommendation") or {}).get("candidate")
    if candidate and (candidate["device"] != plan.get("device") or candidate["n_proc"] != plan.get("n_proc")):
        return {"kind": "benchmark_adoption", "title": "Review the benchmark recommendation",
                "rationale": session["recommendation"]["explanation"],
                "action": "On the Benchmark page, review the recommendation as a configuration proposal. Throughput only — judge learning quality by evaluation afterwards.",
                "evidence": f"benchmark session {session['id']}"}
    return {"kind": "checkpoint_opponent", "title": "Measure against a frozen checkpoint",
            "rationale": "The checkpoint beats the scripted baseline; the next honest question is whether newer checkpoints beat older ones.",
            "action": "Evaluate the newest checkpoint against an older checkpoint (checkpoint opponent in Models) with the same seed and game count.",
            "evidence": f"evaluation {newest.get('candidate_checkpoint_id')} vs {newest.get('opponent_id')}"}
