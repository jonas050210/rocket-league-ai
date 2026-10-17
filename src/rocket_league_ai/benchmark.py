"""Reproducible hardware benchmark: genuine bounded workloads, honest statistics.

Control-plane only: never imports Torch or the simulator. A measurement is a
real, bounded training run executed by the existing training worker; the
per-iteration telemetry it already emits (collection / learning / iteration
seconds, episodes) is aggregated here. Warm-up iterations are excluded,
repeats are summarized with medians, and a recommendation is only made for
configurations that completed stably within resource limits.
"""

import statistics
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import TrainingConfig

MAX_MEASUREMENT_STEPS = 1_000_000
MAX_CANDIDATES = 8
STABILITY_MAX_CV = 0.25
AUTO_DEFAULT_MAX_CV = 0.10
AUTO_DEFAULT_MIN_REPEATS = 3

# 30 ascending stages x 20 seconds = 600 seconds (~10 minutes) full hardware & learning-rate scaling sweep.
# Includes n_proc=20 specifically matched to 20-thread CPUs (e.g. Intel Core i7-12700F) + RTX 4060 Ti 8GB VRAM.
# Every stage satisfies batch_size % (2 * n_proc) == 0 and batch_size % n_minibatches == 0.
HIGH_SCALE_SWEEP_STAGES: list[dict] = [
    # Tier 1: Fast 15s episodes, 512 batch, lr 2e-4, 1 -> 8 envs
    {"n_proc": 1, "batch_size": 512, "episode_timeout_seconds": 15.0, "learning_rate": 0.0002, "n_minibatches": 2},
    {"n_proc": 2, "batch_size": 512, "episode_timeout_seconds": 15.0, "learning_rate": 0.0002, "n_minibatches": 2},
    {"n_proc": 4, "batch_size": 512, "episode_timeout_seconds": 15.0, "learning_rate": 0.0002, "n_minibatches": 2},
    {"n_proc": 8, "batch_size": 512, "episode_timeout_seconds": 15.0, "learning_rate": 0.0002, "n_minibatches": 2},
    # Tier 2: 20s episodes, 1024/1600 batch, lr 2.2e-4, 2 -> 20 envs
    {"n_proc": 2, "batch_size": 1024, "episode_timeout_seconds": 20.0, "learning_rate": 0.00022, "n_minibatches": 4},
    {"n_proc": 4, "batch_size": 1024, "episode_timeout_seconds": 20.0, "learning_rate": 0.00022, "n_minibatches": 4},
    {"n_proc": 8, "batch_size": 1024, "episode_timeout_seconds": 20.0, "learning_rate": 0.00022, "n_minibatches": 4},
    {"n_proc": 16, "batch_size": 1024, "episode_timeout_seconds": 20.0, "learning_rate": 0.00022, "n_minibatches": 4},
    {"n_proc": 20, "batch_size": 1600, "episode_timeout_seconds": 20.0, "learning_rate": 0.00022, "n_minibatches": 4},
    # Tier 3: 30s episodes, 1536/2048/3200 batch, lr 2.5e-4, 4 -> 24 envs
    {"n_proc": 4, "batch_size": 2048, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    {"n_proc": 6, "batch_size": 1536, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    {"n_proc": 8, "batch_size": 2048, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    {"n_proc": 12, "batch_size": 1536, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    {"n_proc": 16, "batch_size": 2048, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    {"n_proc": 20, "batch_size": 3200, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 8},
    {"n_proc": 24, "batch_size": 1536, "episode_timeout_seconds": 30.0, "learning_rate": 0.00025, "n_minibatches": 4},
    # Tier 4: 45s episodes, 3072/4096/6400 batch, lr 2.8e-4, 8 -> 32 envs
    {"n_proc": 8, "batch_size": 4096, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    {"n_proc": 12, "batch_size": 3072, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    {"n_proc": 16, "batch_size": 4096, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    {"n_proc": 20, "batch_size": 6400, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    {"n_proc": 24, "batch_size": 3072, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    {"n_proc": 32, "batch_size": 4096, "episode_timeout_seconds": 45.0, "learning_rate": 0.00028, "n_minibatches": 8},
    # Tier 5: 60s episodes, 6144/6400/8192 batch, lr 3e-4, 12 -> 32 envs
    {"n_proc": 12, "batch_size": 6144, "episode_timeout_seconds": 60.0, "learning_rate": 0.0003, "n_minibatches": 8},
    {"n_proc": 16, "batch_size": 8192, "episode_timeout_seconds": 60.0, "learning_rate": 0.0003, "n_minibatches": 8},
    {"n_proc": 20, "batch_size": 6400, "episode_timeout_seconds": 60.0, "learning_rate": 0.0003, "n_minibatches": 8},
    {"n_proc": 24, "batch_size": 6144, "episode_timeout_seconds": 60.0, "learning_rate": 0.0003, "n_minibatches": 8},
    {"n_proc": 32, "batch_size": 8192, "episode_timeout_seconds": 60.0, "learning_rate": 0.0003, "n_minibatches": 8},
    # Tier 6: Max Hardware 100% Saturation (90s episodes, 12288/12800/16384 batch, lr 3.5e-4, 20 -> 32 envs)
    {"n_proc": 20, "batch_size": 12800, "episode_timeout_seconds": 90.0, "learning_rate": 0.00035, "n_minibatches": 16},
    {"n_proc": 24, "batch_size": 12288, "episode_timeout_seconds": 90.0, "learning_rate": 0.00035, "n_minibatches": 16},
    {"n_proc": 32, "batch_size": 16384, "episode_timeout_seconds": 90.0, "learning_rate": 0.00035, "n_minibatches": 16},
]


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    devices: list[Literal["cpu", "cuda:0"]] = Field(default=["cpu"], min_length=1, max_length=2)
    n_proc_values: list[int] = Field(default=[1], min_length=1, max_length=12)
    batch_size: int = Field(default=512, ge=128, le=16384)
    warmup_iterations: int = Field(default=1, ge=0, le=10)
    measure_iterations: int = Field(default=1900, ge=1, le=1900)
    measurement_seconds: int = Field(default=20, ge=5, le=600)
    min_iterations: int = Field(default=2, ge=1, le=20)
    repeats: int = Field(default=3, ge=1, le=5)
    max_cpu_percent: float = Field(default=100.0, gt=0, le=100)
    max_memory_percent: float = Field(default=95.0, gt=0, le=100)
    max_gpu_memory_percent: float = Field(default=95.0, gt=0, le=100)
    timeout_seconds: int = Field(default=180, ge=30, le=3600)
    auto_default: bool = False
    auto_sweep: bool = False

    @model_validator(mode="after")
    def check_spec(self) -> Self:
        if len(set(self.devices)) != len(self.devices) or len(set(self.n_proc_values)) != len(self.n_proc_values):
            raise ValueError("devices and n_proc_values must not contain duplicates")
        max_n_proc = 32 if self.auto_sweep else 8
        if any(not 1 <= value <= max_n_proc for value in self.n_proc_values):
            raise ValueError(f"n_proc values must be between 1 and {max_n_proc}")
        if not self.auto_sweep and len(self.devices) * len(self.n_proc_values) > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidate configurations per benchmark")
        if not self.auto_sweep:
            for n_proc in self.n_proc_values:
                if self.batch_size % (2 * n_proc):
                    raise ValueError(
                        f"batch_size {self.batch_size} must be divisible by 2 * n_proc ({2 * n_proc}) "
                        f"for {n_proc} environment process(es)"
                    )
        if self.min_iterations > self.effective_measure_iterations:
            raise ValueError(
                f"min_iterations {self.min_iterations} exceeds the "
                f"{self.effective_measure_iterations} measured iterations that fit in the "
                f"{MAX_MEASUREMENT_STEPS:,}-step cap at batch_size {self.batch_size}"
            )
        if self.timeout_seconds <= self.measurement_seconds:
            raise ValueError("timeout_seconds must exceed the measurement window")
        return self

    @classmethod
    def default_auto_sweep(cls, device: Literal["cpu", "cuda:0"] = "cuda:0") -> Self:
        """Build the 10-minute (30 stages x 20s) high-scaling automatic benchmark sweep."""
        return cls(
            devices=[device],
            n_proc_values=[1, 2, 4, 6, 8, 12, 16, 20, 24, 32],
            batch_size=512,
            warmup_iterations=1,
            measure_iterations=1900,
            measurement_seconds=20,
            min_iterations=1,
            repeats=1,
            max_cpu_percent=100.0,
            max_memory_percent=99.0,
            max_gpu_memory_percent=99.0,
            auto_default=True,
            auto_sweep=True,
        )

    @property
    def effective_measure_iterations(self) -> int:
        warmup = self.warmup_iterations
        available = MAX_MEASUREMENT_STEPS // self.batch_size - warmup
        return max(1, min(self.measure_iterations, available))

    def candidates(self) -> list[dict]:
        if self.auto_sweep:
            device = self.devices[0]
            return [
                {
                    "device": device,
                    "n_proc": stage["n_proc"],
                    "batch_size": stage["batch_size"],
                    "episode_timeout_seconds": stage["episode_timeout_seconds"],
                    "learning_rate": stage.get("learning_rate", 0.00025),
                    "n_minibatches": stage.get("n_minibatches", 4),
                    "stage": index + 1,
                }
                for index, stage in enumerate(HIGH_SCALE_SWEEP_STAGES)
            ]
        return [
            {"device": device, "n_proc": n_proc}
            for device in self.devices for n_proc in sorted(self.n_proc_values)
        ]

    def measurement_config(self, base: TrainingConfig, candidate: dict) -> TrainingConfig:
        """A bounded real training config for one candidate; strategy fields stay at the plan's."""
        cand_batch = int(candidate.get("batch_size") or self.batch_size)
        cand_ep_timeout = float(candidate.get("episode_timeout_seconds") or base.episode_timeout_seconds)
        cand_no_touch = min(float(base.no_touch_timeout_seconds), cand_ep_timeout)
        cand_lr = float(candidate.get("learning_rate") or base.learning_rate)
        cand_mb = int(candidate.get("n_minibatches") or base.n_minibatches)
        while cand_mb > 1 and (cand_batch % cand_mb != 0 or cand_batch // cand_mb < 16):
            cand_mb -= 1
        budget = (MAX_MEASUREMENT_STEPS // cand_batch) * cand_batch
        return base.model_copy(update={
            "name": f"Benchmark · {candidate['device']} · {candidate['n_proc']} env · {cand_batch} batch",
            "device": candidate["device"],
            "n_proc": candidate["n_proc"],
            "total_timesteps": budget,
            "timesteps_per_iteration": cand_batch,
            "batch_size": cand_batch,
            "buffer_size": 3 * cand_batch,
            "n_minibatches": cand_mb,
            "learning_rate": cand_lr,
            "pin_memory": candidate["device"].startswith("cuda"),
            "episode_timeout_seconds": cand_ep_timeout,
            "no_touch_timeout_seconds": cand_no_touch,
            "checkpoint_every_iterations": 100,
        })


def aggregate_iterations(iterations: list[dict], warmup: int, batch_size: int) -> dict:
    """Summarize measured (post-warm-up) iterations of one genuine run."""
    measured = iterations[warmup:]
    if not measured:
        raise ValueError("no measured iterations remain after warm-up exclusion")

    def rates(key: str) -> list[float]:
        return [batch_size / item[key] for item in measured if item.get(key)]

    def summary(values: list[float]) -> dict:
        median = statistics.median(values)
        deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"median": median, "min": min(values), "max": max(values),
                "cv": (deviation / median) if median else 0.0}

    baseline = iterations[warmup - 1] if warmup else iterations[0]
    first, last = baseline, iterations[-1]
    episode_span = max((last.get("episodes") or 0) - (first.get("episodes") or 0), 0)
    wall = sum(item.get("iteration_seconds") or 0 for item in measured)
    return {
        "iterations_measured": len(measured),
        "agent_steps_per_second": summary(rates("iteration_seconds")),
        "simulation_steps_per_second": summary(rates("collection_seconds")),
        "ppo_steps_per_second": summary(rates("learning_seconds")),
        "episodes_per_minute": (episode_span / wall * 60) if wall else 0.0,
        "episodes_measured": episode_span,
    }


def combine_repeats(per_repeat: list[dict]) -> dict:
    """Combine per-repeat aggregates honestly."""
    if not per_repeat:
        raise ValueError("no repeat aggregates to combine")

    def combined(key: str) -> dict:
        medians = [repeat[key]["median"] for repeat in per_repeat]
        median = statistics.median(medians)
        between = statistics.pstdev(medians) if len(medians) > 1 else 0.0
        between_cv = (between / median) if median else 0.0
        within_cv = max((repeat[key]["cv"] for repeat in per_repeat), default=0.0)
        return {"median": median, "min": min(medians), "max": max(medians),
                "cv": max(between_cv, within_cv),
                "cv_between_repeats": between_cv, "cv_within_repeats": within_cv,
                "repeat_medians": medians}

    return {
        "repeats": len(per_repeat),
        "iterations_measured": sum(repeat["iterations_measured"] for repeat in per_repeat),
        "agent_steps_per_second": combined("agent_steps_per_second"),
        "simulation_steps_per_second": combined("simulation_steps_per_second"),
        "ppo_steps_per_second": combined("ppo_steps_per_second"),
        "episodes_per_minute": statistics.median([repeat["episodes_per_minute"] for repeat in per_repeat]),
        "episodes_measured": sum(repeat["episodes_measured"] for repeat in per_repeat),
    }


def recommend(results: list[dict], spec: BenchmarkSpec) -> dict:
    """Fastest *stable* candidate, or an explicit refusal to recommend."""
    stable, excluded = [], []
    max_cv = 0.45 if spec.auto_sweep else STABILITY_MAX_CV
    for result in results:
        reasons = []
        if result["status"] != "completed":
            reasons.append(result.get("error") or "measurement did not complete")
        else:
            aggregate = result["aggregate"]
            cv = aggregate["agent_steps_per_second"]["cv"]
            if cv > max_cv:
                reasons.append(
                    f"unstable throughput (CV {cv:.2f} "
                    f"> {max_cv}) across repeats/iterations"
                )
            resources = result.get("resources") or {}
            if resources.get("cpu_percent_peak") is not None and resources["cpu_percent_peak"] > spec.max_cpu_percent:
                reasons.append(f"CPU peak {resources['cpu_percent_peak']:.0f}% exceeds limit {spec.max_cpu_percent:.0f}%")
            if resources.get("memory_percent_peak") is not None and resources["memory_percent_peak"] > spec.max_memory_percent:
                reasons.append(f"memory peak {resources['memory_percent_peak']:.0f}% exceeds limit {spec.max_memory_percent:.0f}%")
            if resources.get("gpu_memory_percent_peak") is not None and resources["gpu_memory_percent_peak"] > spec.max_gpu_memory_percent:
                reasons.append(
                    f"VRAM peak {resources['gpu_memory_percent_peak']:.0f}% exceeds limit "
                    f"{spec.max_gpu_memory_percent:.0f}%"
                )
            if not spec.auto_sweep and result["candidate"]["device"].startswith("cuda") and resources.get("gpu_percent_peak") is None:
                reasons.append("CUDA candidate has no GPU telemetry; utilization and VRAM headroom are unverified")
        (excluded if reasons else stable).append((result, reasons))
    if not stable:
        return {"candidate": None, "auto_default": False,
                "explanation": "No candidate completed stably within the resource limits; no recommendation is made.",
                "excluded": [{"candidate": item[0]["candidate"], "reasons": item[1]} for item in excluded]}
    ranked = sorted(stable, key=lambda item: item[0]["aggregate"]["agent_steps_per_second"]["median"], reverse=True)
    best, _ = ranked[0]
    aggregate = best["aggregate"]
    cv = aggregate["agent_steps_per_second"]["cv"]
    resources = best.get("resources") or {}
    gpu_verified = not best["candidate"]["device"].startswith("cuda") or resources.get("gpu_percent_peak") is not None
    auto_reasons = []
    repeats_verified = spec.repeats >= 2
    if not spec.auto_sweep:
        if not repeats_verified:
            auto_reasons.append("a single repeat cannot establish repeatability; use at least 2")
        if cv > AUTO_DEFAULT_MAX_CV:
            auto_reasons.append(f"CV {cv:.2f} > {AUTO_DEFAULT_MAX_CV} for automatic adoption")
        if spec.repeats < AUTO_DEFAULT_MIN_REPEATS:
            auto_reasons.append(f"{spec.repeats} repeat(s) < {AUTO_DEFAULT_MIN_REPEATS} required for automatic adoption")
        if not gpu_verified:
            auto_reasons.append("GPU telemetry is missing")
    auto_default = (spec.auto_default or spec.auto_sweep) and not auto_reasons
    cand = best["candidate"]
    batch_note = f", batch {cand['batch_size']}" if cand.get("batch_size") else ""
    ep_note = f", episode {cand['episode_timeout_seconds']:.0f}s" if cand.get("episode_timeout_seconds") else ""
    explanation = (
        f"{cand['device']} with {cand['n_proc']} environment process(es){batch_note}{ep_note} was the fastest "
        f"stable configuration: median {aggregate['agent_steps_per_second']['median']:.1f} agent steps/s "
        f"(simulation {aggregate['simulation_steps_per_second']['median']:.1f}/s, PPO "
        f"{aggregate['ppo_steps_per_second']['median']:.1f}/s, CV {cv:.2f}). "
        "This is a throughput recommendation only; it says nothing about learning quality."
    )
    if not repeats_verified and not spec.auto_sweep:
        explanation += " Only one repeat was run, so repeatability across sessions is unverified."
    if auto_default:
        explanation += " Best configuration was automatically adopted as the default for future training runs."
    elif spec.auto_default:
        explanation += " The default plan was not changed: " + "; ".join(auto_reasons) + "."
    else:
        explanation += " Adopting it still requires a reviewed apply_config proposal."
    return {
        "candidate": best["candidate"], "aggregate": aggregate, "explanation": explanation,
        "auto_default": auto_default, "auto_default_blockers": auto_reasons,
        "safety": {"gpu_verified": gpu_verified, "cv": cv, "repeats": spec.repeats,
                   "repeats_verified": repeats_verified,
                   "measurement_seconds": spec.measurement_seconds,
                   "cpu_percent_peak": resources.get("cpu_percent_peak"),
                   "memory_percent_peak": resources.get("memory_percent_peak"),
                   "gpu_memory_percent_peak": resources.get("gpu_memory_percent_peak")},
        "excluded": [{"candidate": item[0]["candidate"], "reasons": item[1]} for item in excluded if item[1]],
    }
