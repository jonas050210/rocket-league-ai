"""Reproducible hardware benchmark: genuine bounded workloads, honest statistics.

Control-plane only: never imports Torch or the simulator. A measurement is a
real, bounded training run executed by the existing training worker; the
per-iteration telemetry it already emits (collection / learning / iteration
seconds, episodes) is aggregated here. Warm-up iterations are excluded,
repeats are summarized with medians, and a recommendation is only made for
configurations that completed every repeat stably within resource limits.
Adoption is restricted to execution fields (device and n_proc): it needs the
spec's ``auto_default`` flag and the strict stability gates, records the
previous plan, and is reversible from the Benchmark page.
"""

import statistics
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import TrainingConfig

# One candidate measurement run may never exceed this many agent steps, so a
# benchmark click can never become an unbounded training session. The run is
# also stopped by wall-clock measurement time (default 20 s), so this is a
# hard safety ceiling rather than the normal stopping condition.
MAX_MEASUREMENT_STEPS = 1_000_000
MAX_CANDIDATES = 8
STABILITY_MAX_CV = 0.25
# A configuration is only safe to adopt as the default when its throughput is
# tightly repeatable. This is deliberately stricter than "stable enough to rank".
AUTO_DEFAULT_MAX_CV = 0.10
AUTO_DEFAULT_MIN_REPEATS = 3


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    devices: list[Literal["cpu", "cuda:0"]] = Field(default=["cpu"], min_length=1, max_length=2)
    n_proc_values: list[int] = Field(default=[1], min_length=1, max_length=4)
    batch_size: int = Field(default=512, ge=128, le=4096)
    warmup_iterations: int = Field(default=1, ge=0, le=10)
    # A safety ceiling, not the normal stop: the 20 s wall-clock window ends
    # the measurement on every realistic host. 1,900 iterations at 512 steps
    # is 973k agent steps, just under the 1M per-run cap, and would only be
    # reached faster than ~95 updates/s.
    measure_iterations: int = Field(default=1900, ge=1, le=1900)
    # Wall-clock measurement window, excluding warm-up and process startup.
    measurement_seconds: int = Field(default=20, ge=5, le=600)
    # Do not stop on time before this many measured iterations exist (slow CPU).
    min_iterations: int = Field(default=2, ge=1, le=20)
    repeats: int = Field(default=3, ge=1, le=5)
    max_cpu_percent: float = Field(default=100.0, gt=0, le=100)
    max_memory_percent: float = Field(default=95.0, gt=0, le=100)
    max_gpu_memory_percent: float = Field(default=95.0, gt=0, le=100)
    timeout_seconds: int = Field(default=180, ge=30, le=3600)
    auto_default: bool = False

    @model_validator(mode="after")
    def check_spec(self) -> Self:
        if len(set(self.devices)) != len(self.devices) or len(set(self.n_proc_values)) != len(self.n_proc_values):
            raise ValueError("devices and n_proc_values must not contain duplicates")
        if any(not 1 <= value <= 8 for value in self.n_proc_values):
            raise ValueError("n_proc values must be between 1 and 8")
        if len(self.devices) * len(self.n_proc_values) > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidate configurations per benchmark")
        # TrainingConfig requires the 1v1 collection wave to divide evenly.
        for n_proc in self.n_proc_values:
            if self.batch_size % (2 * n_proc):
                raise ValueError(
                    f"batch_size {self.batch_size} must be divisible by 2 * n_proc ({2 * n_proc}) "
                    f"for {n_proc} environment process(es)"
                )
        steps = (self.warmup_iterations + self.measure_iterations) * self.batch_size
        if steps > MAX_MEASUREMENT_STEPS:
            raise ValueError(
                f"one measurement run would need {steps:,} agent steps; the benchmark cap is "
                f"{MAX_MEASUREMENT_STEPS:,}. Reduce batch_size or iteration counts."
            )
        if self.min_iterations > self.measure_iterations:
            raise ValueError("min_iterations must not exceed measure_iterations")
        if self.timeout_seconds <= self.measurement_seconds:
            raise ValueError("timeout_seconds must exceed the measurement window")
        return self

    def candidates(self) -> list[dict]:
        return [
            {"device": device, "n_proc": n_proc}
            for device in self.devices for n_proc in sorted(self.n_proc_values)
        ]

    def measurement_config(self, base: TrainingConfig, candidate: dict) -> TrainingConfig:
        """A bounded real training config for one candidate; strategy fields stay at the plan's."""
        # The step budget must not expire before the wall-clock window does, or
        # a fast GPU would finish a "20 s" measurement after a few seconds and
        # under-sample the window. Time is the normal stop; the same value also
        # keeps the run under the hard step cap if a stop is ever missed.
        budget = (MAX_MEASUREMENT_STEPS // self.batch_size) * self.batch_size
        return base.model_copy(update={
            "name": f"Benchmark · {candidate['device']} · {candidate['n_proc']} env",
            "device": candidate["device"], "n_proc": candidate["n_proc"],
            "total_timesteps": budget,
            "timesteps_per_iteration": self.batch_size, "batch_size": self.batch_size,
            "buffer_size": 3 * self.batch_size,
            # Benchmark runs exist to measure, not to accumulate keepable models.
            "checkpoint_every_iterations": 100,
        })


def aggregate_iterations(iterations: list[dict], warmup: int, batch_size: int) -> dict:
    """Summarize measured (post-warm-up) iterations of one genuine run.

    Each iteration dict carries the worker's real telemetry: ``collection_seconds``
    (simulation + inference collection), ``learning_seconds`` (PPO optimization),
    ``iteration_seconds`` (whole update) and cumulative ``episodes``.
    """
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

    # Episode counters are cumulative. The correct baseline is the last
    # warm-up iteration (or zero when there is no warm-up), never iteration 0:
    # with two warm-up iterations, using iteration 0 would count the second
    # warm-up's episodes without counting its time.
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
    """Combine per-repeat aggregates honestly.

    Each repeat was already warm-up-excluded and counted within its own run.
    The combined median is the median of repeat medians, and the reported CV
    describes variability *between repeats* — the number a user actually
    experiences when rerunning a configuration.
    """
    if not per_repeat:
        raise ValueError("no repeat aggregates to combine")

    def combined(key: str) -> dict:
        medians = [repeat[key]["median"] for repeat in per_repeat]
        median = statistics.median(medians)
        between = statistics.pstdev(medians) if len(medians) > 1 else 0.0
        between_cv = (between / median) if median else 0.0
        # A single repeat has zero between-repeat variance; that must not be
        # mistaken for stability. Keep the worst within-run variation too.
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
    """Fastest *stable* candidate, or an explicit refusal to recommend.

    Stability: every repeat completed, throughput coefficient of variation at
    or below the threshold, and sampled resource peaks within the spec limits.
    Maximum utilization is explicitly not the goal; unstable or
    resource-exceeding configurations are excluded with reasons.

    ``auto_default`` is stricter and only covers execution fields (device and
    environment-process count). A CUDA candidate without GPU telemetry is never
    recommended, because its VRAM headroom is unverified.
    """
    stable, excluded = [], []
    for result in results:
        reasons = []
        if result["status"] != "completed":
            reasons.append(result.get("error") or "measurement did not complete")
        else:
            aggregate = result["aggregate"]
            cv = aggregate["agent_steps_per_second"]["cv"]
            if cv > STABILITY_MAX_CV:
                reasons.append(
                    f"unstable throughput (CV {cv:.2f} "
                    f"> {STABILITY_MAX_CV}) across repeats/iterations"
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
            if result["candidate"]["device"].startswith("cuda") and resources.get("gpu_percent_peak") is None:
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
    if not repeats_verified:
        auto_reasons.append("a single repeat cannot establish repeatability; use at least 2")
    if cv > AUTO_DEFAULT_MAX_CV:
        auto_reasons.append(f"CV {cv:.2f} > {AUTO_DEFAULT_MAX_CV} for automatic adoption")
    if spec.repeats < AUTO_DEFAULT_MIN_REPEATS:
        auto_reasons.append(f"{spec.repeats} repeat(s) < {AUTO_DEFAULT_MIN_REPEATS} required for automatic adoption")
    if not gpu_verified:
        auto_reasons.append("GPU telemetry is missing")
    auto_default = spec.auto_default and not auto_reasons
    explanation = (
        f"{best['candidate']['device']} with {best['candidate']['n_proc']} environment process(es) was the fastest "
        f"stable configuration: median {aggregate['agent_steps_per_second']['median']:.1f} agent steps/s "
        f"(simulation {aggregate['simulation_steps_per_second']['median']:.1f}/s, PPO "
        f"{aggregate['ppo_steps_per_second']['median']:.1f}/s, CV {cv:.2f}). "
        "This is a throughput recommendation only; it says nothing about learning quality."
    )
    if not repeats_verified:
        explanation += " Only one repeat was run, so repeatability across sessions is unverified."
    if auto_default:
        explanation += " Device and environment-process count were adopted as the default for future runs."
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
