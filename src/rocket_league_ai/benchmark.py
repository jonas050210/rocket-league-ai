"""Reproducible hardware benchmark: genuine bounded workloads, honest statistics.

Control-plane only: never imports Torch or the simulator. A measurement is a
real, bounded training run executed by the existing training worker; the
per-iteration telemetry it already emits (collection / learning / iteration
seconds, episodes) is aggregated here. Warm-up iterations are excluded,
repeats are summarized with medians, and a recommendation is only made for
configurations that completed every repeat stably within resource limits.
Nothing here applies a configuration: adoption still requires the confirmed
``apply_config`` proposal flow.
"""

import statistics
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import TrainingConfig

# One candidate measurement run may never exceed this many agent steps, so a
# benchmark click can never become an unbounded training session.
MAX_MEASUREMENT_STEPS = 16384
MAX_CANDIDATES = 8
STABILITY_MAX_CV = 0.25


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    devices: list[Literal["cpu", "cuda:0"]] = Field(default=["cpu"], min_length=1, max_length=2)
    n_proc_values: list[int] = Field(default=[1], min_length=1, max_length=4)
    batch_size: int = Field(default=512, ge=128, le=4096)
    warmup_iterations: int = Field(default=1, ge=0, le=10)
    measure_iterations: int = Field(default=3, ge=1, le=20)
    repeats: int = Field(default=2, ge=1, le=5)
    max_cpu_percent: float = Field(default=100.0, gt=0, le=100)
    max_memory_percent: float = Field(default=95.0, gt=0, le=100)
    timeout_seconds: int = Field(default=600, ge=30, le=3600)

    @model_validator(mode="after")
    def check_spec(self) -> Self:
        if len(set(self.devices)) != len(self.devices) or len(set(self.n_proc_values)) != len(self.n_proc_values):
            raise ValueError("devices and n_proc_values must not contain duplicates")
        if any(not 1 <= value <= 8 for value in self.n_proc_values):
            raise ValueError("n_proc values must be between 1 and 8")
        if len(self.devices) * len(self.n_proc_values) > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidate configurations per benchmark")
        steps = (self.warmup_iterations + self.measure_iterations) * self.batch_size
        if steps > MAX_MEASUREMENT_STEPS:
            raise ValueError(
                f"one measurement run would need {steps:,} agent steps; the benchmark cap is "
                f"{MAX_MEASUREMENT_STEPS:,}. Reduce batch_size or iteration counts."
            )
        return self

    def candidates(self) -> list[dict]:
        return [
            {"device": device, "n_proc": n_proc}
            for device in self.devices for n_proc in sorted(self.n_proc_values)
        ]

    def measurement_config(self, base: TrainingConfig, candidate: dict) -> TrainingConfig:
        """A bounded real training config for one candidate; strategy fields stay at the plan's."""
        iterations = self.warmup_iterations + self.measure_iterations
        return base.model_copy(update={
            "name": f"Benchmark · {candidate['device']} · {candidate['n_proc']} env",
            "device": candidate["device"], "n_proc": candidate["n_proc"],
            "total_timesteps": iterations * self.batch_size,
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

    first, last = iterations[0], iterations[-1]
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


def recommend(results: list[dict], spec: BenchmarkSpec) -> dict:
    """Fastest *stable* candidate, or an explicit refusal to recommend.

    Stability: every repeat completed, throughput coefficient of variation at
    or below the threshold, and sampled resource peaks within the spec limits.
    Maximum utilization is explicitly not the goal; unstable or
    resource-exceeding configurations are excluded with reasons.
    """
    stable, excluded = [], []
    for result in results:
        reasons = []
        if result["status"] != "completed":
            reasons.append(result.get("error") or "measurement did not complete")
        else:
            aggregate = result["aggregate"]
            if aggregate["agent_steps_per_second"]["cv"] > STABILITY_MAX_CV:
                reasons.append(
                    f"unstable throughput (CV {aggregate['agent_steps_per_second']['cv']:.2f} "
                    f"> {STABILITY_MAX_CV}) across repeats/iterations"
                )
            resources = result.get("resources") or {}
            if resources.get("cpu_percent_peak") is not None and resources["cpu_percent_peak"] > spec.max_cpu_percent:
                reasons.append(f"CPU peak {resources['cpu_percent_peak']:.0f}% exceeds limit {spec.max_cpu_percent:.0f}%")
            if resources.get("memory_percent_peak") is not None and resources["memory_percent_peak"] > spec.max_memory_percent:
                reasons.append(f"memory peak {resources['memory_percent_peak']:.0f}% exceeds limit {spec.max_memory_percent:.0f}%")
        (excluded if reasons else stable).append((result, reasons))
    if not stable:
        return {"candidate": None,
                "explanation": "No candidate completed stably within the resource limits; no recommendation is made.",
                "excluded": [{"candidate": item[0]["candidate"], "reasons": item[1]} for item in excluded]}
    ranked = sorted(stable, key=lambda item: item[0]["aggregate"]["agent_steps_per_second"]["median"], reverse=True)
    best, _ = ranked[0]
    aggregate = best["aggregate"]
    explanation = (
        f"{best['candidate']['device']} with {best['candidate']['n_proc']} environment process(es) was the fastest "
        f"stable configuration: median {aggregate['agent_steps_per_second']['median']:.1f} agent steps/s "
        f"(simulation {aggregate['simulation_steps_per_second']['median']:.1f}/s, PPO "
        f"{aggregate['ppo_steps_per_second']['median']:.1f}/s, CV {aggregate['agent_steps_per_second']['cv']:.2f}). "
        "This is a throughput recommendation only; it says nothing about learning quality. "
        "Adopting it still requires a reviewed apply_config proposal."
    )
    return {"candidate": best["candidate"], "aggregate": aggregate, "explanation": explanation,
            "excluded": [{"candidate": item[0]["candidate"], "reasons": item[1]} for item in excluded if item[1]]}
