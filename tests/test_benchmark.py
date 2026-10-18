"""Benchmark honesty: reported throughput must match actual execution.

The previous implementation ran the workers one after another inside the
calling process but divided the summed step count by the *longest single*
loop, so ``env_steps_per_s`` claimed roughly ``workers`` times the throughput
the machine really delivered.  These tests pin the wall-clock relationship.
"""

from __future__ import annotations

import time

import pytest

from rlai.hardware import benchmark, benchmark_inference


def test_reported_rate_is_reproducible_from_reported_time():
    result = benchmark("soccer-1v1", seconds=0.6, workers=1, include_inference=False)
    total_steps = sum(worker["steps"] for worker in result["per_worker"])
    assert result["total_steps"] == total_steps
    assert result["env_steps_per_s"] == pytest.approx(total_steps / result["seconds"], rel=1e-9)
    assert result["seconds"] >= 0.6 * 0.95
    assert result["parallel"] is False  # a single worker never spawns
    assert result["sequential_seconds"] == pytest.approx(result["seconds"], rel=0.05)


def test_parallel_workers_really_overlap_in_time():
    started = time.perf_counter()
    result = benchmark("soccer-1v1", seconds=0.6, workers=2, include_inference=False)
    call_seconds = time.perf_counter() - started

    assert result["parallel"] is True, result.get("parallel_error")
    assert result["workers"] == 2
    # the measured phase is the wall clock of the collection, not the sum
    assert result["seconds"] >= 0.6 * 0.95
    assert result["seconds"] < result["sequential_seconds"] * 0.8
    # and the whole call (including process start-up) cannot be shorter than
    # the phase that was measured inside it
    assert result["seconds"] <= call_seconds + 1e-6
    assert result["speedup"] == pytest.approx(
        result["env_steps_per_s"] / result["single_worker_steps_per_s"], rel=1e-9
    )


def test_sequential_mode_reports_sequential_time():
    result = benchmark(
        "soccer-1v1", seconds=0.5, workers=2, parallel=False, include_inference=False
    )
    assert result["parallel"] is False
    assert result["workers"] == 2
    assert result["seconds"] == pytest.approx(result["sequential_seconds"], rel=0.15)
    # two sequential loops of 0.5 s cannot honestly finish in 0.5 s
    assert result["seconds"] >= 0.9


def test_inference_cost_is_measured_separately():
    result = benchmark_inference(hidden=(32,), batch=2, seconds=0.3, device="cpu")
    assert result["decisions"] > 0
    assert result["agent_decisions"] == result["decisions"] * 2
    assert result["decisions_per_s"] > 0
    assert result["milliseconds_per_decision"] > 0
    assert result["parameters"] > 0
    assert result["seconds"] >= 0.3 * 0.9
    # consistency: rate * time == work
    assert result["decisions_per_s"] == pytest.approx(
        result["decisions"] / result["seconds"], rel=1e-9
    )


def test_benchmark_includes_inference_by_default():
    result = benchmark("soccer-1v1", seconds=0.5, workers=1)
    assert "inference" in result
    assert result["inference"]["decisions"] > 0
    assert result["total_seconds"] >= result["seconds"]


def test_worker_count_is_bounded():
    result = benchmark("soccer-1v1", seconds=0.5, workers=64, include_inference=False)
    assert result["workers"] <= 8
