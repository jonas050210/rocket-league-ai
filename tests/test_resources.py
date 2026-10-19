"""The resource guard: how many workers this machine can actually run.

More workers are not automatically faster.  The plan turns the machine's
resources into a concrete budget and explains itself, so the GUI can warn
*before* a run starts and the trainer can refuse an impossible
configuration.
"""

from __future__ import annotations

from rlai.resources import (
    ESTIMATED_WORKER_MEMORY_MB,
    MAX_RECOMMENDED_WORKERS,
    MIN_FREE_MEMORY_MB,
    clamp_workers,
    plan_resources,
)


def test_a_small_machine_gets_a_small_budget():
    plan = plan_resources(
        16,
        cpu_count=2,
        memory_available_mb=2048,
        memory_total_mb=4096,
        disk_free_gb=50.0,
    )
    assert plan.recommended_workers == 1
    assert plan.clamped is True
    assert plan.cpu_count == 2


def test_a_big_machine_gets_the_requested_workers():
    plan = plan_resources(
        4,
        cpu_count=16,
        memory_available_mb=32 * 1024,
        memory_total_mb=64 * 1024,
        disk_free_gb=200.0,
    )
    assert plan.recommended_workers == 4
    assert plan.clamped is False


def test_the_worker_count_is_never_zero():
    plan = plan_resources(
        8,
        cpu_count=1,
        memory_available_mb=10,
        memory_total_mb=1024,
        disk_free_gb=10.0,
    )
    assert plan.recommended_workers >= 1


def test_the_budget_is_never_larger_than_the_hard_limit():
    plan = plan_resources(
        256,
        cpu_count=256,
        memory_available_mb=1024 * 1024,
        memory_total_mb=1024 * 1024,
        disk_free_gb=500.0,
    )
    assert plan.recommended_workers == MAX_RECOMMENDED_WORKERS


def test_oversubscribing_the_cpu_is_reported():
    plan = plan_resources(
        16,
        cpu_count=4,
        memory_available_mb=32 * 1024,
        memory_total_mb=64 * 1024,
        disk_free_gb=100.0,
    )
    assert any("oversubscribe" in warning for warning in plan.warnings)


def test_too_little_memory_is_reported():
    plan = plan_resources(
        8,
        cpu_count=16,
        memory_available_mb=512,
        memory_total_mb=1024,
        disk_free_gb=100.0,
    )
    assert any("swapping" in warning for warning in plan.warnings)


def test_a_machine_without_memory_is_reported():
    plan = plan_resources(
        2,
        cpu_count=8,
        memory_available_mb=MIN_FREE_MEMORY_MB / 2,
        memory_total_mb=1024,
        disk_free_gb=100.0,
    )
    assert any("not safe" in warning for warning in plan.warnings)


def test_a_full_disk_is_reported():
    plan = plan_resources(
        2,
        cpu_count=8,
        memory_available_mb=16 * 1024,
        memory_total_mb=32 * 1024,
        disk_free_gb=0.5,
    )
    assert any("disk" in warning for warning in plan.warnings)


def test_the_clamp_can_be_switched_off():
    workers, plan = clamp_workers(
        12,
        enabled=False,
        cpu_count=2,
        memory_available_mb=1024,
        memory_total_mb=2048,
        disk_free_gb=10.0,
    )
    assert workers == 12
    # the plan still describes the machine, the clamp is simply not applied
    assert plan.recommended_workers == 1


def test_the_clamp_is_applied_by_default():
    workers, plan = clamp_workers(
        12,
        cpu_count=2,
        memory_available_mb=1024,
        memory_total_mb=2048,
        disk_free_gb=10.0,
    )
    assert workers == plan.recommended_workers
    assert workers == 1


def test_the_plan_is_serialisable():
    plan = plan_resources(
        4,
        cpu_count=8,
        memory_available_mb=16 * 1024,
        memory_total_mb=32 * 1024,
        disk_free_gb=100.0,
    )
    data = plan.to_dict()
    assert data["recommended_workers"] == 4
    assert isinstance(data["warnings"], list)
    assert isinstance(data["notes"], list)
    assert data["clamped"] is False


def test_the_memory_estimate_is_documented():
    assert ESTIMATED_WORKER_MEMORY_MB > 0
    assert MIN_FREE_MEMORY_MB > 0


def test_the_requested_worker_count_is_never_below_one():
    plan = plan_resources(
        0,
        cpu_count=4,
        memory_available_mb=8 * 1024,
        memory_total_mb=16 * 1024,
        disk_free_gb=50.0,
    )
    assert plan.requested_workers == 1
    assert plan.recommended_workers == 1
