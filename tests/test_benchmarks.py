"""The lightweight benchmarks.

They are the measurements the project uses to decide whether an
optimisation actually helps, so they have to be *reproducible* and honest
about what they measure.  The tests run them with the smallest possible
workload - they check the shape of the report and the invariants, not a
performance number.
"""

from __future__ import annotations

import pytest

from rlai.benchmarks import (
    benchmark_difficulty,
    benchmark_evaluation_reuse,
    benchmark_threads,
)


def test_the_difficulty_benchmark_measures_every_level():
    report = benchmark_difficulty(levels=("rock", "pro"), steps=180)
    assert report["benchmark"] == "difficulty"
    assert {row["level"] for row in report["results"]} == {"rock", "pro"}
    for row in report["results"]:
        assert row["touches"] >= 0
        assert row["seconds"] >= 0.0
    assert report["strongest"] in {"rock", "pro"}
    assert report["weakest"] in {"rock", "pro"}
    assert report["spread"] >= 0


def test_the_difficulty_benchmark_does_not_claim_a_measured_win_rate():
    report = benchmark_difficulty(levels=("rock",), steps=90)
    assert "not the target win rate" in report["note"]
    assert "ball touches" in report["basis"]


def test_the_evaluation_reuse_benchmark_reports_both_times():
    report = benchmark_evaluation_reuse(games=1, match_seconds=2.0)
    assert report["benchmark"] == "evaluation_arena_reuse"
    assert report["reused_seconds"] >= 0.0
    assert report["fresh_arena_construction_seconds"] >= 0.0
    assert report["games_played"] == 1
    assert "arena construction" in report["note"]


def test_the_thread_benchmark_reports_the_decision_cost():
    report = benchmark_threads(threads=1)
    assert report["benchmark"] == "torch_threads"
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["threads"] == 1
    assert row["decisions_per_s"] > 0.0
    assert row["milliseconds_per_decision"] > 0.0


def test_the_thread_benchmark_restores_the_thread_count():
    import torch

    before = torch.get_num_threads()
    benchmark_threads(threads=1)
    assert torch.get_num_threads() == before


@pytest.mark.parametrize(
    "levels",
    [
        ("rock", "tiny", "medium", "pro"),
    ],
)
def test_every_difficulty_level_can_be_measured(levels):
    report = benchmark_difficulty(levels=levels, steps=90)
    assert len(report["results"]) == len(levels)
