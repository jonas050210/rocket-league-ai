"""Curriculum: promotion only on measured results, per-mode stages."""

from __future__ import annotations

import pytest

from rlai.curriculum import (
    DEFAULT_STAGES,
    HEATSEEKER_STAGES,
    TEAM_STAGES,
    Curriculum,
    Stage,
    stages_for_mode,
)


def _episodes(wins: int, losses: int = 0) -> list[dict]:
    return [{"result": "win"}] * wins + [{"result": "loss"}] * losses


def _stage(index=0, min_iterations=1, win_rate=0.5, min_episodes=4, window=8):
    return Stage(
        name=f"stage-{index}",
        opponent="chase",
        min_iterations=min_iterations,
        win_rate=win_rate,
        min_episodes=min_episodes,
        window=window,
    )


def test_stages_are_mode_specific():
    assert stages_for_mode("soccer-1v1") == DEFAULT_STAGES
    assert stages_for_mode("soccer-2v2") == TEAM_STAGES
    assert stages_for_mode("soccer-3v3") == TEAM_STAGES
    assert stages_for_mode("heatseeker-2v2") == HEATSEEKER_STAGES
    assert all(stage.opponent in {"idle", "chase", "mixed", "self"} for stage in DEFAULT_STAGES)


def test_starts_at_first_stage_and_reports_metrics():
    curriculum = Curriculum(DEFAULT_STAGES)
    assert curriculum.stage is DEFAULT_STAGES[0]
    metrics = curriculum.metrics()
    assert metrics["curriculum/stage"] == 0.0
    assert metrics["curriculum/window_episodes"] == 0.0


def test_no_promotion_before_min_iterations():
    curriculum = Curriculum((_stage(min_iterations=3), _stage(1)))
    curriculum.observe(_episodes(20))
    assert curriculum.index == 0
    assert curriculum.observe(_episodes(20)) is None
    assert curriculum.observe(_episodes(20)) == "stage-1"
    assert curriculum.index == 1


def test_no_promotion_below_win_rate():
    curriculum = Curriculum((_stage(win_rate=0.9, min_episodes=4), _stage(1)))
    curriculum.observe(_episodes(3, 1))
    assert curriculum.index == 0
    curriculum.observe(_episodes(1, 3))
    assert curriculum.index == 0


def test_promotion_resets_window():
    curriculum = Curriculum((_stage(win_rate=0.5, min_episodes=4), _stage(1)))
    assert curriculum.observe(_episodes(4)) == "stage-1"
    assert curriculum.metrics()["curriculum/window_episodes"] == 0.0
    assert curriculum.stage_iterations == 0


def test_last_stage_never_promotes():
    curriculum = Curriculum((_stage(0), _stage(1)))
    curriculum.index = 1
    assert curriculum.finished
    assert curriculum.observe(_episodes(50)) is None


def test_window_is_bounded():
    curriculum = Curriculum((_stage(min_iterations=100, window=5), _stage(1)))
    curriculum.observe(_episodes(3))
    curriculum.observe(_episodes(3))
    assert curriculum.metrics()["curriculum/window_episodes"] == 5.0


def test_counters_can_be_restored_on_resume():
    curriculum = Curriculum(DEFAULT_STAGES, start_index=1, stage_iterations=4, total_iterations=19)
    assert curriculum.index == 1
    assert curriculum.stage_iterations == 4
    assert curriculum.total_iterations == 19
    assert curriculum.describe()["iterations"] == 4


def test_resume_from_index_clamps():
    curriculum = Curriculum(DEFAULT_STAGES, start_index=99)
    assert curriculum.index == len(DEFAULT_STAGES) - 1
    assert Curriculum(DEFAULT_STAGES, start_index=-4).index == 0
    with pytest.raises(ValueError):
        Curriculum(())
    assert curriculum.describe()["finished"] is True
