"""Mastery curriculum: gates, confirmations, retention and diagnostics."""

from __future__ import annotations

import pytest

from rlai.config import TrainingConfig
from rlai.curriculum import (
    GATE_APPROACH,
    GATE_PROGRESS,
    MASTERY_STAGES,
    RETENTION_TOLERANCE,
    Curriculum,
    Gate,
    Stage,
    stages_for_mode,
)


def measurement(rate: float, low: float | None = None, trials: int = 24) -> dict:
    if low is None:
        low = max(0.0, rate - 0.15)
    return {"rate": rate, "ci95": [low, min(1.0, rate + 0.15)], "trials": trials}


def episodes(count: int) -> list[dict]:
    return [
        {"result": "draw", "goals_for": 0, "goals_against": 0, "touches": 2}
        for _ in range(count)
    ]


def ready_curriculum(**kwargs) -> Curriculum:
    """A curriculum that has served its stage long enough to be judged."""
    curriculum = Curriculum(MASTERY_STAGES, stage_iterations=100, **kwargs)
    curriculum.observe(episodes(MASTERY_STAGES[0].min_episodes + 1))
    curriculum.last_event = None
    return curriculum


# ---------------------------------------------------------------------------
# gate plumbing
# ---------------------------------------------------------------------------
def test_gate_validates_its_bounds():
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        Gate("approach", minimum=1.4)
    with pytest.raises(ValueError, match="cannot exceed the minimum"):
        Gate("approach", minimum=0.5, ci_low_minimum=0.9)
    with pytest.raises(ValueError, match="needs a skill name"):
        Gate("  ", minimum=0.5)


def test_unmeasured_gate_blocks_with_an_exact_reason():
    curriculum = Curriculum(MASTERY_STAGES)
    reason = curriculum.blocked_reason()
    assert "approach/rate: not measured yet" in reason
    assert curriculum.observe(episodes(40)) is None
    assert curriculum.index == 0


def test_one_good_evaluation_is_not_mastery():
    curriculum = ready_curriculum()
    assert curriculum.record_gate({"approach": measurement(1.0)}) is None
    assert curriculum.gates.passes_in_a_row == 1
    assert "gate_confirmations" in curriculum.blocked_reason()
    assert curriculum.record_gate({"approach": measurement(1.0)}) == MASTERY_STAGES[1].name
    assert curriculum.last_event == "promotion"
    assert curriculum.index == 1


def test_a_wide_confidence_interval_blocks_promotion():
    """A lucky 4-trial probe must not promote a stage."""
    curriculum = ready_curriculum()
    lucky = measurement(0.95, low=0.10, trials=4)
    curriculum.record_gate({"approach": lucky})
    assert curriculum.index == 0
    assert "95% CI lower bound" in curriculum.blocked_reason()
    assert "measure more games" in curriculum.blocked_reason()


def test_a_failing_measurement_resets_the_confirmation_counter():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.gates.passes_in_a_row == 1
    curriculum.record_gate({"approach": measurement(0.2)})
    assert curriculum.gates.passes_in_a_row == 0
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.gates.passes_in_a_row == 1
    assert curriculum.index == 0


def test_stage_two_needs_both_its_gates():
    curriculum = Curriculum(MASTERY_STAGES, start_index=1, stage_iterations=100)
    curriculum.observe(episodes(60))
    curriculum.record_gate({"approach": measurement(1.0), "progress": measurement(0.2)})
    assert curriculum.index == 1
    assert "progress/rate" in curriculum.blocked_reason()
    curriculum.record_gate({"approach": measurement(1.0), "progress": measurement(0.9)})
    assert curriculum.index == 1  # first confirmation only
    assert curriculum.record_gate(
        {"approach": measurement(1.0), "progress": measurement(0.9)}
    ) == MASTERY_STAGES[2].name


# ---------------------------------------------------------------------------
# retention: a new skill must not destroy an old one
# ---------------------------------------------------------------------------
def test_mastered_skills_are_baselined_and_checked_for_regression():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.record_gate({"approach": measurement(1.0)}) == "control"
    assert curriculum.gates.baselines["approach/rate"] == pytest.approx(1.0)
    assert curriculum.gates.baseline_stage["approach/rate"] == "approach"
    # stage 2 requires the earlier skill to stay within the tolerance
    curriculum.observe(episodes(60))
    curriculum.record_gate(
        {"approach": measurement(0.5), "progress": measurement(0.95)}
    )
    assert curriculum.last_event == "regression"
    assert "retention:approach/rate" in curriculum.blocked_reason()
    assert curriculum.index == 1


def test_a_small_drop_inside_the_tolerance_still_promotes():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.index == 1
    curriculum.observe(episodes(60))
    curriculum.stage_iterations = 100  # a promotion resets the stage clock
    still_ok = 1.0 - RETENTION_TOLERANCE / 2
    assert curriculum.record_gate(
        {"approach": measurement(still_ok), "progress": measurement(0.95)}
    ) is None
    assert curriculum.gates.passes_in_a_row == 1
    assert curriculum.record_gate(
        {"approach": measurement(still_ok), "progress": measurement(0.95)}
    ) == MASTERY_STAGES[2].name


def test_a_noisy_drop_keeps_the_confirmations_that_were_already_earned():
    """A 48-trial probe cannot tell 0.88 from 0.90; that must not reset."""
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.record_gate({"approach": measurement(1.0)}) == "control"
    curriculum.observe(episodes(60))
    curriculum.stage_iterations = 100
    # first confirmation on stage 2
    assert (
        curriculum.record_gate(
            {"approach": measurement(0.97), "progress": measurement(0.95)}
        )
        is None
    )
    assert curriculum.gates.passes_in_a_row == 1
    # 0.80 is below the 0.85 floor, but its 95 % CI reaches 0.94: noise, not
    # a regression.  Promotion stays blocked for this evaluation, yet the
    # confirmation earned above survives.
    noisy = measurement(0.80, low=0.66, trials=48)
    assert (
        curriculum.record_gate({"approach": noisy, "progress": measurement(0.95)})
        is None
    )
    assert curriculum.last_event == "inconclusive"
    assert curriculum.gates.passes_in_a_row == 1
    rows = {row["name"]: row for row in curriculum.retention_rows()}
    row = rows["retention:approach/rate"]
    assert row["ok"] is False and row["conclusive"] is False
    assert "within sampling noise" in row["reason"]
    metrics = curriculum.metrics()
    assert metrics["curriculum/retention/approach/ok"] == 0.0
    assert metrics["curriculum/retention/approach/conclusive"] == 0.0
    # the next clean evaluation completes the promotion instead of starting over
    assert (
        curriculum.record_gate(
            {"approach": measurement(0.97), "progress": measurement(0.95)}
        )
        == MASTERY_STAGES[2].name
    )


def test_a_conclusive_regression_still_resets_the_counter():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    assert curriculum.record_gate({"approach": measurement(1.0)}) == "control"
    curriculum.observe(episodes(60))
    curriculum.stage_iterations = 100
    curriculum.record_gate(
        {"approach": measurement(0.97), "progress": measurement(0.95)}
    )
    assert curriculum.gates.passes_in_a_row == 1
    # 0.40 with a CI of 0.25..0.55: even the optimistic end is far below 0.85
    collapsed = measurement(0.40, low=0.25, trials=48)
    assert (
        curriculum.record_gate({"approach": collapsed, "progress": measurement(0.95)})
        is None
    )
    assert curriculum.last_event == "regression"
    assert curriculum.gates.passes_in_a_row == 0
    rows = {row["name"]: row for row in curriculum.retention_rows()}
    assert rows["retention:approach/rate"]["conclusive"] is True
    assert "within sampling noise" not in rows["retention:approach/rate"]["reason"]


def test_required_skills_accumulate_the_retention_probes():
    curriculum = Curriculum(MASTERY_STAGES, start_index=2)
    assert curriculum.required_skills() == ("approach", "progress", "shooting")
    deeper = Curriculum(MASTERY_STAGES, start_index=4)
    assert "match_chase" in deeper.required_skills()
    assert "defense" in deeper.required_skills()


# ---------------------------------------------------------------------------
# diagnostics and persistence
# ---------------------------------------------------------------------------
def test_diagnostics_name_the_failing_criterion_and_what_to_measure():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(0.2, trials=24)})
    info = curriculum.diagnostics()
    names = {row["name"] for row in info["failing"]}
    assert "approach/rate" in names
    assert any("approach" in hint for hint in info["hints"])
    assert any("gate_confirmations" in hint for hint in info["hints"])
    # a criterion without a skill must not produce a probe reference to "None"
    assert not any("None" in hint for hint in info["hints"])
    assert info["confirmations_required"] == MASTERY_STAGES[0].confirmations


def test_gate_state_survives_a_checkpoint_round_trip():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(1.0)})
    described = curriculum.describe()
    restored = Curriculum(
        MASTERY_STAGES,
        start_index=described["index"],
        stage_iterations=described["iterations"],
        total_iterations=described["total_iterations"],
        window_episodes=described["window"],
        gate_state=described["gate_state"],
    )
    assert restored.gates.passes_in_a_row == 1
    assert restored.gates.evaluations == curriculum.gates.evaluations
    # the restored curriculum promotes on the *next* confirmation, exactly
    # like the original would have
    assert restored.record_gate({"approach": measurement(1.0)}) == "control"


def test_status_contains_the_full_gate_picture():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(0.4)})
    status = curriculum.status()
    gate_rows = [row for row in status["criteria"] if row.get("kind") == "gate"]
    assert gate_rows and gate_rows[0]["skill"] == "approach"
    assert gate_rows[0]["ci95"] is not None
    assert status["gates"]["evaluations"] == 1
    assert status["diagnostics"]["blocked_reason"]


def test_metrics_expose_gate_progress():
    curriculum = ready_curriculum()
    curriculum.record_gate({"approach": measurement(0.9)})
    metrics = curriculum.metrics()
    assert metrics["curriculum/gate_evaluations"] == 1.0
    assert metrics["curriculum/gate_passes_in_a_row"] == 1.0
    assert metrics["curriculum/gate/approach/rate"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# presets and configuration
# ---------------------------------------------------------------------------
def test_presets_are_selectable_and_validated():
    assert stages_for_mode("soccer-1v1") is MASTERY_STAGES
    legacy = stages_for_mode("soccer-1v1", "legacy")
    assert legacy[0].name == "first-touch" and not legacy[0].gates
    with pytest.raises(ValueError, match="unknown curriculum preset"):
        stages_for_mode("soccer-1v1", "nope")
    # heatseeker and the team modes keep their own (window based) lists
    assert not stages_for_mode("heatseeker-2v2")[0].gates


def test_every_mastery_stage_except_the_last_is_gated_and_documented():
    for stage in MASTERY_STAGES[:-1]:
        assert stage.gates, stage.name
        assert stage.purpose, stage.name
        assert stage.confirmations >= 2, stage.name
    assert MASTERY_STAGES[-1].name == "self-play"


def test_configuration_refuses_gates_without_gate_evaluations():
    with pytest.raises(ValueError, match="curriculum_gate_interval is 0"):
        TrainingConfig(curriculum=True, curriculum_gate_interval=0)
    assert TrainingConfig(curriculum=True, curriculum_preset="legacy",
                          curriculum_gate_interval=0).curriculum_preset == "legacy"
    with pytest.raises(ValueError, match="curriculum_preset must be one of"):
        TrainingConfig(curriculum=True, curriculum_preset="magic")


def test_a_stage_without_any_criterion_is_still_refused():
    with pytest.raises(ValueError, match="no promotion criteria"):
        Curriculum(
            (
                Stage("empty", "idle", min_iterations=1, win_rate=0.0),
                Stage("last", "idle", min_iterations=1, win_rate=1.0),
            )
        )


def test_default_gate_thresholds_are_inside_the_measured_range():
    """The thresholds must sit between the passive and the competent reference.

    Reference rates (docs/MASTERY_CURRICULUM.md, experiment 5):
    approach idle 0.06 / chase 1.00, progress idle 0.00 / chase 1.00.
    """
    assert 0.3 < GATE_APPROACH.minimum < 1.0
    assert GATE_APPROACH.ci_low_minimum is not None
    assert 0.3 < GATE_PROGRESS.minimum < 1.0
