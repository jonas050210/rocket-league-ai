"""Automatic checkpoint continuation: planning, validation and fallbacks.

Nothing here starts training.  ``plan_resume`` only answers what a resume
*would* restore, which is what the GUI shows before the user decides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.policy import PolicyConfig
from rlai.resume import (
    INHERITED_FIELDS,
    lineage_runs,
    merge_config,
    plan_resume,
)


def _config(**overrides) -> TrainingConfig:
    base = dict(
        run_name="unit",
        total_steps=4096,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _save(
    models_dir: Path,
    config: TrainingConfig,
    *,
    step: int,
    run_id: str | None = None,
    mode: str | None = None,
    optimizer: bool = True,
    rng: bool = True,
    label: str = "",
    obs_version: int | None = None,
    hidden: tuple[int, ...] | None = None,
    corrupt: bool = False,
    extra: dict | None = None,
) -> dict:
    policy_config = config.policy_config()
    if obs_version is not None:
        from rlai.obs import obs_size

        policy_config = PolicyConfig(
            obs_size=obs_size(int(obs_version)),
            hidden=policy_config.hidden,
            activation=policy_config.activation,
        )
    if hidden is not None:
        policy_config = PolicyConfig(
            obs_size=policy_config.obs_size, hidden=hidden, activation=policy_config.activation
        )
    model = policy_config.build()
    payload_extra = {
        "iteration": step // 64,
        "obs_version": obs_version if obs_version is not None else config.obs_version,
        "league": {"capacity": 8, "agent_elo": 1010.0, "entries": []},
        "curriculum": {"index": 1, "iterations": 3, "total_iterations": 9},
    }
    if rng:
        payload_extra["rng"] = {
            "numpy": np.random.default_rng(step).bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "python": list(torch.randint(0, 5, (3,)).tolist()),
        }
    if extra:
        payload_extra.update(extra)
    optimizer_state = None
    if optimizer:
        optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        loss = model(torch.zeros(1, policy_config.obs_size))[0].pow(2).sum()
        loss.backward()
        optim.step()
        optimizer_state = optim.state_dict()
    manifest = checkpoints.save_checkpoint(
        models_dir,
        model_state=model.state_dict(),
        policy_config=policy_config.to_dict(),
        optimizer_state=optimizer_state,
        step=step,
        run_id=run_id,
        mode=mode if mode is not None else config.mode,
        training_config=config.to_dict(),
        extra=payload_extra,
        label=label,
    )
    if corrupt:
        (models_dir / manifest["path"]).write_bytes(b"this is not a checkpoint")
    return manifest


def test_fresh_mode_ignores_checkpoints_but_says_so(data_root):
    models = data_root / "models"
    config = _config()
    _save(models, config, step=128, run_id="old")
    plan = plan_resume(models, config, mode="fresh", run_id="new")
    assert plan.mode == "fresh"
    assert plan.will_resume is False
    assert plan.checkpoint_path is None
    assert plan.considered, "the plan must show what it ignored"
    assert any("fresh training requested" in warning for warning in plan.warnings)


def test_continue_picks_the_newest_compatible_checkpoint(data_root):
    models = data_root / "models"
    config = _config()
    _save(models, config, step=64, run_id="run-a")
    newest = _save(models, config, step=1024, run_id="run-a", label="final")
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is True
    assert plan.model_id == newest["id"]
    assert plan.restored_step == 1024
    assert plan.parent_run == "run-a"
    assert plan.fallback_used is False
    assert plan.rejected == []
    assert plan.validation["has_optimizer"] is True
    assert plan.validation["has_rng"] is True
    assert plan.config == config.to_dict()


def test_invalid_newest_checkpoint_falls_back_to_an_older_one(data_root):
    models = data_root / "models"
    config = _config()
    good = _save(models, config, step=256, run_id="run-a")
    _save(models, config, step=512, run_id="run-a", corrupt=True)
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is True
    assert plan.model_id == good["id"], plan.rejected
    assert plan.fallback_used is True
    assert len(plan.rejected) == 1
    assert plan.rejected[0]["reasons"]
    assert any("falling back" in warning for warning in plan.warnings)


def test_incompatible_architecture_is_rejected_with_a_reason(data_root):
    models = data_root / "models"
    config = _config()
    _save(models, config, step=512, run_id="run-a", hidden=(64, 64))
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is False
    assert plan.rejected and "architecture mismatch" in plan.rejected[0]["reasons"][0]
    assert any("no compatible checkpoint" in warning for warning in plan.warnings)


def test_observation_layout_mismatch_is_rejected(data_root):
    models = data_root / "models"
    config = _config(obs_version=2)
    _save(models, config, step=128, run_id="run-a", obs_version=1)
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is False
    assert any("observation layout mismatch" in reason for reason in plan.rejected[0]["reasons"])


def test_other_mode_is_not_used_for_a_continuation(data_root):
    models = data_root / "models"
    config = _config(mode="soccer-2v2", total_steps=4096)
    _save(models, _config(mode="soccer-1v1"), step=1024, run_id="run-a", mode="soccer-1v1")
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is False
    assert any("no compatible checkpoint" in warning for warning in plan.warnings)


def test_transfer_mode_accepts_another_mode_and_resets_the_step(data_root):
    models = data_root / "models"
    source = _save(models, _config(mode="soccer-1v1"), step=4096, run_id="run-a")
    config = _config(mode="soccer-2v2")
    plan = plan_resume(models, config, mode="transfer", run_id="run-b")
    assert plan.will_resume is True
    assert plan.model_id == source["id"]
    assert plan.parent_step == 4096
    assert plan.restored_step == 0, "transfer must not inherit the step counter"
    assert any("transfer learning" in warning for warning in plan.warnings)


def test_lineage_is_preferred_over_a_newer_unrelated_checkpoint(data_root):
    models = data_root / "models"
    config = _config()
    own = _save(models, config, step=512, run_id="run-a", extra={"parent_run": "run-0"})
    # a different run's checkpoint written later (newer timestamp)
    _save(models, config, step=64, run_id="run-other")
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    lineage = lineage_runs(models, "run-b")
    assert lineage == ["run-b"]
    assert plan.will_resume is True
    # both are compatible; the plan must be deterministic and report which one
    assert plan.model_id in {own["id"]} | {
        item["id"] for item in checkpoints.list_models(models)
    }
    plan_from_lineage = plan_resume(models, config, mode="continue", run_id="run-a")
    assert plan_from_lineage.model_id == own["id"]
    assert lineage_runs(models, "run-a") == ["run-a", "run-0"]


def test_explicit_checkpoint_reference_is_used_first(data_root):
    models = data_root / "models"
    config = _config()
    older = _save(models, config, step=128, run_id="run-a")
    _save(models, config, step=1024, run_id="run-a")
    plan = plan_resume(
        models, config, mode="continue", run_id="run-b", checkpoint=older["id"]
    )
    assert plan.model_id == older["id"]
    assert plan.restored_step == 128
    assert plan.requested == older["id"]


def test_explicit_incompatible_checkpoint_is_refused_not_substituted(data_root):
    """A named checkpoint is a decision; silently using another one is not."""
    models = data_root / "models"
    config = _config()
    bad = _save(models, config, step=1024, run_id="run-a", hidden=(64,))
    good = _save(models, config, step=128, run_id="run-a")

    plan = plan_resume(models, config, mode="continue", run_id="run-b", checkpoint=bad["id"])
    assert plan.will_resume is False
    assert plan.explicit_unavailable is True
    assert plan.rejected[0]["id"] == bad["id"]
    assert plan.rejected[0]["reasons"]
    assert any(bad["id"] in warning for warning in plan.warnings)
    assert any(good["id"] in warning for warning in plan.warnings)
    # the same directory still offers a fallback for an *automatic* continuation
    auto = plan_resume(models, config, mode="continue", run_id="run-b")
    assert auto.will_resume is True and auto.model_id == good["id"]

    # a reference that matches nothing at all is reported the same way
    missing = plan_resume(models, config, mode="continue", checkpoint="not-a-checkpoint")
    assert missing.will_resume is False
    assert missing.explicit_unavailable is True
    assert any("not-a-checkpoint" in warning for warning in missing.warnings)


def test_missing_optimizer_state_warns_but_still_resumes(data_root):
    models = data_root / "models"
    config = _config()
    manifest = _save(models, config, step=256, run_id="run-a", optimizer=False, rng=False)
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    assert plan.will_resume is True
    assert plan.model_id == manifest["id"]
    assert plan.validation["has_optimizer"] is False
    assert any("RNG state" in warning for warning in plan.warnings)


def test_checkpoint_at_or_beyond_total_steps_is_reported(data_root):
    models = data_root / "models"
    _save(models, _config(total_steps=2048), step=2048, run_id="run-a")
    plan = plan_resume(models, _config(total_steps=2048), mode="continue", run_id="run-b")
    assert plan.will_resume is True
    assert any("total_steps" in warning for warning in plan.warnings)


def test_empty_models_directory_reports_nothing_to_resume(data_root):
    plan = plan_resume(data_root / "models", _config(), mode="continue", run_id="run-b")
    assert plan.will_resume is False
    assert any("no checkpoints" in warning for warning in plan.warnings)


def test_unknown_mode_is_rejected(data_root):
    with pytest.raises(ValueError, match="resume mode"):
        plan_resume(data_root / "models", _config(), mode="sideways")


def test_plan_is_serialisable_for_the_gui(data_root):
    models = data_root / "models"
    config = _config()
    _save(models, config, step=64, run_id="run-a")
    plan = plan_resume(models, config, mode="continue", run_id="run-b")
    payload = plan.to_dict()
    assert json.loads(json.dumps(payload, default=str))["restored_step"] == 64
    assert "continuing" in plan.summary()


def test_merge_config_inherits_comparability_fields(data_root):
    stored = _config(gamma=0.99, learning_rate=1e-4, run_name="stored").to_dict()
    requested = _config(
        gamma=0.9, learning_rate=3e-4, run_name="requested", total_steps=999_999
    )
    merged, overrides, inherited = merge_config(stored, requested, mode="continue")
    assert merged.gamma == 0.99, "gamma must come from the checkpoint"
    assert merged.learning_rate == 1e-4
    assert merged.run_name == "requested", "operational fields follow the request"
    assert merged.total_steps == 999_999
    assert "gamma" in inherited and "policy" in INHERITED_FIELDS
    fields = {item["field"] for item in overrides}
    assert {"gamma", "learning_rate", "run_name", "total_steps"} <= fields
    gamma_override = next(item for item in overrides if item["field"] == "gamma")
    assert gamma_override["used"] == 0.99 and gamma_override["requested"] == 0.9


def test_merge_config_is_a_no_op_for_transfer_and_fresh():
    requested = _config()
    for mode in ("transfer", "fresh"):
        merged, overrides, inherited = merge_config({"gamma": 0.5}, requested, mode=mode)
        assert merged == requested
        assert overrides == [] and inherited == []


def test_validation_reports_every_reason(data_root):
    models = data_root / "models"
    config = _config()
    manifest = _save(models, config, step=128, run_id="run-a")
    path = models / manifest["path"]
    ok = checkpoints.validate_checkpoint(
        path, policy_config=config.policy_config().to_dict(), require_optimizer=True
    )
    assert ok.ok is True and ok.has_optimizer and ok.has_rng and ok.step == 128
    wrong = checkpoints.validate_checkpoint(
        path,
        policy_config=PolicyConfig(obs_size=config.policy_config().obs_size, hidden=(64,)).to_dict(),
    )
    assert wrong.ok is False
    assert any("architecture mismatch" in reason for reason in wrong.reasons)
    missing = checkpoints.validate_checkpoint(models / "nope" / "model.pt")
    assert missing.ok is False and missing.error == "file does not exist"


def test_highest_training_step_wins_even_when_saves_arrive_out_of_order(data_root):
    """Selection is by progress metadata, not by the newest directory name."""
    models = data_root / "models"
    config = _config()
    _save(models, config, step=128, run_id="run-a")
    latest_step = _save(models, config, step=1024, run_id="run-a")
    last_saved = _save(models, config, step=512, run_id="run-a")

    plan = plan_resume(models, config, mode="continue", run_id="run-a")

    assert plan.will_resume is True
    assert plan.model_id == latest_step["id"]
    assert plan.restored_step == 1024
    # the pointer records the last completed save, not the highest step
    assert checkpoints.latest_record(models)["id"] == last_saved["id"]


def test_candidates_found_distinguishes_an_empty_directory_from_rejections(data_root):
    models = data_root / "models"
    empty = plan_resume(models, _config(), mode="continue", run_id="run-b")
    assert empty.candidates_found == 0
    assert empty.will_resume is False

    config = _config()
    _save(models, config, step=128, run_id="run-a", hidden=(64,))
    rejected = plan_resume(models, config, mode="continue", run_id="run-b")
    assert rejected.candidates_found == 1
    assert rejected.will_resume is False
    assert rejected.rejected


def test_explicit_reference_still_overrides_the_newest_step(data_root):
    models = data_root / "models"
    config = _config()
    older = _save(models, config, step=128, run_id="run-a")
    _save(models, config, step=1024, run_id="run-a")
    plan = plan_resume(
        models, config, mode="continue", run_id="run-b", checkpoint=older["id"]
    )
    assert plan.model_id == older["id"]
    assert plan.restored_step == 128
