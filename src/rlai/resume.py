"""Automatic checkpoint continuation.

Three explicitly different things get conflated easily, so they are separate
modes here and stay separate in the CLI, the API and the GUI:

``fresh``
    Start from a new randomly initialised policy.  Compatible checkpoints are
    listed in the plan but ignored - and the plan says so out loud.
``continue``
    Continue an interrupted run *exactly*: weights, optimiser state, step
    counter, RNG state, curriculum stage, opponent-pool ratings and the
    configuration the lineage was trained with.
``transfer``
    Use a checkpoint only as an initialisation (for example 1v1 weights for a
    2v2 run).  Step counter, optimiser and RNG start over, and the report says
    the run is a transfer, not a continuation.

Nothing in this module ever *starts* training.  It only answers "what would
happening if the user pressed continue?", which the GUI shows before the user
decides, and which the trainer consumes once the user has decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import checkpoints
from .config import TrainingConfig

RESUME_MODES = ("fresh", "continue", "transfer")

#: fields that come from the checkpoint *by default* so a continuation stays
#: comparable with the run it continues.  Any of them can still be changed on
#: purpose by naming it in ``explicit_overrides`` (CLI ``--override-config``):
#: that starts a new training *phase* on the same weights and is recorded as
#: such - it is how a fixed reward definition reaches an existing checkpoint.
#:
#: ``episode_seconds`` and ``rollout_steps`` used to live here.  They are
#: operational (they change neither the network nor the objective), and
#: inheriting them silently replaced a requested 60 s horizon with the
#: checkpoint's 15 s in a real run.  The request now wins; the difference is
#: still reported as an override.
INHERITED_FIELDS = (
    "mode",
    "obs_version",
    "policy",
    "reward",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "value_coef",
    "entropy_coef",
    "epochs",
    "minibatch_size",
    "max_grad_norm",
    "clip_value",
    "target_kl",
    "learning_rate",
    "lr_anneal",
    "tick_skip",
)

#: fields the operator is allowed to change when continuing (request wins;
#: differences are listed in the plan/report, never applied silently)
REQUEST_FIELDS = (
    "run_name",
    "total_steps",
    "rollout_steps",
    "episode_seconds",
    "reset_mode",
    "discount_shaping",
    "n_workers",
    "device",
    "seed",
    "opponent",
    "self_play",
    "curriculum",
    "snapshot_interval",
    "checkpoint_interval",
    "keep_checkpoints",
    "eval_opponent",
    "eval_opponents",
    "eval_interval",
    "eval_games",
    "eval_match_seconds",
    "eval_sides",
    "eval_mirror",
    "eval_before",
    "side_randomization",
)


@dataclass
class ResumePlan:
    """What a resume would restore, and why."""

    mode: str = "fresh"
    requested: str | None = None
    checkpoint_path: str | None = None
    model_id: str | None = None
    parent_run: str | None = None
    parent_step: int = 0
    restored_step: int = 0
    config: dict | None = None
    validation: dict | None = None
    overrides: list[dict] = field(default_factory=list)
    inherited: list[str] = field(default_factory=list)
    #: inherited fields the operator explicitly decided to change (phase change)
    explicit_overrides: list[str] = field(default_factory=list)
    #: the configuration the resumed run would *actually* use (after merging)
    effective_config: dict | None = None
    #: inherited fields whose value changes vs the checkpoint (new phase)
    phase_change: list[str] = field(default_factory=list)
    considered: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lineage: list[str] = field(default_factory=list)
    fallback_used: bool = False
    explicit_unavailable: bool = False
    has_model: bool = False
    has_optimizer: bool = False
    has_rng: bool = False
    has_curriculum: bool = False
    has_league: bool = False

    @property
    def will_resume(self) -> bool:
        return self.checkpoint_path is not None

    @property
    def restores(self) -> list[str]:
        """Human-readable list of what the resume actually restores (GUI checklist)."""
        if not self.will_resume:
            return []
        if self.mode == "transfer":
            return [
                "weights (as initialisation only)",
                "step counter restarts at 0",
                "fresh optimiser, RNG and curriculum",
            ]
        items = ["weights"]
        items.append("optimiser state" if self.has_optimizer else "fresh optimiser (none stored)")
        items.append(f"step counter ({self.restored_step})")
        items.append("RNG state" if self.has_rng else "fresh RNG (none stored)")
        items.append("curriculum stage" if self.has_curriculum else "curriculum restarts at stage 0")
        items.append(
            "opponent pool ratings" if self.has_league else "opponent pool starts empty"
        )
        items.append("inherited training config fields")
        return items

    @property
    def missing(self) -> list[str]:
        """Stored state that is absent, so the GUI can warn instead of promising it."""
        if not self.will_resume:
            return ["everything (no checkpoint selected)"]
        if self.mode == "transfer":
            return []
        return [
            label
            for label, present in (
                ("optimizer state", self.has_optimizer),
                ("RNG state", self.has_rng),
                ("curriculum state", self.has_curriculum),
                ("opponent pool ratings", self.has_league),
            )
            if not present
        ]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "requested": self.requested,
            "will_resume": self.will_resume,
            "checkpoint_path": self.checkpoint_path,
            "model_id": self.model_id,
            "parent_run": self.parent_run,
            "parent_step": self.parent_step,
            "restored_step": self.restored_step,
            "config": self.config,
            "validation": self.validation,
            "overrides": self.overrides,
            "inherited": self.inherited,
            "explicit_overrides": self.explicit_overrides,
            "effective_config": self.effective_config,
            "phase_change": self.phase_change,
            "considered": self.considered,
            "rejected": self.rejected,
            "warnings": self.warnings,
            "lineage": self.lineage,
            "fallback_used": self.fallback_used,
            "explicit_unavailable": self.explicit_unavailable,
            "has_model": self.has_model,
            "has_optimizer": self.has_optimizer,
            "has_rng": self.has_rng,
            "has_curriculum": self.has_curriculum,
            "has_league": self.has_league,
            "restores": self.restores,
            "missing": self.missing,
        }

    def summary(self) -> str:
        if not self.will_resume:
            return f"{self.mode} training from scratch" + (
                f" ({len(self.rejected)} candidate(s) rejected)" if self.rejected else ""
            )
        kind = "continuing" if self.mode == "continue" else "transfer from"
        return (
            f"{kind} {self.model_id} at step {self.parent_step}"
            + (f" (fallback after {len(self.rejected)} rejection(s))" if self.fallback_used else "")
        )


def lineage_runs(models_dir: str | Path, run_id: str | None, limit: int = 32) -> list[str]:
    """Run ids that belong to the same training lineage.

    Follows the ``parent_run`` links recorded in checkpoint manifests, newest
    generation first.  Used to prefer "the run this run continues" over an
    unrelated checkpoint that happens to be newer.
    """
    if not run_id:
        return []
    manifests = checkpoints.list_models(models_dir)
    parents = {
        str(item.get("run_id")): item.get("parent_run")
        for item in manifests
        if item.get("run_id")
    }
    seen: list[str] = [str(run_id)]
    current = parents.get(str(run_id))
    while current and str(current) not in seen and len(seen) < limit:
        seen.append(str(current))
        current = parents.get(str(current))
    return seen


def _plain(value: Any) -> Any:
    """Compare configuration values as plain data, not as model objects."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def merge_config(
    checkpoint_config: dict | None,
    requested: TrainingConfig,
    *,
    mode: str = "continue",
    explicit_overrides: Any = (),
) -> tuple[TrainingConfig, list[dict], list[str]]:
    """Combine the stored configuration with the operator's request.

    For ``continue`` the checkpoint wins on everything that would make the
    resumed run incomparable with its parent (mode, observation layout,
    network shape, reward weights, PPO hyper-parameters); the request wins on
    operational knobs (how long, how many workers, episode horizon,
    evaluation schedule).  Every difference is reported as an override with
    the value actually used, so nothing happens silently.

    Fields named in ``explicit_overrides`` are the deliberate exception: the
    request wins even on an inherited field.  That is a *phase change* (same
    weights, changed objective/hyper-parameters) and the override records say
    so (``source: "explicit"``).  Unknown or non-inherited names are refused
    instead of being ignored.  For ``transfer`` and ``fresh`` the request is
    used unchanged.
    """
    explicit = [str(name).strip() for name in (explicit_overrides or ()) if str(name).strip()]
    unknown = [name for name in explicit if name not in INHERITED_FIELDS]
    if unknown:
        raise ValueError(
            f"cannot explicitly override {unknown}: only inherited fields "
            f"{list(INHERITED_FIELDS)} need (and accept) an explicit override; "
            "every other field already follows the request"
        )
    if mode != "continue" or not checkpoint_config:
        return requested, [], []
    overrides: list[dict] = []
    inherited: list[str] = []
    updates: dict[str, Any] = {}
    for name in INHERITED_FIELDS:
        if name not in checkpoint_config:
            continue
        stored = checkpoint_config[name]
        current_value = _plain(getattr(requested, name, None))
        if name in explicit:
            if stored != current_value:
                overrides.append(
                    {
                        "field": name,
                        "checkpoint": stored,
                        "requested": current_value,
                        "used": current_value,
                        "source": "explicit",
                    }
                )
            continue
        inherited.append(name)
        if stored != current_value:
            overrides.append(
                {
                    "field": name,
                    "checkpoint": stored,
                    "requested": current_value,
                    "used": stored,
                    "source": "inherited",
                }
            )
            updates[name] = stored
    for name in REQUEST_FIELDS:
        if name in checkpoint_config:
            stored = checkpoint_config[name]
            current_value = _plain(getattr(requested, name, None))
            if stored != current_value:
                overrides.append(
                    {
                        "field": name,
                        "checkpoint": stored,
                        "requested": current_value,
                        "used": current_value,
                        "source": "request",
                    }
                )
    if not updates:
        return requested, overrides, inherited
    data = requested.to_dict()
    data.update(updates)
    try:
        merged = TrainingConfig.model_validate(data)
    except Exception as exc:
        return requested, overrides + [
            {"field": "*", "error": f"stored configuration could not be merged: {exc}"}
        ], inherited
    return merged, overrides, inherited


def plan_resume(
    models_dir: str | Path,
    config: TrainingConfig,
    *,
    mode: str = "continue",
    run_id: str | None = None,
    checkpoint: str | None = None,
    max_candidates: int = 25,
    explicit_overrides: Any = (),
) -> ResumePlan:
    """Find the newest valid, compatible checkpoint for ``mode``.

    Never raises for a broken or incompatible checkpoint: it reports the
    rejection and keeps looking at older candidates.  Only a ``fresh`` request
    or an empty models directory produces a plan without a checkpoint.
    """
    models_dir = Path(models_dir)
    if mode not in RESUME_MODES:
        raise ValueError(f"resume mode must be one of {RESUME_MODES}, got {mode!r}")

    plan = ResumePlan(mode=mode, requested=checkpoint)
    plan.explicit_overrides = [
        str(name).strip() for name in (explicit_overrides or ()) if str(name).strip()
    ]
    # validate eagerly so a typo fails the *plan*, not the training start
    merge_config({}, config, mode="continue", explicit_overrides=plan.explicit_overrides)
    plan.lineage = lineage_runs(models_dir, run_id)

    candidates = checkpoints.list_checkpoint_candidates(models_dir, mode=None)
    compatible = [
        item
        for item in candidates
        if item.get("mode") in (None, config.mode) or mode == "transfer"
    ]
    if mode == "fresh":
        plan.considered = [_candidate_summary(item) for item in compatible[:max_candidates]]
        if compatible:
            plan.warnings.append(
                f"fresh training requested: {len(compatible)} compatible checkpoint(s) "
                f"ignored, newest is {compatible[0].get('id')} at step {compatible[0].get('step')}"
            )
        return plan

    if not candidates:
        plan.warnings.append(f"no checkpoints in {models_dir}; nothing to resume from")
        return plan

    ordered = _order_candidates(compatible, plan.lineage, config, mode)
    if checkpoint:
        explicit = _explicit_first(models_dir, checkpoint, ordered)
        if explicit is not None:
            ordered = [explicit] + [item for item in ordered if item["id"] != explicit.get("id")]
        else:
            ordered = [
                {
                    "id": None,
                    "absolute_path": str(checkpoint),
                    "step": None,
                    "run_id": None,
                    "mode": None,
                    "created_at": None,
                }
            ] + ordered

    # A missing optimizer state is a limitation, not a rejection: continuing
    # with a fresh optimizer is still much better than starting from scratch.
    # Candidates that do carry one are ordered first (see _order_candidates).
    require_optimizer = False
    for item in ordered[:max_candidates]:
        path = item.get("absolute_path")
        if not path:
            continue
        validation = checkpoints.validate_checkpoint(
            path,
            policy_config=config.policy_config().to_dict(),
            mode=None if mode == "transfer" else config.mode,
            obs_version=config.obs_version,
            require_optimizer=require_optimizer,
        )
        record = _candidate_summary(item)
        record["validation"] = validation.to_dict()
        if not validation.ok:
            record["reasons"] = validation.reasons
            plan.rejected.append(record)
            continue
        if checkpoint and not _matches_reference(models_dir, item, checkpoint):
            # The user named a specific checkpoint.  Substituting an older one
            # without asking would be exactly the silent surprise we avoid.
            plan.explicit_unavailable = True
            reasons = [
                reason
                for entry in plan.rejected
                if _matches_reference(models_dir, entry, checkpoint)
                for reason in entry.get("reasons", [])
            ] or ["the requested checkpoint was not accepted"]
            plan.warnings.append(
                f"requested checkpoint {checkpoint!r} cannot be used ({'; '.join(reasons)}); "
                f"refusing to substitute {item.get('id') or item.get('absolute_path')} "
                "for an explicit request"
            )
            return plan
        plan.considered.append(record)
        plan.checkpoint_path = str(validation.path)
        plan.model_id = item.get("id")
        plan.parent_run = item.get("run_id") or validation.run_id
        plan.parent_step = int(validation.step or 0)
        plan.restored_step = 0 if mode == "transfer" else int(validation.step or 0)
        plan.validation = validation.to_dict()
        plan.fallback_used = bool(plan.rejected)
        plan.config = validation.training_config
        # Show, before anything starts, which configuration the resumed run
        # would actually use and which requested values would *not* be used.
        # (plan.overrides used to stay empty forever, so the preview could
        # not reveal e.g. a horizon being replaced by the checkpoint's.)
        if mode == "continue" and validation.training_config:
            effective, plan.overrides, plan.inherited = merge_config(
                validation.training_config,
                config,
                mode=mode,
                explicit_overrides=plan.explicit_overrides,
            )
            plan.effective_config = effective.to_dict()
            plan.phase_change = [
                item["field"]
                for item in plan.overrides
                if item.get("source") == "explicit"
            ]
            replaced = [
                item["field"]
                for item in plan.overrides
                if item.get("source") == "inherited"
            ]
            if replaced:
                plan.warnings.append(
                    "these requested values will be replaced by the checkpoint's "
                    f"({', '.join(replaced)}); name a field in --override-config "
                    "to change it deliberately instead"
                )
            if plan.phase_change:
                plan.warnings.append(
                    "phase change: continuing the same weights with changed "
                    f"{', '.join(plan.phase_change)} - the value function and "
                    "optimizer moments were fitted to the old values and need "
                    "some iterations to re-adapt"
                )
            stored_total = (validation.training_config or {}).get("total_steps")
            if stored_total and int(stored_total) != int(config.total_steps) and config.lr_anneal:
                plan.warnings.append(
                    f"total_steps changes from {stored_total} to {config.total_steps}: "
                    "the learning-rate schedule is recomputed against the new total, "
                    "so the learning rate will differ from an uninterrupted run"
                )
        plan.has_model = validation.has_model
        plan.has_optimizer = validation.has_optimizer
        plan.has_rng = validation.has_rng
        plan.has_curriculum = validation.has_curriculum
        plan.has_league = validation.has_league
        if mode == "transfer":
            plan.warnings.append(
                f"transfer learning: weights of {plan.model_id or plan.checkpoint_path} are used "
                "as initialisation only; step counter, optimiser and RNG start over"
            )
            if not validation.has_optimizer:
                plan.warnings.append("checkpoint has no optimizer state (irrelevant for transfer)")
        else:
            if not validation.has_rng:
                plan.warnings.append(
                    "checkpoint has no RNG state (written by an older version); rollouts after "
                    "the resume will not be bit-identical to an uninterrupted run"
                )
            if not validation.has_curriculum and config.curriculum:
                plan.warnings.append(
                    "checkpoint has no curriculum state; the curriculum restarts at stage 0"
                )
            if not validation.has_league and config.self_play:
                plan.warnings.append(
                    "checkpoint has no opponent-pool ratings; the pool starts empty"
                )
            if plan.restored_step >= config.total_steps:
                plan.warnings.append(
                    f"checkpoint step {plan.restored_step} already reached total_steps "
                    f"{config.total_steps}; raise total_steps to keep training"
                )
        if plan.fallback_used:
            rejected_ids = ", ".join(
                str(item.get("id") or item.get("absolute_path")) for item in plan.rejected
            )
            plan.warnings.append(
                f"newest checkpoint(s) rejected ({rejected_ids}); falling back to "
                f"{plan.model_id or plan.checkpoint_path}"
            )
        return plan

    # Candidates that were never validated (no readable path) are still shown,
    # but a rejected candidate must not also appear as "considered".
    listed = {
        (item.get("id") or item.get("absolute_path"))
        for item in [*plan.considered, *plan.rejected]
    }
    plan.considered.extend(
        _candidate_summary(item)
        for item in ordered[:max_candidates]
        if (item.get("id") or item.get("absolute_path")) not in listed
    )
    if checkpoint and not plan.will_resume:
        plan.explicit_unavailable = True
        reasons = [
            reason
            for entry in plan.rejected
            if _matches_reference(models_dir, entry, checkpoint)
            for reason in entry.get("reasons", [])
        ]
        if not reasons and not explicit_checkpoint_exists(models_dir, checkpoint):
            reasons = [f"no checkpoint matches {checkpoint!r}"]
        plan.warnings.append(
            f"requested checkpoint {checkpoint!r} cannot be used"
            + (f" ({'; '.join(reasons)})" if reasons else "")
            + "; refusing to start from scratch silently"
        )
    if not plan.will_resume:
        plan.warnings.append(
            f"no compatible checkpoint found for mode={config.mode} "
            f"obs_version={config.obs_version} ({len(plan.rejected)} rejected); "
            "training would start from scratch"
        )
    return plan


def _matches_reference(models_dir: Path, item: dict, reference: str) -> bool:
    """True when a candidate *is* the checkpoint the user asked for by name or path."""
    text = str(reference).strip()
    if not text:
        return False
    if item.get("id") and str(item["id"]) == text:
        return True
    path = item.get("absolute_path")
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for candidate in (
        Path(text),
        Path(text).expanduser(),
        models_dir / text,
        checkpoints.checkpoint_dir(models_dir, text) / checkpoints.MODEL_FILE,
    ):
        try:
            if candidate.resolve() == resolved:
                return True
        except OSError:
            continue
    return False


def explicit_checkpoint_exists(models_dir: str | Path, reference: str) -> bool:
    """True when a user-supplied reference points at a real file."""
    try:
        resolved = checkpoints.resolve_model_path(models_dir, reference, allow_outside=True)
    except (FileNotFoundError, ValueError):
        return False
    return Path(resolved).exists()


def _candidate_summary(item: dict) -> dict:
    return {
        "id": item.get("id"),
        "path": item.get("absolute_path"),
        "step": item.get("step"),
        "run_id": item.get("run_id"),
        "mode": item.get("mode"),
        "created_at": item.get("created_at"),
        "label": item.get("label"),
        "has_optimizer": item.get("has_optimizer"),
        "obs_version": item.get("obs_version"),
        "size_bytes": item.get("size_bytes"),
    }


def _order_candidates(
    candidates: list[dict], lineage: list[str], config: TrainingConfig, mode: str
) -> list[dict]:
    """Newest first, but inside the requested lineage before anything else."""
    def sort_key(item: dict):
        in_lineage = 1 if str(item.get("run_id")) in lineage else 0
        same_mode = 1 if item.get("mode") in (None, config.mode) else 0
        has_optimizer = 1 if item.get("has_optimizer") in (True, None) else 0
        return (
            in_lineage if mode == "continue" else 0,
            same_mode,
            has_optimizer if mode == "continue" else 0,
            item.get("created_at") or "",
            int(item.get("step") or 0),
        )

    return sorted(candidates, key=sort_key, reverse=True)


def _explicit_first(models_dir: Path, reference: str, ordered: list[dict]) -> dict | None:
    """Match an explicit user reference against the known candidates."""
    try:
        resolved = checkpoints.resolve_model_path(models_dir, reference)
    except (FileNotFoundError, ValueError):
        try:
            resolved = checkpoints.resolve_model_path(models_dir, reference, allow_outside=True)
        except (FileNotFoundError, ValueError):
            return None
    for item in ordered:
        if item.get("absolute_path") and Path(item["absolute_path"]) == resolved:
            return item
        if item.get("id") == reference:
            return item
    return {"id": reference, "absolute_path": str(resolved), "step": None, "run_id": None}
