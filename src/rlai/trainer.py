"""Training orchestration: PPO + parallel workers + checkpoints + metrics.

The trainer never decides on its own to start or to continue: it is always
constructed with an explicit configuration and, optionally, an explicit resume
plan produced by :mod:`rlai.resume`.  Every run - finished, stopped or failed -
ends with a persisted statistics report (see :mod:`rlai.report`).
"""

from __future__ import annotations

import json
import os
import random
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import checkpoints, report as report_module
from .checkpoints import prune_candidates, save_checkpoint
from .config import TrainingConfig, ensure_dirs
from .curriculum import Curriculum, stages_for_mode
from .league import OpponentPool, freeze_state
from .policy import PolicyConfig, TorchPolicy, clone_state_dict
from .ppo import Batch, flatten_rollout, learning_rate_at, ppo_update
from .resume import ResumePlan, explicit_checkpoint_exists, merge_config, plan_resume
from .store import Store
from .timing import TrainingClock, utc_now_iso
from .workers import WorkerPool


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def is_best_candidate(result: dict, best_score: float) -> bool:
    """A model can only become "best" after it has actually won a game.

    Untrained policies produce 0:0 draws; marking those as best would put a
    meaningless checkpoint at the top of the GUI and of ``best.json``.  A
    single decisive game is not evidence either: with the loser known, at
    least two decisive games are required before a win rate may dethrone
    the incumbent.
    """
    wins = int(result.get("wins", 0))
    if wins <= 0:
        return False
    if result.get("losses") is not None and wins + int(result.get("losses") or 0) < 2:
        return False
    return float(result.get("win_rate", 0.0)) > best_score


#: periodic evaluations without measurable progress that count as a stall
STALL_WINDOW = 3


def detect_eval_stall(history: list[dict], window: int = STALL_WINDOW) -> bool:
    """True when the last ``window`` evaluations show no progress at all.

    Progress means: a win, a better goal difference, or more ball touches
    than the evaluation right before the window.  Deliberately strict - a
    false alarm costs a log warning, a missed stall costs hours.
    """
    if len(history) < window + 1:
        return False
    recent, before = history[-window:], history[-window - 1]
    if any(int(item.get("wins") or 0) > 0 for item in recent):
        return False
    if max(float(item.get("goal_diff") or 0) for item in recent) > float(
        before.get("goal_diff") or 0
    ):
        return False
    if max(float(item.get("touches") or 0) for item in recent) > float(
        before.get("touches") or 0
    ):
        return False
    return True


def _outcome_counts(episodes: list[dict]) -> tuple[int, int, int]:
    wins = sum(1 for episode in episodes if episode.get("result") == "win")
    losses = sum(1 for episode in episodes if episode.get("result") == "loss")
    draws = sum(1 for episode in episodes if episode.get("result") == "draw")
    return wins, losses, draws


class _InterruptGuard:
    """Turn Ctrl+C / SIGTERM into a safe stop instead of an immediate crash.

    The first signal raises :class:`KeyboardInterrupt` (so a blocking worker
    wait is interrupted promptly).  Inside :meth:`deferred` the signal is
    remembered instead: checkpoint writing must finish before the exception is
    raised, otherwise an interruption in the middle of a save could leave the
    process to be killed before updating its own bookkeeping.  Every further
    signal during cleanup is recorded but ignored, so a second signal cannot
    interrupt the checkpoint and report of the first one.

    ``SIGTERM`` is handled the same way because that is what a supervisor
    (including the GUI job manager) sends to cancel a run; without a handler
    the process would die without writing its interrupt checkpoint.
    """

    def __init__(self) -> None:
        self.count = 0
        self.installed = False
        self._originals: dict[int, object] = {}
        self._defer_depth = 0
        self._deferred = False
        self.shutting_down = False
        #: last signal number that reached :meth:`_handle` (for messages)
        self.last_signum: int | None = None

    def begin_shutdown(self) -> None:
        """From now on signals are noted but never allowed to abort cleanup."""
        self.shutting_down = True

    def describe(self) -> str:
        """Human-readable reason for the message written at shutdown."""
        if self.last_signum == getattr(signal, "SIGTERM", None):
            return "a termination request (SIGTERM)"
        return "a keyboard interrupt (Ctrl+C)"

    def install(self) -> None:
        if self.installed or threading.current_thread() is not threading.main_thread():
            return
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._originals[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            self.installed = True
        except (ValueError, OSError):  # pragma: no cover - platform specific
            self.restore()

    def restore(self) -> None:
        if not self._originals:
            return
        for signum, original in self._originals.items():
            try:
                signal.signal(signum, original)
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass
        self._originals.clear()
        self.installed = False

    def _handle(self, signum, frame) -> None:
        self.count += 1
        if signum is not None:
            self.last_signum = int(signum)
        if self.shutting_down:
            return
        if self._defer_depth:
            # Only a first interrupt may be turned into an exception after the
            # critical section; later ones must not abort cleanup.
            if self.count == 1:
                self._deferred = True
            return
        if self.count == 1:
            raise KeyboardInterrupt

    @contextmanager
    def deferred(self):
        """Delay a first SIGINT until this block has finished."""
        self._defer_depth += 1
        try:
            yield
        finally:
            self._defer_depth -= 1
            if self._defer_depth == 0 and self._deferred:
                self._deferred = False
                raise KeyboardInterrupt


def _reward_component_means(episodes: list[dict]) -> dict[str, float]:
    """Average of every reward component over the finished episodes."""
    totals: dict[str, float] = {}
    counted = 0
    for episode in episodes:
        components = episode.get("reward_components") or {}
        if not components:
            continue
        counted += 1
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if not counted:
        return {}
    return {f"reward/{key}": value / counted for key, value in sorted(totals.items())}


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        run_id: str,
        store: Store,
        *,
        resume_from: str | None = None,
        run_dir: Path | None = None,
        resume_mode: str = "continue",
        resume_plan: ResumePlan | None = None,
        config_override_fields: tuple[str, ...] = (),
    ):
        #: inherited config fields the operator explicitly wants to change
        #: on a ``continue`` resume (a phase change, see rlai.resume)
        self.config_override_fields = tuple(config_override_fields or ())
        #: merge decisions of the resume, for the report
        self.resume_overrides: list[dict] = []
        self.config = config
        self.run_id = run_id
        self.store = store
        paths = ensure_dirs()
        self.models_dir = paths["models"]
        self.run_dir = Path(run_dir) if run_dir else paths["runs"] / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.control_path = self.run_dir / "control.json"
        self.resume_from = resume_from
        self.resume_mode = resume_mode
        self.resume_plan: ResumePlan | None = resume_plan
        self.device = torch.device(config.resolved_device())
        self.model = (
            PolicyConfig.from_dict(config.policy_config().to_dict()).build().to(self.device)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.step = 0
        self.iteration = 0
        self.start_step = 0
        self.pool: WorkerPool | None = None
        self.league = OpponentPool(
            capacity=int(config.pool_capacity),
            duplicate_step_tolerance=int(config.pool_duplicate_step_tolerance),
            duplicate_distance_tolerance=float(config.pool_duplicate_distance_tolerance),
        )
        self.curriculum = (
            Curriculum(stages_for_mode(config.mode, config.curriculum_preset))
            if config.curriculum
            else None
        )
        self.rng = np.random.default_rng(int(config.seed))
        #: best win rate per evaluation protocol (key -> score).  Comparing
        #: against the primary pointer regardless of protocol (the old code)
        #: could block a genuinely better result under a harder protocol -
        #: or crown one measured under an easier one.
        self._best_by_protocol: dict[str, float] = {}
        self.run_best_score = float("-inf")
        #: periodic-eval history for stall detection (see detect_eval_stall)
        self._eval_history: list[dict] = []
        self.started = time.time()
        #: reliable training-time accounting (monotonic clock + persisted
        #: accumulated total); see :mod:`rlai.timing`
        self.clock = TrainingClock(self.run_id, pid=os.getpid())
        self.last_checkpoint_step = -1
        self.last_checkpoint_id: str | None = None
        self.final_manifest: dict | None = None
        self.run_warnings: list[str] = []
        #: True when the latest state was successfully written at shutdown,
        #: False when an interrupted run could not save it.  ``None`` until a
        #: terminal state is known.
        self.resumable: bool | None = None
        self._interrupted = False
        self.interrupt_checkpoint_id: str | None = None
        self.interrupt_checkpoint_error: str | None = None
        self._interrupt_guard = _InterruptGuard()
        self.evaluation_seconds = 0.0
        #: monotonic timestamp of the last recovery checkpoint (crash net)
        self._last_recovery_at = time.monotonic()
        self.gate_seconds = 0.0
        self.gate_probe_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.evaluated_before = False
        #: filled in by :meth:`_apply_resource_limits` before the workers start
        self.resource_plan = None
        #: id of the current session row in the database
        self.session_id: int | None = None
        #: metrics waiting for the next database write
        self._pending_metrics: dict[str, tuple[int, float]] = {}
        #: how many iterations passed since the last metric write
        self._iterations_since_flush = 0
        #: the adaptive-match-dynamics controller (created on demand)
        self._adaptive = None
        #: Torch thread budget of the *trainer* process.  Set before any
        #: rollout is collected so the intended value is the one that is
        #: actually used.
        self.torch_threads = self._resolve_torch_threads()
        self._apply_torch_threads(self.torch_threads)

    # -- logging -----------------------------------------------------------
    @staticmethod
    def _resolve_torch_threads() -> int:
        """Torch threads for the learner process.

        The learner runs one small MLP over a batch; it never benefits from
        more threads than it has cores, and on a shared machine it is polite
        to leave the rest to the simulator workers.
        """
        cores = os.cpu_count() or 1
        return max(1, min(4, cores))

    def _apply_torch_threads(self, threads: int) -> None:
        try:
            torch.set_num_threads(int(threads))
        except Exception:  # pragma: no cover - platform specific
            pass

    def _log(self, level: str, message: str) -> None:
        self.store.log(self.run_id, level, message)
        print(f"[{level.upper()}] {message}", flush=True)

    def _warn(self, message: str) -> None:
        if message not in self.run_warnings:
            self.run_warnings.append(message)
        self._log("WARNING", message)

    def _metric(self, name: str, value: float, step: int | None = None) -> None:
        """Queue a metric for the next flush.

        Writing every metric in its own transaction is pure overhead: one
        iteration produces a few dozen values.  They are collected here
        and written together, which keeps the database from becoming the
        bottleneck of the training loop.
        """
        self._pending_metrics[name] = (
            self.step if step is None else int(step),
            float(value),
        )

    def _flush_metrics(self) -> None:
        """Write the queued metrics in one transaction."""
        if not self._pending_metrics:
            return
        pending = self._pending_metrics
        self._pending_metrics = {}
        rows: dict[int, dict[str, float]] = {}
        for name, (step, value) in pending.items():
            rows.setdefault(int(step), {})[name] = value
        for step, values in rows.items():
            try:
                self.store.log_metrics(self.run_id, step, values)
            except Exception as exc:  # never let the database kill a run
                self._log("WARNING", f"could not write metrics: {exc}")

    # -- control -----------------------------------------------------------
    def _control(self) -> str | None:
        if not self.control_path.exists():
            return None
        try:
            data = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("stop"):
            return "stop"
        if data.get("pause"):
            return "pause"
        return None

    # -- random state ------------------------------------------------------
    def _snapshot_rng(self) -> dict:
        """Everything needed to continue a run with the same rollouts."""
        state: dict[str, object] = {}
        try:
            generator = self.rng.bit_generator.state
            state["numpy"] = json.loads(json.dumps(generator, default=_rng_default))
        except Exception as exc:  # pragma: no cover - generator specific
            self._warn(f"numpy RNG state could not be saved: {exc}")
        try:
            state["torch_cpu"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:  # pragma: no cover - device specific
            self._warn(f"torch RNG state could not be saved: {exc}")
        try:
            state["python"] = list(random.getstate()[1])
        except Exception as exc:  # pragma: no cover
            self._warn(f"python RNG state could not be saved: {exc}")
        return state

    def _restore_rng(self, state: dict) -> None:
        numpy_state = state.get("numpy")
        if isinstance(numpy_state, dict):
            try:
                self.rng.bit_generator.state = numpy_state
            except Exception as exc:
                self._warn(f"numpy RNG state could not be restored: {exc}")
        torch_state = state.get("torch_cpu")
        if isinstance(torch_state, torch.Tensor):
            try:
                torch.set_rng_state(torch_state.to(dtype=torch.uint8, device="cpu"))
            except Exception as exc:
                self._warn(f"torch RNG state could not be restored: {exc}")
        cuda_states = state.get("torch_cuda")
        if isinstance(cuda_states, (list, tuple)) and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(
                    [tensor.to(dtype=torch.uint8, device="cpu") for tensor in cuda_states]
                )
            except Exception as exc:  # pragma: no cover - device specific
                self._warn(f"CUDA RNG state could not be restored: {exc}")
        python_state = state.get("python")
        if isinstance(python_state, (list, tuple)) and python_state:
            try:
                random.setstate((3, tuple(int(value) for value in python_state), None))
            except Exception as exc:  # pragma: no cover
                self._warn(f"python RNG state could not be restored: {exc}")

    # -- lifecycle ---------------------------------------------------------
    def _resolve_plan(self) -> ResumePlan | None:
        """Build (or reuse) the resume plan for this trainer.

        ``continue`` (the default) always asks :func:`resume.plan_resume` to
        discover the newest usable checkpoint, even when the caller supplied no
        explicit reference.  A ``fresh`` run still builds a (read-only) plan so
        the log can state exactly which checkpoints were deliberately ignored.
        """
        if self.resume_plan is not None:
            return self.resume_plan
        return plan_resume(
            self.models_dir,
            self.config,
            mode=self.resume_mode,
            run_id=self.run_id,
            checkpoint=self.resume_from,
            explicit_overrides=self.config_override_fields,
        )

    def _load_resume(self) -> None:
        plan = self._resolve_plan()
        if plan is None:
            return
        self.resume_plan = plan
        for warning in plan.warnings:
            self._warn(warning)
        # Informational plan notes (e.g. "--steps N" resolved to an absolute
        # target) are not warnings: they describe what was asked for.
        for note in getattr(plan, "notes", []) or []:
            self._log("INFO", note)
        if not plan.will_resume:
            reasons = "; ".join(
                reason
                for item in plan.rejected
                for reason in item.get("reasons", [])
            ) or "; ".join(plan.warnings) or "no compatible checkpoint found"

            # An explicit fresh request always wins and is named as such in
            # the log; it is the only way to deliberately ignore progress.
            if self.resume_mode == "fresh":
                if plan.candidates_found:
                    self._log(
                        "INFO",
                        f"fresh training requested: {plan.candidates_found} existing "
                        f"checkpoint(s) in {self.models_dir} deliberately ignored",
                    )
                else:
                    self._log(
                        "INFO",
                        f"fresh training requested and no checkpoint exists in "
                        f"{self.models_dir}; starting a new policy at step 0",
                    )
                return

            if self.resume_from:
                if explicit_checkpoint_exists(self.models_dir, self.resume_from):
                    raise ValueError(
                        f"cannot resume from {self.resume_from!r}: {reasons}"
                    )
                raise FileNotFoundError(
                    f"no usable checkpoint for {self.resume_from!r}: {reasons}"
                )

            if self.resume_mode == "transfer":
                raise ValueError(
                    "transfer learning needs a checkpoint, but none is usable: "
                    + reasons
                    + "; use resume_mode='fresh' for a new policy"
                )

            # ``continue`` is the automatic default.  No checkpoint at all is
            # a normal first start; existing but unusable state is not.  In
            # the latter case starting a fresh model would silently discard
            # progress, so the run is refused with the validation reasons.
            if plan.candidates_found:
                raise ValueError(
                    f"automatic resume found {plan.candidates_found} checkpoint(s) in "
                    f"{self.models_dir}, but none is usable ({reasons}). Refusing to "
                    "start a fresh model and lose that progress; name a checkpoint "
                    "with resume_from, repair the files, or start deliberately with "
                    "resume_mode='fresh'."
                )
            self._log(
                "INFO",
                f"automatic resume: no checkpoint found in {self.models_dir}; "
                "starting a new policy from scratch (fresh training)",
            )
            self.resume_mode = "fresh"
            return

        selection = "requested checkpoint" if self.resume_from else "resume"
        self._log(
            "INFO",
            f"{selection} selected {plan.model_id or plan.checkpoint_path} "
            f"at step {plan.parent_step}",
        )
        path = Path(str(plan.checkpoint_path))
        payload = checkpoints.load_checkpoint(path)
        saved_config = PolicyConfig.from_dict(payload.get("policy_config"))
        target_config = self.config.policy_config()
        if saved_config.to_dict() != target_config.to_dict():
            if plan.mode == "transfer":
                # A transfer may use a different architecture as long as the
                # observation and the action layout match: the layers whose
                # shape changed are re-initialised, the rest is reused.
                self._load_transfer_weights(payload, saved_config, target_config)
            else:
                raise ValueError(
                    "checkpoint policy architecture does not match this run: "
                    f"{saved_config.to_dict()} vs {target_config.to_dict()}"
                )
        else:
            self.model.load_state_dict(payload["model"])
        # The accumulated training time is restored after the step counter
        # is known (below), because the inherited session belongs to the
        # steps that were already trained.
        saved_timing = (payload.get("extra") or {}).get("timing")
        self._saved_timing = saved_timing if isinstance(saved_timing, dict) else None

        explicit = tuple(plan.explicit_overrides or ()) or self.config_override_fields
        merged, overrides, inherited = merge_config(
            payload.get("training_config"),
            self.config,
            mode=plan.mode,
            explicit_overrides=explicit,
        )
        self.resume_overrides = overrides
        if overrides:
            for override in overrides:
                if "error" in override:
                    self._warn(str(override["error"]))
                else:
                    self._log(
                        "INFO",
                        "config field "
                        f"{override['field']!r}: using {override['used']!r} "
                        f"(checkpoint had {override['checkpoint']!r}, request asked for "
                        f"{override['requested']!r}, source={override.get('source', 'inherited')})",
                    )
        phase_change = [
            item["field"] for item in overrides if item.get("source") == "explicit"
        ]
        if phase_change:
            self._warn(
                "phase change: continuing these weights with changed "
                f"{', '.join(phase_change)}; value estimates and optimizer "
                "moments were fitted to the old settings and will re-adapt "
                "over the next iterations"
            )
        self.config = merged
        self.resume_inherited = inherited

        if plan.mode == "transfer":
            self._log(
                "INFO",
                f"transfer learning from {path} (weights only, step counter restarts at 0)",
            )
            self.step = 0
            # A transfer starts a new training lineage: the weights are
            # reused, the training time is not.
            self.clock.restore(None)
            return

        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            self._warn("checkpoint has no optimizer state; starting a fresh optimizer")
        self.step = int(payload.get("step", 0))
        self.start_step = self.step
        # The restored checkpoint already contains this exact state.  Marking
        # its step prevents a redundant copy at the first evaluation or at
        # completion when the run performs no further step; the final save
        # relabels that checkpoint instead of storing the same weights twice.
        if plan.model_id:
            self.last_checkpoint_step = self.step
            self.last_checkpoint_id = str(plan.model_id)
        extra = payload.get("extra") or {}
        self.iteration = int(extra.get("iteration", 0) or 0)
        if isinstance(extra.get("league"), dict):
            self.league = OpponentPool.from_dict(extra["league"])
            self._log(
                "INFO",
                f"restored opponent pool from checkpoint: {len(self.league)} entries, "
                f"{self.league.usable} playable, agent Elo {self.league.agent_elo:.1f}",
            )
            if len(self.league) and not self.league.usable:
                self._log(
                    "WARNING",
                    "pool snapshots carry ratings but no weights (checkpoint written before "
                    "league_states existed, or league_states=false); self-play uses the fixed "
                    "opponent until the next snapshot is added",
                )
            elif self.league.usable and self.league.usable < len(self.league):
                self._log(
                    "WARNING",
                    f"only {self.league.usable} of {len(self.league)} pool entries carry weights; "
                    "the others count for Elo history only",
                )
        if self.curriculum is not None and isinstance(extra.get("curriculum"), dict):
            saved = extra["curriculum"]
            stages = stages_for_mode(self.config.mode, self.config.curriculum_preset)
            # Restore by stage *name*, not by position: a checkpoint written
            # under a different preset (or an older stage list) would otherwise
            # silently continue at whatever stage happens to have that index.
            saved_name = str((saved.get("stage") or {}).get("name") or "")
            names = [stage.name for stage in stages]
            if saved_name and saved_name in names:
                start_index = names.index(saved_name)
            else:
                start_index = int(saved.get("index", 0))
                if saved_name:
                    self._warn(
                        f"checkpoint curriculum stage {saved_name!r} does not exist in the "
                        f"{self.config.curriculum_preset!r} stage list {names}; restoring "
                        f"position {start_index} ({names[min(start_index, len(names) - 1)]}) "
                        "instead - gate confirmations and retention baselines are kept"
                    )
            self.curriculum = Curriculum(
                stages,
                start_index=start_index,
                stage_iterations=int(saved.get("iterations", 0)),
                total_iterations=int(saved.get("total_iterations", 0)),
                window_episodes=saved.get("window") if isinstance(saved.get("window"), list) else None,
                # gate measurements, confirmation counts and retention
                # baselines survive the checkpoint: a resume must not get a
                # second chance at a gate it already failed (or passed)
                gate_state=saved.get("gate_state"),
            )
            self._log(
                "INFO",
                f"restored curriculum stage {self.curriculum.index} "
                f"({self.curriculum.stage.name}); blocked: "
                f"{self.curriculum.blocked_reason() or 'no'}",
            )
        elif self.curriculum is not None:
            self._warn("checkpoint has no curriculum state; the curriculum restarts at stage 0")
        if isinstance(extra.get("rng"), dict) and extra["rng"]:
            self._restore_rng(extra["rng"])
            self._log("INFO", "restored RNG state from checkpoint")
        else:
            self._warn(
                "checkpoint has no RNG state (older format); rollouts after the resume are "
                "reproducible but not identical to an uninterrupted run"
            )
        if isinstance(extra.get("adaptive"), dict) and extra["adaptive"]:
            controller = self._adaptive_controller()
            if controller is not None:
                controller.restore(extra["adaptive"])
                self._log("INFO", "restored the adaptive match dynamics state")
        # learning-rate annealing depends on the step counter; make the
        # optimiser agree with the restored position immediately
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate_at(self.config, self.step)
        self._restore_training_time(plan)
        self._log(
            "INFO",
            f"resumed from {path} at step {self.step} (iteration {self.iteration}, "
            f"parent run {plan.parent_run or 'unknown'})",
        )
        try:
            self.store.update_run(
                self.run_id,
                parent_run=plan.parent_run or "",
                resumed_step=self.start_step,
            )
        except Exception:  # pragma: no cover - older databases without the column
            pass

    def _load_transfer_weights(
        self,
        payload: dict,
        saved_config: PolicyConfig,
        target_config: PolicyConfig,
    ) -> None:
        """Load the weights of a checkpoint into a *different* architecture.

        Only the layers whose shape matches are reused; the rest is
        re-initialised and the run says so.  The observation size and the
        action layout have to match, otherwise the weights would describe a
        different problem.
        """
        if saved_config.obs_size != target_config.obs_size:
            raise ValueError(
                "the checkpoint was trained on a different observation layout "
                f"({saved_config.obs_size} vs {target_config.obs_size} values); "
                "a transfer needs the same observation"
            )
        source = payload.get("model") or {}
        target = self.model.state_dict()
        reused: list[str] = []
        skipped: list[str] = []
        new_state = {}
        for name, tensor in target.items():
            other = source.get(name)
            if other is not None and hasattr(other, "shape") and other.shape == tensor.shape:
                new_state[name] = other
                reused.append(name)
            else:
                new_state[name] = tensor
                skipped.append(name)
        self.model.load_state_dict(new_state)
        self._log(
            "INFO",
            f"transfer learning: reused {len(reused)} of {len(target)} tensors, "
            f"re-initialised {len(skipped)} (the architecture changed)",
        )
        if skipped:
            self._warn(
                "the transferred checkpoint had a different architecture; "
                f"{len(skipped)} layer(s) start from scratch: {', '.join(skipped[:6])}"
            )

    def _restore_training_time(self, plan: ResumePlan) -> None:
        """Restore the accumulated training time of the continued run.

        A checkpoint without duration metadata (an older format) leaves the
        total unknown instead of inventing a value; the run then starts at
        zero and the report says so.
        """
        saved_timing = getattr(self, "_saved_timing", None)
        if not isinstance(saved_timing, dict) or saved_timing.get("accumulated_s") is None:
            self.clock.restore(None)
            self._warn(
                "the checkpoint carries no training-time metadata; the accumulated "
                "training time of this run starts at zero and the historical "
                "duration is reported as unknown"
            )
            return
        self.clock.restore(float(saved_timing["accumulated_s"]))
        self._log(
            "INFO",
            f"restored accumulated training time: "
            f"{self.clock.accumulated_s:.1f}s from the checkpoint",
        )
        # Make the inherited time visible in the session history, so the
        # total of the run is complete and not only the last session.
        parent = plan.parent_run or "an earlier run"
        inherited_id = self.store.start_session(
            self.run_id,
            started_at=saved_timing.get("session_started_at") or utc_now_iso(),
            start_step=0,
            pid=os.getpid(),
        )
        self.store.finish_session(
            inherited_id,
            ended_at=saved_timing.get("measured_at") or utc_now_iso(),
            duration_s=float(self.clock.accumulated_s),
            status="inherited",
            end_step=int(self.start_step),
            note=f"inherited from run {parent}",
        )

    def _save(self, *, label: str = "", force: bool = False) -> dict | None:
        """Write one checkpoint, or reuse the one at the current step.

        A final save must not create a second copy of weights that were already
        written at the same step (for example by the final evaluation).  It
        relabels the existing checkpoint instead; the model file stays the
        single source of truth and no duplicate directory is left behind.
        """
        same_step = (
            self.step == self.last_checkpoint_step and self.last_checkpoint_id is not None
        )
        if same_step:
            if force and label in ("final", "interrupt") and self.last_checkpoint_id:
                try:
                    existing = checkpoints.get_model(
                        self.models_dir, self.last_checkpoint_id
                    )
                    if not (self.models_dir / str(existing["path"])).is_file():
                        raise FileNotFoundError(existing["path"])
                    manifest = checkpoints.set_label(
                        self.models_dir, self.last_checkpoint_id, label
                    )
                    self.store.sync_models([manifest])
                    self.final_manifest = manifest
                    self._log(
                        "INFO",
                        f"checkpoint {manifest['id']} at step {self.step} "
                        f"reused as the {label} checkpoint",
                    )
                    return manifest
                except Exception as exc:
                    self._log(
                        "WARNING",
                        f"could not relabel checkpoint {self.last_checkpoint_id}: {exc}",
                    )
            if not force:
                return None
            # A shutdown save at a step that was already written needs no new
            # file; the existing checkpoint already contains this state.
            if self.last_checkpoint_id:
                try:
                    manifest = checkpoints.get_model(
                        self.models_dir, self.last_checkpoint_id
                    )
                except (KeyError, OSError):
                    manifest = None
                if manifest is not None and (
                    self.models_dir / str(manifest.get("path") or "")
                ).is_file():
                    self.final_manifest = manifest
                    return manifest

        started = time.perf_counter()
        # The accumulated training time is measured with the monotonic clock
        # and covers the sessions that actually ran - never the time the
        # process was stopped.
        accumulated = self.clock.accumulated_s + (self.clock.current_session_s or 0.0)
        manifest = None
        # A Ctrl+C arriving while the file is written is deferred until the
        # checkpoint, the pointer and the bookkeeping are all consistent.
        with self._interrupt_guard.deferred():
            manifest = save_checkpoint(
                self.models_dir,
                model_state=self.model.state_dict(),
                policy_config=self.config.policy_config().to_dict(),
                optimizer_state=self.optimizer.state_dict(),
                step=self.step,
                run_id=self.run_id,
                mode=self.config.mode,
                training_config=self.config.to_dict(),
                extra={
                    "iteration": self.iteration,
                    "obs_version": self.config.obs_version,
                    "league": self.league.to_dict(include_states=self.config.league_states),
                    "curriculum": self.curriculum.describe() if self.curriculum else None,
                    "rng": self._snapshot_rng(),
                    "parent_run": (self.resume_plan.parent_run if self.resume_plan else None),
                    "resume_mode": self.resume_mode,
                    "start_step": self.start_step,
                    "side_randomization": bool(self.config.side_randomization),
                    "timing": {
                        "accumulated_s": float(accumulated),
                        "session_s": self.clock.current_session_s,
                        "session_started_at": (
                            self.clock.timer.started_at if self.clock.timer else None
                        ),
                        "measured_at": utc_now_iso(),
                        "source": "monotonic clock",
                    },
                    "adaptive": (
                        self._adaptive.state() if self._adaptive is not None else None
                    ),
                },
                label=label,
            )
            self.store.register_model(manifest)
            self.last_checkpoint_step = self.step
            self.last_checkpoint_id = manifest["id"]
            self.final_manifest = manifest
            self.store.update_run(
                self.run_id,
                last_checkpoint_at=manifest.get("created_at"),
                last_checkpoint_id=manifest["id"],
            )
            try:
                prune_candidates(
                    self.models_dir,
                    keep=self.config.keep_checkpoints,
                    run_id=self.run_id,
                )
            except Exception as exc:  # never let cleanup kill a run
                self._log("WARNING", f"checkpoint cleanup skipped: {exc}")
        self.checkpoint_seconds += time.perf_counter() - started
        self._metric("time/checkpoint_seconds", time.perf_counter() - started)
        self._log("INFO", f"checkpoint {manifest['id']} saved at step {self.step}")
        return manifest

    # -- adaptive match dynamics --------------------------------------------
    def _adaptive_controller(self):
        """The controller that may adapt the scripted opponent.

        It is created lazily and only when the feature is enabled: a
        controller that is never used must not change anything.

        The adaptation needs a difficulty level to start from, because the
        level defines the opponent's execution quality.  A run against a
        plain scripted opponent therefore gets a warning instead of a
        controller - the opponent's behaviour would otherwise be replaced
        by a difficulty profile.
        """
        if not self.config.adaptive_dynamics:
            return None
        if getattr(self, "_adaptive", None) is not None:
            return self._adaptive
        opponent = str(self.config.opponent or "")
        level = opponent.split(":", 1)[1].strip() if opponent.startswith("difficulty:") else ""
        if not level:
            self._warn(
                "the adaptive match dynamics need a difficulty opponent "
                f"(opponent='difficulty:medium'); the configured opponent "
                f"{opponent!r} keeps its own behaviour and is not adapted"
            )
            return None
        from .adaptive import AdaptationLimits, AdaptationPolicy, AdaptiveController

        self._adaptive = AdaptiveController(
            level,
            policy=AdaptationPolicy(
                target_win_rate=float(self.config.adaptive_target_win_rate),
                tolerance=float(self.config.adaptive_tolerance),
                min_games_between_changes=int(self.config.adaptive_min_games),
                min_seconds_between_changes=float(self.config.adaptive_min_seconds),
                step=float(self.config.adaptive_step),
                enabled=True,
            ),
            limits=AdaptationLimits(),
            monotonic=time.monotonic,
        )
        return self._adaptive

    def _opponent_execution(self) -> dict | None:
        """The adapted execution parameters of the scripted opponent.

        They are only sent when the adaptive match dynamics are enabled and
        the controller has actually changed something - a trained policy is
        never touched.
        """
        controller = getattr(self, "_adaptive", None)
        if controller is None:
            return None
        return dict(controller.current)

    def _apply_adaptation(self, episodes: list[dict]) -> dict | None:
        """Feed the measured episodes to the adaptive controller.

        Only the *scripted* opponent is adapted, and only inside the
        configured limits.  A trained policy is never modified.
        """
        controller = self._adaptive_controller()
        if controller is None or not episodes:
            return None
        games = len(episodes)
        wins = sum(1 for episode in episodes if episode.get("result") == "win")
        result = controller.observe(wins=wins, games=games, at=time.monotonic())
        if result.get("adapted"):
            self._log(
                "INFO",
                f"adaptive dynamics: {result['reason']}; the scripted opponent now "
                f"uses mistake_rate={result['parameters']['mistake_rate']:.2f}, "
                f"reaction_delay={result['parameters']['reaction_delay']:.2f}s, "
                f"boost_efficiency={result['parameters']['boost_efficiency']:.2f}",
            )
            self._metric("adaptive/mistake_rate", result["parameters"]["mistake_rate"])
            self._metric("adaptive/reaction_delay", result["parameters"]["reaction_delay"])
            self._metric("adaptive/boost_efficiency", result["parameters"]["boost_efficiency"])
        return result

    # -- player memory ------------------------------------------------------
    def _remember_players(self, episodes: list[dict]) -> None:
        """Store what the agent learned about the players it met.

        The memory is data about past encounters; it is kept in its own
        table and never becomes part of the model weights.
        """
        if not self.config.player_memory or not episodes:
            return
        from .player_memory import memory_from_rows, rows_from_memory

        memory = memory_from_rows(self.store.list_player_memory())
        for episode in episodes:
            name = episode.get("opponent") or "unknown"
            memory.remember(
                name,
                won=(
                    True
                    if episode.get("result") == "win"
                    else False
                    if episode.get("result") == "loss"
                    else None
                ),
                goals_for=int(episode.get("goals_for") or 0),
                goals_against=int(episode.get("goals_against") or 0),
                tag="match" if episode.get("policy_team") is not None else None,
            )
        memory.prune()
        self.store.replace_player_memory(rows_from_memory(memory))

    # -- retention ----------------------------------------------------------
    def _maybe_recovery_checkpoint(self) -> dict | None:
        """Write a bounded ``recovery`` checkpoint every few minutes.

        The documented checkpoint policy writes a checkpoint only at the end
        of a run or on a graceful stop, so a hard crash (power loss, SIGKILL,
        kernel panic) loses every step since the last run ended.  For a
        multi-hour run that is unacceptable, so this adds a *bounded* net:
        one checkpoint every ``recovery_checkpoint_minutes`` of wall time,
        pruned by the normal ``keep_checkpoints`` retention.

        It is deliberately not a "routine checkpoint every N iterations":
        that design produced about 10,000 directories for a 50M-step run.
        """
        minutes = float(getattr(self.config, "recovery_checkpoint_minutes", 0.0) or 0.0)
        if minutes <= 0:
            return None
        now = time.monotonic()
        if now - self._last_recovery_at < minutes * 60.0:
            return None
        self._last_recovery_at = now
        manifest = self._save(label="recovery")
        if manifest is None:
            return None
        self._log(
            "INFO",
            f"recovery checkpoint {manifest['id']} at step {self.step} "
            f"(every {minutes:g} min of wall time; a hard crash resumes from here)",
        )
        return manifest

    def _apply_retention(self) -> None:
        """Bound the size of the metric, log and episode tables.

        The newest rows are kept, so the charts and the report still
        describe the end of the run.  Pruning is a background chore: a
        failure here must never end a run.
        """
        try:
            removed = 0
            if self.config.metric_retention:
                removed += self.store.prune_metrics(self.run_id, self.config.metric_retention)
            if self.config.episode_retention:
                removed += self.store.prune_episodes(
                    self.run_id,
                    self.config.episode_retention,
                    # keep the beginning of the run: the report compares the
                    # first and last episode window to judge learning
                    protect_first=report_module.LEARNING_EVIDENCE_WINDOW,
                )
            if self.config.log_retention:
                removed += self.store.prune_logs(self.run_id, self.config.log_retention)
            if removed:
                self._log("INFO", f"retention removed {removed} old rows")
        except Exception as exc:  # pragma: no cover - defensive
            self._log("WARNING", f"retention cleanup skipped: {exc}")

    # -- time accounting ----------------------------------------------------
    def _start_session(self) -> None:
        """Open the training session and persist its start.

        The session row is written *before* the first step, so a crash
        leaves a trace that the recovery path can find.
        """
        started_at = utc_now_iso()
        self.session_id = self.store.start_session(
            self.run_id,
            started_at=started_at,
            start_step=self.step,
            pid=os.getpid(),
        )
        self.clock.start_session(start_step=self.step, session_id=self.session_id)
        timer = self.clock.timer
        assert timer is not None
        self.store.update_run(
            self.run_id,
            started_at=timer.started_at,
            duration_s=self.clock.accumulated_s,
        )

    def _finish_session(self, status: str, *, note: str = "") -> None:
        """Close the session and store its measured duration."""
        if self.clock.timer is None:
            # already closed (or never opened)
            return
        record = self.clock.stop_session(status, end_step=self.step, note=note)
        if self.session_id is not None:
            self.store.finish_session(
                self.session_id,
                ended_at=record.ended_at or utc_now_iso(),
                duration_s=record.duration_s,
                status=record.status,
                end_step=self.step,
                note=record.note,
            )
        self.store.update_run(
            self.run_id,
            duration_s=self.clock.accumulated_s,
            session_duration_s=record.duration_s,
        )
        self._log(
            "INFO",
            f"session {record.status} after {record.duration_s or 0.0:.1f}s "
            f"(accumulated training time {self.clock.accumulated_s:.1f}s)",
        )

    def _recover_interrupted_sessions(self) -> None:
        """Close sessions that a previous process left open.

        A crashed process cannot measure its own duration, so the session
        is recorded as *interrupted* with an unknown duration.  The
        accumulated total comes from the checkpoint, which is the last place
        a measured value was written.
        """
        recovered = self.store.recover_interrupted_sessions(
            self.run_id,
            note="recovered after the process ended without closing the session",
        )
        if recovered:
            self._warn(
                f"{recovered} session(s) of this run were left open by a previous "
                "process and are recorded as interrupted; their duration is unknown"
            )

    # -- resource limits ----------------------------------------------------
    def _apply_resource_limits(self) -> None:
        """Clamp the worker count to what this machine can actually run.

        More workers do not automatically mean more throughput: every extra
        RocketSim arena costs memory and a core, and past the core count the
        workers simply take turns.  The clamp is applied *before* the pool
        starts, and the decision is recorded as a warning so it shows up in
        the report and in the GUI.
        """
        from .resources import plan_resources

        plan = plan_resources(self.config.n_workers)
        for warning in plan.warnings:
            self._warn(warning)
        for note in plan.notes:
            self._log("INFO", note)
        if plan.clamped and self.config.auto_worker_limit:
            self._log(
                "WARNING",
                f"reducing workers from {self.config.n_workers} to "
                f"{plan.recommended_workers} for this machine "
                f"({plan.cpu_count} logical cores, "
                f"{plan.memory_available_mb / 1024:.1f} GB free)",
            )
            self.config = self.config.model_copy(
                update={"n_workers": int(plan.recommended_workers)}
            )
        self.resource_plan = plan

    # -- opponent selection --------------------------------------------------
    def _select_opponents(
        self,
    ) -> tuple[list[str | None], list, list[str | None], str, str, float]:
        """Per-worker (names, weights, pool ids, stage, reset mode, seconds).

        Every worker draws its own opponent: in `mixed` stages each worker
        independently faces chase or random, in self-play each worker gets
        its own PFSP snapshot draw.  A whole batch against a single
        opponent is correlated data (and one lucky/unlucky snapshot could
        dominate an entire PPO update).  The reset distribution and episode
        horizon belong to the stage: a touch drill wants randomized close
        starts and short episodes, a match stage wants longer ones.
        """
        count = max(1, int(self.config.n_workers))
        seconds = float(self.config.episode_seconds)
        if self.curriculum is not None:
            stage = self.curriculum.stage
            reset_mode = stage.reset_mode
            if stage.episode_seconds is not None:
                seconds = float(stage.episode_seconds)
            if stage.opponent == "self" and self.league.usable:
                picks = [self.league.sample(self.rng) for _ in range(count)]
                return (
                    [None] * count,
                    [entry.state for entry in picks],
                    [entry.id for entry in picks],
                    stage.name, reset_mode, seconds,
                )
            if stage.opponent == "mixed":
                names = [str(self.rng.choice(["chase", "random"])) for _ in range(count)]
                return names, [None] * count, [None] * count, stage.name, reset_mode, seconds
            if stage.opponent == "self":
                # The pool has no playable snapshot yet (first iteration, or a
                # checkpoint written without league states).  "self" is not a
                # buildable opponent, so fall back to a scripted one instead
                # of handing the worker a name it cannot resolve.
                fallback = self._self_play_fallback()
                return [fallback] * count, [None] * count, [None] * count, stage.name, reset_mode, seconds
            return [stage.opponent] * count, [None] * count, [None] * count, stage.name, reset_mode, seconds
        reset_mode = str(self.config.reset_mode)
        if self.config.self_play and self.league.usable:
            picks = [self.league.sample(self.rng) for _ in range(count)]
            return (
                [None] * count,
                [entry.state for entry in picks],
                [entry.id for entry in picks],
                "self-play", reset_mode, seconds,
            )
        if self.config.opponent == "self":
            fallback = self._self_play_fallback()
            return [fallback] * count, [None] * count, [None] * count, "fixed", reset_mode, seconds
        return [self.config.opponent] * count, [None] * count, [None] * count, "fixed", reset_mode, seconds

    def _self_play_fallback(self) -> str:
        """Scripted opponent to use while no self-play snapshot exists yet."""
        fallback = "chase" if self.config.opponent in ("self", "chase") else self.config.opponent
        if fallback not in ("idle", "random", "chase") and not fallback.startswith("model:"):
            fallback = "chase"
        self._warn(
            "self-play requested but the opponent pool has no playable snapshot; "
            f"using the scripted {fallback!r} opponent until the first snapshot exists"
        )
        return fallback

    def _policy_teams(self) -> list[int] | None:
        """Per-worker team assignment (``None`` keeps everybody on blue)."""
        if not self.config.side_randomization:
            return None
        count = max(1, int(self.config.n_workers))
        if count == 1:
            # with a single worker alternate between the sides per iteration
            return [int(self.iteration % 2)]
        teams = [int(index % 2) for index in range(count)]
        # deterministic shuffle so a resumed run continues the same pattern
        self.rng.shuffle(teams)
        return teams

    # -- curriculum gate evaluation ------------------------------------------
    def _gate_measurements(self) -> dict[str, dict]:
        """Measure every skill the active curriculum currently needs.

        Probes are state-based and independent of the training reward; the
        ``match_*`` skills are real match evaluations against a fixed
        opponent.  Skills mastered in *earlier* stages are measured in the
        same call so the retention comparison uses one protocol.
        """
        from .evaluation import evaluate_policy
        from .skills import MATCH_SKILLS, evaluate_skills, match_measurement

        assert self.curriculum is not None
        skills = self.curriculum.required_skills()
        policy = TorchPolicy(self.model, device=self.device)
        measurements: dict[str, dict] = {}
        probes = [skill for skill in skills if skill not in MATCH_SKILLS]
        try:
            if probes:
                suite = evaluate_skills(
                    policy,
                    probes,
                    mode=self.config.mode,
                    trials=int(self.config.curriculum_gate_trials),
                    seed=int(self.config.seed) + 770_001,
                    sides=("blue", "orange"),
                    mirror=bool(self.config.eval_mirror),
                    obs_version=self.config.obs_version,
                    device=str(self.device),
                )
                measurements.update(suite["results"])
                self.gate_probe_seconds += float(suite.get("duration_s") or 0.0)
            for skill in skills:
                if skill not in MATCH_SKILLS:
                    continue
                result = evaluate_policy(
                    policy,
                    mode=self.config.mode,
                    opponent=MATCH_SKILLS[skill],
                    games=int(self.config.eval_games),
                    seed=int(self.config.seed) + 880_001,
                    match_seconds=float(self.config.eval_match_seconds),
                    episode_seconds=min(float(self.config.eval_match_seconds), 30.0),
                    device=str(self.device),
                    obs_version=self.config.obs_version,
                    sides=tuple(self.config.eval_sides),
                    mirror=bool(self.config.eval_mirror),
                )
                measurements[skill] = match_measurement(result)
        finally:
            self.model.train()
        return measurements

    def _run_gate_evaluation(self) -> dict | None:
        """One independent gate evaluation; may promote, block or demote."""
        if self.curriculum is None or self.curriculum.finished:
            return None
        stage_name = self.curriculum.stage.name
        started = time.perf_counter()
        measurements = self._gate_measurements()
        seconds = time.perf_counter() - started
        self.gate_seconds += seconds
        transition = self.curriculum.record_gate(measurements)
        for key, value in self.curriculum.metrics().items():
            self._metric(key, value)
        self._metric("time/gate_seconds", seconds)
        summary = ", ".join(
            f"{gate.skill}={float(measurement.get('rate') or 0.0):.2f}"
            f"[n={int(measurement.get('trials') or 0)}]"
            for gate in self.curriculum.stage.gates
            for measurement in [measurements.get(gate.skill) or {}]
        )
        self._log(
            "INFO",
            f"gate[{stage_name}] {summary or 'no gates'} in {seconds:.1f}s -> "
            f"{self.curriculum.gates.passes_in_a_row}/"
            f"{self.curriculum.stage.confirmations} confirmations; blocked by: "
            f"{self.curriculum.blocked_reason() or 'nothing'}",
        )
        event = self.curriculum.last_event
        if transition and event == "promotion":
            self._log_curriculum_transition("promotion")
        elif event == "regression":
            failing = [
                row
                for row in self.curriculum.retention_rows()
                if not row["ok"]
            ]
            self._warn(
                "skill regression: "
                + "; ".join(f"{row['name']} {row['reason']}" for row in failing)
                + f" - stage {stage_name!r} is NOT counted as mastered; the "
                "confirmation counter was reset"
            )
        self._flush_metrics()
        return measurements

    def _log_curriculum_transition(self, event: str) -> None:
        stage = self.curriculum.stage
        assert self.curriculum is not None
        self._log(
            "INFO",
            f"curriculum {event} -> {stage.name} "
            f"(opponent {stage.opponent}, resets {stage.reset_mode}, "
            f"episodes {stage.episode_seconds or self.config.episode_seconds:.0f}s; "
            f"gates {[gate.name for gate in stage.gates] or 'window criteria'}; "
            f"blocked by: {self.curriculum.blocked_reason() or 'nothing'})",
        )

    # -- evaluation ----------------------------------------------------------
    def _evaluation_opponents(self) -> list[str]:
        if self.config.eval_opponents:
            return list(self.config.eval_opponents)
        return [self.config.eval_opponent]

    def _run_evaluation(self, phase: str = "periodic") -> dict | None:
        from .evaluation import evaluate_policy, evaluate_suite

        started = time.perf_counter()
        opponents = self._evaluation_opponents()
        policy = TorchPolicy(self.model, device=self.device)
        common = dict(
            mode=self.config.mode,
            games=self.config.eval_games,
            # Fixed seed on purpose: the same kickoffs every evaluation, so
            # results stay inside ONE protocol and remain comparable across
            # checkpoints (a rotating seed silently created a new protocol -
            # and a new best-pointer file - on every single evaluation).
            seed=int(self.config.seed) + 900_001,
            match_seconds=self.config.eval_match_seconds,
            episode_seconds=min(self.config.eval_match_seconds, 30.0),
            device=str(self.device),
            obs_version=self.config.obs_version,
            sides=tuple(self.config.eval_sides),
            mirror=bool(self.config.eval_mirror),
        )
        try:
            if len(opponents) > 1:
                suite = evaluate_suite(policy, opponents=opponents, **common)
                result = self._flatten_suite(suite)
            else:
                result = evaluate_policy(policy, opponent=opponents[0], **common)
        finally:
            self.model.train()
        seconds = time.perf_counter() - started
        self.evaluation_seconds += seconds
        self._metric("time/eval_seconds", seconds)
        # the evaluation metrics belong to this step, so they are written
        # together with the rest of the iteration
        self._flush_metrics()

        result["phase"] = phase
        if phase == "before":
            self.evaluated_before = True
        for key, value in (
            ("eval/win_rate", result.get("win_rate")),
            ("eval/win_rate_low", (result.get("win_rate_ci95") or [None, None])[0]),
            ("eval/win_rate_high", (result.get("win_rate_ci95") or [None, None])[1]),
            ("eval/goals_for", result.get("goals_for")),
            ("eval/goals_against", result.get("goals_against")),
            ("eval/goal_diff", result.get("goal_diff")),
            ("eval/games", result.get("games")),
            ("eval/touches", float(result.get("touches") or 0.0)),
            ("eval/draw_rate", (result.get("draws") or 0) / max(result.get("games") or 1, 1)),
            ("eval/seconds", seconds),
        ):
            if value is not None:
                self._metric(key, float(value))
        for name, per_opponent in (result.get("per_opponent") or {}).items():
            if per_opponent.get("win_rate") is not None:
                self._metric(f"eval/{name}/win_rate", float(per_opponent["win_rate"]))
                self._metric(f"eval/{name}/goal_diff", float(per_opponent.get("goal_diff") or 0))

        protocol = result.get("protocol") or {}
        self._log(
            "INFO",
            f"eval[{phase}] vs {result.get('opponent')}: {result.get('wins')}W "
            f"{result.get('losses')}L {result.get('draws')}D goals "
            f"{result.get('goals_for')}:{result.get('goals_against')} "
            f"(95% CI {(result.get('win_rate_ci95') or [0, 0])[0]:.2f}-"
            f"{(result.get('win_rate_ci95') or [0, 0])[1]:.2f}) in {seconds:.1f}s",
        )
        # Training writes no routine periodic checkpoints.  A periodic
        # evaluation therefore keeps its metrics but has no model artefact to
        # attach them to.  Only the final evaluation of a completed run is
        # linked to the single final checkpoint (written at the same step,
        # either just before this call or by the fallback below).
        if phase == "after" and self.last_checkpoint_step != self.step:
            self._save(label="final", force=True)
        model_id = self.last_checkpoint_id if phase == "after" else None
        games_played = (
            int(result.get("wins") or 0)
            + int(result.get("losses") or 0)
            + int(result.get("draws") or 0)
        )
        if games_played:
            rows = result.get("rows") or [
                dict(result, model_id=model_id, run_id=self.run_id)
            ]
            for row in rows:
                row.setdefault("model_id", model_id)
                row.setdefault("run_id", self.run_id)
                row["phase"] = phase
                row.setdefault("protocol", protocol)
                self.store.add_evaluation(row)
        score = result.get("win_rate") or 0.0
        if games_played:
            self.run_best_score = max(self.run_best_score, float(score))
        if phase == "periodic" and games_played:
            self._track_eval_stall(result)
        if not games_played or model_id is None:
            return result

        self.store.set_model_eval_score(model_id, float(score))
        # A 0 % win rate is not a "best" model: keep the pointer empty until the
        # policy actually won a game, so the GUI never offers a checkpoint as
        # "best" that has never scored.  Scores are only compared inside one
        # protocol, so "best" always means "best under the same measurement" -
        # including against whatever a previous run left on disk.
        key = checkpoints.protocol_key(protocol or None)
        incumbent = float(
            checkpoints.best_record(self.models_dir, protocol=protocol or None).get(
                "score", float("-inf")
            )
        )
        best_score = max(incumbent, self._best_by_protocol.get(key, float("-inf")))
        if is_best_candidate(result, best_score):
            record = checkpoints.set_best(
                self.models_dir,
                model_id,
                float(score),
                metric="win_rate",
                protocol=protocol or None,
            )
            self._best_by_protocol[key] = float(record.get("score", score))
            self._log(
                "INFO",
                f"new best for protocol {key}: {model_id} (win rate "
                f"{float(score):.2f} vs {result.get('opponent')})",
            )
        return result

    def _save_after_interrupt(self, reason: str | None = None) -> tuple[bool, str]:
        """Stop the workers, then persist the latest resumable state.

        Called from the ``KeyboardInterrupt`` handler and from the graceful
        ``stop`` control path.  Closing the pool first releases the simulator
        processes; only then is the checkpoint written, so no worker can still
        be running while the state is snapshotted.  A failure to write is
        reported as such - the run must not claim to be resumable when its
        latest state is not on disk.
        """
        reason = reason or self._interrupt_guard.describe()
        self._interrupt_guard.begin_shutdown()
        self._warn(f"training interrupted by {reason}; stopping workers and saving state")
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception as exc:  # pragma: no cover - defensive
                self._warn(f"could not close all workers cleanly: {exc}")
        try:
            manifest = self._save(label="interrupt", force=True)
        except BaseException as exc:  # the shutdown path must still finish
            self.interrupt_checkpoint_error = f"{type(exc).__name__}: {exc}"
            self.resumable = False
            previous = self.last_checkpoint_id or "none"
            message = (
                f"interrupted by {reason}; the latest state could NOT be saved "
                f"({self.interrupt_checkpoint_error}); the previous checkpoint "
                f"{previous} was left untouched"
            )
            self._log("ERROR", message)
            return False, message
        if manifest is None:  # pragma: no cover - force=True always returns one
            self.resumable = False
            message = f"interrupted by {reason}; no checkpoint could be written"
            self._log("ERROR", message)
            return False, message
        self.interrupt_checkpoint_id = str(manifest.get("id"))
        self.resumable = True
        message = (
            f"interrupted by {reason}; resumable checkpoint "
            f"{self.interrupt_checkpoint_id} saved at step {manifest.get('step')}"
        )
        self._log("INFO", message)
        return True, message

    def _track_eval_stall(self, result: dict) -> None:
        """Warn (once per stall) when independent evaluation stops moving.

        Training reward can keep climbing while play does not; the periodic
        evaluation is the ground truth, so three flat evaluations in a row
        are reported loudly instead of being discovered in the final report.
        """
        self._eval_history.append(
            {
                "goal_diff": float(result.get("goal_diff") or 0),
                "wins": int(result.get("wins") or 0),
                "touches": float(result.get("touches") or 0),
            }
        )
        stalled = detect_eval_stall(self._eval_history)
        self._metric("eval/stalled", 1.0 if stalled else 0.0)
        if stalled and not getattr(self, "_stall_warned", False):
            self._stall_warned = True
            hint = ""
            if self.curriculum is not None:
                hint = (
                    f" Curriculum stage {self.curriculum.stage.name!r} is blocked by: "
                    f"{self.curriculum.blocked_reason()}."
                )
            self._warn(
                "evaluation stall: no win and no goal-difference or touch "
                f"improvement across the last {STALL_WINDOW} periodic evaluations."
                + hint
                + " Check charts/touches and reward components before spending "
                "more steps - more of the same training rarely fixes a stall."
            )
        elif not stalled:
            self._stall_warned = False

    @staticmethod
    def _flatten_suite(suite: dict) -> dict:
        """Turn a multi-opponent suite into one row plus per-opponent rows."""
        strength = suite.get("strength") or {}
        rows = []
        per_opponent = {}
        for name, result in (suite.get("results") or {}).items():
            per_opponent[name] = {
                "win_rate": result.get("win_rate"),
                "goal_diff": result.get("goal_diff"),
                "games": result.get("games"),
                "wins": result.get("wins"),
                "losses": result.get("losses"),
                "draws": result.get("draws"),
            }
            row = dict(result)
            row["opponent"] = name
            row["protocol"] = suite.get("protocol")
            row["kind"] = "suite-member"
            row["details"] = {"games": result.get("details", {}).get("games", [])}
            rows.append(row)
        headline = {
            "mode": suite.get("mode"),
            "opponent": "+".join(suite.get("opponents") or []),
            "kind": "suite",
            "games": int(strength.get("games") or 0),
            "wins": int(strength.get("wins") or 0),
            "losses": int(strength.get("losses") or 0),
            "draws": int(strength.get("draws") or 0),
            "goals_for": int(strength.get("goals_for") or 0),
            "goals_against": int(strength.get("goals_against") or 0),
            "win_rate": strength.get("win_rate"),
            "win_rate_ci95": strength.get("win_rate_ci95") or [0.0, 1.0],
            "goal_diff": strength.get("goal_diff"),
            "touches": sum(
                int(item.get("touches") or 0) for item in (suite.get("results") or {}).values()
            ),
            "protocol": suite.get("protocol"),
            "duration_s": suite.get("duration_s"),
            "per_opponent": per_opponent,
            "diagnostic": suite.get("diagnostic"),
            "rows": rows,
            "details": {"suite": {name: {k: v for k, v in item.items() if k != "details"}
                                  for name, item in (suite.get("results") or {}).items()}},
        }
        if not headline["games"]:
            # no strength opponent was played: fall back to the diagnostics so
            # the number in the GUI is still a real measurement
            diagnostic = suite.get("diagnostic") or {}
            for key in ("games", "wins", "losses", "draws", "goals_for", "goals_against"):
                headline[key] = int(diagnostic.get(key) or 0)
            headline["win_rate"] = diagnostic.get("win_rate")
            headline["win_rate_ci95"] = diagnostic.get("win_rate_ci95") or [0.0, 1.0]
            headline["goal_diff"] = diagnostic.get("goal_diff")
            headline["opponent"] = "+".join(diagnostic.get("opponents") or []) + " (diagnostic)"
        return headline

    # -- main loop -----------------------------------------------------------
    def run(self) -> dict:
        # The CLI may hand us a fresh run id; JobManager normally creates the
        # row beforehand, but the trainer must not depend on that.
        if self.store.get_run(self.run_id) is None:
            self.store.create_run(
                self.config.run_name,
                self.config.mode,
                self.config.to_dict(),
                self.config.total_steps,
                run_id=self.run_id,
            )
        self.store.update_run(
            self.run_id,
            status="starting",
            pid=os.getpid(),
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        self._log(
            "INFO",
            f"starting run {self.run_id} mode={self.config.mode} device={self.device} "
            f"obs_v{self.config.obs_version} resume_mode={self.resume_mode}",
        )
        self._recover_interrupted_sessions()
        if self.device.type == "cpu":
            # The thread budget was already applied in __init__; re-assert it
            # here so a resumed run cannot inherit a different setting.
            self._apply_torch_threads(self.torch_threads)
            self._log(
                "INFO",
                f"torch threads for the learner: {self.torch_threads}",
            )

        exit_status = "completed"
        exit_message = ""
        self._interrupt_guard.install()
        try:
            self._load_resume()
            self.store.update_run(
                self.run_id,
                total_steps=self.config.total_steps,
                metrics_json=json.dumps(
                    {"step": self.step, "total_steps": self.config.total_steps}
                ),
            )
            self._apply_resource_limits()
            self._start_session()
            self.pool = WorkerPool(self.config)
            self.store.update_run(self.run_id, status="running")
            self.league.add(f"initial-{self.run_id[:6]}", self.step, self._snapshot_state())
            if self.curriculum is not None:
                stage = self.curriculum.stage
                self._log(
                    "INFO",
                    f"curriculum stage {self.curriculum.index}/"
                    f"{len(self.curriculum.stages) - 1}: {stage.name} vs "
                    f"{stage.opponent} (resets {stage.reset_mode}); gates "
                    f"{[gate.name for gate in stage.gates] or 'window criteria'} "
                    f"every {self.config.curriculum_gate_interval} iterations "
                    f"x {self.config.curriculum_gate_trials} trials; promotion "
                    f"blocked by: {self.curriculum.blocked_reason() or 'nothing'}",
                )
            if self.config.eval_interval and self.config.eval_before:
                self._log("INFO", "measuring the policy before the first update")
                self._run_evaluation(phase="before")
            batch_size = self.config.rollout_steps * self.config.n_workers
            if self.step >= self.config.total_steps:
                exit_message = (
                    f"nothing to do: restored step {self.step} already reached total_steps "
                    f"{self.config.total_steps}"
                )
                self._warn(exit_message)
            while self.step < self.config.total_steps:
                control = self._control()
                if control == "stop":
                    exit_status = "stopped"
                    exit_message = "stopped by user"
                    break
                if control == "pause":
                    if self.store.get_run(self.run_id)["status"] != "paused":
                        self.store.update_run(self.run_id, status="paused")
                        self._log("INFO", "paused")
                    time.sleep(1.0)
                    continue
                if self.store.get_run(self.run_id)["status"] == "paused":
                    self.store.update_run(self.run_id, status="running")
                    self._log("INFO", "resumed")

                iteration_started = time.perf_counter()
                (
                    opponent_names,
                    opponent_states,
                    opponent_ids,
                    stage_name,
                    reset_mode,
                    episode_seconds,
                ) = self._select_opponents()
                started = time.perf_counter()
                rollouts = self.pool.collect(
                    self._cpu_state(),
                    self.config.rollout_steps,
                    self.config.seed + self.iteration,
                    opponent_names=opponent_names,
                    opponent_states=opponent_states,
                    policy_teams=self._policy_teams(),
                    opponent_execution=self._opponent_execution(),
                    reset_mode=reset_mode,
                    episode_seconds=episode_seconds,
                )
                collect_seconds = time.perf_counter() - started

                batches = [
                    flatten_rollout(rollout, rollout["last_values"], self.config)
                    for rollout in rollouts
                ]
                batch = Batch(
                    obs=np.concatenate([b.obs for b in batches]),
                    actions=np.concatenate([b.actions for b in batches]),
                    logprobs=np.concatenate([b.logprobs for b in batches]),
                    advantages=np.concatenate([b.advantages for b in batches]),
                    returns=np.concatenate([b.returns for b in batches]),
                    values=np.concatenate([b.values for b in batches]),
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate_at(self.config, self.step)

                update_started = time.perf_counter()
                stats = ppo_update(
                    self.model, self.optimizer, batch, self.config, device=self.device
                )
                update_seconds = time.perf_counter() - update_started
                self.step += batch_size
                self.iteration += 1

                episodes = [ep for rollout in rollouts for ep in rollout["episodes"]]
                wins, losses, draws = _outcome_counts(episodes)
                self._apply_adaptation(episodes)
                self._remember_players(episodes)
                # Elo decay is per snapshot: with per-worker draws, each
                # rollout's results belong to the snapshot that worker faced
                recorded = False
                for rollout in rollouts:
                    entry_id = opponent_ids[int(rollout.get("worker", 0)) % len(opponent_ids)]
                    if entry_id:
                        n_w, n_l, n_d = _outcome_counts(rollout["episodes"])
                        if n_w + n_l + n_d:
                            self.league.record(entry_id, n_w, n_l, n_d)
                            recorded = True
                if recorded:
                    self._metric("league/agent_elo", self.league.agent_elo)
                    self._metric("league/pool_size", float(len(self.league)))

                if self.curriculum is not None:
                    transitioned = self.curriculum.observe(episodes)
                    for key, value in self.curriculum.metrics().items():
                        self._metric(key, value)
                    event = self.curriculum.last_event
                    if transitioned and event in ("promotion", "demotion"):
                        self._log_curriculum_transition(event)
                    elif event == "stagnation":
                        self._warn(
                            f"curriculum stage {self.curriculum.stage.name!r} has not "
                            f"promoted for {self.curriculum.stage_iterations} iterations; "
                            f"blocked by: {self.curriculum.blocked_reason()}. Training "
                            "continues on this stage - check reward/touch metrics "
                            "instead of waiting for the stage to move by itself."
                        )

                self._log_iteration(
                    episodes,
                    stats,
                    rollouts,
                    collect_seconds,
                    update_seconds,
                    batch_size,
                    stage_name,
                )

                if self.iteration % self.config.snapshot_interval == 0:
                    self.league.add(
                        f"step-{self.step}-{self.iteration}", self.step, self._snapshot_state()
                    )

                # Gate evaluations are independent of the periodic match
                # evaluation: they measure the *skills* the active stage is
                # about to be promoted on, on both sides, with fixed seeds.
                if (
                    self.curriculum is not None
                    and not self.curriculum.finished
                    and self.config.curriculum_gate_interval
                    and self.iteration % self.config.curriculum_gate_interval == 0
                ):
                    self._run_gate_evaluation()

                if self.config.eval_interval and self.iteration % self.config.eval_interval == 0:
                    self._run_evaluation()

                self.store.update_run(
                    self.run_id,
                    step=self.step,
                    metrics_json=json.dumps(
                        {
                            "step": self.step,
                            "total_steps": self.config.total_steps,
                            "charts/episodic_return": _mean([e["reward"] for e in episodes]),
                            "charts/win_rate": _mean(
                                [1.0 if e.get("result") == "win" else 0.0 for e in episodes]
                            ),
                            "time/sps": batch_size / max(collect_seconds + update_seconds, 1e-6),
                            "curriculum/stage": (
                                float(self.curriculum.index) if self.curriculum else 0.0
                            ),
                            "phase": stage_name,
                        }
                    ),
                )
                self._metric("time/iteration_seconds", time.perf_counter() - iteration_started)
                # Crash net: bounded, time based, pruned by keep_checkpoints.
                self._maybe_recovery_checkpoint()
                # the queued metrics are written in one transaction, at
                # most every ``metric_flush_interval`` iterations
                self._iterations_since_flush += 1
                if self._iterations_since_flush >= int(self.config.metric_flush_interval):
                    self._flush_metrics()
                    self._iterations_since_flush = 0

            if exit_status == "stopped":
                # A cancelled run persists its exact state as the interrupt
                # checkpoint.  This happens before any closing evaluation: the
                # supervisor that requested the stop waits a bounded time for
                # the process, and a multi-minute measurement could otherwise
                # be terminated before the weights reach the disk.
                self._interrupted = True
                _, exit_message = self._save_after_interrupt("a stop request")
                self.resumable = self.interrupt_checkpoint_id is not None
            else:
                # Reaching total_steps normally is the one point at which a
                # completed run publishes a checkpoint.  It is written before
                # the closing measurement so the final weights are on disk
                # even if that evaluation is interrupted.
                self._save(label="final", force=True)
                if self.config.eval_interval and self.step > self.start_step:
                    self._run_evaluation(phase="after")
                elif self.config.eval_interval:
                    self._warn(
                        "no final evaluation: the run completed no training step, so an "
                        "'after' measurement would be identical to the baseline"
                    )
                self.resumable = bool(self.final_manifest)
                exit_message = exit_message or "training completed"
        except KeyboardInterrupt:
            # Ctrl+C: stop the workers, then write whatever state exists.  The
            # exception is swallowed so the session, run row and report are
            # still closed consistently; the CLI sees a normal "stopped" exit.
            self._interrupted = True
            exit_status = "stopped"
            _, exit_message = self._save_after_interrupt()
        except Exception as exc:  # pragma: no cover - requires a failure to exercise
            exit_status = "failed"
            exit_message = str(exc)
            self.resumable = False
            self.store.log(self.run_id, "ERROR", traceback.format_exc())
            self._log("ERROR", f"training failed: {exc}")
        finally:
            if self.pool is not None:
                self.pool.close()
            # The session is closed before the report is written, so the
            # report contains the measured duration of this session.
            try:
                self._finish_session(exit_status)
            except Exception as exc:  # pragma: no cover - defensive
                self._log("WARNING", f"could not close the training session: {exc}")
            self._flush_metrics()
            self._apply_retention()
            self.store.update_run(
                self.run_id,
                status=exit_status,
                step=self.step,
                ended_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                message=exit_message,
            )
            self._log("INFO", f"run {exit_status}: {exit_message}")
            self._write_report(exit_status)
            self._interrupt_guard.restore()
        return {
            "status": exit_status,
            "step": self.step,
            "start_step": self.start_step,
            "steps_completed": max(0, self.step - self.start_step),
            "message": exit_message,
            "resume_mode": self.resume_mode,
            "parent_run": self.resume_plan.parent_run if self.resume_plan else None,
            "checkpoint": self.interrupt_checkpoint_id or self.last_checkpoint_id,
            "resumable": bool(self.resumable),
            "interrupted": self._interrupted,
            "report": str(self.run_dir / "report.json"),
            "warnings": list(self.run_warnings),
            "best_win_rate": (
                None if self.run_best_score == float("-inf") else self.run_best_score
            ),
        }

    # -- reporting ---------------------------------------------------------
    def _write_report(self, status: str) -> None:
        try:
            final_checkpoint = None
            # An interrupted run whose shutdown save failed must not present
            # an older periodic checkpoint as its final resumable state.
            if self.final_manifest and not (
                self._interrupted and self.resumable is False
            ):
                final_checkpoint = {
                    "model_id": self.final_manifest.get("id"),
                    "path": str(self.models_dir / self.final_manifest["path"]),
                    "step": self.final_manifest.get("step"),
                    "label": self.final_manifest.get("label"),
                    "size_bytes": self.final_manifest.get("size_bytes"),
                }
            resume_info = self.resume_plan.to_dict() if self.resume_plan else {
                "mode": self.resume_mode,
                "will_resume": False,
                "warnings": [],
            }
            resume_info["mode"] = self.resume_mode
            resume_info["restored_step"] = self.start_step
            resume_info["inherited_fields"] = list(getattr(self, "resume_inherited", []))
            resume_info["interrupted"] = bool(self._interrupted)
            resume_info["resumable"] = self.resumable
            # the merge decisions actually applied by this process (the plan
            # may predate them); the report must show the effective config's
            # provenance, not only the final values
            if self.resume_overrides:
                resume_info["overrides"] = self.resume_overrides
            resume_info["phase_change"] = [
                item["field"]
                for item in (resume_info.get("overrides") or [])
                if item.get("source") == "explicit"
            ]
            payload = report_module.build_run_report(
                self.store,
                self.run_id,
                models_dir=self.models_dir,
                run_dir=self.run_dir,
                config=self.config.to_dict(),
                resume=resume_info,
                curriculum=self.curriculum.status() if self.curriculum else None,
                device=str(self.device),
                workers=int(self.config.n_workers),
                wall_seconds=max(0.0, time.time() - self.started),
                final_checkpoint=final_checkpoint,
                extra_warnings=self.run_warnings,
            )
            payload["status"] = status
            path = report_module.persist_report(payload, self.store, run_dir=self.run_dir)
            self._log(
                "INFO",
                "training report written"
                + (f" to {path}" if path else " (database only)")
                + f": {payload['steps']['completed']} steps in "
                f"{payload['duration_s'] if payload['duration_s'] is not None else 'unknown'} s",
            )
        except Exception as exc:  # a broken report must never mask the run result
            self._log("ERROR", f"could not build the training report: {exc}")
            self.store.log(self.run_id, "ERROR", traceback.format_exc())

    def _snapshot_state(self) -> dict:
        """Immutable CPU copy of the weights (self-play snapshot)."""
        return freeze_state(clone_state_dict(self.model.state_dict()))

    def _cpu_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def _log_iteration(
        self,
        episodes,
        stats,
        rollouts,
        collect_seconds,
        update_seconds,
        batch_size,
        stage_name="fixed",
    ):
        env_steps = sum(r["steps"] for r in rollouts)
        sps = env_steps / max(collect_seconds + update_seconds, 1e-9)
        sim_seconds = sum(float(r.get("sim_seconds", 0.0)) for r in rollouts)
        infer_seconds = sum(float(r.get("infer_seconds", 0.0)) for r in rollouts)
        opponent_name = stage_name
        self.store.log_metrics(
            self.run_id,
            self.step,
            {
                **stats,
                **_reward_component_means(episodes),
                "charts/episodic_return": _mean([e["reward"] for e in episodes]),
                "charts/episode_length": _mean([e["length"] for e in episodes]),
                "charts/win_rate": _mean([1.0 if e.get("result") == "win" else 0.0 for e in episodes]),
                "charts/draw_rate": _mean([1.0 if e.get("result") == "draw" else 0.0 for e in episodes]),
                "charts/goals_for": _mean([e["goals_for"] for e in episodes]),
                "charts/goals_against": _mean([e["goals_against"] for e in episodes]),
                "charts/touches": _mean([e.get("touches", 0) for e in episodes]),
                "charts/shots": _mean([e.get("shots", 0) for e in episodes]),
                "charts/episodes": float(len(episodes)),
                "charts/truncation_rate": _mean(
                    [1.0 if e.get("truncated") else 0.0 for e in episodes]
                ),
                "time/sps": sps,
                "time/collect_seconds": collect_seconds,
                "time/update_seconds": update_seconds,
                "time/sim_seconds": sim_seconds / max(len(rollouts), 1),
                "time/infer_seconds": infer_seconds / max(len(rollouts), 1),
                "time/eval_seconds": self.evaluation_seconds,
                "time/checkpoint_seconds": self.checkpoint_seconds,
                "time/total_steps": float(self.step),
            },
        )
        for episode in episodes:
            self.store.log_episode(
                self.run_id,
                episode["episode"],
                episode["reward"],
                episode["length"],
                episode["goals_for"],
                episode["goals_against"],
                episode["result"],
                step=self.step,
                truncated=bool(episode.get("truncated")),
                opponent=episode.get("opponent") or opponent_name,
                touches=int(episode.get("touches", 0)),
                shots=int(episode.get("shots", 0)),
            )
        self._log(
            "INFO",
            f"iter {self.iteration} step {self.step}/{self.config.total_steps} "
            f"stage={stage_name} sps={sps:.0f} return={_mean([e['reward'] for e in episodes]):.3f} "
            f"winrate={_mean([1.0 if e.get('result') == 'win' else 0.0 for e in episodes]):.2f} "
            f"kl={stats.get('losses/approx_kl', 0.0):.4f}",
        )


def _rng_default(value):
    """JSON fallback for RNG state values (numpy integers and arrays)."""
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def run_training(
    config: TrainingConfig,
    run_id: str,
    store: Store,
    *,
    resume_from: str | None = None,
    resume_mode: str = "continue",
    resume_plan: ResumePlan | None = None,
    config_override_fields: tuple[str, ...] = (),
) -> dict:
    return Trainer(
        config,
        run_id,
        store,
        resume_from=resume_from,
        resume_mode=resume_mode,
        resume_plan=resume_plan,
        config_override_fields=config_override_fields,
    ).run()
