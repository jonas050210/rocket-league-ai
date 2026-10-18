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
import time
import traceback
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
from .workers import WorkerPool


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def is_best_candidate(result: dict, best_score: float) -> bool:
    """A model can only become "best" after it has actually won a game.

    Untrained policies produce 0:0 draws; marking those as best would put a
    meaningless checkpoint at the top of the GUI and of ``best.json``.
    """
    return int(result.get("wins", 0)) > 0 and float(result.get("win_rate", 0.0)) > best_score


def _outcome_counts(episodes: list[dict]) -> tuple[int, int, int]:
    wins = sum(1 for episode in episodes if episode.get("result") == "win")
    losses = sum(1 for episode in episodes if episode.get("result") == "loss")
    draws = sum(1 for episode in episodes if episode.get("result") == "draw")
    return wins, losses, draws


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
    ):
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
        self.league = OpponentPool(capacity=8)
        self.curriculum = (
            Curriculum(stages_for_mode(config.mode)) if config.curriculum else None
        )
        self.rng = np.random.default_rng(int(config.seed))
        self.best_score = float(
            checkpoints.best_record(self.models_dir).get("score", float("-inf"))
        )
        self.run_best_score = float("-inf")
        self.started = time.time()
        self.last_checkpoint_step = -1
        self.last_checkpoint_id: str | None = None
        self.final_manifest: dict | None = None
        self.run_warnings: list[str] = []
        self.evaluation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.evaluated_before = False

    # -- logging -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        self.store.log(self.run_id, level, message)
        print(f"[{level.upper()}] {message}", flush=True)

    def _warn(self, message: str) -> None:
        if message not in self.run_warnings:
            self.run_warnings.append(message)
        self._log("WARNING", message)

    def _metric(self, name: str, value: float, step: int | None = None) -> None:
        self.store.log_metric(self.run_id, self.step if step is None else step, name, float(value))

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
        """Build (or reuse) the resume plan for this trainer."""
        if self.resume_plan is not None:
            return self.resume_plan
        if not self.resume_from and self.resume_mode == "fresh":
            return None
        if not self.resume_from and self.resume_mode == "continue":
            # Automatic continuation: only when the caller explicitly asked for
            # it through ``resume_mode``; a plain fresh run stays fresh.
            return None
        return plan_resume(
            self.models_dir,
            self.config,
            mode=self.resume_mode,
            run_id=self.run_id,
            checkpoint=self.resume_from,
        )

    def _load_resume(self) -> None:
        plan = self._resolve_plan()
        if plan is None:
            return
        self.resume_plan = plan
        for warning in plan.warnings:
            self._warn(warning)
        if not plan.will_resume:
            if self.resume_from:
                reasons = "; ".join(
                    reason
                    for item in plan.rejected
                    for reason in item.get("reasons", [])
                ) or "; ".join(plan.warnings) or "no compatible checkpoint found"
                if explicit_checkpoint_exists(self.models_dir, self.resume_from):
                    raise ValueError(
                        f"cannot resume from {self.resume_from!r}: {reasons}"
                    )
                raise FileNotFoundError(
                    f"no usable checkpoint for {self.resume_from!r}: {reasons}"
                )
            return

        path = Path(str(plan.checkpoint_path))
        payload = checkpoints.load_checkpoint(path)
        saved_config = PolicyConfig.from_dict(payload.get("policy_config"))
        if saved_config.to_dict() != self.config.policy_config().to_dict():
            raise ValueError(
                "checkpoint policy architecture does not match this run: "
                f"{saved_config.to_dict()} vs {self.config.policy_config().to_dict()}"
            )
        self.model.load_state_dict(payload["model"])

        merged, overrides, inherited = merge_config(
            payload.get("training_config"), self.config, mode=plan.mode
        )
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
                        f"{override['requested']!r})",
                    )
        self.config = merged
        self.resume_inherited = inherited

        if plan.mode == "transfer":
            self._log(
                "INFO",
                f"transfer learning from {path} (weights only, step counter restarts at 0)",
            )
            self.step = 0
            return

        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            self._warn("checkpoint has no optimizer state; starting a fresh optimizer")
        self.step = int(payload.get("step", 0))
        self.start_step = self.step
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
            self.curriculum = Curriculum(
                stages_for_mode(self.config.mode),
                start_index=int(saved.get("index", 0)),
                stage_iterations=int(saved.get("iterations", 0)),
                total_iterations=int(saved.get("total_iterations", 0)),
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
        # learning-rate annealing depends on the step counter; make the
        # optimiser agree with the restored position immediately
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate_at(self.config, self.step)
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

    def _save(self, *, label: str = "", force: bool = False) -> dict | None:
        if not force and self.step == self.last_checkpoint_step:
            return None
        started = time.perf_counter()
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
            },
            label=label,
        )
        self.store.register_model(manifest)
        self.last_checkpoint_step = self.step
        self.last_checkpoint_id = manifest["id"]
        self.final_manifest = manifest
        try:
            prune_candidates(self.models_dir, keep=self.config.keep_checkpoints, run_id=self.run_id)
        except Exception as exc:  # never let cleanup kill a run
            self._log("WARNING", f"checkpoint cleanup skipped: {exc}")
        self.checkpoint_seconds += time.perf_counter() - started
        self._metric("time/checkpoint_seconds", time.perf_counter() - started)
        self._log("INFO", f"checkpoint {manifest['id']} saved at step {self.step}")
        return manifest

    # -- opponent selection --------------------------------------------------
    def _select_opponent(self) -> tuple[str | None, dict | None, str | None, str]:
        """Return (opponent name, opponent weights, pool entry id, stage name)."""
        if self.curriculum is not None:
            stage = self.curriculum.stage
            if stage.opponent == "self" and self.league.usable:
                entry = self.league.sample(self.rng)
                return None, entry.state, entry.id, stage.name
            if stage.opponent == "mixed":
                name = str(self.rng.choice(["chase", "random"]))
                return name, None, None, stage.name
            return stage.opponent, None, None, stage.name
        if self.config.self_play and self.league.usable:
            entry = self.league.sample(self.rng)
            return None, entry.state, entry.id, "self-play"
        return self.config.opponent, None, None, "fixed"

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
            seed=int(self.config.seed) + 900_001 + self.iteration,
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
        manifest = self._save()
        if manifest is not None:
            self.last_checkpoint_id = manifest["id"]
        if not self.last_checkpoint_id or not (
            int(result.get("wins") or 0)
            + int(result.get("losses") or 0)
            + int(result.get("draws") or 0)
        ):
            return result

        model_id = self.last_checkpoint_id
        rows = result.get("rows") or [dict(result, model_id=model_id, run_id=self.run_id)]
        for row in rows:
            row.setdefault("model_id", model_id)
            row.setdefault("run_id", self.run_id)
            row["phase"] = phase
            row.setdefault("protocol", protocol)
            self.store.add_evaluation(row)
        score = result.get("win_rate") or 0.0
        self.store.set_model_eval_score(model_id, float(score))
        # A 0 % win rate is not a "best" model: keep the pointer empty until the
        # policy actually won a game, so the GUI never offers a checkpoint as
        # "best" that has never scored.  Scores are only compared inside one
        # protocol, so "best" always means "best under the same measurement".
        if is_best_candidate(result, self.best_score):
            record = checkpoints.set_best(
                self.models_dir,
                model_id,
                float(score),
                metric="win_rate",
                protocol=protocol or None,
            )
            self.best_score = float(record.get("score", score))
            self._log(
                "INFO",
                f"new global best {model_id} (win rate {float(score):.2f} vs "
                f"{result.get('opponent')}, protocol {record.get('protocol_key')})",
            )
        self.run_best_score = max(self.run_best_score, float(score))
        return result

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
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        exit_status = "completed"
        exit_message = ""
        try:
            self._load_resume()
            self.store.update_run(
                self.run_id,
                total_steps=self.config.total_steps,
                metrics_json=json.dumps(
                    {"step": self.step, "total_steps": self.config.total_steps}
                ),
            )
            self.pool = WorkerPool(self.config)
            self.store.update_run(self.run_id, status="running")
            self.league.add(f"initial-{self.run_id[:6]}", self.step, self._snapshot_state())
            if self.curriculum is not None:
                self._log(
                    "INFO",
                    f"curriculum stage 0/{len(self.curriculum.stages) - 1}: "
                    f"{self.curriculum.stage.name} vs {self.curriculum.stage.opponent}",
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
                opponent_name, opponent_state, opponent_id, stage_name = self._select_opponent()
                started = time.perf_counter()
                rollouts = self.pool.collect(
                    self._cpu_state(),
                    self.config.rollout_steps,
                    self.config.seed + self.iteration,
                    opponent_state=opponent_state,
                    opponent=opponent_name,
                    policy_teams=self._policy_teams(),
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
                if opponent_id:
                    self.league.record(opponent_id, wins, losses, draws)
                    self._metric("league/agent_elo", self.league.agent_elo)
                    self._metric("league/pool_size", float(len(self.league)))

                if self.curriculum is not None:
                    promoted = self.curriculum.observe(episodes)
                    for key, value in self.curriculum.metrics().items():
                        self._metric(key, value)
                    if promoted:
                        stage = self.curriculum.stage
                        self._log(
                            "INFO",
                            f"curriculum promotion -> {stage.name} "
                            f"(opponent {stage.opponent}, target {stage.win_rate:.0%})",
                        )
                        self.curriculum.reset_window()

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

                if self.config.eval_interval and self.iteration % self.config.eval_interval == 0:
                    self._run_evaluation()

                if self.iteration % self.config.checkpoint_interval == 0:
                    self._save()
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

            if self.config.eval_interval and self.step > self.start_step:
                self._run_evaluation(phase="after")
            elif self.config.eval_interval:
                self._warn(
                    "no final evaluation: the run completed no training step, so an "
                    "'after' measurement would be identical to the baseline"
                )
            self._save(label="final", force=True)
            exit_message = exit_message or "training completed"
        except Exception as exc:  # pragma: no cover - requires a failure to exercise
            exit_status = "failed"
            exit_message = str(exc)
            self.store.log(self.run_id, "ERROR", traceback.format_exc())
            self._log("ERROR", f"training failed: {exc}")
        finally:
            if self.pool is not None:
                self.pool.close()
            self.store.update_run(
                self.run_id,
                status=exit_status,
                step=self.step,
                ended_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                message=exit_message,
            )
            self._log("INFO", f"run {exit_status}: {exit_message}")
            self._write_report(exit_status)
        return {
            "status": exit_status,
            "step": self.step,
            "start_step": self.start_step,
            "steps_completed": max(0, self.step - self.start_step),
            "message": exit_message,
            "resume_mode": self.resume_mode,
            "parent_run": self.resume_plan.parent_run if self.resume_plan else None,
            "checkpoint": self.last_checkpoint_id,
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
            if self.final_manifest:
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
            payload = report_module.build_run_report(
                self.store,
                self.run_id,
                models_dir=self.models_dir,
                run_dir=self.run_dir,
                config=self.config.to_dict(),
                resume=resume_info,
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
) -> dict:
    return Trainer(
        config,
        run_id,
        store,
        resume_from=resume_from,
        resume_mode=resume_mode,
        resume_plan=resume_plan,
    ).run()
