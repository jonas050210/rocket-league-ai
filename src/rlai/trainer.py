"""Training orchestration: PPO + parallel workers + checkpoints + metrics."""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import checkpoints
from .checkpoints import prune_candidates, save_checkpoint
from .config import TrainingConfig, ensure_dirs
from .curriculum import Curriculum, stages_for_mode
from .league import OpponentPool
from .policy import PolicyConfig, TorchPolicy
from .ppo import Batch, flatten_rollout, learning_rate_at, ppo_update
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


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        run_id: str,
        store: Store,
        *,
        resume_from: str | None = None,
        run_dir: Path | None = None,
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
        self.device = torch.device(config.resolved_device())
        self.model = (
            PolicyConfig.from_dict(config.policy_config().to_dict()).build().to(self.device)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.step = 0
        self.iteration = 0
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

    # -- logging -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        self.store.log(self.run_id, level, message)
        print(f"[{level.upper()}] {message}", flush=True)

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

    # -- lifecycle ---------------------------------------------------------
    def _load_resume(self) -> None:
        if not self.resume_from:
            return
        payload = torch.load(Path(self.resume_from), map_location="cpu", weights_only=False)
        saved_config = PolicyConfig.from_dict(payload.get("policy_config"))
        if saved_config.to_dict() != self.config.policy_config().to_dict():
            raise ValueError(
                "checkpoint policy architecture does not match this run: "
                f"{saved_config.to_dict()} vs {self.config.policy_config().to_dict()}"
            )
        self.model.load_state_dict(payload["model"])
        if payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload.get("step", 0))
        extra = payload.get("extra") or {}
        if isinstance(extra.get("league"), dict):
            self.league = OpponentPool.from_dict(extra["league"])
            self._log(
                "INFO",
                f"restored opponent-pool ratings from checkpoint ({len(self.league)} entries)",
            )
            if len(self.league) and not self.league.usable:
                self._log(
                    "INFO",
                    "pool snapshots carry ratings but no weights; using the fixed "
                    "opponent until the next snapshot is added",
                )
        if self.curriculum is not None and isinstance(extra.get("curriculum"), dict):
            saved = extra["curriculum"]
            self.curriculum = Curriculum(
                stages_for_mode(self.config.mode),
                start_index=int(saved.get("index", 0)),
                stage_iterations=int(saved.get("iterations", 0)),
                total_iterations=int(saved.get("total_iterations", 0)),
            )
        self._log("INFO", f"resumed from {self.resume_from} at step {self.step}")

    def _save(self, *, label: str = "", force: bool = False) -> dict | None:
        if not force and self.step == self.last_checkpoint_step:
            return None
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
                "league": self.league.to_dict(),
                "curriculum": self.curriculum.describe() if self.curriculum else None,
            },
            label=label,
        )
        self.store.register_model(manifest)
        self.last_checkpoint_step = self.step
        self.last_checkpoint_id = manifest["id"]
        try:
            prune_candidates(self.models_dir, keep=self.config.keep_checkpoints, run_id=self.run_id)
        except Exception as exc:  # never let cleanup kill a run
            self._log("WARNING", f"checkpoint cleanup skipped: {exc}")
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

    # -- evaluation ----------------------------------------------------------
    def _run_evaluation(self) -> dict:
        from .evaluation import evaluate_policy

        policy = TorchPolicy(self.model, device=self.device)
        try:
            result = evaluate_policy(
                policy,
                mode=self.config.mode,
                opponent=self.config.eval_opponent,
                games=self.config.eval_games,
                seed=int(self.config.seed) + 900_001 + self.iteration,
                match_seconds=self.config.eval_match_seconds,
                episode_seconds=min(self.config.eval_match_seconds, 30.0),
                device=str(self.device),
                obs_version=self.config.obs_version,
            )
        finally:
            self.model.train()
        for key, value in (
            ("eval/win_rate", result["win_rate"]),
            ("eval/win_rate_low", result["win_rate_ci95"][0]),
            ("eval/win_rate_high", result["win_rate_ci95"][1]),
            ("eval/goals_for", result["goals_for"]),
            ("eval/goals_against", result["goals_against"]),
            ("eval/goal_diff", result["goal_diff"]),
            ("eval/games", result["games"]),
        ):
            self._metric(key, value)
        self._log(
            "INFO",
            f"eval vs {result['opponent']}: {result['wins']}W {result['losses']}L "
            f"{result['draws']}D goals {result['goals_for']}:{result['goals_against']} "
            f"(95% CI {result['win_rate_ci95'][0]:.2f}-{result['win_rate_ci95'][1]:.2f})",
        )
        manifest = self._save()
        if manifest is not None:
            self.last_checkpoint_id = manifest["id"]
        if self.last_checkpoint_id and result["wins"] + result["losses"] + result["draws"]:
            result.update(model_id=self.last_checkpoint_id, run_id=self.run_id)
            self.store.add_evaluation(result)
            self.store.set_model_eval_score(self.last_checkpoint_id, result["win_rate"])
            # A 0 % win rate is not a "best" model: keep the pointer empty
            # until the policy actually won a game, so the GUI never offers a
            # checkpoint as "best" that has never scored.
            if is_best_candidate(result, self.best_score):
                record = checkpoints.set_best(
                    self.models_dir,
                    self.last_checkpoint_id,
                    result["win_rate"],
                    metric="win_rate",
                )
                self.best_score = float(record.get("score", result["win_rate"]))
                self._log(
                    "INFO",
                    f"new global best {self.last_checkpoint_id} "
                    f"(win rate {result['win_rate']:.2f} vs {result['opponent']})",
                )
            self.run_best_score = max(self.run_best_score, result["win_rate"])
        return result

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
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._log(
            "INFO",
            f"starting run {self.run_id} mode={self.config.mode} device={self.device} "
            f"obs_v{self.config.obs_version}",
        )
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        exit_status = "completed"
        exit_message = ""
        try:
            self._load_resume()
            self.pool = WorkerPool(self.config)
            self.store.update_run(self.run_id, status="running")
            self.league.add(f"initial-{self.run_id[:6]}", self.step, self._cpu_state())
            if self.curriculum is not None:
                self._log(
                    "INFO",
                    f"curriculum stage 0/{len(self.curriculum.stages) - 1}: "
                    f"{self.curriculum.stage.name} vs {self.curriculum.stage.opponent}",
                )
            batch_size = self.config.rollout_steps * self.config.n_workers
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

                opponent_name, opponent_state, opponent_id, stage_name = self._select_opponent()
                started = time.perf_counter()
                rollouts = self.pool.collect(
                    self._cpu_state(),
                    self.config.rollout_steps,
                    self.config.seed + self.iteration,
                    opponent_state=opponent_state,
                    opponent=opponent_name,
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
                        f"step-{self.step}-{self.iteration}", self.step, self._cpu_state()
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
                        }
                    ),
                )

            if self.config.eval_interval:
                self._run_evaluation()
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
                ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                message=exit_message,
            )
            self._log("INFO", f"run {exit_status}: {exit_message}")
        return {
            "status": exit_status,
            "step": self.step,
            "message": exit_message,
            "best_win_rate": (
                None if self.run_best_score == float("-inf") else self.run_best_score
            ),
        }

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
        self.store.log_metrics(
            self.run_id,
            self.step,
            {
                **stats,
                "charts/episodic_return": _mean([e["reward"] for e in episodes]),
                "charts/episode_length": _mean([e["length"] for e in episodes]),
                "charts/win_rate": _mean([1.0 if e.get("result") == "win" else 0.0 for e in episodes]),
                "charts/draw_rate": _mean([1.0 if e.get("result") == "draw" else 0.0 for e in episodes]),
                "charts/goals_for": _mean([e["goals_for"] for e in episodes]),
                "charts/goals_against": _mean([e["goals_against"] for e in episodes]),
                "charts/episodes": float(len(episodes)),
                "time/sps": sps,
                "time/collect_seconds": collect_seconds,
                "time/update_seconds": update_seconds,
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
            )
        self._log(
            "INFO",
            f"iter {self.iteration} step {self.step}/{self.config.total_steps} "
            f"stage={stage_name} sps={sps:.0f} return={_mean([e['reward'] for e in episodes]):.3f} "
            f"winrate={_mean([1.0 if e.get('result') == 'win' else 0.0 for e in episodes]):.2f} "
            f"kl={stats.get('losses/approx_kl', 0.0):.4f}",
        )


def run_training(
    config: TrainingConfig,
    run_id: str,
    store: Store,
    *,
    resume_from: str | None = None,
) -> dict:
    return Trainer(config, run_id, store, resume_from=resume_from).run()
