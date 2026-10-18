"""Training orchestration: PPO + parallel workers + checkpoints + metrics."""

from __future__ import annotations

import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .checkpoints import prune_candidates, save_checkpoint
from .config import TrainingConfig, ensure_dirs
from .policy import PolicyConfig
from .ppo import Batch, flatten_rollout, learning_rate_at, ppo_update
from .store import Store
from .workers import WorkerPool

CHECKPOINT_INTERVAL_DEFAULT = 10


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


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
        self.snapshots: list[dict] = []
        self.started = time.time()
        self.last_checkpoint_step = -1

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
            extra={"iteration": self.iteration},
            label=label,
        )
        self.store.register_model(manifest)
        self.last_checkpoint_step = self.step
        try:
            prune_candidates(self.models_dir, keep=self.config.keep_checkpoints, run_id=self.run_id)
        except Exception as exc:  # never let cleanup kill a run
            self._log("WARNING", f"checkpoint cleanup skipped: {exc}")
        self._log("INFO", f"checkpoint {manifest['id']} saved at step {self.step}")
        return manifest

    def run(self) -> dict:
        self.store.update_run(
            self.run_id,
            status="starting",
            pid=__import__("os").getpid(),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._log(
            "INFO", f"starting run {self.run_id} mode={self.config.mode} device={self.device}"
        )
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        exit_status = "completed"
        exit_message = ""
        try:
            self._load_resume()
            self.pool = WorkerPool(self.config)
            self.store.update_run(self.run_id, status="running")
            self.snapshots = [self._cpu_state()]
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

                opponent_state = self._pick_opponent()
                started = time.perf_counter()
                rollouts = self.pool.collect(
                    self._cpu_state(),
                    self.config.rollout_steps,
                    self.config.seed + self.iteration,
                    opponent_state=opponent_state,
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
                self._log_iteration(
                    episodes, stats, rollouts, collect_seconds, update_seconds, batch_size
                )

                if self.iteration % self.config.snapshot_interval == 0:
                    self.snapshots.append(self._cpu_state())
                    self.snapshots = self.snapshots[-8:]

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
                            "charts/win_rate": _mean([1.0 if e["won"] else 0.0 for e in episodes]),
                            "time/sps": batch_size / max(collect_seconds + update_seconds, 1e-6),
                        }
                    ),
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
                ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                message=exit_message,
            )
            self._log("INFO", f"run {exit_status}: {exit_message}")
        return {"status": exit_status, "step": self.step, "message": exit_message}

    def _cpu_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def _pick_opponent(self):
        if not self.config.self_play:
            return None
        return self.snapshots[random.randrange(len(self.snapshots))]

    def _log_iteration(
        self, episodes, stats, rollouts, collect_seconds, update_seconds, batch_size
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
                "charts/win_rate": _mean([1.0 if e["won"] else 0.0 for e in episodes]),
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
            f"sps={sps:.0f} return={_mean([e['reward'] for e in episodes]):.3f} "
            f"winrate={_mean([1.0 if e['won'] else 0.0 for e in episodes]):.2f}",
        )


def run_training(
    config: TrainingConfig,
    run_id: str,
    store: Store,
    *,
    resume_from: str | None = None,
) -> dict:
    return Trainer(config, run_id, store, resume_from=resume_from).run()
