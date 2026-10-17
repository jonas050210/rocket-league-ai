"""Real upstream PPO/collector with update-boundary control, telemetry and atomic saves.

No replacement algorithm, toy environment, terminal emulation or silent CPU fallback.
Only upstream lifecycle/persistence defects identified in docs/AUDIT.md are adapted.
"""

import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
from rlgym_learn import (
    BaseConfigModel,
    EnvAction,
    LearningCoordinator,
    LearningCoordinatorConfigModel,
    ProcessConfigModel,
    SerdeTypesModel,
)
from rlgym_learn.pyany_serde import NumpySerdeConfig, PyAnySerdeType
from rlgym_learn_algos.ppo import (
    BasicCritic,
    DiscreteFF,
    ExperienceBufferConfigModel,
    GAETrajectoryProcessor,
    GAETrajectoryProcessorConfigModel,
    PPOAgentController,
    PPOAgentControllerConfigModel,
    PPOLearnerConfigModel,
    PPOMetricsLogger,
    SeparateActorCritic,
)

from ..checkpoints import CheckpointStore
from ..config import TrainingConfig
from ..persistence import atomic_json
from .buffer import PortableExperienceBuffer
from .environment import EnvironmentFactory, Float32DefaultObs
from .protocol import WorkerChannel, finite


def numpy_rng_json(rng_state):
    name, keys, position, has_gauss, cached = rng_state
    return [name, keys.tolist(), position, has_gauss, cached]


def numpy_rng_restore(state):
    return state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]


def tuples(value):
    return tuple(tuples(item) for item in value) if isinstance(value, list) else value


class TelemetryLogger(PPOMetricsLogger):
    def __init__(self):
        super().__init__()
        self.latest = {}

    def collect_agent_metrics(self, data):
        ppo, trajectory = data.ppo_data, data.trajectory_processor_data
        self.latest = finite({
            "reward": trajectory.average_reward,
            "segment_return": trajectory.average_undiscounted_episodic_return,
            "return_std": trajectory.return_standard_deviation,
            "entropy": ppo.actor_entropy, "kl_divergence": ppo.kl_divergence,
            "critic_loss": ppo.critic_loss, "clip_fraction": ppo.sb3_clip_fraction,
            "actor_update_magnitude": ppo.actor_update_magnitude,
            "critic_update_magnitude": ppo.critic_update_magnitude,
            "steps_per_second": data.timesteps_collected / max(data.iteration_time, 1e-9),
            "collection_seconds": data.timestep_collection_time,
            "learning_seconds": ppo.batch_consumption_time,
            "iteration_seconds": data.iteration_time,
        })

    def collect_env_metrics(self, data):
        # Learn 2.0.0 fails to advance the startup offset after shared_info deserialization.
        # Do not enable that channel. Fixed reward components are audited from real timesteps.
        pass

    def report_metrics(self):
        # The worker emits a structured snapshot once the upstream update has fully returned.
        pass


class FoundationController(PPOAgentController):
    def __init__(self, app_config: TrainingConfig, run_dir: Path, channel: WorkerChannel):
        self.app_config, self.run_dir, self.channel = app_config, run_dir, channel
        self.checkpoint_store = CheckpointStore(run_dir)
        self.telemetry = TelemetryLogger()
        self.episodes = 0
        self.goal_episodes = 0
        self.episode_returns = []
        self.partial_returns = {}
        self.last_checkpoint_step = -1
        self.reward_goal_abs = 0.0
        self.reward_touch = 0.0

        def actor_critic_factory(obs_space, action_space, dtype, device, name):
            return SeparateActorCritic(
                DiscreteFF(obs_space[1], action_space[1], app_config.hidden_sizes, dtype, device),
                BasicCritic(obs_space[1], app_config.hidden_sizes, dtype, device),
            )

        def optimizers_factory(actor_critic, kwargs, name):
            return [torch.optim.Adam(actor_critic.actor.parameters(), **kwargs["actor"]),
                    torch.optim.Adam(actor_critic.critic.parameters(), **kwargs["critic"])]

        super().__init__(
            actor_critic_factory, optimizers_factory,
            PortableExperienceBuffer(GAETrajectoryProcessor(), pin_memory=app_config.pin_memory),
            metrics_logger=self.telemetry,
        )

    def _load_from_checkpoint(self):
        # Boundary-only format: no pickled/in-flight controller trajectories or old monotonic times.
        path = Path(self.config.subcontroller_config.checkpoint_load_folder)
        state = json.loads((path / "ppo_agent.json").read_text())
        if state["iteration_timesteps"] != 0:
            raise RuntimeError("Only completed-update checkpoints can be recovered by this adapter")
        self.cur_iteration = state["cur_iteration"]
        self.cumulative_timesteps = state["cumulative_timesteps"]
        self.iteration_timesteps = 0
        self.current_env_trajectories.clear()
        self.iteration_trajectories.clear()
        self.iteration_shared_infos.clear()
        self.iteration_start_time = self.timestep_collection_start_time = time.perf_counter()

    def restore_application_state(self, checkpoint: Path):
        state = json.loads((checkpoint / "application.json").read_text())
        self.episodes = state["metrics"].get("episodes", 0)
        self.goal_episodes = state["metrics"].get("goal_episodes", 0)
        self.episode_returns = state["episode_returns"]
        self.telemetry.latest = state["metrics"]
        # Upstream load resets seeds, so restore RNG after the entire coordinator is initialized.
        random.setstate(tuples(state["python_rng"]))
        np.random.set_state(numpy_rng_restore(state["numpy_rng"]))
        self.experience_buffer.rng.set_state(numpy_rng_restore(state["buffer_rng"]))
        rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["torch"])
        if self.app_config.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])

    def process_timestep_data(self, data):
        for env_id, (timesteps, _, _) in data.items():
            if not timesteps:
                continue
            totals = self.partial_returns.setdefault(env_id, {})
            for step in timesteps:
                totals[step.agent_id] = totals.get(step.agent_id, 0.0) + step.reward
                goal = (10.0 if step.reward > 0 else -10.0) if step.terminated else 0.0
                self.reward_goal_abs += abs(goal)
                self.reward_touch += step.reward - goal
            if all(step.terminated or step.truncated for step in timesteps):
                self.episodes += 1
                self.goal_episodes += int(any(step.terminated for step in timesteps))
                self.episode_returns.append(sum(totals.values()) / len(totals))
                self.episode_returns = self.episode_returns[-100:]
                self.partial_returns.pop(env_id, None)
        previous_iteration = self.cur_iteration
        super().process_timestep_data(data)
        if self.cur_iteration != previous_iteration:
            self.telemetry.latest["reward_components"] = {
                "goal_absolute_sum": self.reward_goal_abs, "touch_sum": self.reward_touch,
            }
            self.reward_goal_abs = self.reward_touch = 0.0
        self.channel.last_progress = time.monotonic()

    def get_env_actions(self, env_obs_data_dict, env_state_info_dict):
        # Upstream clears current_env_trajectories at an update and can lose terminal flags.
        # Use the collector's authoritative flags, including when a goal coincides with an update.
        actions, to_step = {}, {}
        for env_id, obs_data in env_obs_data_dict.items():
            _, _, terminated, truncated = env_state_info_dict[env_id]
            done = terminated is not None and all(
                terminated[agent] or truncated[agent] for agent in obs_data[0]
            )
            if done:
                actions[env_id] = EnvAction.RESET()
            else:
                to_step[env_id] = obs_data
        if to_step:
            actions.update({env_id: EnvAction.STEP(action_list=action_list)
                            for env_id, action_list in self.get_actions(to_step).items()})
        self.process_env_actions(actions)
        return 0, actions

    def handle_env_closes(self, reasons):
        if reasons:
            raise RuntimeError(
                "A simulation process exited unexpectedly. Check worker logs and mesh installation; "
                "recover the last valid checkpoint after fixing the cause."
            )

    def snapshot(self):
        return finite({
            **self.telemetry.latest, "steps": self.cumulative_timesteps,
            "iterations": self.cur_iteration,
            "updates": self.learner.cumulative_model_updates,
            "episodes": self.episodes, "goal_episodes": self.goal_episodes,
            "episodic_return": float(np.mean(self.episode_returns)) if self.episode_returns else None,
            "win_rate": None, "device": self.app_config.device,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated() if self.app_config.device.startswith("cuda") else None,
            "gpu_reserved_bytes": torch.cuda.memory_reserved() if self.app_config.device.startswith("cuda") else None,
        })

    def save_checkpoint(self):
        if self.iteration_timesteps != 0:
            raise RuntimeError("Refusing to save partially collected or partially updated PPO state")
        if self.last_checkpoint_step == self.cumulative_timesteps:
            return
        metrics = self.snapshot()

        def writer(path):
            self.learner.save_checkpoint(path / "ppo_learner")
            self.experience_buffer.save_checkpoint(path / "experience_buffer")
            atomic_json(path / "ppo_agent.json", {
                "cur_iteration": self.cur_iteration, "iteration_timesteps": 0,
                "cumulative_timesteps": self.cumulative_timesteps,
            })
            atomic_json(path / "application.json", {
                "metrics": metrics, "episode_returns": self.episode_returns,
                "numpy_rng": numpy_rng_json(np.random.get_state()),
                "python_rng": random.getstate(),
                "buffer_rng": numpy_rng_json(self.experience_buffer.rng.get_state()),
            })
            torch.save({"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if self.app_config.device.startswith("cuda") else []},
                       path / "rng.pt")

        manifest = self.checkpoint_store.publish(
            self.cumulative_timesteps, self.app_config.digest(), writer,
            {"metrics": metrics, "resume": "completed PPO update; environments reset", "evaluated": False},
        )
        self.last_checkpoint_step = self.cumulative_timesteps
        self.channel.emit("checkpoint", checkpoint_id=manifest["id"], step=manifest["step"])


def coordinator_config(config: TrainingConfig, run_dir: Path, checkpoint: Path | None):
    obs_width = Float32DefaultObs(zero_padding=1).get_obs_space("blue-0")[1]
    space_serde = PyAnySerdeType.TUPLE((PyAnySerdeType.STRING(), PyAnySerdeType.INT()))
    return LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            random_seed=config.seed, timestep_limit=config.total_timesteps,
            flinks_folder=str(run_dir / "shmem_flinks"),
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float32, config=NumpySerdeConfig.STATIC(
                    shape=(obs_width,), allocation_pool_warning_size=None,
                )),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64, config=NumpySerdeConfig.STATIC(
                    shape=(1,), allocation_pool_warning_size=None,
                )),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=space_serde, action_space_serde_type=space_serde,
                shared_info_serde_type=None,
            ),
        ),
        process_config=ProcessConfigModel(
            n_proc=config.n_proc, render=False, recalculate_agent_id_every_step=False,
            min_frac_process_responses_per_collection=1.0,
        ),
        agent_controller_config=PPOAgentControllerConfigModel(
            timesteps_per_iteration=config.timesteps_per_iteration, save_every_ts=10**18,
            checkpoint_load_folder=str(checkpoint) if checkpoint else None,
            random_seed=config.seed, save_mid_iteration_data_in_checkpoint=False,
            run_name="foundation", run_suffix="", n_checkpoints_to_keep=1_000_000,
            learner_config=PPOLearnerConfigModel(
                device=config.device, n_epochs=config.n_epochs, batch_size=config.batch_size,
                n_minibatches=config.n_minibatches, ent_coef=config.entropy_coefficient,
                clip_range=config.clip_range, max_grad_norm=config.max_grad_norm,
                optimizer_named_parameter_group_kwargs={
                    "actor": {"lr": config.learning_rate}, "critic": {"lr": config.learning_rate},
                },
            ),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=config.buffer_size, device="cpu",
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(
                    gamma=config.gamma, lmbda=config.gae_lambda, standardize_rewards=True,
                ),
            ),
        ),
        # Never let the upstream retention writer see managed checkpoints.
        agent_controller_save_folder=str(run_dir / "unused-upstream-checkpoints"),
    )


class ControlledCoordinator(LearningCoordinator):
    def __init__(self, config: TrainingConfig, run_dir: Path, channel: WorkerChannel, checkpoint=None):
        self.app_config, self.channel = config, channel
        controller = FoundationController(config, run_dir, channel)
        resolved = coordinator_config(config, run_dir, checkpoint)
        # Save a portable resolved artifact, not the host-specific runtime paths.
        artifact = json.loads(resolved.model_dump_json(polymorphic_serialization=True))
        artifact["base_config"]["flinks_folder"] = "./shmem_flinks"
        artifact["agent_controller_save_folder"] = "./unused-upstream-checkpoints"
        if checkpoint:
            artifact["agent_controller_config"]["checkpoint_load_folder"] = "<verified-parent-checkpoint>"
        atomic_json(run_dir / "resolved_config.json", artifact)
        # Match Learn's process context. With min_frac_responses=1.0, every environment
        # publishes exactly one response per wave; a shared semaphore is a deterministic
        # release fence, not a delay or a replacement collector. See docs/UPSTREAM.md.
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self.response_fence = mp.get_context(method).Semaphore(0)
        super().__init__(EnvironmentFactory(config, self.response_fence), controller, config=resolved)
        if checkpoint:
            controller.restore_application_state(checkpoint)
        self.cumulative_timesteps = controller.cumulative_timesteps
        # Initialization is not rollout throughput. Recovery also discards old monotonic times.
        controller.iteration_start_time = controller.timestep_collection_start_time = time.perf_counter()
        self.channel.last_progress = time.monotonic()

    def run_controlled(self):
        """Keep the real collector and controller; replace only upstream terminal lifecycle."""
        controller = self.agent_controller
        controller.save_checkpoint()
        self.channel.emit("ready", metrics=controller.snapshot())
        previous_obs, previous_info = {}, {}
        last_report = 0.0
        while self.cumulative_timesteps < self.app_config.total_timesteps:
            collected, closed, obs, timestep_data, info, spaces = self.env_process_interface.collect_env_responses(
                previous_obs, previous_info,
            )
            self.cumulative_timesteps += collected
            controller.handle_env_closes(closed)
            previous_iteration = controller.cur_iteration
            controller.process_timestep_data(timestep_data)
            controller.set_space_types(spaces)
            updated = controller.cur_iteration != previous_iteration
            if updated:
                if any(not torch.isfinite(parameter).all() for parameter in controller.learner.actor_critic.parameters()):
                    raise RuntimeError("PPO parameters became non-finite; last valid checkpoint preserved")
                if controller.cur_iteration % self.app_config.checkpoint_every_iterations == 0:
                    controller.save_checkpoint()
            if updated or time.monotonic() - last_report > 0.5:
                self.channel.emit("metrics", metrics=controller.snapshot(), updated=updated)
                last_report = time.monotonic()
            request = self.channel.poll()
            at_boundary = controller.iteration_timesteps == 0
            if at_boundary and request:
                controller.save_checkpoint()
                if request == "stop":
                    self.channel.emit("state", status="stopping", message="Checkpoint committed; shutting down simulation processes")
                    return "stopped"
                pause_started = time.perf_counter()
                self.channel.emit("state", status="paused", message="Paused at a completed PPO update", metrics=controller.snapshot())
                while self.channel.poll() == "pause":
                    time.sleep(0.05)
                if self.channel.request == "stop":
                    return "stopped"
                paused_for = time.perf_counter() - pause_started
                controller.iteration_start_time += paused_for
                controller.timestep_collection_start_time += paused_for
                self.channel.last_progress = time.monotonic()
                self.channel.emit("state", status="running", message="Training resumed")
            if self.cumulative_timesteps >= self.app_config.total_timesteps:
                break
            _, env_actions = controller.get_env_actions(obs, info)
            if not env_actions:
                raise RuntimeError("Collector returned no environments; stopping the run")
            self.env_process_interface.send_env_actions(env_actions)
            for _ in env_actions:
                self.response_fence.release()
            previous_obs, previous_info = obs, info
        controller.save_checkpoint()
        return "completed"
