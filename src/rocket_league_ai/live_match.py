"""Live interactive Rocket League match controller (Real RLBot launcher + live RocketSim arena).

Supports:
1. Launching real Rocket League via RLBot (`rlbot.cfg` + `run_rlbot_match.py` + Epic/Steam/WSL bridge)
   so the user can play Human vs AI or watch AI vs AI inside the actual Rocket League game client.
2. Running a real-time interactive RocketSim 1v1 match loop in the background that streams live
   frames, scores, boost meters, and controller telemetry to the Web Control Room and responds
   immediately to Web UI buttons and keyboard hotkeys (Start/Stop, Pause/Resume, Reset Kickoff,
   Next Game, Speed multiplier).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np

from .config import TrainingConfig
from .training.evaluate_core import DECISIONS_PER_SECOND, extract_frame


def _smart_chaser_action(obs: np.ndarray, action_table: np.ndarray, rng: np.random.Generator) -> int:
    """Built-in responsive 1v1 ball-chaser policy over the 90 LookupTableAction rows.

    Used when no trained checkpoint is loaded yet (or for the opponent in demo mode)
    so the live simulation always plays active, realistic 1v1 Rocket League.
    """
    # DefaultObs layout:
    # [0..3]: ball pos / 2300, [3..6]: ball vel / 2300
    # [52..55]: car pos / 2300, [55..58]: car forward, [58..61]: car up
    ball_pos = obs[0:3] * 2300.0
    car_pos = obs[52:55] * 2300.0
    car_fwd = obs[55:58]
    on_ground = obs[69] > 0.5
    boost_amt = obs[67] * 100.0

    to_ball = ball_pos - car_pos
    dist_2d = float(math.hypot(to_ball[0], to_ball[1])) + 1e-6
    target_yaw = math.atan2(float(to_ball[1]), float(to_ball[0]))
    car_yaw = math.atan2(float(car_fwd[1]), float(car_fwd[0]))
    angle_diff = (target_yaw - car_yaw + math.pi) % (2 * math.pi) - math.pi

    desired_steer = float(np.clip(angle_diff * 2.2, -1.0, 1.0))
    aligned = abs(angle_diff) < 0.38
    want_boost = aligned and dist_2d > 450.0 and boost_amt > 10.0
    want_jump = dist_2d < 280.0 and ball_pos[2] > 140.0 and on_ground and rng.random() < 0.35
    want_slide = abs(angle_diff) > 1.35 and on_ground

    # Score all 90 actions in LookupTableAction: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    best_idx = 0
    best_score = -1e9
    for idx in range(len(action_table)):
        row = action_table[idx]
        throttle, steer, pitch, _yaw, _roll, jump, boost, handbrake = row
        if throttle < 0.5:
            continue
        score = 2.0 * throttle - 1.8 * abs(steer - desired_steer)
        if want_boost and boost > 0.5:
            score += 1.2
        elif not want_boost and boost > 0.5:
            score -= 0.6
        if want_jump and jump > 0.5:
            score += 1.5
        elif not want_jump and jump > 0.5:
            score -= 1.5
        if want_slide and handbrake > 0.5:
            score += 0.8
        elif not want_slide and handbrake > 0.5:
            score -= 0.8
        if not on_ground:
            score -= 0.5 * abs(pitch)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def write_rlbot_match_files(
    export_dir: Path,
    mode: str = "human_vs_ai",
    *,
    team_size: int = 1,
    boost_mode: str = "standard",
    arena_map: str = "DFHStadium",
) -> dict[str, str]:
    """Write `rlbot.cfg` and `run_rlbot_match.py` into an exported RLBot checkpoint folder."""
    is_human = mode == "human_vs_ai"
    total_players = 2 * max(1, min(2, int(team_size)))
    boost_mutator = "Unlimited" if boost_mode == "unlimited" else ("No Boost" if boost_mode == "starved" else "Default")
    participants = []
    for i in range(total_players):
        team_num = 0 if i < total_players // 2 else 1
        p_type = "human" if (is_human and i == 0) else "rlbot"
        p_cfg = "None" if p_type == "human" else "./bot.cfg"
        participants.append(
            f"participant_config_{i} = {p_cfg}\n"
            f"participant_team_{i} = {team_num}\n"
            f"participant_type_{i} = {p_type}\n"
            f"participant_bot_skill_{i} = 1.0\n"
        )
    rlbot_cfg = (
        "[RLBot Configuration]\n"
        "extension_path = None\n\n"
        "[Team Configuration]\n"
        "Team Blue Color = 0\n"
        "Team Blue Name = Blue\n"
        "Team Orange Color = 0\n"
        "Team Orange Name = Orange\n\n"
        "[Match Configuration]\n"
        f"num_participants = {total_players}\n"
        "game_mode = Soccer\n"
        f"game_map = {arena_map}\n"
        "skip_replays = False\n"
        "start_without_countdown = False\n"
        "existing_match_behavior = Restart_If_Different\n"
        "enable_lockstep = False\n"
        "enable_rendering = False\n"
        "enable_state_setting = True\n"
        "auto_save_replay = False\n\n"
        "[Mutator Configuration]\n"
        "Match Length = 5 Minutes\n"
        "Max Score = Unlimited\n"
        "Overtime = Unlimited\n"
        f"Boost Amount = {boost_mutator}\n\n"
        "[Participant Configuration]\n" + "".join(participants)
    )
    runner_py = (
        '"""Standalone RLBot match launcher for Rocket League AI."""\n'
        "import pathlib\nimport time\n\n"
        "HERE = pathlib.Path(__file__).resolve().parent\n\n"
        "def main():\n"
        "    from rlbot.setup_manager import SetupManager\n"
        "    manager = SetupManager()\n"
        "    manager.load_config(config_location=str(HERE / 'rlbot.cfg'))\n"
        "    manager.connect_to_game()\n"
        "    manager.launch_early_start_bot_processes()\n"
        "    manager.start_match()\n"
        "    manager.launch_bot_processes()\n"
        "    while not manager.has_received_metadata:\n"
        "        manager.try_recieve_agent_metadata()\n"
        "        time.sleep(0.1)\n"
        "    manager.infinite_loop()\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (export_dir / "rlbot.cfg").write_text(rlbot_cfg)
    (export_dir / "run_rlbot_match.py").write_text(runner_py)
    return {"rlbot.cfg": str(export_dir / "rlbot.cfg"), "run_rlbot_match.py": str(export_dir / "run_rlbot_match.py")}


class LiveMatchController:
    """Manages live RocketSim matches and optional real Rocket League (RLBot) client launches."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._paused = False
        self._reset_requested = False
        self._speed = 1.0
        self._mode: str = "ai_vs_ai"
        self._scenario: str = "mixed-scenarios-v1"
        self._team_size: int = 1
        self._boost_mode: str = "standard"
        self._arena_map: str = "DFHStadium"
        self._status: Literal["idle", "running", "paused", "stopped", "error"] = "idle"
        self._error: str | None = None
        self._client_process: subprocess.Popen | None = None
        self._client_status: str = "idle"
        self._client_message: str = "Bereit zum Starten (Live-Arena + RLBot Launcher)"
        self._export_dir: str | None = None
        self._checkpoint_info: dict | None = None
        self._frames: deque[dict] = deque(maxlen=240)
        self._current_frame: dict | None = None
        self._score: list[int] = [0, 0]
        self._game_number: int = 1
        self._decisions: int = 0
        self._last_goal: dict | None = None
        self._human_controls: dict[str, float] | None = None
        self._human_controls_until: float = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": "paused" if (self._status == "running" and self._paused) else self._status,
                "paused": self._paused,
                "mode": self._mode,
                "scenario": self._scenario,
                "team_size": self._team_size,
                "boost_mode": self._boost_mode,
                "arena_map": self._arena_map,
                "speed": self._speed,
                "score": list(self._score),
                "game_number": self._game_number,
                "elapsed_seconds": round(self._decisions / DECISIONS_PER_SECOND, 1),
                "last_goal": self._last_goal,
                "client_status": self._client_status,
                "client_message": self._client_message,
                "export_dir": self._export_dir,
                "checkpoint": self._checkpoint_info,
                "current_frame": self._current_frame,
                "recent_frames": list(self._frames)[-60:],
                "error": self._error,
            }

    def start(
        self,
        *,
        mode: str = "human_vs_ai",
        scenario: str = "mixed-scenarios-v1",
        team_size: int = 1,
        boost_mode: str = "standard",
        arena_map: str = "DFHStadium",
        speed: float = 1.0,
        launch_client: bool = True,
        run_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> dict:
        self.stop()
        with self.lock:
            self._stop_event.clear()
            self._paused = False
            self._reset_requested = False
            self._mode = mode if mode in ("human_vs_ai", "ai_vs_ai", "sim_live") else "human_vs_ai"
            self._scenario = scenario if scenario in ("kickoff-v1", "mixed-scenarios-v1") else "mixed-scenarios-v1"
            self._team_size = 2 if int(team_size) == 2 else 1
            self._boost_mode = boost_mode if boost_mode in ("standard", "unlimited", "starved") else "standard"
            self._arena_map = arena_map if arena_map in ("DFHStadium", "ChampionsField", "Mannfield_Night", "NeonFields", "UtopiaColiseum") else "DFHStadium"
            self._speed = float(max(0.25, min(4.0, speed)))
            self._score = [0, 0]
            self._game_number = 1
            self._decisions = 0
            self._last_goal = None
            self._frames.clear()
            self._current_frame = None
            self._error = None
            self._status = "running"

        # Resolve checkpoint if available
        target_ref = None
        if run_id and checkpoint_id:
            target_ref = {"run_id": run_id, "checkpoint_id": checkpoint_id}
        else:
            target_ref = self.orchestrator.latest_checkpoint_ref()

        bundle = None
        if target_ref:
            try:
                bundle = self.orchestrator.export_rlbot_bundle(target_ref["run_id"], target_ref["checkpoint_id"])
                export_dir = Path(bundle["export_dir"])
                write_rlbot_match_files(
                    export_dir,
                    mode=self._mode,
                    team_size=self._team_size,
                    boost_mode=self._boost_mode,
                    arena_map=self._arena_map,
                )
                with self.lock:
                    self._export_dir = str(export_dir)
                    self._checkpoint_info = {
                        "run_id": bundle["run_id"],
                        "checkpoint_id": bundle["checkpoint_id"],
                        "step": bundle["step"],
                    }
            except Exception as exc:
                with self.lock:
                    self._checkpoint_info = None
                    self._client_message = f"Checkpoint-Export Hinweis: {exc}"

        if launch_client and self._mode in ("human_vs_ai", "ai_vs_ai"):
            self._try_launch_real_rocket_league()
        else:
            with self.lock:
                self._client_status = "sim_active"
                self._client_message = "Live RocketSim 1v1 Arena läuft im Web Control Room."

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="live-match-loop")
        self._thread.start()
        # Wait briefly for the first frame so the caller immediately gets an initial arena state
        for _ in range(15):
            if self._current_frame is not None or self._status == "error":
                break
            time.sleep(0.02)
        return self.snapshot()

    def _try_launch_real_rocket_league(self) -> None:
        """Attempt to start real Rocket League + RLBot runner when installed on the host/WSL."""
        export_dir = self._export_dir
        try:
            import importlib.util

            has_rlbot = importlib.util.find_spec("rlbot") is not None
        except Exception:
            has_rlbot = False

        if has_rlbot and export_dir:
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(Path(export_dir) / "run_rlbot_match.py")],
                    cwd=export_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with self.lock:
                    self._client_process = proc
                    self._client_status = "rlbot_launched"
                    self._client_message = (
                        f"RLBot Match-Runner (PID {proc.pid}) gestartet! Rocket League verbindet sich mit {export_dir}."
                    )
                return
            except OSError:
                pass

        # Check Windows / WSL2 Epic Games / Steam protocol launcher
        cmd_exe = shutil.which("cmd.exe")
        if cmd_exe or os.name == "nt":
            try:
                epic_url = (
                    "com.epicgames.launcher://apps/"
                    "9773aa1aa54f4f7b80e44bef04986cea%3A530145df28a24424923f5828cc9031a1%3ASugar"
                    "?action=launch&silent=true"
                )
                if os.name == "nt":
                    os.startfile(epic_url)  # type: ignore[attr-defined]
                elif cmd_exe:
                    subprocess.Popen(
                        [cmd_exe, "/c", "start", "", epic_url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                with self.lock:
                    self._client_status = "game_client_launched"
                    self._client_message = (
                        "Rocket League über Windows/Epic gestartet + RLBot-Match-Dateien bereit unter "
                        f"{export_dir or 'exports/'}. Live-Simulation läuft parallel!"
                    )
                return
            except OSError:
                pass

        with self.lock:
            self._client_status = "sim_live_ready"
            self._client_message = (
                "Kein lokaler RocketLeague.exe-Client in dieser Linux-Umgebung gefunden – "
                "interaktive Live-Simulation (RocketSim 120Hz) läuft direkt im Browser! "
                + (f"RLBot-Match-Paket bereit unter: {export_dir}" if export_dir else "")
            )

    def control(
        self,
        action: str,
        *,
        speed: float | None = None,
        scenario: str | None = None,
        boost_mode: str | None = None,
        arena_map: str | None = None,
        human_controls: dict[str, float] | None = None,
    ) -> dict:
        with self.lock:
            if action == "toggle_pause":
                self._paused = not self._paused
            elif action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
            elif action in ("reset_kickoff", "next_game"):
                self._reset_requested = True
                if action == "next_game":
                    self._game_number += 1
                self._paused = False
            elif action == "set_speed" and speed is not None:
                self._speed = float(max(0.25, min(4.0, speed)))
            elif action == "set_scenario" and scenario in ("kickoff-v1", "mixed-scenarios-v1"):
                self._scenario = scenario
                self._reset_requested = True
            elif action == "set_boost" and boost_mode in ("standard", "unlimited", "starved"):
                self._boost_mode = boost_mode
                self._reset_requested = True
            elif action == "set_map" and arena_map in ("DFHStadium", "ChampionsField", "Mannfield_Night", "NeonFields", "UtopiaColiseum"):
                self._arena_map = arena_map
            elif action == "human_input" and human_controls is not None:
                self._human_controls = dict(human_controls)
                self._human_controls_until = time.monotonic() + 0.65
            else:
                raise ValueError(f"Unknown live match control action: {action}")
        return self.snapshot()

    def stop(self) -> dict:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        with self.lock:
            self._thread = None
            if self._client_process and self._client_process.poll() is None:
                try:
                    self._client_process.terminate()
                except OSError:
                    pass
            self._client_process = None
            self._status = "stopped" if self._status != "idle" else "idle"
            self._paused = False
            self._client_status = "stopped"
            self._client_message = "Live-Match gestoppt."
        return self.snapshot()

    def _load_checkpoint_actor(self):
        """Try loading the PyTorch DiscreteFF actor from the active checkpoint if torch is installed."""
        info = self._checkpoint_info
        if not info:
            return None
        try:
            import torch
            from rlgym_learn_algos.ppo import BasicCritic, DiscreteFF, SeparateActorCritic

            run = self.orchestrator.store.run(info["run_id"])
            cfg = TrainingConfig.model_validate(run["config"])
            ckpt_path = self.orchestrator.checkpoints(info["run_id"]).path(info["checkpoint_id"])
            actor_critic = SeparateActorCritic(
                DiscreteFF(92, 90, cfg.hidden_sizes, torch.float32, torch.device("cpu")),
                BasicCritic(92, cfg.hidden_sizes, torch.float32, torch.device("cpu")),
            )
            actor_critic.load_state_dict(
                torch.load(ckpt_path / "ppo_learner" / "actor_critic.pt", map_location="cpu", weights_only=True)
            )
            actor_critic.eval()
            actor = actor_critic.actor

            def act_fn(obs: np.ndarray) -> int:
                with torch.no_grad():
                    probs = actor.get_output([np.asarray(obs, dtype=np.float32)])
                    return int(probs.argmax(dim=-1).item())

            return act_fn
        except Exception:
            return None

    def _run_loop(self) -> None:
        try:
            from rlgym.rocket_league.action_parsers import LookupTableAction

            from .training.environment import build_environment

            env_cfg = TrainingConfig(
                device="cpu",
                team_size=self._team_size,  # type: ignore[arg-type]
                boost_mode=self._boost_mode,  # type: ignore[arg-type]
                arena_map=self._arena_map,  # type: ignore[arg-type]
                environment_version=self._scenario,  # type: ignore[arg-type]
                episode_timeout_seconds=120.0,
                no_touch_timeout_seconds=30.0,
                self_play_enabled=True,
                seed=int(time.time()) % 100000,
            )
            env = build_environment(env_cfg)
            action_table = LookupTableAction()._lookup_table
            rng = np.random.default_rng(env_cfg.seed)
            trained_actor = self._load_checkpoint_actor()

            obs = env.reset()
            while not self._stop_event.is_set():
                with self.lock:
                    paused = self._paused
                    reset_req = self._reset_requested
                    speed = self._speed
                    game_num = self._game_number
                    score_tuple = (self._score[0], self._score[1])
                    boost_mode = self._boost_mode

                if reset_req:
                    with self.lock:
                        self._reset_requested = False
                        self._decisions = 0
                    env.boost_mode = boost_mode
                    obs = env.reset()
                    continue

                if paused:
                    time.sleep(0.05)
                    continue

                env.boost_mode = boost_mode
                chosen_actions: dict[str, int] = {}
                with self.lock:
                    hc = self._human_controls if time.monotonic() < self._human_controls_until else None
                for agent_id, agent_obs in obs.items():
                    is_blue = str(agent_id).startswith("blue")
                    if agent_id == "blue-0" and hc is not None:
                        # Map human browser WASD/Arrow + Boost/Jump controls to closest LookupTableAction row
                        t_in = float(hc.get("throttle", 1.0))
                        s_in = float(hc.get("steer", 0.0))
                        b_in = float(hc.get("boost", 0.0))
                        j_in = float(hc.get("jump", 0.0))
                        h_in = float(hc.get("handbrake", 0.0))
                        best_i, best_d = 8, 1e9
                        for idx_row in range(len(action_table)):
                            r = action_table[idx_row]
                            dist = (
                                2.0 * abs(r[0] - t_in)
                                + 2.0 * abs(r[1] - s_in)
                                + 1.5 * abs(r[6] - b_in)
                                + 2.0 * abs(r[5] - j_in)
                                + 1.2 * abs(r[7] - h_in)
                                + 0.5 * (abs(r[2]) + abs(r[3]) + abs(r[4]))
                            )
                            if dist < best_d:
                                best_d, best_i = dist, idx_row
                        chosen_actions[agent_id] = best_i
                    elif trained_actor is not None and (is_blue or self._mode == "ai_vs_ai"):
                        chosen_actions[agent_id] = trained_actor(agent_obs)
                    else:
                        chosen_actions[agent_id] = _smart_chaser_action(agent_obs, action_table, rng)

                blue_act = chosen_actions.get("blue-0", 8)
                orange_act = chosen_actions.get("orange-0", 8)
                frame = extract_frame(env.state, game_num, self._decisions, score_tuple)
                if frame is not None:
                    frame["candidate_action"] = int(blue_act)
                    frame["opponent_action"] = int(orange_act)
                    frame["candidate_action_vector"] = [float(v) for v in action_table[int(blue_act)]]
                    frame["opponent_action_vector"] = [float(v) for v in action_table[int(orange_act)]]
                    with self.lock:
                        self._current_frame = frame
                        self._frames.append(frame)

                actions = {agent_id: np.array([act], dtype=np.int64) for agent_id, act in chosen_actions.items()}
                obs, rewards, terminated, truncated = env.step(actions)
                with self.lock:
                    self._decisions += 1

                if all(terminated.values()):
                    blue_rew = float(rewards.get("blue-0", 0.0))
                    with self.lock:
                        if blue_rew > 5.0:
                            self._score[0] += 1
                            self._last_goal = {"team": "blue", "time": round(self._decisions / DECISIONS_PER_SECOND, 1)}
                        elif blue_rew < -5.0:
                            self._score[1] += 1
                            self._last_goal = {"team": "orange", "time": round(self._decisions / DECISIONS_PER_SECOND, 1)}
                        self._game_number += 1
                        self._decisions = 0
                    obs = env.reset()
                elif all(truncated.values()):
                    with self.lock:
                        self._game_number += 1
                        self._decisions = 0
                    obs = env.reset()

                sleep_time = max(0.01, (1.0 / DECISIONS_PER_SECOND) / max(0.25, speed))
                time.sleep(sleep_time)
            env.close()
        except Exception as exc:
            with self.lock:
                self._status = "error"
                self._error = str(exc)
