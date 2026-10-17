"""Small, strict application config. Never import training libraries in the service."""

import hashlib
import json
import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(default="Foundation · 1v1", min_length=1, max_length=80)
    device: Literal["cuda:0", "cpu"] = "cuda:0"
    seed: int = Field(default=123, ge=0, le=2**31 - 1)
    n_proc: int = Field(default=1, ge=1, le=32)
    total_timesteps: int = Field(default=8192, ge=128, le=1_000_000_000)
    timesteps_per_iteration: int = Field(default=512, ge=128, le=131072)
    batch_size: int = Field(default=512, ge=128, le=131072)
    buffer_size: int = Field(default=1536, ge=128, le=393216)
    n_minibatches: int = Field(default=2, ge=1, le=64)
    n_epochs: int = Field(default=1, ge=1, le=4)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    entropy_coefficient: float = Field(default=0.01, ge=0, le=0.1)
    clip_range: float = Field(default=0.2, ge=0.05, le=0.5)
    max_grad_norm: float = Field(default=0.5, ge=0.1, le=5.0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    checkpoint_every_iterations: int = Field(default=2, ge=1, le=500)
    episode_timeout_seconds: float = Field(default=30, ge=1, le=300)
    no_touch_timeout_seconds: float = Field(default=10, ge=1, le=60)
    pin_memory: bool = False
    # Supports 1v1 (2 bots on field) and 2v2 (4 bots on field) with shared 92-float obs contract.
    team_size: Literal[1, 2] = 1
    tick_skip: Literal[8] = 8
    boost_mode: Literal["standard", "unlimited", "starved"] = "standard"
    arena_map: Literal[
        "DFHStadium",
        "ChampionsField",
        "Mannfield_Night",
        "NeonFields",
        "UtopiaColiseum",
    ] = "DFHStadium"
    environment_version: Literal["kickoff-v1", "mixed-scenarios-v1"] = "kickoff-v1"
    reward_version: Literal[
        "goal10-touch0.1-transition-v1",
        "goal10-touch0.1-shaped-v2",
        "goal10-touch0.1-pro-v3",
    ] = "goal10-touch0.1-transition-v1"
    # Opt-in dense shaping (v2 and v3). Defaults are zero so foundation reward is bit-identical.
    reward_velocity_to_ball_weight: float = Field(default=0.0, ge=0, le=2.0)
    reward_ball_to_goal_weight: float = Field(default=0.0, ge=0, le=2.0)
    # Pro v3 reward weights (boost management, defensive positioning/saves, aerial/power shots, speedflip/mechanics).
    reward_boost_weight: float = Field(default=0.0, ge=0, le=2.0)
    reward_defensive_weight: float = Field(default=0.0, ge=0, le=2.0)
    reward_aerial_shot_weight: float = Field(default=0.0, ge=0, le=2.0)
    reward_mechanics_weight: float = Field(default=0.0, ge=0, le=2.0)
    self_play_enabled: bool = False

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if not 1 <= len(self.hidden_sizes) <= 4 or any(
            type(size) is not int or not 16 <= size <= 512 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain 1–4 integer widths between 16 and 512")
        if self.batch_size != self.timesteps_per_iteration:
            raise ValueError("batch_size must equal timesteps_per_iteration in the foundation adapter")
        if not self.batch_size <= self.buffer_size <= 3 * self.batch_size:
            raise ValueError("buffer_size must be between one and three batches")
        if self.batch_size % self.n_minibatches or self.batch_size // self.n_minibatches < 16:
            raise ValueError("n_minibatches must divide batch_size with at least 16 samples per minibatch")
        if self.total_timesteps % self.timesteps_per_iteration:
            raise ValueError("total_timesteps must be a multiple of timesteps_per_iteration")
        if self.timesteps_per_iteration % (2 * self.team_size * self.n_proc):
            raise ValueError("timesteps_per_iteration must be divisible by 2 * team_size * n_proc (agent steps)")
        if self.no_touch_timeout_seconds > self.episode_timeout_seconds:
            raise ValueError("no_touch_timeout_seconds must not exceed episode_timeout_seconds")
        if self.device == "cpu" and self.pin_memory:
            raise ValueError("pin_memory requires an explicitly selected CUDA device")
        v2_shaping = self.reward_velocity_to_ball_weight + self.reward_ball_to_goal_weight
        v3_shaping = (
            self.reward_boost_weight
            + self.reward_defensive_weight
            + self.reward_aerial_shot_weight
            + self.reward_mechanics_weight
        )
        total_shaping = v2_shaping + v3_shaping
        if total_shaping > 0 and self.reward_version == "goal10-touch0.1-transition-v1":
            raise ValueError("dense shaping weights require reward_version goal10-touch0.1-shaped-v2 or goal10-touch0.1-pro-v3")
        if v3_shaping > 0 and self.reward_version != "goal10-touch0.1-pro-v3":
            raise ValueError("pro shaping weights (boost/defensive/aerial/mechanics) require reward_version goal10-touch0.1-pro-v3")
        if self.reward_version in {"goal10-touch0.1-shaped-v2", "goal10-touch0.1-pro-v3"} and total_shaping == 0:
            raise ValueError(f"{self.reward_version} requires at least one positive shaping weight")
        return self

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


TRAINING_PRESETS: dict[str, dict] = {
    "all_in_one_complete": {
        "id": "all_in_one_complete",
        "label": "All-in-One Master AI (Alles gleichzeitig + 100% Hardware)",
        "description": "Trainiert ab Sekunde 1 ALLE Fähigkeiten gleichzeitig: Kickoff, Speedflip, Torschüsse, Saves, Boost, Aerials, Wall-Reads, Recovery & Self-Play mit maximaler CPU/GPU-Auslastung.",
        "default_steps": 50_000_000,
        "overrides": {
            "name": "All-in-One Master AI",
            "team_size": 1,
            "boost_mode": "standard",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.35,
            "reward_ball_to_goal_weight": 0.65,
            "reward_boost_weight": 0.35,
            "reward_defensive_weight": 0.45,
            "reward_aerial_shot_weight": 0.50,
            "reward_mechanics_weight": 0.60,
            "self_play_enabled": True,
            "n_epochs": 2,
            "learning_rate": 0.00025,
            "entropy_coefficient": 0.008,
            "episode_timeout_seconds": 60.0,
            "no_touch_timeout_seconds": 15.0,
        },
    },
    "warmup": {
        "id": "warmup",
        "label": "Quick Warmup",
        "description": "Schneller Testlauf für Anstoß-Kontakte und erste PPO-Updates.",
        "default_steps": 262_144,
        "overrides": {
            "name": "Preset · Quick Warmup",
            "environment_version": "kickoff-v1",
            "reward_version": "goal10-touch0.1-transition-v1",
            "reward_velocity_to_ball_weight": 0.0,
            "reward_ball_to_goal_weight": 0.0,
            "reward_boost_weight": 0.0,
            "reward_defensive_weight": 0.0,
            "reward_aerial_shot_weight": 0.0,
            "self_play_enabled": False,
            "n_epochs": 1,
            "learning_rate": 0.0002,
            "entropy_coefficient": 0.01,
            "episode_timeout_seconds": 20.0,
            "no_touch_timeout_seconds": 8.0,
        },
    },
    "kickoff_basics": {
        "id": "kickoff_basics",
        "label": "Grundlagen & Kickoff",
        "description": "Zuverlässig zum Ball fahren, sichere Kickoffs und erste Torschüsse.",
        "default_steps": 1_000_000,
        "overrides": {
            "name": "Preset · Grundlagen & Kickoff",
            "environment_version": "kickoff-v1",
            "reward_version": "goal10-touch0.1-shaped-v2",
            "reward_velocity_to_ball_weight": 0.4,
            "reward_ball_to_goal_weight": 0.2,
            "reward_boost_weight": 0.0,
            "reward_defensive_weight": 0.0,
            "reward_aerial_shot_weight": 0.0,
            "self_play_enabled": False,
            "n_epochs": 1,
            "learning_rate": 0.0002,
            "entropy_coefficient": 0.01,
            "episode_timeout_seconds": 30.0,
            "no_touch_timeout_seconds": 10.0,
        },
    },
    "allround_striker": {
        "id": "allround_striker",
        "label": "Allround Striker & Positioning",
        "description": "Angriffs-, Verteidigungs- und Open-Play-Szenarien mit Boost- & Schuss-Shaping.",
        "default_steps": 10_000_000,
        "overrides": {
            "name": "Preset · Allround Striker",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.35,
            "reward_ball_to_goal_weight": 0.5,
            "reward_boost_weight": 0.2,
            "reward_defensive_weight": 0.25,
            "reward_aerial_shot_weight": 0.15,
            "self_play_enabled": False,
            "n_epochs": 2,
            "learning_rate": 0.0002,
            "entropy_coefficient": 0.01,
            "episode_timeout_seconds": 45.0,
            "no_touch_timeout_seconds": 12.0,
        },
    },
    "pro_selfplay": {
        "id": "pro_selfplay",
        "label": "Pro 1v1 Self-Play & Mechanics",
        "description": "Volles Pro-Reward-System inkl. Speedflip, Fast-Aerials, Recovery und 1v1 Self-Play.",
        "default_steps": 50_000_000,
        "overrides": {
            "name": "Preset · Pro 1v1 Self-Play",
            "team_size": 1,
            "boost_mode": "standard",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.25,
            "reward_ball_to_goal_weight": 0.6,
            "reward_boost_weight": 0.3,
            "reward_defensive_weight": 0.4,
            "reward_aerial_shot_weight": 0.35,
            "reward_mechanics_weight": 0.4,
            "self_play_enabled": True,
            "n_epochs": 2,
            "learning_rate": 0.00018,
            "entropy_coefficient": 0.008,
            "episode_timeout_seconds": 60.0,
            "no_touch_timeout_seconds": 15.0,
        },
    },
    "mechanics_speedflip": {
        "id": "mechanics_speedflip",
        "label": "Mechanics · Speedflip, Aerials & Wave-Dash",
        "description": "Intensives Mechanik-Training (Speedflip-Kickoffs, Fast-Aerials, Wall-to-Air, Clean Recovery) mit viel Boost.",
        "default_steps": 25_000_000,
        "overrides": {
            "name": "Preset · Pro Mechanics & Speedflip",
            "team_size": 1,
            "boost_mode": "unlimited",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.45,
            "reward_ball_to_goal_weight": 0.5,
            "reward_boost_weight": 0.15,
            "reward_defensive_weight": 0.2,
            "reward_aerial_shot_weight": 0.65,
            "reward_mechanics_weight": 0.85,
            "self_play_enabled": True,
            "n_epochs": 2,
            "learning_rate": 0.00018,
            "entropy_coefficient": 0.01,
            "episode_timeout_seconds": 45.0,
            "no_touch_timeout_seconds": 12.0,
        },
    },
    "team_2v2_pro": {
        "id": "team_2v2_pro",
        "label": "2v2 Team Self-Play (4 Bots auf dem Feld)",
        "description": "4 KI-Autos gleichzeitig pro Arena (2 Blue vs 2 Orange) für schnelles Team-, Pass- & Rotations-Training.",
        "default_steps": 50_000_000,
        "overrides": {
            "name": "Preset · 2v2 Team Self-Play",
            "team_size": 2,
            "boost_mode": "standard",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.25,
            "reward_ball_to_goal_weight": 0.65,
            "reward_boost_weight": 0.35,
            "reward_defensive_weight": 0.45,
            "reward_aerial_shot_weight": 0.4,
            "reward_mechanics_weight": 0.45,
            "self_play_enabled": True,
            "n_epochs": 2,
            "learning_rate": 0.00018,
            "entropy_coefficient": 0.008,
            "episode_timeout_seconds": 60.0,
            "no_touch_timeout_seconds": 15.0,
        },
    },
    "champion_master": {
        "id": "champion_master",
        "label": "Champion Long-Run",
        "description": "Meisterschafts-Finetuning mit Fokus auf Speedflips, Power-Shots, Aerials, Saves und taktischem Self-Play.",
        "default_steps": 150_000_000,
        "overrides": {
            "name": "Preset · Champion Long-Run",
            "team_size": 1,
            "boost_mode": "standard",
            "environment_version": "mixed-scenarios-v1",
            "reward_version": "goal10-touch0.1-pro-v3",
            "reward_velocity_to_ball_weight": 0.15,
            "reward_ball_to_goal_weight": 0.75,
            "reward_boost_weight": 0.35,
            "reward_defensive_weight": 0.5,
            "reward_aerial_shot_weight": 0.55,
            "reward_mechanics_weight": 0.6,
            "self_play_enabled": True,
            "n_epochs": 3,
            "learning_rate": 0.00015,
            "entropy_coefficient": 0.005,
            "episode_timeout_seconds": 90.0,
            "no_touch_timeout_seconds": 15.0,
        },
    },
}

ROADMAP_MILESTONES: list[dict] = [
    {
        "id": "milestone-1",
        "stage": 1,
        "title": "Stufe 1 · Ball Chaser, Kickoff & Speedflip-Start",
        "target_steps": 1_000_000,
        "preset": "all_in_one_complete",
        "summary": "Trainiert ab Schritt 1 für alles gleichzeitig: Anstoß, Speedflip-Ansatz, Ballkontakte, Boost und erste Tore.",
        "skills": [
            "Anstoß & Speedflip-Kickoff sicher treffen",
            "Direkt mit Boost zum Ball beschleunigen",
            "Erste direkte Tore & Paraden",
        ],
    },
    {
        "id": "milestone-2",
        "stage": 2,
        "title": "Stufe 2 · Striker, Boost & Saves",
        "target_steps": 10_000_000,
        "preset": "all_in_one_complete",
        "summary": "Gezielte Power-Shots aufs gegnerische Tor, 34-Boost-Pad-Routing und sichere Defensive.",
        "skills": [
            "Gezielte Schüsse aufs gegnerische Tor",
            "Boost-Pads aktiv einsammeln & einteilen",
            "Paraden (Saves) bei Gegnerschüssen",
        ],
    },
    {
        "id": "milestone-3",
        "stage": 3,
        "title": "Stufe 3 · Pro Mechanics (Speedflip, Aerials & Recovery)",
        "target_steps": 50_000_000,
        "preset": "all_in_one_complete",
        "summary": "Beherrschung von Speedflip-Kickoffs, Fast-Aerials, Wall-Reads und 4-Rad-Recoveries.",
        "skills": [
            "Speedflip-Kickoffs & Supersonic-Tempo",
            "Hohe Bälle (Fast Aerials) & Wand-Abpraller",
            "Saubere 4-Rad-Landung & Wave-Dash-Recovery",
        ],
    },
    {
        "id": "milestone-4",
        "stage": 4,
        "title": "Stufe 4 · Tactical Self-Play & Dribble",
        "target_steps": 150_000_000,
        "preset": "pro_selfplay",
        "summary": "Dribblings, Shadow-Defense, starke Abschlüsse und intensives Self-Play.",
        "skills": [
            "Shadow-Defense bei 1v1-Kontern",
            "Starke Power-Shots & hohe Ballkontakte",
            "Druckvolle 1v1- & 2v2-Zweikämpfe gewinnen",
        ],
    },
    {
        "id": "milestone-5",
        "stage": 5,
        "title": "Stufe 5 · Champion Master Bot",
        "target_steps": 500_000_000,
        "preset": "champion_master",
        "summary": "Langzeit-Meisterschaftstraining mit vollem Pro-Reward-Shaping, Speedflips und Aerial-Kontrolle.",
        "skills": [
            "Meisterhafte Aerials, Speedflips & Präzisionsschüsse",
            "Konstante Defensive gegen starke Gegner",
            "Maximales Niveau auf deiner Hardware (RTX 4060 Ti + i7-12700F)",
        ],
    },
]


def align_timesteps(target_steps: int, batch_size: int) -> int:
    """Round target_steps up to a valid multiple of batch_size within [batch_size, 1_000_000_000]."""
    step_unit = max(batch_size, 128)
    clamped = max(step_unit, min(int(target_steps), 1_000_000_000))
    multiples = max(1, math.ceil(clamped / step_unit))
    aligned = multiples * step_unit
    if aligned > 1_000_000_000:
        aligned = (1_000_000_000 // step_unit) * step_unit
    return max(step_unit, aligned)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def changes(before: TrainingConfig, after: TrainingConfig) -> dict:
    old, new = before.model_dump(mode="json"), after.model_dump(mode="json")
    return {key: {"before": old[key], "after": value} for key, value in new.items() if old[key] != value}
