"""Deterministic match evaluation.

A "game" is a fixed amount of game time (default 120 s).  Goals do not end
the game: the environment episode is reset after each goal and scoring
continues, exactly like a real match.  Seeds, deterministic action selection
and fixed opponents make results reproducible.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .envs import MatchEnv
from .modes import get_mode
from .opponents import ModelOpponent, build_opponent
from .policy import PolicyConfig, TorchPolicy


def load_policy(model_path: str | Path, device: str = "cpu") -> TorchPolicy:
    payload = torch.load(Path(model_path), map_location=device, weights_only=False)
    config = PolicyConfig.from_dict(payload.get("policy_config"))
    policy = TorchPolicy.from_config(config, device=device)
    policy.model.load_state_dict(payload["model"])
    policy.model.eval()
    return policy


def _opponent_for(name: str, seed: int, policy: TorchPolicy, device: str):
    if name == "self":
        return ModelOpponent(policy, name="self", deterministic=True)
    return build_opponent(name, seed=seed, device=device)


def evaluate_model(
    model_path: str | Path,
    *,
    mode: str,
    opponent: str = "chase",
    games: int = 20,
    seed: int = 1000,
    match_seconds: float = 120.0,
    device: str = "cpu",
    episode_seconds: float = 30.0,
) -> dict:
    spec = get_mode(mode)
    policy = load_policy(model_path, device=device)
    started = time.perf_counter()
    results = []
    for game_index in range(games):
        env = MatchEnv(
            spec,
            _opponent_for(opponent, seed + game_index, policy, device),
            seed=seed + game_index * 7919,
            episode_seconds=episode_seconds,
        )
        obs = env.reset(episode_seed=seed + game_index * 7919)
        goals_for = 0
        goals_against = 0
        ticks = 0
        target_ticks = int(match_seconds * 120)
        steps = 0
        while ticks < target_ticks and steps < target_ticks:
            agents = env.blue_agents
            actions, _, _ = policy.act({agent: obs[agent] for agent in agents}, deterministic=True)
            result = env.step(actions)
            steps += 1
            ticks += env.tick_skip
            if result.info.goal_scored:
                if result.info.scoring_team == 0:
                    goals_for += 1
                elif result.info.scoring_team == 1:
                    goals_against += 1
            if result.done or result.truncated:
                obs = env.reset()
            else:
                obs = result.obs
        if goals_for > goals_against:
            outcome = "win"
        elif goals_for < goals_against:
            outcome = "loss"
        else:
            outcome = "draw"
        results.append(
            {
                "game": game_index + 1,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "outcome": outcome,
                "ticks": ticks,
            }
        )
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    goals_for = sum(r["goals_for"] for r in results)
    goals_against = sum(r["goals_against"] for r in results)
    return {
        "mode": mode,
        "opponent": opponent,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win_rate": wins / max(games, 1),
        "goal_diff": goals_for - goals_against,
        "seed": seed,
        "match_seconds": match_seconds,
        "device": device,
        "duration_s": time.perf_counter() - started,
        "details": {"games": results},
    }
