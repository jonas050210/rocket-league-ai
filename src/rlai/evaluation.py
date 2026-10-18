"""Deterministic match evaluation.

A "game" is a fixed amount of game time (default 120 s).  Goals do not end
the game: the environment episode is reset after each goal and scoring
continues, exactly like a real match.  Seeds, deterministic action selection
and fixed opponents make results reproducible.

The same code path evaluates a checkpoint from disk and the live model inside
the trainer, so training-time metrics and manual evaluations are comparable.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .envs import MatchEnv
from .modes import ModeSpec, get_mode
from .obs import version_for_obs_size
from .opponents import ModelOpponent, build_opponent
from .policy import PolicyConfig, TorchPolicy


def load_policy(model_path: str | Path, device: str = "cpu") -> TorchPolicy:
    payload = torch.load(Path(model_path), map_location=device, weights_only=False)
    config = PolicyConfig.from_dict(payload.get("policy_config"))
    policy = TorchPolicy.from_config(config, device=device)
    policy.model.load_state_dict(payload["model"])
    policy.model.eval()
    return policy


def policy_obs_version(policy: TorchPolicy) -> int:
    """The observation layout a checkpoint was trained with."""
    return version_for_obs_size(int(policy.model.obs_size))


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a win rate.

    With 20 games the uncertainty is roughly ±20 percentage points; reporting
    it prevents over-reading small evaluation sets.
    """
    if games <= 0:
        # No games means no information, not a certain 0 % win rate.
        return (0.0, 1.0)
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    margin = (
        z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _opponent_for(name: str, seed: int, policy: TorchPolicy, device: str, spec: ModeSpec):
    if name == "self":
        return ModelOpponent(policy, name="self", deterministic=True)
    return build_opponent(name, seed=seed, device=device)


def evaluate_policy(
    policy: TorchPolicy,
    *,
    mode: str,
    opponent: str = "chase",
    games: int = 20,
    seed: int = 1000,
    match_seconds: float = 120.0,
    device: str = "cpu",
    episode_seconds: float = 30.0,
    obs_version: int | None = None,
    stochastic: bool = False,
) -> dict:
    spec = get_mode(mode)
    version = policy_obs_version(policy) if obs_version is None else int(obs_version)
    started = time.perf_counter()
    results = []
    for game_index in range(games):
        game_seed = seed + game_index * 7919
        env = MatchEnv(
            spec,
            _opponent_for(opponent, game_seed, policy, device, spec),
            seed=game_seed,
            episode_seconds=episode_seconds,
            obs_version=version,
        )
        obs = env.reset(episode_seed=game_seed)
        goals_for = 0
        goals_against = 0
        ticks = 0
        touches = 0
        previous_touches = 0
        target_ticks = int(match_seconds * 120)
        while ticks < target_ticks:
            agents = env.blue_agents
            actions, _, _ = policy.act(
                {agent: obs[agent] for agent in agents}, deterministic=not stochastic
            )
            result = env.step(actions)
            ticks += env.tick_skip
            if result.info.goal_scored:
                if result.info.scoring_team == 0:
                    goals_for += 1
                elif result.info.scoring_team == 1:
                    goals_against += 1
            current_touches = sum(int(env.state.cars[a].ball_touches) for a in agents)
            # Counters reset with the episode, so add the delta when the
            # counter grows and the new value when it was reset.
            touches += (
                current_touches - previous_touches
                if current_touches >= previous_touches
                else current_touches
            )
            previous_touches = current_touches
            if result.done or result.truncated:
                obs = env.reset()
                previous_touches = 0
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
                "seed": game_seed,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "outcome": outcome,
                "ticks": ticks,
                "touches": touches,
            }
        )
    wins = sum(1 for r in results if r["outcome"] == "win")
    losses = sum(1 for r in results if r["outcome"] == "loss")
    draws = sum(1 for r in results if r["outcome"] == "draw")
    goals_for = sum(r["goals_for"] for r in results)
    goals_against = sum(r["goals_against"] for r in results)
    low, high = wilson_interval(wins, games)
    decisive = wins + losses
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
        "win_rate_ci95": [round(low, 4), round(high, 4)],
        # Draws are common when two weak policies never score; report the
        # share of wins among games that produced a goal difference too.
        "win_rate_decisive": wins / decisive if decisive else None,
        "goal_diff": goals_for - goals_against,
        "touches": sum(r["touches"] for r in results),
        "seed": seed,
        "match_seconds": match_seconds,
        "device": device,
        "obs_version": version,
        "stochastic": stochastic,
        "duration_s": time.perf_counter() - started,
        "details": {"games": results},
    }


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
    stochastic: bool = False,
) -> dict:
    policy = load_policy(model_path, device=device)
    result = evaluate_policy(
        policy,
        mode=mode,
        opponent=opponent,
        games=games,
        seed=seed,
        match_seconds=match_seconds,
        device=device,
        episode_seconds=episode_seconds,
        stochastic=stochastic,
    )
    result["model"] = str(model_path)
    return result
