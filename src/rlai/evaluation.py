"""Deterministic, side-balanced match evaluation.

A "game" is a fixed amount of game time (default 120 s).  Goals do not end
the game: the environment episode is reset after each goal and scoring
continues, exactly like a real match.  Seeds, deterministic action selection
and fixed opponents make results reproducible.

Two rules keep the numbers honest:

* **Side balance.**  Rocket League is symmetric, but a checkpoint that only
  ever trained on blue sees out-of-distribution absolute field features when
  it plays orange.  A series can therefore be played on both sides, either
  with the observation mirrored into the trained side (default, measures
  strength) or raw (measures what the checkpoint really does on that side).
  Both variants are recorded in the result and in the protocol.
* **Protocol identity.**  Every evaluation carries a ``protocol`` describing
  every knob that can change the outcome.  Only results with the same
  protocol are comparable, and only those may update a "best model" pointer.

The same code path evaluates a checkpoint from disk and the live model inside
the trainer, so training-time metrics and manual evaluations are comparable.
"""

from __future__ import annotations

import math
import time
from pathlib import Path


from .envs import MatchEnv
from .modes import ModeSpec, get_mode
from .obs import version_for_obs_size
from .opponents import ModelOpponent, build_opponent
from .policy import MirroredPolicy, TorchPolicy

#: team ids used by RLGym / RocketSim
BLUE_TEAM = 0
ORANGE_TEAM = 1
SIDE_TEAM = {"blue": BLUE_TEAM, "orange": ORANGE_TEAM}

#: opponents that only prove the pipeline works - they say nothing about strength
DIAGNOSTIC_OPPONENTS = ("idle", "random")
#: opponents whose results describe playing strength
STRENGTH_OPPONENTS = ("chase", "mixed", "self")


def load_policy(model_path: str | Path, device: str = "cpu") -> TorchPolicy:
    """Load a checkpoint for evaluation (shared, safe loader)."""
    from .checkpoints import load_policy as _load_policy

    return _load_policy(model_path, device=device)


def policy_obs_version(policy) -> int:
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


def _opponent_for(name: str, seed: int, policy, device: str, spec: ModeSpec):
    """Build the opponent.  ``policy`` is already the view it should use."""
    if name == "self":
        return ModelOpponent(policy, name="self", deterministic=True)
    return build_opponent(name, seed=seed, device=device)


def _summarise(games: list[dict]) -> dict:
    wins = sum(1 for game in games if game["outcome"] == "win")
    losses = sum(1 for game in games if game["outcome"] == "loss")
    draws = sum(1 for game in games if game["outcome"] == "draw")
    goals_for = sum(game["goals_for"] for game in games)
    goals_against = sum(game["goals_against"] for game in games)
    low, high = wilson_interval(wins, len(games))
    decisive = wins + losses
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "win_rate": wins / len(games) if games else 0.0,
        "win_rate_ci95": [round(low, 4), round(high, 4)],
        "win_rate_decisive": wins / decisive if decisive else None,
        "goal_diff": goals_for - goals_against,
        "touches": sum(game["touches"] for game in games),
    }


def evaluate_policy(
    policy,
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
    sides: tuple[str, ...] | list[str] | str = ("blue",),
    mirror: bool = True,
) -> dict:
    """Play a reproducible series against one opponent.

    ``sides`` selects which team the policy controls.  ``("blue", "orange")``
    plays a balanced series (the extra game goes to blue).  ``mirror=True``
    rewrites the observation of the non-trained side into the trained side's
    frame; ``mirror=False`` feeds it raw, which measures how the checkpoint
    really behaves on that side.
    """
    spec = get_mode(mode)
    version = policy_obs_version(policy) if obs_version is None else int(obs_version)
    if isinstance(sides, str):
        sides = (sides,)
    side_list = [str(side).lower() for side in sides] or ["blue"]
    unknown = [side for side in side_list if side not in SIDE_TEAM]
    if unknown:
        raise ValueError(f"unknown side(s) {unknown}; use 'blue' and/or 'orange'")
    started = time.perf_counter()

    # Two views of the same checkpoint: the raw one (blue, or orange when no
    # mirroring is wanted) and the mirrored one.  Built once per series.
    mirrored_view = MirroredPolicy(policy, obs_version=version) if mirror else None

    def view_for(team: int):
        if team == BLUE_TEAM or mirrored_view is None:
            return policy
        return mirrored_view

    played: list[dict] = []
    # One arena per *side* is enough: the environment is reset between games,
    # so a fresh RocketSim arena per game is pure overhead (it dominated the
    # evaluation cost before).  The opponent has to be rebuilt when the side
    # changes, because the opponent controls the other team.
    envs: dict[int, MatchEnv] = {}
    opponents: dict[int, object] = {}
    try:
        for game_index in range(games):
            game_seed = seed + game_index * 7919
            side = side_list[game_index % len(side_list)]
            team = SIDE_TEAM[side]
            active = view_for(team)
            env = envs.get(team)
            if env is None:
                game_opponent = _opponent_for(
                    opponent, game_seed, view_for(1 - team), device, spec
                )
                env = MatchEnv(
                    spec,
                    game_opponent,
                    seed=game_seed,
                    episode_seconds=episode_seconds,
                    obs_version=version,
                    policy_team=team,
                )
                envs[team] = env
                opponents[team] = game_opponent
            else:
                # the arena is reused, but the seed and the opponent change
                env.seed = game_seed
                env.policy_team = team
                env.opponent = _opponent_for(
                    opponent, game_seed, view_for(1 - team), device, spec
                )
                opponents[team] = env.opponent
            obs = env.reset(episode_seed=game_seed)
            goals_for = 0
            goals_against = 0
            ticks = 0
            touches = 0
            previous_touches = 0
            target_ticks = int(match_seconds * 120)
            while ticks < target_ticks:
                agents = env.policy_agents
                actions, _, _ = active.act(
                    {agent: obs[agent] for agent in agents}, deterministic=not stochastic
                )
                result = env.step(actions)
                ticks += env.tick_skip
                if result.info.goal_scored:
                    if result.info.scoring_team == team:
                        goals_for += 1
                    elif result.info.scoring_team == 1 - team:
                        goals_against += 1
                current_touches = sum(int(env.state.cars[a].ball_touches) for a in agents)
                # Counters reset with the episode, so add the delta when the
                # counter grows and add the new value when it was reset.
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
            played.append(
                {
                    "game": game_index + 1,
                    "seed": game_seed,
                    "side": side,
                    "mirrored": bool(team != BLUE_TEAM and mirror),
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "outcome": outcome,
                    "ticks": ticks,
                    "touches": touches,
                }
            )
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if callable(close):
                close()

    summary = _summarise(played)
    side_results = {
        side: _summarise([game for game in played if game["side"] == side])
        for side in sorted({game["side"] for game in played})
    }
    protocol = {
        "mode": mode,
        "opponent": opponent,
        "games": int(games),
        "seed": int(seed),
        "match_seconds": float(match_seconds),
        "episode_seconds": float(episode_seconds),
        "sides": sorted(set(side_list)),
        "mirror": bool(mirror),
        "stochastic": bool(stochastic),
        "obs_version": int(version),
        "opponent_class": classify_opponent(opponent),
    }
    return {
        "mode": mode,
        "opponent": opponent,
        "opponent_class": classify_opponent(opponent),
        "games": summary["games"],
        "wins": summary["wins"],
        "losses": summary["losses"],
        "draws": summary["draws"],
        "goals_for": summary["goals_for"],
        "goals_against": summary["goals_against"],
        "win_rate": summary["win_rate"],
        "win_rate_ci95": summary["win_rate_ci95"],
        # Draws are common when two weak policies never score; report the
        # share of wins among games that produced a goal difference too.
        "win_rate_decisive": summary["win_rate_decisive"],
        "goal_diff": summary["goal_diff"],
        "touches": summary["touches"],
        "seed": seed,
        "match_seconds": match_seconds,
        "device": device,
        "obs_version": version,
        "stochastic": stochastic,
        "sides": sorted(set(side_list)),
        "mirror": bool(mirror),
        "side_results": side_results,
        "protocol": protocol,
        "duration_s": time.perf_counter() - started,
        "details": {"games": played},
    }


def classify_opponent(name: str) -> str:
    """``diagnostic`` for pipeline sanity checks, ``strength`` otherwise."""
    key = str(name).strip()
    if key in DIAGNOSTIC_OPPONENTS:
        return "diagnostic"
    return "strength"


def evaluate_suite(
    policy,
    *,
    mode: str,
    opponents: list[str] | tuple[str, ...] = ("chase",),
    games: int = 10,
    seed: int = 1000,
    match_seconds: float = 120.0,
    device: str = "cpu",
    episode_seconds: float = 30.0,
    obs_version: int | None = None,
    stochastic: bool = False,
    sides: tuple[str, ...] | list[str] = ("blue", "orange"),
    mirror: bool = True,
) -> dict:
    """Evaluate against several opponents under one shared protocol.

    Results are grouped: ``strength`` (chase, self-play snapshots, other
    checkpoints) describes playing strength, ``diagnostic`` (idle, random)
    only proves that observation, reward and scoring still work.  Mixing the
    two into one headline number is what makes "win rate" meaningless.
    """
    started = time.perf_counter()
    opponent_list = [str(name) for name in opponents] or ["chase"]
    per_opponent: dict[str, dict] = {}
    for index, name in enumerate(opponent_list):
        per_opponent[name] = evaluate_policy(
            policy,
            mode=mode,
            opponent=name,
            games=games,
            seed=seed + index * 104_729,
            match_seconds=match_seconds,
            device=device,
            episode_seconds=episode_seconds,
            obs_version=obs_version,
            stochastic=stochastic,
            sides=sides,
            mirror=mirror,
        )

    groups: dict[str, list[dict]] = {"strength": [], "diagnostic": []}
    for name, result in per_opponent.items():
        groups[classify_opponent(name)].append(result)

    def aggregate(results: list[dict]) -> dict:
        if not results:
            return {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "goals_for": 0,
                "goals_against": 0,
                "win_rate": None,
                "win_rate_ci95": [0.0, 1.0],
                "goal_diff": 0,
                "opponents": [],
                "measured": False,
            }
        games = sum(item["games"] for item in results)
        wins = sum(item["wins"] for item in results)
        losses = sum(item["losses"] for item in results)
        draws = sum(item["draws"] for item in results)
        goals_for = sum(item["goals_for"] for item in results)
        goals_against = sum(item["goals_against"] for item in results)
        low, high = wilson_interval(wins, games)
        return {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "win_rate": wins / games if games else 0.0,
            "win_rate_ci95": [round(low, 4), round(high, 4)],
            "goal_diff": goals_for - goals_against,
            "opponents": [item["opponent"] for item in results],
            "measured": True,
        }

    strength = aggregate(groups["strength"])
    diagnostic = aggregate(groups["diagnostic"])
    protocol = {
        "mode": mode,
        "opponents": opponent_list,
        "games_per_opponent": int(games),
        "seed": int(seed),
        "match_seconds": float(match_seconds),
        "episode_seconds": float(episode_seconds),
        "sides": sorted(set(str(side).lower() for side in sides)),
        "mirror": bool(mirror),
        "stochastic": bool(stochastic),
        "obs_version": obs_version if obs_version is not None else policy_obs_version(policy),
        "kind": "suite",
    }
    return {
        "mode": mode,
        "kind": "suite",
        "opponents": opponent_list,
        "results": per_opponent,
        "strength": strength,
        "diagnostic": diagnostic,
        # headline score: strength opponents only, or None when none were played
        "score": strength["win_rate"],
        "score_metric": "win_rate_vs_strength",
        "games": strength["games"] + diagnostic["games"],
        "wins": strength["wins"] + diagnostic["wins"],
        "losses": strength["losses"] + diagnostic["losses"],
        "draws": strength["draws"] + diagnostic["draws"],
        "goals_for": strength["goals_for"] + diagnostic["goals_for"],
        "goals_against": strength["goals_against"] + diagnostic["goals_against"],
        "win_rate": strength["win_rate"] if strength["measured"] else diagnostic["win_rate"],
        "protocol": protocol,
        "duration_s": time.perf_counter() - started,
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
    sides: tuple[str, ...] | list[str] | str = ("blue",),
    mirror: bool = True,
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
        sides=sides,
        mirror=mirror,
    )
    result["model"] = str(model_path)
    return result
