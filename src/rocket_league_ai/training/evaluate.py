"""Frozen-checkpoint evaluation subprocess: real RocketSim games, no training.

Loads only the actor/critic weights of a verified checkpoint (inference on
CPU, greedy argmax, gradients disabled), plays side-balanced seeded games
against a scripted baseline, and writes a result manifest plus a recorded
trajectory for the replay viewer. Weights are never updated here.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from ..checkpoints import CheckpointStore
from ..config import TrainingConfig
from ..evaluation import EvaluationSpec
from ..persistence import atomic_json
from ..store import checked_id
from .evaluate_core import play_game


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    spec = EvaluationSpec.model_validate(request["spec"])
    root = Path(request["root"])

    run_dir = root / "runs" / checked_id(spec.run_id)
    config = TrainingConfig.model_validate_json((run_dir / "config.json").read_text())
    store = CheckpointStore(run_dir)
    checkpoint = store.require_valid(spec.checkpoint_id, config.digest())
    checkpoint_path = store.path(checkpoint["id"])

    import numpy as np
    import torch
    from rlgym_learn_algos.ppo import BasicCritic, DiscreteFF, SeparateActorCritic

    from .environment import build_environment

    torch.set_num_threads(1)
    actor_critic = SeparateActorCritic(
        DiscreteFF(92, 90, config.hidden_sizes, torch.float32, torch.device("cpu")),
        BasicCritic(92, config.hidden_sizes, torch.float32, torch.device("cpu")),
    )
    state_dict = torch.load(checkpoint_path / "ppo_learner" / "actor_critic.pt",
                            map_location="cpu", weights_only=True)
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    for parameter in actor_critic.parameters():
        parameter.requires_grad_(False)
    actor = actor_critic.actor

    def candidate_action(agent_id, observation):
        with torch.no_grad():
            actions, _ = actor.get_actions([agent_id], [np.asarray(observation, dtype=np.float32)],
                                           deterministic=True)
        return int(np.asarray(actions).reshape(-1)[0])

    eval_config = config.model_copy(update={
        "episode_timeout_seconds": spec.max_game_seconds,
        "no_touch_timeout_seconds": min(config.no_touch_timeout_seconds, spec.max_game_seconds),
    })
    env = build_environment(eval_config)
    max_decisions = math.ceil(spec.max_game_seconds * 15)
    games, frames = [], []
    sides = []
    for index in range(spec.games):
        side = "blue" if index % 2 == 0 else "orange"
        sides.append(side)
        games.append(play_game(
            env, candidate_action, spec.opponent_id, index=index, candidate_side=side,
            seed=spec.seed + index, max_decisions=max_decisions,
            record=index < spec.record_games, frames=frames,
        ))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "result.json", {
        "candidate_id": spec.checkpoint_id,
        "games": [game.model_dump() for game in games],
    })
    if frames:
        atomic_json(args.output / "trajectory.json", {
            "schema_version": 1, "source": "evaluation recording; decision-rate frames from RocketSim state",
            "candidate_checkpoint_id": spec.checkpoint_id, "opponent_id": spec.opponent_id,
            "candidate_side_by_game": sides, "frames": frames[:20000],
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
