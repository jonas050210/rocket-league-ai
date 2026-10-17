"""Frozen-checkpoint evaluation subprocess: real RocketSim games, no training.

Loads only verified checkpoint weights (greedy CPU inference, gradients
disabled) for the candidate — and optionally for a frozen checkpoint
opponent — and plays side-balanced seeded games against a scripted or
checkpoint opponent. Games may run in parallel worker processes; seeds and
side assignment depend only on the game index, never on scheduling, so
parallel and sequential runs of the same spec are identical in protocol.
Writes a result manifest, a recorded trajectory and progress for the backend.
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..checkpoints import CheckpointStore
from ..config import TrainingConfig
from ..evaluation import EvaluationSpec
from ..persistence import atomic_json
from ..store import checked_id
from .evaluate_core import play_game

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

_WORKER: dict = {}


def _load_actor(config: TrainingConfig, checkpoint_path: Path):
    import numpy as np
    import torch
    from rlgym_learn_algos.ppo import BasicCritic, DiscreteFF, SeparateActorCritic

    actor_critic = SeparateActorCritic(
        DiscreteFF(92, 90, config.hidden_sizes, torch.float32, torch.device("cpu")),
        BasicCritic(92, config.hidden_sizes, torch.float32, torch.device("cpu")),
    )
    actor_critic.load_state_dict(torch.load(checkpoint_path / "ppo_learner" / "actor_critic.pt",
                                            map_location="cpu", weights_only=True))
    actor_critic.eval()
    for parameter in actor_critic.parameters():
        parameter.requires_grad_(False)
    actor = actor_critic.actor

    def act(agent_id, observation):
        import torch as _torch

        with _torch.no_grad():
            probs = actor.get_output([np.asarray(observation, dtype=np.float32)])
            action = int(probs.argmax(dim=-1).item())
            return action, float(probs[0, action].item())

    return act


def _init_worker(payload: dict):
    import numpy as np
    import torch

    from .environment import build_environment

    torch.set_num_threads(1)
    spec = EvaluationSpec.model_validate(payload["spec"])
    config = TrainingConfig.model_validate(payload["config"])
    eval_config = config.model_copy(update={
        "episode_timeout_seconds": spec.max_game_seconds,
        "no_touch_timeout_seconds": min(config.no_touch_timeout_seconds, spec.max_game_seconds),
        "environment_version": spec.scenario_environment or config.environment_version,
    })
    _WORKER["env"] = build_environment(eval_config)
    _WORKER["candidate"] = _load_actor(config, Path(payload["checkpoint"]))
    _WORKER["opponent"] = None
    if payload.get("opponent_checkpoint"):
        opponent_config = TrainingConfig.model_validate(payload["opponent_config"])
        _WORKER["opponent"] = _load_actor(opponent_config, Path(payload["opponent_checkpoint"]))
    _WORKER["spec"] = spec
    _WORKER["np"] = np
    from rlgym.rocket_league.action_parsers import LookupTableAction

    _WORKER["action_table"] = LookupTableAction()._lookup_table


def _play_one(index: int):
    spec: EvaluationSpec = _WORKER["spec"]
    side = "blue" if index % 2 == 0 else "orange"
    frames: list = []
    opponent_fn = None
    if _WORKER["opponent"] is not None:
        opponent_actor = _WORKER["opponent"]
        opponent_fn = lambda agent_id, observation, rng: opponent_actor(agent_id, observation)[0]  # noqa: E731
    game = play_game(
        _WORKER["env"], _WORKER["candidate"], spec.opponent_id, index=index, candidate_side=side,
        seed=spec.seed_for_game(index), max_decisions=math.ceil(spec.max_game_seconds * 15),
        record=index < spec.record_games, frames=frames, opponent_action_fn=opponent_fn,
        action_table=_WORKER["action_table"],
    )
    return {"game": game.model_dump(), "frames": frames, "side": side}


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
    payload = {"spec": spec.model_dump(mode="json"), "config": config.model_dump(mode="json"),
               "checkpoint": str(store.path(checkpoint["id"])), "opponent_checkpoint": None,
               "opponent_config": None}
    if spec.opponent_checkpoint_id:
        opponent_dir = root / "runs" / checked_id(spec.opponent_run_id)
        opponent_config = TrainingConfig.model_validate_json((opponent_dir / "config.json").read_text())
        opponent_store = CheckpointStore(opponent_dir)
        opponent = opponent_store.require_valid(spec.opponent_checkpoint_id, opponent_config.digest())
        payload["opponent_checkpoint"] = str(opponent_store.path(opponent["id"]))
        payload["opponent_config"] = opponent_config.model_dump(mode="json")

    args.output.mkdir(parents=True, exist_ok=True)
    games, frames, sides = [], [], []
    if spec.parallel_workers <= 1:
        _init_worker(payload)
        for index in range(spec.total_games):
            outcome = _play_one(index)
            games.append(outcome["game"])
            frames.extend(outcome["frames"])
            sides.append(outcome["side"])
            atomic_json(args.output / "progress.json", {"completed": index + 1, "games": spec.total_games})
    else:
        import multiprocessing as mp

        completed = 0
        with ProcessPoolExecutor(max_workers=spec.parallel_workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker, initargs=(payload,)) as pool:
            futures = [pool.submit(_play_one, index) for index in range(spec.total_games)]
            outcomes = {}
            for future in as_completed(futures):
                outcome = future.result()
                outcomes[outcome["game"]["index"]] = outcome
                completed += 1
                atomic_json(args.output / "progress.json", {"completed": completed, "games": spec.total_games})
        for index in range(spec.total_games):
            games.append(outcomes[index]["game"])
            frames.extend(outcomes[index]["frames"])
            sides.append(outcomes[index]["side"])

    atomic_json(args.output / "result.json", {"candidate_id": spec.checkpoint_id, "games": games})
    if frames:
        atomic_json(args.output / "trajectory.json", {
            "schema_version": 1, "source": "evaluation recording; decision-rate frames from RocketSim state",
            "candidate_checkpoint_id": spec.checkpoint_id,
            "opponent_id": spec.opponent_id if not spec.opponent_checkpoint_id else f"checkpoint:{spec.opponent_checkpoint_id}",
            "candidate_side_by_game": sides, "frames": frames[:20000],
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
