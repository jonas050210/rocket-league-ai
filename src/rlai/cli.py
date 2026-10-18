"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import TrainingConfig, ensure_dirs
from .store import Store


def _store() -> Store:
    paths = ensure_dirs()
    return Store(paths["root"] / "rlai.sqlite3")


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    from .api import create_app

    if args.token:
        os.environ["RLAI_AUTH_TOKEN"] = args.token
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .trainer import run_training

    config = TrainingConfig.load(args.config_file)
    if args.max_steps:
        config = config.model_copy(update={"total_steps": int(args.max_steps)})
    store = _store()
    run_id = args.run_id
    if not run_id:
        run_id = store.create_run(
            config.run_name, config.mode, config.to_dict(), config.total_steps
        )
        store.log(run_id, "INFO", "run created from CLI")
    result = run_training(config, run_id, store, resume_from=args.resume_from)
    print(json.dumps(result))
    return 0 if result["status"] in {"completed", "stopped"} else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_model

    result = evaluate_model(
        args.model,
        mode=args.mode,
        opponent=args.opponent,
        games=args.games,
        seed=args.seed,
        match_seconds=args.match_seconds,
        device=args.device,
    )
    store = Store(Path(args.store)) if args.store else _store()
    result["model_id"] = args.model_id
    store.add_evaluation(result)
    store.log(
        None,
        "INFO",
        f"evaluation {args.model_id or args.model} vs {args.opponent}: "
        f"{result['wins']}W {result['losses']}L {result['draws']}D",
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .hardware import benchmark, diagnostics

    print(json.dumps(diagnostics(), indent=2))
    print(json.dumps(benchmark(args.mode, args.seconds, args.workers), indent=2))
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import LiveUnavailable, launch_match

    paths = ensure_dirs()
    try:
        result = launch_match(paths, args.model_id, args.mode)
    except LiveUnavailable as exc:
        print(f"live match unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlai", description="Rocket League AI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local web application")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--token", default=None)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    train = sub.add_parser("train", help="run a training job")
    train.add_argument("--config-file", required=True)
    train.add_argument("--run-id", default=None)
    train.add_argument("--resume-from", default=None)
    train.add_argument("--max-steps", type=int, default=None)
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--model-id", default=None)
    evaluate.add_argument("--mode", default="soccer-1v1")
    evaluate.add_argument("--opponent", default="chase")
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--match-seconds", type=float, default=120.0)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--store", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    bench = sub.add_parser("benchmark", help="detect hardware and measure simulation throughput")
    bench.add_argument("--mode", default="soccer-1v1")
    bench.add_argument("--seconds", type=float, default=3.0)
    bench.add_argument("--workers", type=int, default=1)
    bench.set_defaults(func=cmd_benchmark)

    live = sub.add_parser("live", help="launch a live RLBot match (Windows)")
    live.add_argument("--mode", default="soccer-1v1")
    live.add_argument("--model-id", default=None)
    live.set_defaults(func=cmd_live)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
