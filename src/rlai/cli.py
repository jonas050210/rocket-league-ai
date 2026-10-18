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


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    from .api import create_app
    from .config import AppSettings

    # one validated settings object drives both the warning and the socket
    try:
        settings = AppSettings.from_env(host=args.host, port=args.port, auth_token=args.token)
    except Exception as exc:  # pydantic ValidationError -> a readable refusal
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            for error in exc.errors():
                print(
                    f"ERROR: {'.'.join(str(part) for part in error['loc'])}: {error['msg']}",
                    file=sys.stderr,
                )
            raise SystemExit(2) from exc
        raise
    if settings.auth_token:
        os.environ["RLAI_AUTH_TOKEN"] = settings.auth_token
    if settings.weak_token:
        print(
            "WARNING: the configured token is shorter than 12 characters and is easy to "
            "guess; prefer a long random value, e.g. "
            "python -c 'import secrets; print(secrets.token_urlsafe(32))'",
            file=sys.stderr,
        )
    if settings.exposed_without_token:
        print(
            f"WARNING: serving on {settings.host} without a token exposes the training API "
            "to every machine that can reach this port, including starting and stopping runs. "
            "Bind to 127.0.0.1 (the default) or set --token / RLAI_AUTH_TOKEN.",
            file=sys.stderr,
        )
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)
    return 0


def _resolve_resume(args: argparse.Namespace, config: TrainingConfig) -> tuple[str, str | None]:
    """Decide the explicit resume mode and (optional) checkpoint reference."""
    mode = args.resume_mode
    if mode is None:
        mode = "continue" if (args.resume_from or args.auto_resume) else "fresh"
    if mode not in ("fresh", "continue", "transfer"):
        raise SystemExit(f"--resume-mode must be fresh, continue or transfer, got {mode!r}")
    reference = args.resume_from
    if mode == "fresh" and (reference or args.auto_resume):
        print(
            "NOTE: --resume-mode fresh ignores the checkpoint; nothing is restored.",
            file=sys.stderr,
        )
        reference = None
    return mode, reference


def cmd_train(args: argparse.Namespace) -> int:
    from .resume import plan_resume
    from .trainer import run_training

    config = TrainingConfig.load(args.config_file)
    if args.max_steps:
        config = config.model_copy(update={"total_steps": int(args.max_steps)})
    store = _store()
    paths = ensure_dirs()
    resume_mode, reference = _resolve_resume(args, config)

    run_id = args.run_id
    plan = None
    if resume_mode != "fresh":
        plan = plan_resume(
            paths["models"],
            config,
            mode=resume_mode,
            run_id=run_id,
            checkpoint=reference,
        )
        _print({"resume_plan": plan.to_dict()})
        if args.plan_only:
            return 0
        if not plan.will_resume and reference:
            print(
                f"ERROR: cannot resume from {reference!r}: "
                + "; ".join(
                    reason for item in plan.rejected for reason in item.get("reasons", [])
                ),
                file=sys.stderr,
            )
            return 2
        for warning in plan.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    elif args.plan_only:
        _print({"resume_plan": {"mode": "fresh", "will_resume": False}})
        return 0

    if not run_id:
        run_id = store.create_run(
            config.run_name, config.mode, config.to_dict(), config.total_steps
        )
        store.log(run_id, "INFO", f"run created from CLI (resume_mode={resume_mode})")
    result = run_training(
        config,
        run_id,
        store,
        resume_from=str(plan.checkpoint_path) if plan and plan.will_resume else None,
        resume_mode=resume_mode,
        resume_plan=plan,
    )
    print(json.dumps(result, default=str))
    return 0 if result["status"] in {"completed", "stopped"} else 1


def cmd_resume_plan(args: argparse.Namespace) -> int:
    """Show what a resume *would* restore.  Never starts training."""
    from .resume import plan_resume

    config = TrainingConfig.load(args.config_file)
    if args.max_steps:
        config = config.model_copy(update={"total_steps": int(args.max_steps)})
    paths = ensure_dirs()
    plan = plan_resume(
        paths["models"],
        config,
        mode=args.mode,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
    )
    _print(plan.to_dict())
    print(plan.summary(), file=sys.stderr)
    return 0 if (plan.will_resume or args.mode == "fresh") else 3


def cmd_report(args: argparse.Namespace) -> int:
    """Print the persisted statistics report of a run."""
    from .report import build_run_report, compare_runs

    store = _store()
    if args.compare:
        run_ids = [item.strip() for item in args.compare.split(",") if item.strip()]
        _print(compare_runs(store, run_ids))
        return 0
    run_id = args.run_id
    if not run_id:
        runs = store.list_runs(limit=1)
        if not runs:
            print("no runs recorded yet", file=sys.stderr)
            return 3
        run_id = runs[0]["id"]
    stored = store.get_report(run_id)
    if args.rebuild:
        # explicit rebuild wins over a stored report: the stored copy can predate
        # a fix in the report builder or have been written by an older version
        paths = ensure_dirs()
        report = build_run_report(
            store,
            run_id,
            models_dir=paths["models"],
            run_dir=paths["runs"] / run_id,
        )
    elif stored and stored.get("report"):
        report = stored["report"]
    else:
        print(
            f"no report stored for run {run_id}; pass --rebuild to compute it from the database",
            file=sys.stderr,
        )
        return 3
    if args.summary:
        _print(
            {
                key: report.get(key)
                for key in (
                    "run_id",
                    "status",
                    "mode",
                    "steps",
                    "duration_s",
                    "steps_per_second",
                    "throughput",
                    "hardware",
                    "resume",
                    "evaluation",
                    "final_checkpoint",
                    "warnings",
                )
            }
        )
    else:
        _print(report)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_model, evaluate_suite, load_policy

    store = Store(Path(args.store)) if args.store else _store()
    sides = tuple(side.strip() for side in (args.sides or "blue").split(",") if side.strip())
    opponents = [name.strip() for name in (args.opponents or "").split(",") if name.strip()]
    if not opponents:
        opponents = [args.opponent]

    if len(opponents) > 1:
        policy = load_policy(args.model, device=args.device)
        result = evaluate_suite(
            policy,
            mode=args.mode,
            opponents=opponents,
            games=args.games,
            seed=args.seed,
            match_seconds=args.match_seconds,
            device=args.device,
            stochastic=args.stochastic,
            sides=sides,
            mirror=not args.no_mirror,
        )
        result["model"] = str(args.model)
        result["model_id"] = args.model_id
        # one database row per opponent, all under the shared protocol
        for name, item in (result.get("results") or {}).items():
            row = dict(item)
            row["model_id"] = args.model_id
            row["phase"] = args.phase
            row["protocol"] = result.get("protocol")
            store.add_evaluation(row)
    else:
        result = evaluate_model(
            args.model,
            mode=args.mode,
            opponent=opponents[0],
            games=args.games,
            seed=args.seed,
            match_seconds=args.match_seconds,
            device=args.device,
            stochastic=args.stochastic,
            sides=sides,
            mirror=not args.no_mirror,
        )
        result["model_id"] = args.model_id
        result["phase"] = args.phase
        store.add_evaluation(result)
    store.log(
        None,
        "INFO",
        f"evaluation {args.model_id or args.model} vs {result.get('opponent') or opponents}: "
        f"{result.get('wins')}W {result.get('losses')}L {result.get('draws')}D "
        f"sides={','.join(sides)}",
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .hardware import benchmark, diagnostics

    print(json.dumps(diagnostics(), indent=2))
    print(
        json.dumps(
            benchmark(
                args.mode,
                args.seconds,
                args.workers,
                parallel=not args.sequential,
                include_inference=not args.no_inference,
            ),
            indent=2,
        )
    )
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import LiveUnavailable, launch_match, list_matches, preflight

    paths = ensure_dirs()
    if args.list:
        print(json.dumps(list_matches(paths), indent=2))
        return 0
    if args.preflight:
        report = preflight(paths)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    try:
        result = launch_match(
            paths,
            args.model_id,
            args.mode,
            human_team=args.human_team,
            fill=args.fill,
            human=not args.no_human,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    except LiveUnavailable as exc:
        print(f"live match unavailable: {exc}", file=sys.stderr)
        print("run `rlai live --preflight` for a per-check report", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlai", description="Rocket League AI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local web application")
    serve.add_argument(
        "--host",
        default=None,
        help="interface to bind (default 127.0.0.1, or RLAI_HOST; "
        "0.0.0.0 exposes the API to the network)",
    )
    serve.add_argument("--port", type=int, default=None, help="default 8000, or RLAI_PORT")
    serve.add_argument("--token", default=None)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    train = sub.add_parser("train", help="run a training job")
    train.add_argument("--config-file", required=True)
    train.add_argument("--run-id", default=None)
    train.add_argument(
        "--resume-from",
        default=None,
        help="checkpoint id or path to continue/transfer from (validated before use)",
    )
    train.add_argument(
        "--auto-resume",
        action="store_true",
        help="continue from the newest valid, compatible checkpoint of this lineage",
    )
    train.add_argument(
        "--resume-mode",
        default=None,
        choices=["fresh", "continue", "transfer"],
        help="fresh (default), continue (restore everything) or transfer (weights only)",
    )
    train.add_argument(
        "--plan-only",
        action="store_true",
        help="print the resume plan and exit without training anything",
    )
    train.add_argument("--max-steps", type=int, default=None)
    train.set_defaults(func=cmd_train)

    plan = sub.add_parser(
        "resume-plan", help="show which checkpoint a resume would restore (never trains)"
    )
    plan.add_argument("--config-file", required=True)
    plan.add_argument("--mode", default="continue", choices=["fresh", "continue", "transfer"])
    plan.add_argument("--run-id", default=None)
    plan.add_argument("--checkpoint", default=None, help="explicit checkpoint id or path")
    plan.add_argument("--max-steps", type=int, default=None)
    plan.set_defaults(func=cmd_resume_plan)

    report = sub.add_parser("report", help="print the persisted training statistics report")
    report.add_argument("--run-id", default=None, help="defaults to the newest run")
    report.add_argument("--summary", action="store_true", help="only the headline numbers")
    report.add_argument(
        "--rebuild",
        action="store_true",
        help="recompute the report from the database, ignoring any stored copy",
    )
    report.add_argument(
        "--compare",
        default=None,
        help="comma-separated run ids to compare instead of printing one report",
    )
    report.set_defaults(func=cmd_report)

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--model-id", default=None)
    evaluate.add_argument("--mode", default="soccer-1v1")
    evaluate.add_argument("--opponent", default="chase")
    evaluate.add_argument(
        "--opponents",
        default=None,
        help="comma-separated list for a multi-opponent suite (separates diagnostics "
        "from strength opponents)",
    )
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--match-seconds", type=float, default=120.0)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument(
        "--sides",
        default="blue",
        help="'blue', 'orange' or 'blue,orange' for a side-balanced series",
    )
    evaluate.add_argument(
        "--no-mirror",
        action="store_true",
        help="do not rewrite the observation of the non-trained side (measures raw behaviour)",
    )
    evaluate.add_argument(
        "--phase",
        default="manual",
        help="label stored with the evaluation: before, periodic, after or manual",
    )
    evaluate.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of argmax (behaviour of the training policy)",
    )
    evaluate.add_argument("--store", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    bench = sub.add_parser("benchmark", help="detect hardware and measure simulation throughput")
    bench.add_argument("--mode", default="soccer-1v1")
    bench.add_argument("--seconds", type=float, default=3.0)
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument(
        "--sequential",
        action="store_true",
        help="run the workers one after another in this process (honest, slower)",
    )
    bench.add_argument(
        "--no-inference", action="store_true", help="skip the separate inference measurement"
    )
    bench.set_defaults(func=cmd_benchmark)

    live = sub.add_parser("live", help="launch a live RLBot match (Windows)")
    live.add_argument("--mode", default="soccer-1v1")
    live.add_argument("--model-id", default=None)
    live.add_argument(
        "--human-team",
        default="blue",
        choices=["blue", "orange"],
        help="team the human plays on (the policy always takes the other team)",
    )
    live.add_argument(
        "--fill",
        default="psyonix",
        choices=["psyonix", "policy"],
        help="who completes the human team in 2v2/3v3",
    )
    live.add_argument(
        "--no-human",
        action="store_true",
        help="watch a policy-only match (both teams controlled by the checkpoint)",
    )
    live.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="launcher exits after N seconds (0 waits for the match to end)",
    )
    live.add_argument(
        "--dry-run",
        action="store_true",
        help="write and validate the match files without starting RLBot",
    )
    live.add_argument(
        "--preflight",
        action="store_true",
        help="check and print every live prerequisite, then exit",
    )
    live.add_argument(
        "--list",
        action="store_true",
        help="list recent live matches with launcher status and last log line",
    )
    live.set_defaults(func=cmd_live)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
