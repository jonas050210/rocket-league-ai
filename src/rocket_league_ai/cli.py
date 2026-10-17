"""Explicit entry points. Opening the application never starts a training run."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from .metadata import package_versions


def doctor(probe: bool):
    result = {
        "python": sys.version.split()[0], "platform": sys.platform,
        "training_packages": package_versions(),
        "desktop_installed": importlib.util.find_spec("PySide6") is not None,
        "cuda": {"probed": False, "available": None},
    }
    if probe:
        try:
            import torch

            available = torch.cuda.is_available()
            result["cuda"] = {"probed": True, "available": available, "torch": torch.__version__,
                              "runtime": torch.version.cuda}
            if available:
                value = torch.ones((64, 64), device="cuda:0")
                value @ value
                torch.cuda.synchronize()
                result["cuda"].update(name=torch.cuda.get_device_name(0),
                                       capability=list(torch.cuda.get_device_capability(0)), matmul="passed")
            from .training.environment import initialize_simulator

            meshes = initialize_simulator()
            import RocketSim as rsim

            arena = rsim.Arena(rsim.GameMode.SOCCAR)
            arena.add_car(rsim.Team.BLUE, rsim.CarConfig.OCTANE)
            arena.add_car(rsim.Team.ORANGE, rsim.CarConfig.OCTANE)
            arena.step(8)
            result["simulation"] = {"soccar_meshes": len(list((meshes / "soccar").glob("*.cmf"))),
                                    "ticks": arena.tick_count, "status": "passed"}
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(result, indent=2))
    return 1 if result.get("error") or (probe and not result["cuda"].get("available")) else 0


def main():
    parser = argparse.ArgumentParser(prog="rocket-ai", description="Rocket League AI — local-first training control")
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend", help="Start the independent control-plane service; no training auto-start")
    backend.add_argument("--host", default="127.0.0.1")
    backend.add_argument("--port", type=int, default=8765)
    backend.add_argument("--preview-readonly", action="store_true", help="Allow unauthenticated read-only diagnostics; writes still need the local key")
    desktop = commands.add_parser("desktop", help="Open the native desktop shell, starting a detached backend if needed")
    desktop.add_argument("--backend-url", default="http://127.0.0.1:8765")
    desktop.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    desktop.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    browser = commands.add_parser("browser-url", help="Print a local capability URL; keep its key private")
    browser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    for child in (backend, desktop, browser):
        child.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RLAI_DATA_DIR", ".rlai")))
    check = commands.add_parser("doctor", help="Inspect packages; --probe checks actual CUDA and RocketSim without training")
    check.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor(args.probe))
    if args.command == "backend":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(args.data_dir, preview_readonly=args.preview_readonly),
                    host=args.host, port=args.port, access_log=False)
    elif args.command == "browser-url":
        path = args.data_dir / "access-token"
        if not path.is_file():
            parser.error("Start rocket-ai backend for this workspace first.")
        print(args.backend_url.rstrip("/") + "/#token=" + path.read_text().strip())
    else:
        try:
            from .desktop import launch

            raise SystemExit(launch(args))
        except ImportError as error:
            parser.exit(2, f"Desktop dependency unavailable: {error}\nInstall the desktop extra and Ubuntu GUI libraries listed in docs/SETUP.md. The backend can run headlessly.\n")
        except (RuntimeError, OSError) as error:
            parser.exit(2, f"Desktop startup failed: {error}\nSee docs/SETUP.md for backend connection and WSLg recovery.\n")
