# Rocket League AI

A local-first **desktop control room for headless RocketSim reinforcement learning**. Real RLGym Learn / PPO / PyTorch training, an independent backend, explicit user control, live monitoring and recoverable checkpoints.

**Phase 2 foundation + measured tooling — not a competitive agent.** Includes a genuine hardware benchmark (real bounded runs, fastest-stable recommendation), frozen-checkpoint evaluation against scripted baselines with confidence intervals, and a recorded-replay viewer. No 3D/live renderer or original-game automation is implemented. Missing metrics/results are labeled honestly; nothing starts training merely because you open the app.

## Start on Ubuntu / WSL 2

Use Ubuntu 24.04, Python 3.12 and uv 0.13.1. Install the native Qt/WSLg prerequisites in **[docs/SETUP.md](docs/SETUP.md)** first. Python 3.11 is also supported; native Windows Python is not the training/application platform.

```bash
uv sync --frozen --all-extras --python 3.12
uv run --no-sync rocket-ai doctor --probe
uv run --no-sync rocket-ai backend
uv run --no-sync rocket-ai browser-url  # private capability URL; open in your browser
```

The Web Control Room connects to the **detached local backend**. Close/reopen the browser tab without stopping an active run. Start the 1-click Auto-Benchmark (`POST /api/benchmarks/auto`) to test 30 ascending hardware stages (20s per stage, up to 32 parallel environments and 16,384 batch steps) and automatically adopt the fastest stable configuration.

## Seven views in the Web Control Room

- **Dashboard:** real run/episode/reward/speed signals, host resources and readiness.
- **Roadmap:** 5 progressive AI milestones (`1M`, `10M`, `50M`, `150M`, `500M` steps) with live progress bars, estimated time remaining based on your measured average `steps/s`, and 1-click **Start Training** buttons that always continue from the newest valid checkpoint.
- **Training:** simplified setup with 5 curated **Presets**, **Max Steps**, and live **Estimated Duration (ca. Zeit)**. Clicking **Start Training** starts immediately without confirmation modals and automatically resumes from the newest valid checkpoint.
- **Monitoring:** durable live logs, warnings/errors, worker heartbeat and backend health.
- **Models:** verified checkpoint inventory/metadata, 1-click **RLBot Export** (`bot.cfg` + `agent.py`), frozen-checkpoint evaluation, Elo ratings, and league matches.
- **Visualization:** recorded-replay viewer rendering genuine evaluation trajectories (top-down schematic, play/pause/scrub/follow-ball); **not** a live 3D renderer.
- **Benchmark:** simplified 1-click **Start benchmark** / **Cancel benchmark** running a 30-stage × 20-second (~10 min) ascending hardware sweep (`n_proc` 1 → 32, `batch_size` 512 → 16,384, `episode_timeout_seconds` 15s → 90s) and automatically adopting the winning configuration for training.

Checkpoints restore real learner/optimizer/buffer/GAE/RNG state at completed updates; simulator episodes reset. Recovery creates a new linked run and always requires confirmation. There is no automatic pruning or best-model promotion.

## Verify

```bash
uv run --no-sync ruff check src tests
uv run --no-sync pytest -q -m 'not desktop'  # includes actual bounded CPU subprocess tests
npm ci                                    # browser test tooling only; not needed at runtime
npm test                                  # drives real training through the actual UI
```

Verified in the development sandbox: **112 Python tests pass**, with 30 explicit skips because the locked training extra (Torch/RocketSim) is not installed there. CI installs every extra and runs the full Python set plus the native Qt/Xvfb and Chromium lifecycle tests; native tests require a display/WSLg or Xvfb and explicit opt-in. Exact commands, warnings and hardware limits are in **[docs/VERIFICATION.md](docs/VERIFICATION.md)**. GPU/WSLg/long-duration certification on the target i7-12700F / RTX 4060 Ti 8 GB / 32 GB host is a separate gate; no throughput or win-rate claim is inferred from CPU/browser tests.

## Project map

| Document | Purpose |
|---|---|
| **[PROJECT.md](PROJECT.md)** | Authoritative current state, constraints, contracts and defaults |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Implemented process/module/lifecycle/storage boundaries |
| [ROADMAP.md](ROADMAP.md) | Evidence-led next gates; historical phase-number reconciliation |
| [docs/SETUP.md](docs/SETUP.md) | WSL, native GUI, dependencies, launch and test commands |
| [docs/TARGET_HOST.md](docs/TARGET_HOST.md) | Prepared (not executed) RTX 4060 Ti certification, benchmark sweeps, baseline and shaping-experiment procedures |
| [docs/RECOVERY.md](docs/RECOVERY.md) | Failure diagnosis, explicit recovery, backup and security limits |
| [docs/AUDIT.md](docs/AUDIT.md) | Audit performed before implementation; corrected assumptions |
| [docs/UPSTREAM.md](docs/UPSTREAM.md) | Verified release defects and narrowly scoped adaptations |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | Exact executed commands/results and remaining verification |
| [RESEARCH.md](RESEARCH.md) | Preserved Phase 1 research with prominent Phase 2 corrections |

The offline UI has no npm runtime dependency. Exact Python versions/hashes are in `uv.lock`; browser test dependencies are in `package-lock.json`. Keep generated models, logs, capabilities and test artifacts out of Git. Work is reviewed through feature-branch pull requests.
