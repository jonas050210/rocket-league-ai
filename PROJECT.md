# Rocket League AI — project reference

**Authoritative current state · v0.3.0 · 2026-10-17 · Web Control Room + High-Scale Benchmark + Roadmap + Pro v3 Shaping + RLBot Export**

This project is a local-first **web control room for real, headless RocketSim reinforcement learning** (`src/rocket_league_ai/`, `tests/`). The legacy PySide6 desktop wrapper (`desktop.py`) has been removed; all controls live in the 7-tab Web Control Room (`Dashboard`, `Roadmap`, `Training`, `Monitoring`, `Models`, `Visualization`, `Benchmark`).

## 1. Goal, platform and boundaries

Target: **Ubuntu on WSL 2**, Intel i7-12700F, RTX 4060 Ti **8 GB**, 32 GB host RAM. The implementation is Linux-only; the desktop runs in Ubuntu/WSLg, not a separate native Windows Python environment. Python 3.12 is recommended; 3.11 is supported. Setup: [docs/SETUP.md](docs/SETUP.md).

Deliver a small, observable, recoverable learning application before optimizing throughput or training for strength:

- A native PySide6/Qt WebEngine window with an offline dark six-view client.
- An independent local FastAPI service supervising an independent training subprocess.
- A real 1v1 Soccar environment, released RLGym Learn collector and released upstream PPO/PyTorch learner.
- Explicitly initiated runs, update-boundary pause/resume/stop, failure reporting, verified checkpoints and confirmed recovery.
- Immutable experiment configuration, reproducibility metadata (including actual training source hash), structured durable events and conservative advisory monitoring.

**Improvement-plan delivery (2026-10-17):** CI contracts repaired for the six-view UI; benchmark statistics are per-repeat (warm-up excluded per repeat, CV across repeat medians); evaluation scales to parallel game workers and frozen checkpoint opponents with progress/stop and additive schema migration; opt-in `shaped-v2` reward components and `mixed-scenarios-v1` starting states are available as separately approved experiment variables (defaults unchanged); PPO clip/grad-norm are validated config fields; Models compares checkpoints with CI-aware verdicts; the dashboard proposes one evidence-linked next experiment. Prepared but **not executed**: RTX/WSLg certification, GPU sweeps, 400-game baseline, shaping experiment, budget increase — see [docs/TARGET_HOST.md](docs/TARGET_HOST.md).

**Implemented since the foundation:** a genuine hardware benchmark (20-second real runs, warm-up exclusion, median/CV statistics, explainable fastest-stable recommendation; device and `n_proc` can be auto-adopted when strict stability gates pass, with the previous plan kept for revert), a frozen-checkpoint evaluator against seeded scripted baselines (`random-v1`, `idle-v1`) with side-balanced games, Wilson confidence intervals and recorded replays, and a top-down replay viewer for those recordings. **Not implemented:** frozen opponent pool/league, curriculum, 3D/live renderer, original-game/RLBot control, deployment export, 2v2/3v3, competitive training, automatic strategy changes, automatic best-model promotion or automatic deletion.

## 2. Non-negotiable control rules

1. **Opening, closing, reconnecting or restarting the GUI never starts training.** A service restart never resumes a run automatically.
2. Closing the GUI leaves the service and worker intact. Stopping the service requests a safe worker stop; a killed service causes worker stdin EOF and cooperative shutdown.
3. Starting a run, applying a materially changed plan, recovering, replacing the designated best or deleting a checkpoint requires a **persisted proposal with rationale and expected impact**, followed by explicit user confirmation of its digest. The GUI requires an unchecked-by-default review checkbox.
4. Proposals expire after ten minutes, are single-use, bind the approved-plan revision and recheck mutable preconditions. Pending reviews expire on service restart. Interrupted side effects are not automatically retried.
5. Configurations are immutable per run. Applying a plan affects future runs only. No hot hyperparameter or strategy changes.
6. Pause/stop are cooperative requests, not instant simulator snapshots. The GUI displays `pausing`/`stopping` until acknowledgement at a completed PPO update.
7. Failures preserve previous committed checkpoints. Recovery is an explicitly confirmed **new linked experiment**, never an overwrite.
8. Recommendations are read-only explanations. Designation is **manual**, not an evaluation claim. The designated best and the last valid recovery checkpoint of a run are protected from deletion.
9. No silent CPU fallback. A CUDA request without working CUDA fails with an actionable explanation. CPU mode is an explicitly approved correctness-test configuration.
10. Automated tests deliberately initiate bounded CPU workers in isolated test workspaces. This does not enable training on normal application startup.

## 3. Implemented vertical slice

| View | Implemented behavior | Honest boundary |
|---|---|---|
| Dashboard | Connection/run state, agent steps, episodes, measured reward/throughput history, entropy, measured host CPU/RAM and best-effort GPU telemetry, activity, report | Missing readings show unavailable; win rate says not evaluated |
| Training | Approved-plan editor (validated draft, JSON, reviewed apply), review/start, pause/resume/stop, budget/progress, immutable metadata, experiment history, recovery action | Learner/update-boundary acknowledgement, not bit-exact match continuation |
| Monitoring | Cursor-resumed structured logs, warning/error filters, text search, worker heartbeat, resources, backend health, advisory insights | GPU NVML information may be unsupported under WSL |
| Models | Real inventory, SHA-256 integrity, metadata inspection, confirmed recovery/designation/deletion, protections; frozen-checkpoint evaluation vs scripted baselines with Wilson intervals and checkpoint comparison | Scripted baselines only; no league, no rank claim; designation remains manual |
| Visualization | Recorded-replay viewer: top-down schematic rendering of genuine evaluation trajectories (play/pause/scrub/speed/follow-ball), independent of training | Recorded replays only; no live or 3D renderer (NumPy/renderer conflict documented) |
| Benchmark | 20-second wall-clock sweeps of device × n_proc, warm-up exclusion, median/CV statistics, CPU/GPU/VRAM peaks, explainable recommendation; auto-adopts only device/n_proc after strict gates and keeps the previous plan for revert | Throughput only; no learning-quality claim; CUDA without GPU telemetry is excluded |

The client uses relative same-origin URLs and local assets only. No CDN, JavaScript framework, telemetry service, W&B account or npm runtime is needed. The browser diagnostic client and native window use exactly the same application UI.

## 4. Architecture and module ownership

```text
Qt desktop process (or diagnostic browser)
    offline UI, async HTTP, session-only capability, no Torch/simulator imports
               │ authenticated same-origin HTTP
Independent FastAPI backend process
    Orchestrator + SQLite WAL + events + proposals + checkpoint inventory + host monitor
               │ versioned JSON lines over stdin/stdout; separate process group
Independent training worker
    ControlledCoordinator → upstream RLGym Learn collector → real upstream PPO/PyTorch
               │ shared memory / local UDP + response-publication fence
    1–8 headless RocketSim/RLGym environment processes (all responses per collection)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for lifecycle, persistence, API and decisions. The service and GUI do not import Torch, create the training CUDA context, or simulate matches. Qt may use graphics acceleration for its own UI; that is not a training CUDA context.

| Location | Responsibility |
|---|---|
| `config.py`, `domain.py` | Frozen validated application config, hashes/diffs, lifecycle and future extension contracts |
| `api.py`, `cli.py`, `desktop.py`, `web/` | Local capability-protected API, entry points, native shell, six offline views |
| `orchestration.py` | Single-worker ownership, durable proposal execution, subprocess supervision, update-boundary controls, recovery/model actions |
| `training/environment.py` | Actual released RocketSim/RLGym composition and environment contract guards |
| `training/adapter.py`, `buffer.py`, `protocol.py`, `worker.py` | Narrow upstream compatibility/lifecycle adapters; real collector, actor/critic, Adam and PPO remain upstream |
| `store.py`, `persistence.py`, `checkpoints.py`, `metadata.py` | SQLite state/events, atomic files, process leases, verified immutable checkpoint publication and run metadata |
| `monitoring.py` | Measured psutil/NVIDIA telemetry and read-only recommendations |
| `tests/`, `tests/ui/` | Storage/API/autonomy, real-stack/subprocess, opt-in native Qt, actual-browser lifecycle verification |

## 5. Verified dependency profile

Exact transitive versions and hashes: **`uv.lock`**. Runtime groups are separately installable.

| Role | Exact selected package |
|---|---|
| Simulation | `rocketsim==2.2.1` (import **`RocketSim`**) |
| Environment | `rlgym-api==2.0.0`, `rlgym-rocket-league[sim]==2.0.1` |
| Collector / PPO | `rlgym-learn==2.0.0`, `rlgym-learn-algos==0.4.2` |
| Numeric / learner | `numpy==1.26.4`, `torch==2.9.1` (resolved Linux CUDA 12.8 build) |
| Desktop | `PySide6==6.10.2` and matching Qt components |
| Service | `fastapi==0.135.1`, `starlette==0.52.1`, `pydantic==2.14.1`, `uvicorn==0.42.0`, `psutil==7.2.2` |
| Python checks | `pytest==9.1.1`, `httpx==0.28.1`, `ruff==0.11.8` |
| Browser checks only | `@playwright/test==1.64.1`, `@sparticuz/chromium==153.0.0` |

**Release source beats GitHub HEAD examples.** The published Rocket League environment requires NumPy **<2**; current rlviser-py requires NumPy 2, so it is excluded rather than forced into a broken environment. The shipped engine constructor is not the newer HEAD API. Pydantic is pinned new enough for Learn 2.0.0's serialization API.

Known upstream defects and local adaptations are recorded in [docs/UPSTREAM.md](docs/UPSTREAM.md), including a reproduced Rust startup offset defect and a fast-response UDP drain race. No installed third-party source is monkeypatched, and no patched binary is committed.

The RocketSim wheel works through its ABI3 filename, but its internal `WHEEL` tags incorrectly declare CPython 3.9. `uv pip check` reports that metadata defect; standard pip dependency checking passes. Do not hide or patch it.

## 6. Environment, agent and learning contracts

- Fixed **1v1 Soccar**, Octane cars, kickoff-only mutator. One shared policy controls both teams; there is no opponent history or rating system.
- Bundled **16** soccar collision meshes; checked and explicitly initialized. No game client or extracted game assets are required for this slice.
- `DefaultObs(zero_padding=1)` produces **92 contiguous float32 values**. `LookupTableAction` exposes **90** discrete actions; `RepeatAction` repeats eight physics ticks.
- Physics is 120 Hz; decisions are 15 Hz per car. A simulator transition contributes **two agent timesteps**, not one frame and not one episode.
- Reward is `10 * GoalReward + 0.1 * TouchReward`. Component sums are audited from actual transitions. Touch reward can still encourage low-value/repeated contacts; this is a transparent foundation reward, not a strength guarantee.
- `ball_touches` is a **per-transition** count in this release. Two adjacent contact-bearing transitions can both report one. Do not subtract adjacent counts. The shipped no-touch timer correctly resets after a touch.
- A goal is true termination; idle/absolute time limits truncate. If both happen together, goal termination wins. No stepping after done until reset. GAE bootstraps truncations, not goal terminations.
- Real `SeparateActorCritic(DiscreteFF, BasicCritic)`, separate Adam optimizers, upstream PPO, CPU experience buffer and Rust GAE. No toy environment or substitute learning algorithm.
- The response fence prevents publishing an environment response before the released collector finishes sending/draining. It depends on the fixed all-environment response barrier; partial-response collection/dynamic env counts require re-verification.

Approved-plan defaults are the executable `TrainingConfig`: explicit `cuda:0`, seed 123, one environment process, 8,192 agent steps, 512-step collection/batch, 1,536 buffer capacity, two minibatches, one PPO epoch, hidden widths 256×3, learning rate 0.0002, entropy 0.01, gamma 0.99, lambda 0.95, checkpoints every two iterations, 30-second absolute/10-second no-touch timeout, pinned memory off. These are conservative bounded foundation defaults, **not a measured optimum for the target host**.

## 7. Durable workspace and recovery

Default workspace: `.rlai/`, ignored by Git. Use `--data-dir` or `RLAI_DATA_DIR` to select another **Linux filesystem** directory.

```text
workspace/
  access-token                  private local capability, mode 0600
  backend.lock                  advisory flock; do not remove a live lock file
  active-worker.lock            survives loss of supervisor ownership until worker exits
  control.sqlite3[-wal,-shm]    runs, settings, proposals, structured event cursor
  runs/<run-id>/
    config.json, metadata.json  immutable experiment evidence
    resolved_config.json       portable resolved upstream config
    resume.json                only for a confirmed recovered run
    checkpoints/
      .pending-<id>/           unpublished interrupted staging, never recoverable
      step-<12 digits>-<id>/   manifest + learner + optimizers + buffer/GAE + RNG + counters
      latest.json             advisory pointer; scanning verified manifests is authoritative
```

Publication is stage → required-file validation → file hashes/fsync → manifest/fsync → atomic same-filesystem rename → directory fsync → atomic pointer. A damaged latest pointer or corrupt newest checkpoint does not hide older verified checkpoints. Malformed manifests, unexpected files, symlinks, unsafe paths and config mismatches are refused before trusted-local loading.

**Restored:** actor, critic, both optimizers, experience data, GAE running statistics, Python/NumPy/buffer/PyTorch RNGs (and CUDA RNGs when selected), learner counters and completed-episode summary. **Not restored:** simulator state, in-flight trajectories or partial episode returns. New episodes reset. This is update-boundary learner recovery, **not bit-exact continuation**. The original config, package versions, Python major/minor profile and recorded training/config/storage source hash are required; cross-driver/OS/hardware numerical equivalence is not promised.

Checksums detect accidental damage, not malicious replacement. The upstream buffer contains trusted-local pickle data; never import untrusted checkpoints. There is no automatic pruning, external model import, backup/restore wizard or migration system. Reports are status exports, not full backups. See [docs/RECOVERY.md](docs/RECOVERY.md).

## 8. Running and checking

```bash
# Full developer environment, on Ubuntu with Python 3.12 and uv 0.13.1 installed:
uv sync --frozen --all-extras --python 3.12
uv run --no-sync rocket-ai doctor --probe   # exit 1 when CUDA is unavailable; never trains
uv run --no-sync rocket-ai desktop         # starts/connects backend; never auto-starts training

# Explicit service-only mode:
uv run --no-sync rocket-ai backend
# Optional private local browser capability URL (do not share its key):
uv run --no-sync rocket-ai browser-url

# Fast checks / real-stack checks:
uv run --no-sync ruff check src tests
uv run --no-sync pytest -q -m 'not integration and not desktop'
uv run --no-sync pytest -q -m 'integration and not desktop'

# Browser testing (isolated real CPU workers, Linux x86_64; Node 22.22.3+):
npm ci
npm test
# ONLY on a host independently verified to have no accessible CUDA:
RLAI_TEST_CUDA_UNAVAILABLE=1 npm test
```

A CPU smoke plan can be entered in the Training plan panel: explicitly choose CPU, 512 total steps, collection/batch 128, buffer 384, hidden widths `[32,32]`, two minibatches, one-second episode/no-touch limits. Validate, review/apply, then start (an unchanged plan starts without a second dialog). This tests correctness without pretending to benchmark an RTX GPU. Native test requirements and commands are in [docs/SETUP.md](docs/SETUP.md).

## 9. Verification and known limits

The audit was completed before the first code write; the original repository contained only four Markdown planning documents. [docs/AUDIT.md](docs/AUDIT.md) records findings, preservation and decisions. [docs/VERIFICATION.md](docs/VERIFICATION.md) is the exact execution record; do not infer a passed gate from an implemented interface or CI configuration.

Locally verified on Debian 12 / Python 3.11.2 / CPU-only sandbox:

- Real meshes/arena/ticks and actual PPO parameter changes; single- and two-environment bounded runs.
- API authorization, explicit confirmation, storage failures/corruption, no autostart, lifecycle, hard worker kill, watchdog, real backend death/restart and graceful service shutdown.
- Recovery comparison of actual weights, Adam state, experience buffer, GAE statistics, RNG states and counters.
- All six browser views and real GUI-driven start/pause/resume/stop/recovery, log filtering, report download, manual designation/confirmed deletion, connection loss and window reconnect.

Separately verified on **Ubuntu 24.04 / Python 3.12.15 CI**: **98 non-desktop tests, 2 actual native Qt/Xvfb tests and 5 actual-browser tests**, all with zero skips. Native tests traverse the six views, close/reopen during a real paused worker and verify detached-backend bootstrap. Local non-desktop tests: **98 passed**; local browser tests: **5 passed**.

**Still target-host gates:** CUDA math/training and VRAM/throughput on the actual RTX 4060 Ti, WSL resource allocation, WSLg interactive behavior, long-duration stability and power-loss testing. The recorded native CI pass is not interactive WSLg or native save-dialog certification. No skill, win-rate or rank claim is made. Historical throughput/CPU-penalty estimates remain hypotheses.

## 10. Next phase and change discipline

[ROADMAP.md](ROADMAP.md) defines measurable next gates: target-host CUDA/native certification, a bounded soak with checkpoint recovery and GUI reconnect, then a separately approved frozen-opponent evaluator before any strength recommendation or promotion. No larger mode, curriculum or live-game control is part of this foundation delivery.

Work on the feature/session branch, run relevant tests, commit changes and use a pull request for review. Keep generated data, credentials, browser traces and model binaries out of Git. Dependency/strategy changes require explicit review and new evidence. For reproducible valuable runs, commit training code first and use the frozen environment; a dirty flag/source hash identifies changes but does not archive an uncommitted source tree. Do not edit source/dependencies while a worker is active. Preserve [RESEARCH.md](RESEARCH.md); its Phase 2 correction notice supersedes withdrawn Phase 1 claims. The original architecture/roadmap are retained under `docs/history/` as historical design, not executable instructions.
