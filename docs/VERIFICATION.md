# Phase 2 — execution record

**2026-10-17 · branch `arena/135f4561-rocket-league-ai` · application 0.2.0**

This records what was **executed**, not merely implemented. [PROJECT.md](../PROJECT.md) is the current reference; [UPSTREAM.md](UPSTREAM.md) explains reproduced release defects. All runs below are deliberately initiated, bounded correctness tests in disposable/ignored workspaces. They are not competitive training or target-host benchmarks.

## 1. Environment and exact profile

Local sandbox: Debian 12 Linux x86_64, Python **3.11.2**, Node **22.22.3**, uv **0.13.1**, about **3.8 GiB** RAM, no CUDA device/driver/display/Xvfb. Python 3.12 binary downloads failed with a GitHub TLS handshake EOF; TLS verification was not disabled. The lock declares Python >=3.11,<3.13 and recommends 3.12.

Installed and actually imported: RocketSim 2.2.1, RLGym API 2.0.0 / Rocket League 2.0.1, Learn 2.0.0, Algos 0.4.2, NumPy 1.26.4, Torch **2.9.1+cu128** (runtime **12.8**, `cuda.is_available() == False`). Full installed venv occupies approximately 7.3 GiB on disk; do not confuse that with measured RAM/VRAM. Desktop wheel installation alone is not native launch evidence.

## 2. Local command results

Commands run at repository root. `.venv/bin/...` explicitly selects the supported local 3.11 interpreter, avoiding an implicit `.python-version` download.

| Executed command | Result / scope |
|---|---|
| `uv --system-certs sync --frozen --all-extras --python /usr/bin/python3.11` | Passed; exact locked training/desktop/developer profile installed |
| `.venv/bin/ruff check src tests` | **Passed**, including the final loop-capture fix and source-profile regressions |
| `.venv/bin/python -m compileall -q src` | **Passed** |
| `npm run check` | **Passed**, actual offline client syntax |
| `.venv/bin/pytest -q -m 'not desktop'` | **98 passed, 2 deselected, 1 external warning, 45.09 s** (final sequential local Python run before CI submission) |
| `RLAI_TEST_CUDA_UNAVAILABLE=1 npm test` | **5 passed, about 65.9 s**; real GUI lifecycle case 36.4 s. This run overlapped the Python suite, so do not treat timing as a benchmark |
| `uv --system-certs run --no-sync --python /usr/bin/python3.11 --with pip python -m pip check` | **Passed**: `No broken requirements found.` Temporary pip checker, no lock/dependency modification |
| `uv --system-certs build --python /usr/bin/python3.11 --no-python-downloads` | **Passed**: sdist + `rocket_league_ai-0.2.0-py3-none-any.whl`; inspected 28 wheel entries including every offline UI and training asset |
| `.venv/bin/python -m rocket_league_ai doctor --probe` | Simulation **passed**: 16 meshes, 8 ticks; CUDA unavailable; **exit 1 intentionally**. No CUDA matmul/training was possible |
| `RLAI_NATIVE_TEST=1 .venv/bin/pytest -q -m desktop` | **Environment-blocked: 2 failed, 98 deselected, 1.48 s**. Both fail at actual desktop import with missing `libGL.so.1`; neither is falsely skipped or counted as a native pass |

One external deprecation remains: Starlette 0.52.1's test client references AnyIO's deprecated `anyio.abc.BlockingPortal` alias. It is reported, not filtered out or patched.

`uv pip check` separately reports RocketSim's malformed internal CPython-3.9 WHEEL tags; the ABI3 filename resolves and real imports/steps pass on local 3.11. Standard pip checks the dependency graph successfully. This packaging warning is not hidden and does not certify every Python/platform ABI.

A bare `uv build` initially tried `.python-version`'s unavailable 3.12 download; the explicit system-3.11 build above passed. Isolated scratch native-library probes reached `gbm_bo_get_fd_for_plane` missing from an old GBM copy. No system libraries were installed/replaced, ELF symbols patched or unofficial native recipe promoted. The production setup uses normal Ubuntu packages.

## 3. What the 98 Python tests actually exercise

### Unit, API, authorization and storage

- Frozen strict config, unknown/non-finite input, explicit devices, budgets/batch/minibatch/env/timeout relationships, deterministic config hash and field differences.
- Startup has no runs/worker/training side effect; metadata reads do not initialize Torch/CUDA in the backend.
- Capability authorization, foreign-origin refusal, read-only preview, JSON validation with no side effect, literal-true confirmation, digest matching, rejection, expiry, replay, plan revision and restart-expired review handling.
- Atomic approved-plan/revision transaction with rollback injection; uncertain/interrupted executing actions are never automatically retried.
- Checkpoint writer failure/partial staging, malformed schemas/IDs/types, missing/changed files, fresh integrity after cached inspection, unsafe paths/symlinks, corrupt latest pointer/newest payload and older-valid fallback.
- Single backend/worker ownership, model action target/manifest binding, stale best-pointer protection, designated-best and sole-valid protection, explicit deletion with retained other valid checkpoints.
- Recovery refusal for changed packages, training/config/storage source hash or Python major/minor; source rechecked between review and confirmation, before any new worker/run.

Unit checkpoint payloads are explicitly byte fixtures for filesystem/API logic. They are never loaded by PPO and never claimed as learned models.

### Actual RocketSim/RLGym semantics

- Real 16-mesh/two-car arena, initialization idempotence after successful init, seeded kickoff, contiguous float32 observation width 92 and discrete action count 90.
- Repeating an action advances **eight physics ticks**; done/reset guard and real idle truncation.
- Actual goal crossing produces ±10 reward/true termination; goal beats simultaneous timeout. Audited goal/touch component sums come from real transitions.
- Actual engine callback/state extraction reports adjacent touches `1,1`, then idle `0`; shipped touch reward and idle reset behavior, not the withdrawn cumulative-delta replacement.
- Actual GAE bootstraps a truncation (5.5 in the controlled fixture), not goal termination (1.0).
- Response fence lets the initial reset return, computes a real step to tick 8, blocks its publication until the permit, then releases the genuine result. No fake metrics or simulator is used.

### Actual learner and service subprocesses

- Upstream collector/PPO with **one and two** env processes completes exactly **512 agent timesteps**, updates actor and critic tensors and leaves finite parameters.
- Real pause: update-boundary acknowledgement, counters stable while paused, verified checkpoint, real resume progress and stop/ownership release.
- Three delayed pause/resume cycles after at least 1,024 collected steps, each requiring acknowledgement and >=512 subsequent steps within the regression deadline. This targets the five-second upstream keepalive stall, not an RTX latency SLA.
- Actual worker SIGKILL preserves the last committed model and never auto-restarts; SIGSTOP triggers the test's shortened 1.5-second watchdog.
- Actual backend SIGKILL while the learner is paused causes stdin EOF cleanup, valid checkpoint/lease release, restart `interrupted` status/no autostart, then a separately confirmed linked recovery.
- Actual backend SIGTERM requests graceful worker stop, records terminal state and lifespan cleanup; Uvicorn intentionally re-raises SIGTERM (Python return `-15`).
- Real recovery compares serialized **actor/critic, both Adam states, experience buffer, GAE return-running statistics, Python/NumPy/buffer/Torch RNGs, learner counters and episode summary**. It starts new simulator episodes, so this does not imply bit-exact mid-match continuation.

## 4. Actual-browser coverage (5 tests)

Chromium **153**, Playwright **1.64.1**, Linux x86_64, one worker/no retries. Test services bind ports 8871/8872; unique `.rlai/ui-tests/<uuid>` data directories. No runtime app CDN or third-party asset request. Container-only `--no-sandbox` is not a native production recommendation; web security is not disabled.

1. **Read-only preview:** six views, actual offline resources, no page errors/horizontal overflow at 1440/1000/420 px, mutation disabled and honest missing evaluator/renderer.
2. **Authentication:** prompt, wrong-key rejection, startup/same-document capability-fragment removal, no automatic training.
3. **Real GUI vertical slice:** invalid JSON/validation-no-side-effect, rationale/impact/difference/checkbox gating, cancellation then explicit approval; actual PPO steps/episodes/reward/logs; pause; filter logs; disconnect/reconnect; resume; close/reopen browser page while run persists; stop; inspect a real checkpoint; confirmed manual designation/protected deletion; cancel then confirm another real model deletion; token-free measured report download; explicitly confirmed recovery child and stop.
4. **Disconnect safety:** cached configuration write controls disabled without losing the local draft; reconnect restores usable controls.
5. **CUDA refusal:** on this independently verified no-CUDA host, explicitly requested `cuda:0` fails visibly and never substitutes CPU. Other hosts skip this **host-specific** browser case unless explicitly opted in after checking no CUDA.

Screenshots/reports/failure traces are ignored `test-results/`, not committed. Test keys are disposable; equivalent real-workspace traces must not be published. Native Qt browser-engine behavior is **not inferred** from this Chromium result.

### Investigation that changed the implementation

Early browser runs exposed 5–15-second Rust collection stalls. Faulthandler and release/HEAD source inspection identified signal-before-response-drain ordering plus five-second child keepalives. Temporary tracing was removed after the deterministic publication fence was implemented. The full suite then passed (including an earlier 5/5 run in 44.6 s). A measured corrected run acknowledged pause in ~53 ms, resume ~16 ms and stop ~111 ms. These are isolated local CPU control observations, not skill/target-throughput claims or a latency guarantee under load.

## 5. Native Qt / Python 3.12 CI status

**Configured, result pending at initial commit.** See `.github/workflows/verify.yml` and the feature-branch pull request for the actual run. Do not count a configured workflow as executed verification.

The job uses Ubuntu 24.04/Python 3.12 and normal apt native dependencies, the exact uv/npm locks, lint/compile/package, all non-desktop tests, actual native Qt under Xvfb, then actual-browser tests. Official action revisions are SHA-pinned, permission is contents-read only, and no CUDA pass is asserted on a CPU runner.

The two opt-in native tests actually launch the Qt window, traverse six routes through its real WebEngine client, save PNGs and close it. They test idle/no-autostart, a real paused learner surviving close/reopen with unchanged counters and subsequent resume/stop, and launcher bootstrap of a detached backend surviving the window. `RLAI_NATIVE_TEST=1` makes missing deps/display/readiness a **failure**, not a skip. Without the flag, ordinary headless runs deselect/skip the tests with a reason.

Xvfb/native CI, even if green, is not WSLg interactive certification, native file-dialog usability certification or RTX execution. Those remain host gates.

## 6. Unpassed target / future gates

- Actual RTX 4060 Ti CUDA math/training/recovery, 8-GB VRAM headroom and throughput.
- WSL 2 allocation/filesystem/driver behavior, interactive WSLg window and report-save/cancel interaction.
- Thirty-minute target-host lifecycle/resource/disk-growth soak; sudden power-loss storage testing.
- Evaluator/frozen opponents/side-balanced confidence intervals. No wins, rank, strength recommendations or promoted “best” policy are fabricated.
- Renderer compatibility, curriculum/league, larger match modes, deployment/original-game control: not implemented, not tested, not part of this delivery.

[ROADMAP.md](../ROADMAP.md) supplies measurable next criteria. Passing correctness checks never authorizes automatic strategy changes, training starts, model promotion or valuable-data deletion.
