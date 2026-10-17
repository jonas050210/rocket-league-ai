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
| `.venv/bin/ruff check src tests scripts` | **Passed**, including the final loop-capture fix and source-profile regressions |
| `.venv/bin/python -m compileall -q src scripts` | **Passed** |
| `npm run check` | **Passed**, actual offline client syntax |
| `.venv/bin/pytest -q -m 'not desktop' --junitxml=.rlai/verification/python-final.xml` | **98 passed, 2 deselected, 1 external warning, 44.90 s** (final sequential local Python run) |
| `RLAI_TEST_CUDA_UNAVAILABLE=1 npm test` | **5 passed, 47.7 s** wall time; real GUI lifecycle case 27.5 s; JUnit suite time 45.836 s, zero skips. Timing is not a target-host benchmark |
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

## 5. Observed native Qt / Python 3.12 CI result

**PASSED**, not just configured. [Run 39037128792](https://github.com/jonas050210/rocket-league-ai/actions/runs/39037128792), commit **`51cd0f6`**, Ubuntu **24.04**, Python **3.12.15**, Node **22.22.3**. Entire job passed in **2 min 47 s**, including native dependency installation, locked setup, lint/compile/package and artifact upload.

| Executed CI command (environment in workflow) | Observed JUnit result |
|---|---|
| `uv run --no-sync pytest -q -m 'not desktop' --junitxml=.rlai/python-junit.xml` | **98 tests, 0 failures/errors/skips, 53.994 s** |
| `RLAI_NATIVE_TEST=1 xvfb-run -a -s '-screen 0 1440x1000x24' uv run --no-sync pytest -q -m desktop --basetemp=native-test-results --junitxml=.rlai/native-junit.xml` | **2 tests, 0 failures/errors/skips, 13.577 s** |
| `RLAI_TEST_CUDA_UNAVAILABLE=1 npm test` | **5 actual-browser tests, 0 failures/errors/skips, 41.432 s** suite time |

These counts/timings were read from actual GitHub Check annotations generated from JUnit XML, not guessed from job names. `scripts/report_junit.py` emits bounded diagnostic annotations without extra token permissions; the workflow retains the original test exit status. Raw log downloads redirect to a host unavailable in this sandbox. The **`ui-verification`** CI artifact contains disposable browser screenshots/report/JUnit plus actual native PNGs, not private workspace data.

The two native tests launch the **actual Qt window/WebEngine**, traverse all six routes and save PNGs. They verify idle/no-autostart, a real paused CPU learner surviving native close **and reopen** with unchanged counters and subsequent independent resume/stop, and launcher bootstrap of a detached backend surviving the window. `RLAI_NATIVE_TEST=1` makes missing deps/display/readiness a **failure**, not a skip. Normal headless test commands deselect them explicitly.

An earlier [run 39036179565](https://github.com/jonas050210/rocket-league-ai/actions/runs/39036179565) passed Python 3.12 non-desktop tests but failed the native step. Qt smoke callback values were then serialized explicitly as JSON rather than relying on binding-specific object conversion, and bounded JUnit diagnostics were added. [Run 39036654355](https://github.com/jonas050210/rocket-league-ai/actions/runs/39036654355) subsequently passed all steps (98 non-desktop / 2 native), followed by the complete measured run above. The artifact action is now a verified Node-24 revision; the old Node-20 action warning no longer appears. No missing-dependency or readiness failure was reclassified as a native pass.

Local Debian still lacks the native runtime: the failed local tests in §2 remain environment-blocked. The **actual native success is Ubuntu/Xvfb CI**, not a claimed local Qt window. CI uses standard apt libraries and software rendering/container-only sandbox flags, with exact uv/npm locks and SHA-pinned first-party actions. It is not a WSLg/GPU environment.

Xvfb success does **not** certify interactive WSLg, the native file-save/cancel dialog, RTX execution or long-duration stability. Those remain host gates. Later documentation-only commits do not alter this verified application profile; the [PR's current check](https://github.com/jonas050210/rocket-league-ai/pull/2) is authoritative for its latest head.

## 6. Unpassed target / future gates

- Actual RTX 4060 Ti CUDA math/training/recovery, 8-GB VRAM headroom and throughput.
- WSL 2 allocation/filesystem/driver behavior, interactive WSLg window and report-save/cancel interaction.
- Thirty-minute target-host lifecycle/resource/disk-growth soak; sudden power-loss storage testing.
- Evaluator/frozen opponents/side-balanced confidence intervals. No wins, rank, strength recommendations or promoted “best” policy are fabricated.
- Renderer compatibility, curriculum/league, larger match modes, deployment/original-game control: not implemented, not tested, not part of this delivery.

[ROADMAP.md](../ROADMAP.md) supplies measurable next criteria. Passing correctness checks never authorizes automatic strategy changes, training starts, model promotion or valuable-data deletion.

## Addendum — benchmark / evaluation / replay delivery (2026-10-17, this session)

Environment: Debian 12, Python 3.11.2 venv with the locked training stack pip-installed (torch 2.9.1+cu128, `cuda.is_available() == False`), 2 cores, ~3 GiB RAM, no GPU/display. This is a CPU correctness environment, not the RTX target host.

Executed and verified:

| Command / flow | Result |
|---|---|
| `pytest -q -m 'not desktop'` (full suite incl. real-training integration) | **114 passed, 2 deselected** in ~59 s (98 pre-existing + 16 new) |
| `ruff check src tests`; `node --check src/rocket_league_ai/web/app.js` | Passed (the previously broken benchmark UI JavaScript now parses) |
| New genuine integration: real benchmark measurement run | Passed — bounded 384-step CPU run measured through the existing worker |
| New genuine integration: real evaluation of a trained checkpoint | Passed — side-balanced games vs `idle-v1`, trajectory frames recorded, kickoff ball at centre verified |
| Live backend + HTTP benchmark | Session completed: **1,529 agent steps/s** (2,485 sim steps/s, 10,514 PPO steps/s, 352.7 episodes/min, batch 128, 1 env, this sandbox) with a fastest-stable recommendation |
| Live HTTP evaluation vs `random-v1` (384-step checkpoint, 4 games) | **0W/0L/4D, goals 0–0, win rate 0% (95% CI 0–49%)** — an honest weak baseline; no improvement claimed |
| Replay artifact | 302 decision-rate frames over 4 games; sides alternate blue/orange; first-frame car/ball kickoff positions correct |

Not executed here (target-host gates, unchanged): CUDA/RTX 4060 Ti throughput sweeps, WSLg native desktop certification, 400-game evaluation comparisons, 30-minute soak, sudden-power-loss storage test. Browser UI tests (`npm test`) were not re-run in this sandbox; the UI was verified by syntax check, API-level flows and the live backend above.

## Addendum — improvement-plan delivery (2026-10-17, this session)

Executed in the CPU sandbox (no GPU/display; CUDA gates remain open):

- P0: desktop/browser contracts updated to the real six views and the
  implemented evaluation capability; benchmark aggregation is per repeat
  (constructed-data test proves warm-ups of later repeats no longer leak);
  run metadata no longer claims evaluation is unimplemented.
- P1: genuine integration test plays a parallel (2 workers) mirror
  evaluation — candidate vs its own frozen checkpoint, 4 games, completed
  with progress persisted; checkpoint opponents are labelled
  `checkpoint:<id>`; additive `progress` migration verified against an
  old-schema database.
- P2: shaped-v2 + mixed-scenarios config trains genuinely (256 steps) and
  saves a valid checkpoint; mixed scenarios produce varied real resets;
  shaping direction/normalization unit-tested from physics state.
- P3: snapshot evaluations are light (detail verified only via the
  evaluations API), next-experiment suggestions unit-tested, checkpoint
  comparison and replay goal markers are frontend-only honest renderings.

Not executed: browser/native CI suites for these commits (running on the
PR), all RTX/WSLg items, the 400-game baseline, any shaping/budget
experiment. See docs/TARGET_HOST.md for the prepared procedures.


## Addendum — extension packages A–E (2026-10-17, this session)

- **A**: scenario-override evaluation (kickoff/mixed), multi-seed wrap
  scheduling with per-game persisted seeds, replay frames carry the
  chosen candidate action, its policy probability and the decoded
  control vector (lookup table incl. idle action 8); action readout and
  JSON export in the viewer. Unit tests incl. FakeEnv action recording.
- **B**: chronological Elo (K=32, start 1500) over checkpoint-vs-
  checkpoint evaluations only; scripted baselines never contribute.
  Experiment journal diffs run configs and only claims an effect when
  Wilson intervals separate. Partial/legacy records are tolerated.
- **C**: league round-robin service over 2–6 frozen checkpoints, one
  side-balanced match per pair through the normal evaluation pipeline;
  genuine integration test trains a real run, runs a 2-checkpoint
  league to completion and asserts both enter the Elo table.
- **D**: retention candidates (beyond five newest per run; designated
  best and last-valid per run protected; suggestions only, deletion
  stays a confirmed individual proposal), measured workspace size with
  a >5 GB warning, recovery candidates panel, terminal-transition
  toasts, per-candidate benchmark initialization overhead.
- **E**: Playwright coverage extended to Models (evaluation + league)
  and Benchmark surfaces in the read-only preview; `npm run test:ui`
  passes locally (4 passed, 1 platform-conditional skip).
## Addendum — benchmark-first iteration (2026-10-17, this session)

Implemented (code, not target-host evidence):

- **Benchmark window:** `BenchmarkSpec.measurement_seconds=20` by default; a
  candidate is stopped at an update boundary by wall clock, with
  `min_iterations` protecting slow CPUs and `measure_iterations` as a high ceiling (1,900).
  A hard `timeout_seconds` still bounds a stuck run. `repeats` defaults to 3.
- **Statistics:** warm-up episode counts use the last warm-up iteration as the
  baseline; repeat CV is `max(between-repeat, within-run)`, so one repeat can
  no longer be labelled stable by zero between-repeat variance. VRAM peaks
  and missing CUDA telemetry participate in exclusion.
- **Automatic adoption:** when requested (`auto_default`), only `device` and
  `n_proc` are written to the plan, with `plan_revision` increment, an audit
  event, a stored previous plan and a UI revert. Requires three repeats,
  CV ≤ 0.10 and GPU telemetry for CUDA.
- **Disposable measurements:** benchmark runs are marked
  `purpose=benchmark`, filtered from Training history/checkpoint inventory and
  their run directories are removed after the session. SQLite keeps the
  lightweight benchmark record.
- **CPU diagnosis:** Monitoring now reports `cpu_threads_busy` and per-core
  utilization, so "10% CPU" on a 20-thread host is explained as roughly two
  busy threads rather than an idle machine.
- **UI:** Configuration route moved into Training as the Approved plan panel
  (legacy `#configuration` redirect); six navigation views; 20 s benchmark
  form, throughput chart and session history; pure black/white theme with a
  minimum 12 px readability pass; replay renderer uses an explicit field
  mapping (x ±4096, y ±5120, goal 1,785.51 uu wide), not a hard-coded scale.
- **League HTTP gap:** `/api/leagues` and `/api/leagues/{id}/stop` now exist,
  and a static client→API route contract test prevents the previous 404.
- **Replay:** empty state draws an outlined field instead of a green canvas;
  `requestAnimationFrame` playback, zoom, follow, altitude cue and a geometry
  self-test exposed as `window.__RLAI_GEOMETRY__.selfTest()`.

Executed in the CPU sandbox (Python 3.11.2, no training packages installed in
the temporary venv):

- `ruff check src tests` — all checks passed.
- `node --check` for `web/app.js` and `tests/ui/control-room.spec.cjs`;
  `compileall -q src`; `create_app()` constructs.
- `pytest -q` — **112 passed, 30 skipped, 0 failed**. Every skip is the
  explicit `requires_training` marker: the integration modules and three API
  lifecycle tests launch the real worker/evaluator and therefore need the
  locked training extra. The pure/API suite includes the new client→API route
  contract test, which fails if the client calls an unregistered endpoint.
- `pytest tests/test_benchmark.py -q` — 12 passed, 1 skipped (same marker).
  This includes the 20-second budget test, warm-up baseline, single-repeat CV,
  auto-default gates and the end-to-end auto-adoption test (which previously
  emitted an unhandled worker-thread warning from run-directory cleanup; the
  cleanup is now best-effort).
- Geometry self-test via Node:
  `{"fieldWidth":715,"fieldHeight":572,"orangeRight":true,"blueLeft":true,"positiveXUp":true,"goalWidth":124.67184082031247}`.
  The Playwright spec now pins 715 / 572 / 124.6718 and samples the halfway
  line (`world (0, 3000)`) so a mapping regression fails loudly.
- Live backend smoke on `127.0.0.1:8899`: `/api/health` ready (`training_auto_start:false`);
  `/api/snapshot` contains `benchmark`, `benchmark_sessions`,
  `benchmark_adoption` and `visualization_available:true`; `/api/leagues` 200;
  invalid benchmark spec 422; an otherwise valid benchmark returns 409 with
  the explicit `Training extra is missing` message; `/assets/app.js` 200
  (`text/javascript`, CSP `script-src 'self'`).

### Browser suite (Chromium via `npm ci` + `@sparticuz/chromium`)

Run with the sandbox venv: `RLAI_TEST_PYTHON=/tmp/rlvenv/bin/python PYTHONPATH=src npm test`.
The suite starts two real backends (control on 8871, read-only preview on 8872).

| Test | Result |
|---|---|
| read-only preview: six views, honest unavailable features, offline assets, responsive layout | **passed** |
| capability prompt, wrong-key rejection, URL fragment removal, no auto-start | **passed** |
| cached draft disables controls on disconnect and survives reconnect | **passed** |
| twenty-second smoke: every view keeps rendering under polling | **passed** (22.6 s: 3.3 s per view across all six, zero page errors) |
| real GUI vertical slice (validate → review → approve → pause → …) | fails only at the real-training step: this sandbox has no Torch/RocketSim, so the Pause button never becomes enabled. Validation, diff, cancel, approval and the unchanged-plan inline start all executed before that point. |
| actual worker CUDA failure | skipped (`RLAI_TEST_CUDA_UNAVAILABLE` opt-in only) |

Two defects were found and fixed by these runs:

1. The header row overflowed at 420 px because the top bar
   was a single 72 px flex row; it now wraps and right-aligns its status group,
   so `scrollWidth <= innerWidth` holds at 420 px.
2. The halfway line used the vertical coordinate for both endpoints, so it was
   not drawn at all in the empty state. It is now drawn from
   `px(-x, 0)` to `px(x, 0)`. This is exactly the regression the new
   geometry/pixel assertions exist to catch; the earlier pixel check sampled a
   point that the field outline also covered, which is why it passed before.

A third defect was found by code review against the refresh loop, not by a
browser run: clicking "Advanced · complete validated configuration" opened the
JSON editor, and the next one-second poll rebuilt the panel and closed it
again. Opening the disclosure now enters the same edit-guard as typing, so the
documented "edit the complete JSON" workflow is actually usable.

The 20-second benchmark was **not executed** because this sandbox has no
RocketSim/Torch/GPU; the 409 above is the honest boundary, not a pass. Native
Qt, WSLg, the target-host sweep and the 400-game baseline remain unexecuted. No
throughput, skill or hardware claim is made from this sandbox.
