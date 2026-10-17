# Phase 2 — pre-implementation repository audit

Audit completed **2026-10-17**, before implementation. Baseline: `c06d97a47a4cde078f079bb158217e868f861091`; working branch: `arena/135f4561-rocket-league-ai`. The checkout was clean.

## Complete inventory and actual state

`git ls-files`, `git ls-tree -r HEAD`, a filesystem walk (including hidden directories), Git history, and merged PR #1 were inspected. The only project directory was the repository root. All four tracked files were read in full:

| File | Baseline lines / bytes | Finding |
|---|---:|---|
| `README.md` | 1 / 19 | Project title only. |
| `RESEARCH.md` | 1,223 / 123,513 | Valuable evidence-tagged Phase 1 survey, not executable verification. Several source interpretations are wrong (below). |
| `ARCHITECTURE.md` | 611 / 48,437 | Explicitly design-only; D1–D11 and future training/deployment topology. No GUI, application service, control protocol, or durable confirmation mechanism. |
| `ROADMAP.md` | 444 / 40,141 | Planning-only phases 0–7 and B1–B7. Its "Phase 2" means throughput tuning, not this request's foundation phase. No milestone has actually passed. |

There was **no implementation, dependency manifest/lock, environment, test, CI, configuration, dataset, checkpoint, benchmark result, secret file, or established code convention to preserve**. The only established workflow was a feature branch and a reviewed documentation PR merged to main. Work will stay on the session branch and be committed; no merge to main is implied.

## Verification method

Checked actual PyPI JSON metadata, published source distributions and the RocketSim Linux wheel, not only GitHub HEAD or similarly named libraries. Extracted source under a scratch cache outside the repository. Inspected RLGym's complete default observation, action/reward/done APIs, transition engine, car state contract, and RLGym Learn's coordinator, process lifecycle, configuration, PPO controller/learner, metrics, experience buffer and GAE persistence.

Evidence:
- [RLGym Rocket League 2.0.1 release metadata/source](https://pypi.org/pypi/rlgym-rocket-league/2.0.1/json)
- [RocketSim 2.2.1 wheel metadata](https://pypi.org/pypi/rocketsim/2.2.1/json)
- [RLGym Learn 2.0.0](https://pypi.org/pypi/rlgym-learn/2.0.0/json)
- [RLGym Learn Algos 0.4.2](https://pypi.org/pypi/rlgym-learn-algos/0.4.2/json)
- Inspected GitHub heads: RLGym `2fc8d435663454b7fe45d2556777efd65cba8c2d`, Learn `250379331d21946afaed384dd0fadc857f3df6b6`, Algos `466b8e30f911914602a5cb3ba6053eabc53ab0e3`. These are evidence references, **not** unpinned installation targets.

## Inconsistencies, corrections and risks

1. **Critical: D11 and the alleged cumulative-touch bugs are incorrect.** In the *published* `rlgym-rocket-league==2.0.1`, `api/car.py` documents `ball_touches` as "number of ball touches since last state was sent". `sim/rocketsim_engine.py::_get_state` assigns `_touches[id]` to `car.ball_touches`, then sets `_touches[id] = 0`. GitHub HEAD does the same. The shipped `TouchReward` pays for a contact-bearing transition; `NoTouchTimeoutCondition` correctly measures idle time following an earlier contact. A delta of two adjacent transition counts loses consecutive equal-count touches. The research's farming arithmetic and proposed replacements must **not** drive implementation. Verify the actual engine contract and test consecutive contacts plus idle-after-touch. Keep the research as history with a prominent correction.
2. **Package name is not import name.** The `rocketsim` 2.2.1 wheel exports `RocketSim` (capital R/S), not `rocketsim`. The old roadmap's `import rocketsim` acceptance check is invalid.
3. **GitHub HEAD is not the PyPI API.** Published `RocketSimEngine.__init__` accepts only `rlbot_delay=True`; HEAD adds `game_mode`. The release includes 16 soccar meshes, not the additional modes implied by HEAD. Use release APIs, soccar only, explicitly check meshes and initialize before constructing the engine.
4. **"Exact resume" is overstated.** Upstream controller/buffer persistence does not save simulator state or all random-number generator states; current in-flight environment trajectories are not fully serialized. It also restores monotonic timestamps from a previous process and drops GAE load context in the buffer. Foundation recovery should be declared **update-boundary learner recovery with environment reset**, not bit-exact match continuation. Persist RNG/telemetry state and restore learner/buffer/GAE state explicitly where needed.
5. **Upstream checkpoint writes are not atomic and prune automatically.** A directory is visible while being written; retention calls `shutil.rmtree`. This conflicts with crash safety and mandatory deletion confirmation. Override only the checkpoint boundary: stage, hash, fsync, atomic rename, then update the latest pointer. No automatic pruning. Never save partially updated parameters after a PPO failure.
6. **Coordinator is terminal-oriented and swallows training exceptions.** Its keyboard pause and `start()` exception handler cannot be the desktop lifecycle contract. A narrowly scoped adapter should retain the real collector/PPO and replace terminal control with cooperative IPC, explicit acknowledgements, propagated failures, and supervisor deadlines.
7. **Desktop/orchestration are absent from the old design.** GUI lifetime must not own the learner, CUDA context, simulation processes, or run state. Introduce a local service boundary and one native desktop shell sharing a web client (also usable for local diagnostics). Never render the environment in the training loop.
8. **Autonomy is not an authorization system.** The historical promotion gate says "never silently replaced" but has no implementation. Consequential actions need immutable persisted proposals, rationale/impact, explicit single-use confirmation, stale-proposal rejection and audit events. Training never starts on application launch or recovery.
9. **Roadmap scope conflicts with this phase.** The old first milestone demands 50 M steps, ≥400 evaluation games, rendering, and a 50k-step CUDA sweep. None is a prerequisite to the requested small foundation slice. Retain these gates as future work, not completed work, and add an explicitly named Phase 2 foundation milestone.
10. **Unresolved recommendations remain hypotheses.** Eight initial env processes, core affinity, the 26% frequency-to-throughput inference, 50× CPU penalty, ≥24 GB WSL allocation and all throughput claims are not measurements here. CPU-only bounded correctness tests are justified; they do not satisfy B1/B3. Default production device stays explicit CUDA, without fallback.
11. **Internal planning inconsistencies.** Research recommends two PPO epochs but says to gate them behind B4; roadmap starts one and adds two later. D7 asks for a curriculum mutator immediately while roadmap defers it. Use kickoff-only and one epoch for this phase; do not add inert abstractions or new learning strategies. The parameter-count table labels a value as per-network although actor and critic have different output widths. Potential-based invariance also requires the discounted form `gamma * Phi(s_next) - Phi(s)`, not an unqualified difference when gamma < 1.
12. **Platform/security assumptions need limits.** This execution environment is Debian 12, Python 3.11, ~4 GB RAM, no GPU/`nvidia-smi`, no display and few native GUI libraries—not the target machine. WSL/CUDA/WSLg certification remains local. Local HTTP controls need bearer capabilities and no cross-origin permission; public diagnostic preview must be read-only. Checkpoint hashes detect corruption, not malicious replacement; upstream pickle-based buffers must remain trusted local artifacts.
13. **The release has a real NumPy constraint missed by the earlier research.** Resolving the proposed lock failed: published `rlgym-rocket-league==2.0.1` declares `numpy>=1.19,<2` and `rocketsim>=2.1` (no upper bound); GitHub HEAD declares different constraints. PyPI JSON lacks `requires_dist` for this legacy sdist, so source `setup.json` / generated wheel metadata are authoritative. Use **NumPy 1.26.4**, explicitly pin RocketSim 2.2.1, and omit `rlviser-py` (its current NumPy 2 requirement is incompatible). Do not suppress dependency resolution or pretend HEAD is the release. Runtime compatibility still needs the real-stack tests.
14. **Missing foundations:** strict cross-field configuration validation, immutable run configs, process ownership/leases, watchdogs, termination/truncation tests, atomic metadata, log cursors/reconnect, checkpoint inventory and recovery tests, evaluated-vs-unevaluated model labels, setup/recovery instructions, packaging and CI all need implementation.

## What is preserved / intentionally deferred

Preserve the evidence-tagged research, useful stack comparison, multi-agent extension path, benchmark protocols, deployment boundary and offline-only rule. Add corrections rather than erase the historical findings. Supersede design-only architecture/roadmap status with implemented-state sections and clearly separate later milestones.

No 2v2/3v3, curriculum, competitive training, opponent league, original-game automation, automatic strategy changes, auto-promotion, automatic data deletion or fabricated evaluation results. Evaluation/opponent/visualization extension contracts may be defined without claiming an implemented evaluator/renderer.

## Implementation plan derived from the audit

1. Pin and resolve the published RocketSim/RLGym/Learn/Algos/PyTorch stack, separately optional desktop dependencies; support Python 3.11–3.12 with Python 3.12 recommended for WSL.
2. Implement validated config, durable experiments/events/proposals, atomic verified checkpoints and a process supervisor.
3. Integrate the real Rust-backed RLGym Learn collector and upstream PPO through a small lifecycle/checkpoint adapter; use bounded CPU tests here, explicit CUDA preflight on target.
4. Implement six responsive dark desktop views, asynchronous same-origin HTTP, reconnect and explicit confirmation dialogs; keep the backend independent.
5. Execute unit, real-stack integration and GUI tests, document exact results/limitations and next measurable gates, then commit on the session branch.

## Integration findings discovered after the initial audit

The initial audit was source/document based, not a claim that runtime compatibility had already passed. Subsequent real-stack and browser experiments added these findings; see [UPSTREAM.md](UPSTREAM.md) for source/repair/removal conditions and [VERIFICATION.md](VERIFICATION.md) for execution evidence.

15. **Released Learn shared-info startup parser panics.** The parser failed to advance the shared-info byte offset before decoding spaces. Two real worker attempts reproduced Rust panics. Disable that optional serde channel and audit fixed reward components from real timesteps; do not monkeypatch a binary or fabricate metrics.
16. **CPU pinning/GAE load path are not portable as shipped.** Algos hardcodes pinned allocations on a no-driver CPU host and omits the GAE checkpoint context. A narrow portable-buffer subclass makes pinning explicit and restores actual GAE/buffer RNG state. Real CPU PPO/recovery tests verify this, not CUDA throughput.
17. **Fast environment responses can be drained as stale.** Released Learn and inspected HEAD signal child events before draining parent sockets. Actual browser-driven pause delays reached 5–15 seconds. Real Rust traces and the child's five-second keepalive identified a response-loss race. A deterministic shared semaphore delays only response publication until send/drain returns; initial reset bypasses it. All-environment, fixed-count STEP/RESET waves are required. No sleeps, timeout inflation or fake progress substitute for the repair.
18. **Controller update can erase terminal reset intent.** PPO clears its trajectory map. Derive RESET from authoritative collected done flags, including an episode ending at the update boundary.
19. **GUI authenticity requires separate native evidence.** Actual Chromium tests pass six routes and real training/control/persistence, but they do not prove Qt/WSLg. Explicit native tests fail on missing runtime libraries rather than disappearing as skips. Ubuntu 24.04/3.12/Xvfb CI subsequently passed **2 actual native tests plus the full Python/browser suites**; the result is recorded separately. No RTX/WSLg certification is claimed in this sandbox.
20. **Reproducible recovery needs an implementation profile.** Commit + dirty flag alone does not identify edited training code. Run metadata records a deterministic training/config/storage source hash; recovery checks it, exact training packages and Python major/minor at proposal, confirmation and worker load. A changed strategy is not silently resumed under an old immutable config.
21. **Uvicorn SIGTERM return is not the cleanup verdict.** Version 0.42 re-raises its signal after successful lifespan cleanup. The real process regression checks durable stopped/checkpoint/ownership/log evidence and expects `-15`, not a misleading exit-zero-only assertion.

## Resulting implementation and preservation

- Added real optional training/desktop profiles, exact Python/npm locks, Linux setup, packaging, six-view offline native-shell client and an independent capability-protected service/worker.
- Added strict config/immutable experiments, structured SQLite event cursors, process leases/watchdog/EOF shutdown, durable review/confirmation, atomic verified checkpoints and explicit linked recovery.
- Actual upstream simulator/collector/PPO/actor/critic/Adam/GAE remain in use. Adaptations replace only broken compatibility or unsuitable terminal/persistence boundaries.
- Kept all original research text with prominent withdrawals; preserved original architecture/roadmap verbatim after an archive banner in `docs/history/`. Root current docs now describe code/evidence, not a future training system.
- Local evidence: **98 non-desktop Python tests and 5 actual-browser tests passed**; distribution builds and standard dependency checks passed. Native execution is environment-blocked locally, but **98 non-desktop / 2 native Qt / 5 browser** passed on Ubuntu 24.04/Python 3.12 CI with zero skips; see evidence and precise limits in [VERIFICATION.md](VERIFICATION.md).
- Recovery/designation/deletion are reviewed human actions. No competitive agent, larger modes, fake evaluation, automatic best replacement, valuable-data pruning or original-game control was added.

The deliverable remains on the session feature branch and follows the existing commit/pull-request review workflow. No merge or later-phase training is authorized by this audit or a green test suite.
