# Rocket League AI — evidence-led roadmap

**Updated 2026-10-17.** Current-state authority: [PROJECT.md](PROJECT.md); architecture: [ARCHITECTURE.md](ARCHITECTURE.md); exact checks: [docs/VERIFICATION.md](docs/VERIFICATION.md).

**Naming correction:** this delivery's **Phase 2 foundation** is the user's application/engineering phase. The Phase 1 planning document used “Phase 2” for throughput tuning. They are not the same milestone. The original 444-line roadmap and benchmark procedures are preserved in [docs/history/ROADMAP_PHASE1.md](docs/history/ROADMAP_PHASE1.md), subject to the [audit corrections](docs/AUDIT.md).

A checkbox records evidence, not an aspiration. An interface, a configured CI job or an estimated steps/s figure does not pass a gate. No automatic move into competitive training follows this delivery.

## F — Phase 2 foundation delivery

**Objective:** launch a real, observable desktop-controlled learning slice with explicit user authority and recoverable state, without training a competitive agent.

### Implemented and locally verified

- [x] Audit all original files/dependencies/workflow before code changes; preserve research and design history.
- [x] Resolve and pin published RocketSim/RLGym/Learn/Algos/PyTorch packages; record NumPy/API/wheel constraints rather than force incompatible libraries.
- [x] Independent API/backend and worker; no Torch in service/GUI; no startup/reconnect autotraining.
- [x] Real 1v1 kickoff-only headless simulation, 92 float32 observation, 90 actions, tick skip 8, actual upstream PPO parameter updates.
- [x] Single- and two-environment real bounded CPU runs to exact update/step budgets.
- [x] GUI client: Dashboard, Training, Monitoring, Models, Configuration, Visualization; measured metrics/logs, honest missing evaluator/renderer labels.
- [x] Actual browser GUI → backend → real training lifecycle; validate/reject/approve, pause/resume/stop, disconnect/reconnect, close/reopen window, inventory/metadata/manual designation and report download.
- [x] Consequential proposal rationale/impact, checkbox/digest confirmation, stale/expired/replayed/restarted review rejection; no automatic promotion/deletion/config change.
- [x] Atomic checkpoint publication, integrity checks, malformed/corrupt/partial checkpoint tests, older-valid fallback, protected deletion and original-run preservation.
- [x] Real recovery equality for actor/critic, Adam, buffer, GAE running statistics, RNG and counters; simulator-reset limitation explicit.
- [x] Real worker kill, frozen-worker watchdog, backend kill/restart with stdin EOF cleanup, explicit recovery and graceful service SIGTERM.
- [x] Correct goal/truncation/GAE behavior and actual released engine touch callback semantics.
- [x] Reproduce and contain upstream shared-info startup defect and early-response UDP race; deterministic response fence and regression checks.

### Delivery evidence that must remain explicit

- [x] Native Qt launch/close/reopen/bootstrap: **2 actual Qt tests passed** under Ubuntu 24.04/Xvfb; verified separately from browser tests.
- [x] Python **3.12.15** full profile in CI: **98 non-desktop + 2 native + 5 browser** tests passed with zero skips; local non-desktop execution also passed on 3.11.2.
- [ ] Target WSLg/RTX 4060 Ti certification: **not done in this CPU-only sandbox**.

The first two delivery verification gates have actual results in `docs/VERIFICATION.md`, not merely a configured job. The sandbox still lacks the native GUI runtime; the native pass is on Ubuntu/Xvfb CI with standard packages. The unchecked RTX/WSLg item is a separate operator-host gate, not an inferred pass or authorization to start another phase.

## N0 — Next: target-host certification (before tuning or skill claims)

**Entry:** foundation tests pass; user explicitly authorizes any test training plan. Keep current plan/checkpoints unchanged until reviewed.

**Procedure / measurable acceptance:**

1. On the user's Windows/WSL 2 host, record OS/WSL/kernel/Python/lock hash, NVIDIA Windows driver, `nvidia-smi`, visible RAM/CPU allocation and storage location.
2. `rocket-ai doctor --probe` exits **0**, reports CUDA available, exact device `RTX 4060 Ti`, and completes an actual CUDA matmul plus 16-mesh/eight-tick simulation probe. No CPU fallback is accepted as a pass.
3. Native `rocket-ai desktop` opens in WSLg and all six routes are usable. Explicitly start a bounded CUDA plan, verify real update/episode/log/checkpoint counters; pause/resume/stop all acknowledge at update boundaries.
4. Close and reopen the native window during an active run; the same worker/run remains connected, no duplicate or automatic new run. Confirm native report-save/cancel behavior interactively.
5. Record peak allocated/reserved CUDA memory and host RSS during the run. No OOM, non-finite loss/parameters or missing worker acknowledgement. Do not claim a safe 8-GB budget from a CPU test.
6. Confirm recovery into a new linked CUDA run. Validate config/package equality, monotonic learner counters, valid old/new manifests and reset-episode disclosure.
7. Save a concise result row with commands, steps, timings, VRAM, lock hash and limitations. Failures retain checkpoints and require user review.

**Non-goals:** “maximum n_proc,” skill/rank, automatic CPU/GPU selection, renderer compatibility overrides, large-mode training.

## N1 — Next: reliability soak on the target host

**Entry:** N0 pass and explicit approval of a fixed, bounded experiment. The current config caps one run at one million agent steps; do not silently raise that safety limit.

- Run a **30-minute operator-supervised soak** using separately initiated bounded runs as needed, not an unbounded background experiment.
- Maintain measured update/lifecycle progress; perform at least three GUI reconnects and three pause/resume cycles.
- Explicitly stop at least once, verify all manifests, corrupt a **disposable test copy** of a newest checkpoint and demonstrate older-valid fallback; never damage valuable originals for a test.
- Interrupt a disposable service process, demonstrate no automatic restart, verify worker lease release, and recover only after review.
- Record maximum/median API health latency during training, peak worker/simulation RSS, VRAM, checkpoint size/growth, acknowledgement latency and logs. No orphaned process groups after normal shutdown and no invalid manifest offered for recovery.
- Review disk growth: there is no automatic checkpoint pruning. If space is insufficient, propose a retention/deletion plan for explicit approval instead of deleting data.

Pass/fail must identify the host and fault protocol. Unit fault injection is not evidence of physical power-loss durability. True sudden-power-loss certification remains a separate storage test.

## N2 — Frozen-opponent evaluator (foundation implemented 2026-10-17)

**Status:** implemented and verified on CPU in this sandbox. `training/evaluate.py` plays genuine side-balanced RocketSim games with a frozen checkpoint actor (greedy CPU inference) against seeded `random-v1`/`idle-v1` baselines; results persist with Wilson 95% intervals, goals and per-game records; checkpoints are compared by lower confidence bound in Models. A 384-step foundation checkpoint measured 0W/0L/4D vs random — a working baseline, not strength. Remaining for the full gate: 400-game comparisons on the target host, a frozen *learned* opponent pool/league, and promotion proposals backed by this evidence (designation is still manual).

## N2-history — Original evaluator gate (superseded by the status above)

**Why before recommendations:** training reward, entropy and checkpoint age do not establish playing strength.

**Scope:** implement `Evaluator`/`OpponentPolicy` contracts using a genuine scripted/random baseline and frozen inference-only snapshots. No opponent league, curriculum or original-game control yet.

**Acceptance:**

- Versioned evaluation manifest: candidate/opponent/config hashes, seeds, environment/reward versions, inference mode, sides, goals/wins/losses/draws and exact games.
- At least **400 side-balanced games** per declared comparison, seeded and capped by explicit evaluation time limits; report draws and confidence intervals, not just a percentage.
- Opponent weights do not change during evaluation. Evaluation runs separately from throughput measurement.
- GUI displays actual results and evidence-linked advisory candidates; no synthetic win rate or recommendation.
- Promotion/replacement remains a separate, user-confirmed proposal with rationale/impact and expected best-pointer precondition. A statistical gate is necessary evidence, not authority to act automatically.
- Tests cover side balance, deterministic inference, frozen weights, counting, seed/config traceability, timeout draws and replay protection.

No competitive threshold/rank is promised by a 400-game harness. It can faithfully measure a weak baseline policy.

## N3 — Measured throughput characterization (benchmark implemented 2026-10-17)

**Status:** the benchmark harness is implemented and verified on CPU in this sandbox: candidates (device × 1–8 env processes) run genuine bounded training runs through the existing worker; warm-up iterations are excluded; simulation/PPO/episode throughput, resource peaks and stability (CV ≤ 0.25) produce an explainable fastest-stable recommendation that is never auto-applied. First sandbox measurement (batch 128, 1 env): 1,529 agent steps/s. Remaining: the B2–B5 sweeps on the RTX target host below, which no CPU number substitutes for.

## N3-history — Original throughput gates (still the target-host procedure)

Preserve the intent of historical gates B1–B5, but use the actual locked stack and current safety contracts. No benchmark is currently passed on the RTX target.

| Historical gate | Current next action | Required evidence |
|---|---|---|
| B1 platform | Covered by N0 CUDA/WSLg checks; renderer is independently deferred | Actual GPU/matmul/simulation and host resource record |
| B2 simulation/process scaling | Fixed-seed headless sweeps of 1, 2, 4, 8 env processes, within visible allocation | Warm-up excluded, fixed protocol, CPU/RSS, three repeats and tail latency |
| B3 end-to-end learner | Bounded fixed-config CUDA learner runs | Agent steps/s split into collection/learning/checkpoint cost, VRAM and loss health |
| B4 learner/network/epochs | Only after N2 and explicit strategy approval | Compare learning/evaluation per wall-clock, not just raw steps/s |
| B5 affinity/WSL allocation | Optional operator changes after baseline | Before/after measurements, actual visible cores/RAM, no automatic host config edit |
| B6 deployment parity | Future original-game phase only | Versioned actor artifact and observation/action parity tests |
| B7 live-game timing | Future offline/RLBot phase only | Measured p50/p95/p99 end-to-end latency and safe runtime integration |

Fix the response-race compatibility profile before comparing collectors. The current response fence assumes complete collection waves; reducing the response fraction/dynamic processes is not a free tuning switch. Any upgrade or collector topology change must re-run semantic, concurrency and lifecycle tests.

The previous “50× CPU penalty,” frequency-to-throughput extrapolations and core-count recommendations are hypotheses, not this repository's measurements. A 512-step CPU smoke is not B3, a WSL benchmark, or evidence of competitive learning.

## Later, explicitly out of this phase

1. Evaluation-backed reward/curriculum experiments, one change at a time with explicit user approval.
2. Frozen opponent history/self-play league and checkpoint comparison based on real N2 evidence.
3. Larger networks or match modes only after measured memory/throughput/evaluation gates; 2v2/3v3 are not included here.
4. Compatible observation/selected-match visualization; keep all training environments headless. Verify NumPy/renderer dependency compatibility in an isolated profile first.
5. CPU-only deployment actor, manifest and observation/action parity; then separately scoped **offline/RLBot** original-game integration. No automated online game control.
6. Competitive validation only with defensible evaluation and compute/time records; no rank promise from planning documents.

## Review / completion discipline

- Every implementation claim points to code **and an executed verification record**.
- Keep `PROJECT.md`, architecture, roadmap, setup/recovery and the execution record synchronized after a meaningful change.
- Preserve useful research; clearly withdraw disproved claims rather than quietly reusing them.
- Keep `.rlai`, binaries, test traces, capabilities and generated datasets out of commits.
- Commit on the feature/session branch and submit through pull-request review. No automatic merge, strategy change, best-model promotion or valuable-data deletion is implied by a green test suite.
