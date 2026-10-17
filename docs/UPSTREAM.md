# Released-stack compatibility record

**Checked 2026-10-17.** Exact installed profile: `uv.lock`. This record explains narrowly scoped adaptations in `training/`; it is not permission to install GitHub HEAD over the lock.

## 1. Identity and source evidence

| Distribution | Inspected release / actual interface |
|---|---|
| `rocketsim` | 2.2.1 Linux ABI3 wheel; imports **`RocketSim`** |
| `rlgym-api` | 2.0.0; real `RLGym` reset/step component API |
| `rlgym-rocket-league` | 2.0.1 published sdist and installed wheel; 16 bundled soccar meshes; `RocketSimEngine(rlbot_delay=True)` |
| `rlgym-learn` | 2.0.0 published Rust/Python source; collector, serde, process control |
| `rlgym-learn-algos` | 0.4.2 source; actual PPO/controller/buffer/GAE persistence |
| `torch` | 2.9.1 resolved Linux CUDA 12.8 profile; actual CPU operations verified locally |

PyPI release records: [RocketSim](https://pypi.org/pypi/rocketsim/2.2.1/json), [Rocket League environment](https://pypi.org/pypi/rlgym-rocket-league/2.0.1/json), [Learn](https://pypi.org/pypi/rlgym-learn/2.0.0/json), [Algos](https://pypi.org/pypi/rlgym-learn-algos/0.4.2/json).

GitHub source heads inspected during the audit: RLGym `2fc8d435663454b7fe45d2556777efd65cba8c2d`, Learn `250379331d21946afaed384dd0fadc857f3df6b6`, Algos `466b8e30f911914602a5cb3ba6053eabc53ab0e3`. These are evidence references, not install targets. Release source/metadata wins where they disagree.

## 2. Published dependency/API differences

**NumPy:** the legacy Rocket League sdist's PyPI JSON lacks `requires_dist`. Its extracted `setup.json`/generated wheel metadata explicitly require `numpy>=1.19,<2` and `rocketsim>=2.1` (no upper bound). Resolving NumPy 2 failed correctly. We pin **1.26.4** and RocketSim 2.2.1. Current rlviser-py requires NumPy 2, so no compatible renderer is claimed or forced into this profile.

**Engine:** HEAD adds a `game_mode` constructor argument; release 2.0.1 does not have it. Use released Soccar APIs and bundled meshes, not a copied HEAD snippet.

**Pydantic:** Learn 2.0.0 uses newer serialization interfaces. The selected 2.14.1 satisfies the actual Learn requirement; an older FastAPI-example pin is not interchangeable.

**Wheel tags:** RocketSim's wheel filename is ABI3-compatible and imports/steps successfully on Python 3.11 locally and 3.12 in CI, but the internal WHEEL tags incorrectly declare `cp39-cp39`. `uv pip check` warns; standard `pip check` passes the dependency graph. Metadata is not patched. This remains an upstream packaging defect, not a claimed universal ABI certification.

## 3. Shared-info startup offset defect (reproduced)

**Release source:** Learn 2.0.0 `src/env_process_interface.rs::collect_start_response_data`.

After deserializing shared info, the startup parser retrieves the object but does not advance the byte offset before reading observation/action spaces. With shared-info serde enabled, two real worker launches panicked in Rust with an invalid slice offset (`8385531473611417793` into 16,248 bytes); cleanup then hit another panic.

**Application adaptation:** `shared_info_serde_type=None` in `coordinator_config`. Do not deserialize shared info for metrics. The fixed goal/touch decomposition and completion counts are computed from actual collected timestep rewards/flags. No simulated values are substituted. The environment's local shared info still exists for reward calculations.

**Evidence:** real worker completion, episode/metric/checkpoint tests and GUI lifecycle tests. Enabling the broken channel remains unsupported. Remove this workaround only after an upstream release fixes the parser and real startup/data tests pass with the intended serde.

## 4. Fast-response drain race (reproduced during browser testing)

**Release source:** Learn 2.0.0 `src/env_process_interface.rs::send_env_actions`:

```text
write action to shared memory
signal environment event
DRAIN parent response socket
mark awaiting flag false
```

A fast environment can complete/publish a **new** response between signal and drain. The sender drains that response, then collection blocks until the environment's five-second idle keepalive. Rust collection retains the Python GIL while polling, which also delays the worker's Python heartbeat/control threads. Observed browser-driven pause delays were approximately 4.6, 9.7 and 14.9 seconds despite preceding 128-agent-step iterations taking about 0.06 seconds. Tracebacks located the stall in the real Rust collect call; the source ordering and five-second keepalive explain the failure. It also exists in the inspected HEAD.

**Deterministic adaptation, without patching the binary:**

- `ControlledCoordinator` creates a semaphore in the same forkserver/spawn context as Learn.
- Each real environment computes its step/reset result, then waits before returning it into Rust's response serialization.
- After the actual upstream `send_env_actions` call returns (therefore after its drain), the coordinator releases one permit per action.
- The first startup reset bypasses the fence, so startup handshake cannot deadlock.
- With **all environments required per collection wave** and fixed process count, each environment consumes exactly one permit before the next wave. No result/observation/reward is fabricated, no sleep is added to simulate progress, and the collector/PPO remain upstream.

**Scope:** the adapter only emits STEP/RESET, uses `min_frac_process_responses_per_collection=1.0` and has no dynamic env process tuning. Partial-response collection, DEFER scheduling or dynamic process changes require redesign/re-verification, not a silent toggle.

**Evidence:** actual single- and two-environment budget completion, a real-environment pre-publication blocking regression, repeated delayed pause/resume cycles with a bounded deadline, backend failure/recovery tests, and the entire browser lifecycle after the fix. One measured corrected browser run acknowledged pause in about 0.053 seconds and stop in about 0.11 seconds; these are isolated CPU correctness observations, **not a target-host performance guarantee**.

**Removal condition:** a verified upstream release drains before signalling / otherwise proves the race fixed, plus concurrency/lifecycle tests without the fence. Do not replace this with a guessed environment sleep or simply increase test timeouts to hide lost responses.

## 5. CPU buffer allocation and GAE restoration

**Source:** Algos 0.4.2 `ppo/experience_buffer.py`, `experience_buffer_numpy.py` and `gae_trajectory_processor.py`.

- Upstream hardcodes pinned tensor buffers. On this CPU/no-driver host, `pin_memory=True` fails. `PortableExperienceBuffer` changes lazy tensor buffer allocation according to explicit config and uses pageable CPU tensors for CPU mode. CPU config with pinning is rejected.
- The buffer does not pass its checkpoint folder into the GAE processor's derived load context. The local adapter explicitly loads that context and restores GAE return-running statistics.
- Loaded tensor buffers are reconstructed through local overridden helpers; no Torch/global upstream monkeypatch. Sampling RNG state is separately persisted/restored.

**Evidence:** real PPO CPU updates and equality of recovered buffer arrays/tensors, GAE state/counts and buffer RNG. CUDA pinned-memory performance remains a target-host measurement.

## 6. Controller lifecycle/checkpoint differences

**Source:** Algos `ppo_agent_controller.py`, `ppo_learner.py`; Learn `learning_coordinator.py`.

- Upstream `start()` is terminal-oriented and catches errors; it may try to save after a failure. The local cooperative loop calls the real collector/controller/learner and propagates exceptions, acknowledges pause/stop at completed updates and never publishes a failure-path partial learner state.
- Upstream checkpoint writes expose partially written directories and prune with `shutil.rmtree`. Managed checkpoints instead use the application's staged/fsynced/hashed atomic store; upstream periodic save is disabled and pointed at an unused path, never managed inventory. No automatic deletion.
- Upstream controller recovery includes in-flight trajectory pickles and old `perf_counter` values. Our format deliberately saves/restores completed boundaries, resets simulator episodes and monotonic timings, persists application/RNG state, and labels the limitation honestly.
- A PPO update clears current trajectory maps; using only that map can lose terminal-reset intent. `FoundationController.get_env_actions` uses authoritative collected terminated/truncated flags to issue RESET, including a terminal step coinciding with an update.
- Initialization is excluded from iteration timing; paused duration is removed from collection/iteration clocks. Reported throughput is actual measured collection/update time, not an extrapolated simulator benchmark.

**Evidence:** actual parameter changes, complete budgets, paused counters stable, resumed counters increase, stop/failure/kill/watchdog handling, real backend crash, and restored learner/optimizer/buffer/GAE/RNG/counters. None proves bit-exact simulator continuation, CUDA correctness on absent hardware or competitive strength.

## 7. The touch-counter claim was wrong, not an upstream defect

Released `Car.ball_touches` means “touches since last state was sent.” `RocketSimEngine._get_state` copies `_touches[id]` and resets it to zero. The shipped `TouchReward` pays on a contact-bearing transition; the shipped `NoTouchTimeoutCondition` correctly resets idle time following a contact.

The Phase 1 claim that these are cumulative episode counts, and its proposed adjacent-count delta reward/timer, are withdrawn. Adjacent contact transitions can both report one; subtracting them loses a real touch. `AuditedReward` composes shipped reward functions rather than implementing the incorrect replacement. Regression tests exercise the actual engine callback/state extraction twice, followed by idle timeout, and real goal/timeout behavior. Callback injection in that regression is explicitly a semantic test, **not a learned-contact or skill demonstration**.

## 8. Upgrade policy

Keep the lock, source evidence, compatibility code, tests and docs synchronized. On an upstream bump:

1. Resolve actual released metadata and inspect changed source/API/mesh/serde/checkpoint contracts.
2. Re-run semantic/GAE/storage/autonomy tests and real one-/multi-env workers, repeated lifecycle and recovery equality.
3. Re-run actual-browser and native Qt checks. Separately re-certify CUDA/WSLg on the target.
4. Remove an adaptation only when the exact defect is fixed and a regression still passes without it.
5. A strategy/config change is a proposal for user review, not an automatic consequence of a package update. No cross-version recovery or existing checkpoint conversion is silently performed.
