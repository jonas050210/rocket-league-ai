# Rocket League AI — implemented architecture

**Phase 2 foundation · 2026-10-17 · v0.2.0**

Authority: [PROJECT.md](PROJECT.md). Evidence: [audit](docs/AUDIT.md), [upstream adaptations](docs/UPSTREAM.md), [verification](docs/VERIFICATION.md). Next gates: [ROADMAP.md](ROADMAP.md).

This file describes code that exists. Where an integration is not implemented or not certified on the target host, it says so. The original 611-line design is preserved in [docs/history/ARCHITECTURE_PHASE1.md](docs/history/ARCHITECTURE_PHASE1.md); it is not a setup guide.

## 1. Process boundaries

```text
Desktop: PySide6 QMainWindow + off-the-record QWebEngineProfile
   └─ local HTML/CSS/JS (six views; no Node runtime, external fonts or CDN)
      └─ relative /api requests + capability header, asynchronous fetch/polling
                         │
Independent service: FastAPI / Uvicorn
   ├─ Orchestrator: sole run supervisor, RLock, process-group ownership
   ├─ Store: SQLite WAL/FULL, settings/runs/events/proposals
   ├─ CheckpointStore: immutable inventory/integrity, no Torch deserialization
   ├─ SystemMonitor: psutil and best-effort nvidia-smi, separate sampler thread
   └─ training Popen: new session/process group, JSON stdin + structured stdout
                         │
Training worker (imports Torch ONLY here)
   ├─ WorkerChannel: control receiver + heartbeat, progress age, EOF → stop
   ├─ ControlledCoordinator: real LearningCoordinator/EnvProcessInterface
   ├─ FoundationController: upstream PPOAgentController/PPOLearner
   │   ├─ SeparateActorCritic(DiscreteFF, BasicCritic), separate Adam optimizers
   │   └─ PortableExperienceBuffer + actual Rust GAETrajectoryProcessor
   └─ 1–8 forkserver/spawn environment processes
       └─ RLGym v2 + real RocketSim arena; no renderer
```

The GUI does not own the backend, training CUDA context, simulator or durable state. Native launcher bootstrap is detached (`start_new_session`); GUI exit is deliberately not wired to a stop request. The native page refuses foreign navigation/new windows. A native download-save handler handles JSON report exports.

The backend defaults to loopback. Hosted diagnostics explicitly bind `0.0.0.0` and opt into read-only unauthenticated GETs; relative browser requests also work behind a preview proxy. A preview must never expose a real workspace's capability.

## 2. Module boundaries and dependency direction

```text
web / desktop / CLI → API → Orchestrator → Store / CheckpointStore / Monitor
                                   └─ process protocol → training worker
training worker → validated config / persistence / training adapters → released RL libraries
```

- `config.py` and `domain.py` import neither Torch nor RLGym. `TrainingConfig` is frozen, forbids unknown fields/non-finite floats and enforces cross-field relationships.
- `metadata.py` reads installed package metadata without importing the training stack. Each run records source commit/dirty flag, lock hash, package/Python/platform versions, actual training/config/storage source hash, seed through immutable config, environment/reward contracts and recovery semantics.
- `persistence.py` is Linux-specific: `flock`, atomic replace and file/directory fsync. Store the workspace on WSL's Linux filesystem, not `/mnt/c`.
- `training/` owns observations/actions/reward composition, collector configuration, policy factories, PPO control and serialization adapters. It is loaded only by a confirmed worker launch or explicit diagnostic/test.
- `domain.py` defines `Evaluator`, `OpponentPolicy`, `EvaluationResult` and `ObservationVisualizer`. They are future contracts; there are no fake implementations or synthesized outcomes.

No plugin registry, event bus, distributed queue, web build pipeline or speculative curriculum hierarchy is introduced. The interfaces needed for later self-play/evaluation exist without pretending to implement those phases.

## 3. Service/worker lifecycle

States: `starting`, `running`, `pausing`, `paused`, `resuming`, `stopping`, `stopped`, `completed`, `failed`, `interrupted`.

| Request/event | Rule |
|---|---|
| Start | Only an approved, current single-use proposal; service idle and worker lease free |
| `ready` | Training stack/real environment constructed and initial checkpoint committed; starting → running |
| Pause | running → pausing immediately; worker finishes PPO update, saves, acknowledges paused |
| Resume | paused → resuming; worker resumes real collection and acknowledges running |
| Stop | Any non-stopping active state → stopping; worker finishes update, saves and cleans up |
| Budget exhausted | Completed update, final checkpoint, simulator cleanup, completed acknowledgement |
| Worker exception | failed, actionable error and logs; no failure-path checkpoint of partial PPO state |
| Unexpected exit | failed if no terminal acknowledgement; kill orphan descendants; never auto-restart |
| Watchdog | Missing heartbeat or stalled progress beyond 120 seconds terminates worker group; only acknowledged paused state is exempt from the progress deadline |
| Service SIGTERM | Cooperative stop, wait up to 20 seconds, then kill group if necessary; preserve previous committed state |
| Service killed | Stdin EOF asks worker to stop; broken stdout is safely discarded, not a reason to skip checkpoint/cleanup |
| Service restarted | Unfinished durable runs become interrupted; pending proposals expire; no training starts |

A backend lease prevents two supervisors for the same workspace. An inherited worker lease prevents a restarted service from launching a duplicate while a prior worker is still releasing resources. A raw lock file is not a stale process: ownership is the kernel lock, never presence/absence of a filename.

Control messages and `RLAI_EVENT ` stdout records have protocol version 1. Non-protocol upstream output is retained as logs. SQLite events have monotonic sequence cursors for reconnects. The worker reports actual steps/iterations/optimizer counters/episodes/reward/PPO timings; the supervisor never invents progress.

**Boundary semantics:** pause is not immediate cancellation, and a long CUDA update can delay acknowledgement. The service remains responsive and keeps displaying the pending state. The watchdog is a final containment deadline, not a performance optimizer. There is one active learner per workspace; concurrent evaluation is not implemented.

## 4. Confirmation and authorization

All mutations require the workspace bearer capability (`access-token`, mode 0600) and reject foreign browser origins. There are no control cookies and no permissive CORS. Normal API reads also require the key, except public health. An explicit diagnostic-preview flag permits unauthenticated GETs only; writes still require the capability.

The native shell injects the key via a URL fragment, which the client consumes/removes on initial or same-document navigation. It lives only in the off-record/session client. Polling never confirms an action. Stale client controls are disabled after a disconnect, including the cached configuration form.

Consequential flow:

1. Validate input without side effects.
2. Persist proposal ID, kind/payload, rationale, expected impact, plan revision, digest and ten-minute expiry.
3. Present changes/rationale/impact with an unchecked review checkbox.
4. Accept **literal `true`** confirmation plus exact digest only once.
5. Under the orchestrator lock, revalidate idleness, revision, integrity/config/manifest hash and model protections.
6. Mark executing, perform the action, persist approved/failed result and audit event.

Actions: `apply_config`, `start_run`, `recover_run`, `designate_checkpoint`, `delete_checkpoint`. Ordinary pause/resume/stop are user-initiated direct controls and cannot change strategy. Plan plus revision publish in one SQLite transaction. A process launch/filesystem action cannot be atomic with SQLite; an interruption during execution is marked uncertain/failed and requires inventory review, not automatic retry.

Designation/deletion proposals bind the prior best pointer. Deletion is idle-only, cannot remove a designated best or the sole valid checkpoint of its run, and never occurs through retention. No automatic recommendation executor exists.

## 5. Real environment and agent

The code composes the **released** APIs:

- `MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator())`.
- `RocketSimEngine(rlbot_delay=True)` with 16 checked soccar meshes and explicitly idempotent successful initialization.
- `Float32DefaultObs(DefaultObs(zero_padding=1))`: width 92, contiguous float32.
- `RepeatAction(LookupTableAction(), repeats=8)`: 90 discrete actions, int64 shape `(1,)` serialized to the collector.
- `AuditedReward`: real shipped goal and touch rewards, weights 10 and 0.1. Per-transition component sums are reported from real timesteps because broken upstream shared-info serialization is disabled.
- `GoalCondition` true termination; no-touch and absolute timeout truncation, with goal precedence if simultaneous.
- `EpisodeRLGym` enforces reset-after-done and finite step observations; renderer `None`.

A physics transition is eight ticks / 1/15 simulated second. Two cars yield two agent timesteps. Training both teams with one policy is not a frozen-opponent evaluation, win rate or league.

The policy/critic are separate MLPs with separate Adam optimizers. Batch size equals collection size in this bounded adapter. There is one PPO epoch by default and fixed all-environment collection. Cross-field config checks prevent incomplete update budgets and incompatible minibatches/process counts.

## 6. Upstream integration repairs

See [docs/UPSTREAM.md](docs/UPSTREAM.md) for source locations, evidence and removal conditions.

1. Disable shared-info serde: Learn 2.0.0 fails to advance the startup-space offset after that field and panics. Reward components come from real transitions, not a fake channel.
2. Response-publication semaphore: the released Rust sender signals the environment and *then* drains the response socket. A fast new response can be discarded, causing five-second keepalive stalls and delayed controls. The environment computes its real result but waits to publish until the real sender/drain returns. The first startup reset is exempt. All responses per wave and fixed env count are required.
3. Portable buffer: do not hardcode pinned allocations on CPU/no-driver hosts; restore GAE checkpoint context and buffer RNG explicitly.
4. Boundary checkpoint adapter: avoid upstream non-atomic/pruning writer, in-flight trajectory pickles and stale monotonic timestamps.
5. Authoritative collector done flags trigger reset even if a PPO update cleared the controller trajectory map.
6. Cooperative non-terminal run loop propagates failures instead of using upstream keyboard handling/exception-swallowing `start()`.

These are local subclasses/composition, **not** process-wide monkeypatches, forced dependency overrides, vendored third-party forks, sleeping to hide the race, or replacement PPO/simulation code. Any upstream upgrade must repeat regression tests before removing an adaptation.

## 7. Persistence and checkpoint contract

SQLite WAL with synchronous FULL owns mutable run state, approved plan/revision, proposals and event log. Config and metadata files are written atomically per new run. Original runs remain unchanged on recovery; a parent run/checkpoint pair records lineage.

Checkpoint format `rlgym-learn-ppo-v1`:

```text
manifest.json                         schema, immutable ID/step, config hash, created, file SHA-256/bytes
ppo_agent.json                        completed iteration + cumulative timesteps
application.json                      telemetry, episode history, Python/NumPy/buffer RNGs
rng.pt                                PyTorch CPU/CUDA RNG state
ppo_learner/actor_critic.pt            real actor and critic state
ppo_learner/optimizers.pt, misc.json   Adam states + learner counter
experience_buffer/experience_buffer.zip
experience_buffer/trajectory_processor.json   actual experience + GAE running statistics
```

Only completed update boundaries are saved. Required files are staged, flushed/hashed, validated, directory-fsynced and atomically renamed before `latest.json` updates. Scanning fully verified manifests is authoritative; partial staging and damaged pointers are ignored, and older valid checkpoints remain offered.

Manifest schemas/types, ID/step agreement, file inventory, hashes/bytes, relative paths and symlinks are checked. Inventory integrity caching includes size, timestamps, inode and ctime; consequential actions require a fresh check. Wrong config/package versions, Python major/minor profile or recorded training/config/storage source hash are refused before recovery (and rechecked on confirmation and in the worker).

Loading restores learner, optimizers, buffer, GAE, RNGs and counters, but starts new simulator episodes and drops partial episode trajectories/returns. CPU recovery equality is tested against real serialized state. CUDA/cross-machine numerical equivalence and sudden power-loss certification remain target-host gates. Hashes are damage detection, not signatures; upstream buffer pickles are trusted-local only.

There is no schema migration, user-data deletion scheduler, model import endpoint or full backup tool. A report is measured status plus the first 1,000 durable events, not a full run backup. Full events remain in SQLite; the UI keeps a bounded recent buffer.

## 8. Observability and frontend behavior

Polling uses `/api/snapshot` plus an events cursor; async fetch has a timeout, automatic reconnect and retained last-known state. DOM updates preserve config drafts and disable writes when disconnected/busy. Requests are gated to avoid duplicate proposal submissions; server single-use checks remain authoritative.

Charts contain only actual update events. `reward` is mean transition reward; completed-episode return is separately named. Upstream segment return is not mislabeled as completed-episode win rate. Throughput excludes stack construction and accounts for pause duration. CPU/GPU readings name their source; unavailable NVML/CUDA values remain null/unverified.

NVIDIA sampling runs separately from API/training and times out. WSL does not support every NVML field. Recommendations cover measured resource pressure and recovery/preflight checks; no competitive recommendation is fabricated.

## 9. Phase 1 decision reconciliation

| Original decision | Current status |
|---|---|
| D1 Learn/Algos, not old rlgym-ppo | Retained with actual release pins and narrow documented compatibility repairs; the old no-NumPy-conflict rationale was wrong |
| D2 1v1 first | Implemented; no larger modes |
| D3 90 discrete / tick skip 8 | Implemented and tested |
| D4 one shared policy before multi-controller | Implemented; opponent history remains future |
| D5 frozen historical opponents | Contract only; not implemented |
| D6 separate evaluation process | Future design; no evaluator exists |
| D7 curriculum abstraction from day one | Deferred; real kickoff-only mutator, no inert curriculum scaffold |
| D8 two-tier automatic retention | Superseded by preserved-by-default inventory and explicit confirmed deletion/protections |
| D9 CPU traced deployment actor | Deferred; no game/deployment code |
| D10 serialized config | Implemented as validated application config + portable resolved upstream artifact + immutable metadata |
| D11 delta-based touch/no-touch replacements | **Withdrawn**: published engine counters reset per state; shipped touch/no-touch semantics are correct |

Current architecture adds independent service/native UI, explicit approval, crash-safe storage, process leases and verified recovery, which were absent from the original design. Quantitative benchmarks and competitive gates remain future measurements, not implied by this foundation.

## Addendum — benchmark, evaluation and replay (2026-10-17)

- `benchmark.py` (control plane, no Torch): `BenchmarkSpec` (at most eight candidates of device × environment processes, a default 20-second wall-clock measurement window per genuine run, and a 1,000,000-agent-step safety cap), warm-up-excluding median/CV aggregation of the worker's real per-iteration telemetry, and an explainable fastest-stable recommendation. `Orchestrator.start_benchmark` executes each candidate as a disposable training run through the existing worker lifecycle, samples host/CPU/GPU/VRAM peaks, persists sessions/results in SQLite (`benchmarks`, `benchmark_results`), and removes the throwaway run directories afterwards. When the spec sets `auto_default` and the strict gates pass (three repeats, CV ≤ 0.10, GPU telemetry for CUDA), only `device` and `n_proc` are written to the approved plan; the previous plan is recorded for a reviewed revert. Strategy fields are never changed by a benchmark.
- `evaluation.py` (control plane): `EvaluationSpec`, per-game records, side-balance validation, Wilson 95% intervals and lower-bound checkpoint comparison. `training/evaluate.py` is a separate frozen-inference subprocess: verified checkpoint weights only, greedy CPU argmax, seeded `random-v1`/`idle-v1` opponents, genuine RocketSim games; `training/evaluate_core.py` holds the simulator-agnostic counting loop (goal = true termination with scorer reward +10; truncation = draw) and decision-rate trajectory extraction from real `GameState`. Results and `evaluations/<id>/trajectory.json` persist in the workspace.
- Mutual exclusion: at most one auxiliary workload (benchmark or evaluation) runs at a time, and neither runs concurrently with a training worker (`Orchestrator.auxiliary`, enforced by `_require_idle`). Backend restart marks running benchmarks/evaluations `interrupted` without inventing partial results.
- API: `POST/GET /api/benchmarks`, `POST /api/benchmarks/stop`, `POST/GET /api/evaluations`, `GET /api/evaluations/{id}/trajectory`. The replay viewer renders recorded trajectories on a canvas (top-down schematic); it is not a live or 3D renderer and cannot affect training.
