> **Historical Phase 1 design, preserved verbatim below (2026-10-17).** Not current setup or implementation. Touch-counter, NumPy/renderer and exact-resume assumptions were withdrawn. Read [PROJECT.md](../../PROJECT.md), the current [ARCHITECTURE.md](../../ARCHITECTURE.md) and [audit](../AUDIT.md). Original relative links below are relative to the repository root.

# Rocket League AI — System Architecture

**Companion to** [`RESEARCH.md`](RESEARCH.md) (evidence and citations) and [`ROADMAP.md`](ROADMAP.md) (phases and acceptance criteria).
**Status:** design only. Nothing here is implemented. Evidence tags **[V]/[C]/[R]/[E]/[U]** are defined in `RESEARCH.md` §0.

---

## 1. Design goals and non-goals

### 1.1 Goals

| # | Goal | How the architecture serves it |
|---|---|---|
| G1 | **Maximum useful training progress per unit of compute** | CPU-bound env farm, GPU-only learner, float32 obs, tunable batched inference, no wasted serialisation |
| G2 | **Stable learning** | Winning-dominated reward with annealed shaping, GAE + reward/advantage standardisation, gradient clipping, KL/entropy/clip-fraction telemetry |
| G3 | **Reproducibility** | pydantic config serialised to `config.json`, pinned dependency lock, per-run metadata, seeded envs, retained policy snapshots |
| G4 | **Measurable competitive strength** | Out-of-process evaluation against a frozen opponent set, Wilson-interval promotion gate, reward-component auditing |
| G5 | **Clean deployment boundary** | Training never depends on the game; the game never depends on the trainer. One-way artifact hand-off plus a parity test |
| G6 | **Smallest viable start, clear growth path** | Phase 1 is one policy + one process pool. Every extension is a swap of one component, not a rewrite |

### 1.2 Non-goals (explicitly out of scope for the first iterations)

- 3v3, or any mode other than 1v1 initially.
- Attention / entity-based networks (EARL). `DefaultObs` + MLP first.
- Continuous or multi-discrete action spaces.
- Imitation learning / replay pretraining / teacher distillation.
- Multi-machine distribution (no Redis, no RPC).
- Training against the live game. **RLBot is evaluation-only.**
- Online play of any kind. Offline/RLBot modes only — see `RESEARCH.md` §7.5.

---

## 2. Component architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  WSL 2 (Ubuntu)  —  TRAINING HOST                                                    │
│                                                                                      │
│  ┌────────────────────────────── LEARNER PROCESS (1) ────────────────────────────┐  │
│  │  LearningCoordinator  (rlgym_learn)                                            │  │
│  │   ├─ reads config.json  (pydantic LearningCoordinatorConfigModel)              │  │
│  │   ├─ EnvProcessInterface ── UDP control + shared memory ──► env processes      │  │
│  │   └─ AgentController                                                           │  │
│  │        ├─ Phase 1: PPOAgentController  (one policy, both cars)                 │  │
│  │        └─ Phase 3+: MultiAgentController                                       │  │
│  │             ├─ "current"   PPOAgentController   (learns)                       │  │
│  │             ├─ "hist_0..k" FrozenPolicyController (inference only, opponent pool)│ │
│  │             └─ "baseline"  ScriptedController / RandomController               │  │
│  │                                                                                │  │
│  │   PPOLearner (torch, cuda:0)                                                   │  │
│  │    ├─ SeparateActorCritic( DiscreteFF(92→90), BasicCritic(92→1) )              │  │
│  │    ├─ NumpyExperienceBuffer + GAETrajectoryProcessor (Rust, CPU)               │  │
│  │    ├─ Adam(actor), Adam(critic) — separate optimizers                          │  │
│  │    └─ checkpoint writer ──► agent_controller_checkpoints/                      │  │
│  │   MetricsLogger ──► console + Weights & Biases                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                    ▲                                   │                             │
│      obs (f32) / action (i64) / reward / done          │                             │
│                    │                                   ▼                             │
│  ┌─────────────────┴────────── ENV PROCESS POOL (n_proc, CPU) ──────────────────┐   │
│  │  env_0 … env_{n-1}   one RLGym v2 env each, Rust mio/shmem driver             │   │
│  │    RocketSimEngine(RocketSim Arena, 120 Hz, rlbot_delay=True)                 │   │
│  │    DefaultObs(f32) · RepeatAction(LookupTableAction, 8) · CombinedReward      │   │
│  │    GoalReward + TouchEventReward*  ·  GoalCondition / IdleTouch* / Timeout    │   │
│  │    MutatorSequence(FixedTeamSizeMutator(1,1), CurriculumStateMutator)         │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
│  ┌──────────────── EVALUATOR (separate process, on demand) ───────────────────────┐  │
│  │  build env · load two frozen checkpoints · greedy argmax · fixed seeds          │  │
│  │  N games, side-balanced ──► results.jsonl ──► wilson/promotion verdict          │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌──────────────── EXPORTER ──────────────────────────────────────────────────────┐  │
│  │  actor_critic.pt ──► torch.jit.trace(actor) ──► policy.pt  +  obs builder copy  │  │
│  │  numerical equivalence test vs. training actor on fixed obs                     │  │
│  └───────────────────────────────────┬────────────────────────────────────────────┘  │
└──────────────────────────────────────┼───────────────────────────────────────────────┘
                                       │  one-way file copy to the Windows filesystem
┌──────────────────────────────────────▼───────────────────────────────────────────────┐
│  WINDOWS HOST  —  DEPLOYMENT / EVALUATION ONLY                                       │
│   Rocket League (offline, FPS capped 120, VSync off)                                 │
│   RLBotServer (RLBot v5) ──TCP+FlatBuffers──► bot process                            │
│     bot: rlbot.managers.Bot.get_output(GamePacket) → ControllerState                 │
│       ├─ rlgym_compat V2 GameState  (GamePacket → rlgym-shaped state)                │
│       ├─ ported DefaultObs          (must match the training obs builder)            │
│       ├─ torch.jit policy, CPU, greedy argmax, tick_skip 8                           │
│       └─ observation parity test vs. RocketSim                                       │
│   Dependencies: rlbot, torch(CPU), numpy, rlgym-compat ONLY. No rocketsim,           │
│   no rlgym_learn, no CUDA.                                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Component responsibilities

*\* = custom component required because the shipped RLGym v2 equivalent has the wrong semantics — see §4.2 and decision D11.*

| Component | Owns | Must never |
|---|---|---|
| **Env processes** | Physics stepping, obs construction, reward computation, termination/truncation, state setting | Know anything about policies, networks or checkpoints |
| **AgentController** | Which policy acts for which agent id in which env; reset/step/defer/close decisions | Do gradient work |
| **PPOLearner** | Networks, optimizers, GAE consumption, PPO update, checkpointing | Touch RocketSim or an env process |
| **MetricsLogger** | Telemetry sinks | Influence control flow |
| **Evaluator** | Match scheduling, seeding, statistics, promotion verdict | Mutate a training checkpoint |
| **Exporter** | Producing a self-contained inference artifact | Import training-only dependencies |
| **RLBot bot** | Real-time control within ~8.3 ms | Import `rlgym_learn`, `rocketsim`, or CUDA |

---

## 3. Process topology for the target machine

**[V]** Hardware: i7‑12700F = 8 P‑cores (HyperThreaded) + 4 E‑cores = 12 cores / 20 threads; RTX 4060 Ti 8 GB; 32 GB RAM.

**[E] Recommended initial layout:**

| Role | Count | Placement | Rationale |
|---|---|---|---|
| Learner (control + GPU) | 1 | any core; consider an E-core pair | Mostly I/O-bound during collection, bursty during update |
| Env processes | start **8**, sweep 8/12/16/18/20 | P-cores first | One RocketSim arena each; the dominant CPU consumer |
| Evaluator | 0 (on demand) | run while training is **paused** | Otherwise throughput numbers are meaningless |
| rlviser renderer | 0 or 1 | best-effort | Unreliable under WSLg — `RESEARCH.md` §8.3 |

**Why not 20 env processes immediately [E]:** the collector waits for `min_frac_process_responses_per_collection` (default **0.45**) of process responses per inference pass **[V]**, so stragglers gate the batch. On a hybrid CPU, an env process on an E-core (max 3.6 GHz) is ~26 % slower than one on a P-core (max 4.9 GHz) **[C]** and becomes a systematic straggler. Oversubscription also multiplies per-process RSS. **Find `n_proc` by measurement (gate B2/B3), not by assumption.** The one concrete hypothesis worth testing is pinning env processes to P-cores (`taskset` / `numactl`) — **`[U]` U3**.

**Serialisation choices [V][E]:**
- `agent_id_serde_type = STRING` (agent ids are `"blue-0"` etc.)
- `obs_serde_type = NUMPY(np.float32, config=STATIC(shape=(92,)))` — derive the shape from `DefaultObs.get_obs_space()`, never hard-code (`RESEARCH.md` §5.2.4 trap)
- `action_serde_type = NUMPY(np.int64, config=STATIC(shape=(1,)))`
- `reward_serde_type = FLOAT`
- `shared_info_serde_type = None` unless ball-prediction-style shared state is added
- `recalculate_agent_id_every_step = False` — **required** by `PPOAgentController` **[V]**
- `OPENBLAS_NUM_THREADS=1` set before any numpy import — **required** **[V]**

---

## 4. Environment layer

### 4.1 Composition

```
RLGym(
  state_mutator    = MutatorSequence(
                       FixedTeamSizeMutator(blue_size=1, orange_size=1),
                       CurriculumStateMutator(stage=...),      # custom — see 4.3
                     ),
  obs_builder      = Float32DefaultObs(zero_padding=1, ...),   # DefaultObs subclass, f32
  action_parser    = RepeatAction(LookupTableAction(), repeats=8),
  reward_fn        = CombinedReward(
                       (GoalReward(), 10.0),
                       (TouchEventReward(), 0.1),      # custom, delta-based — see 4.2 warning
                       (AnnealedReward(VelocityBallToGoalReward(), 0.1→0.02), ...),
                       (AnnealedReward(SpeedTowardBallReward(),  0.1→0.0),  ...),
                     ),
  termination_cond = GoalCondition(),
  truncation_cond  = AnyCondition(IdleTouchCondition(30), TimeoutCondition(300)),  # IdleTouchCondition is custom — see 4.4
  transition_engine= RocketSimEngine(rlbot_delay=True, game_mode=rsim.GameMode.SOCCAR),
  renderer         = None,                                     # optional RLViserRenderer
)
```

Observation width is **92** for 1v1 **[C]** (`RESEARCH.md` §4.3). Action space is **90 discrete**, held 8 ticks → **15 decisions/s** **[V]**.

### 4.2 Reward architecture

Three layers, in priority order:

1. **Objective (constant, dominant):** `GoalReward` × 10. Never annealed. **[V]** The shipped class is verified zero-sum and one-shot: it returns `+1` for the scoring team and `−1` for the conceding team, on the goal step only. If an asymmetric concede penalty is wanted later (the `aggression_bias` idea ZealanL describes **[R]**), it must be a **custom `RewardFunction`** — `GoalReward.__init__` takes no arguments, so the knob does not exist in the shipped class.
2. **Minimal viability (constant, small):** a custom `TouchEventReward` × 0.1 — prevents doing nothing, without the stalling exploit described below.
3. **Shaping (annealed to ~0):** velocity-toward-ball, ball-velocity-toward-goal, optional air-time × ball-height. Annealing is driven by **global `cumulative_timesteps`**, not per-process step counts (a documented bug to avoid — `RESEARCH.md` §9.2).

#### ⚠ Two shipped components do not mean what their names say — verified from source

Both are documented in full in `RESEARCH.md` §4.4.1 / §4.4.2. They are the single most likely source of a silently broken Phase 1 run, because every metric still looks plausible.

**(a) `TouchReward` is cumulative, not per-event. Do not use it.**
`reset()` is a no-op and the predicate is `state.cars[agent].ball_touches > 0` — the *cumulative* episode count **[V]**. It therefore returns `+1` on **every step after the first touch**. **[C]** At the commonly copied weight 0.1, against `GoalReward × 10`:

| Episode (agent steps) | Touch-term total | Ratio to one goal |
|---|---|---|
| 600 | +30 to +55 | 3 – 5.5 × |
| 1 500 | +120 to +145 | 12 – 14.5 × |
| 4 500 (`TimeoutCondition(300)` max) | +420 to +445 | 42 – 44.5 × |

The reward would be dominated by "touch once, then stall". **Required:** a ~10-line `TouchEventReward` storing the previous per-agent `ball_touches` in `reset()` and returning `max(0, current − previous)`.

**(b) `NoTouchTimeoutCondition` never fires after the first touch.**
It tests `any(car.ball_touches > 0 for car in state.cars.values())` — again cumulative, not a delta **[V]**. So once any car touches the ball, it returns `False` permanently; its real meaning is *"30 s of no contact from kickoff"*. **Consequence [C]:** the only remaining truncation is `TimeoutCondition(300)` ⇒ a maximum of **4 500 agent steps** per episode, and idle episodes are never cut short. **Required if genuine idle truncation is wanted:** a ~15-line custom `IdleTouchCondition` comparing per-car `ball_touches` deltas.

**Design rules [E]:**
- Prefer **potential differences** `Φ(s_t) − Φ(s_{t−1})` over raw per-step values. That is the form proven not to alter the optimal policy (Ng et al., ICML 1999), and it is how Necto implements its distance/alignment/win-probability/boost terms **[V]**.
- **Read the source of every shipped component before weighting it.** The two failures above are invisible in logs: reward still rises, entropy still falls, and the bot still learns — the wrong thing.
- **Log every reward component separately** (the `Reward/*` naming convention `rlgymppo_rs` uses **[V]**). This is the primary defence against silent reward farming, and it makes the promotion gate's rule 5 (`RESEARCH.md` §10.4) computable.
- Every component declares, in one line, **how it could be exploited**. If you cannot write that line, do not add the term.

### 4.3 Curriculum state setter — required custom work

**[V]** RLGym v2 ships only `KickoffMutator`, `FixedTeamSizeMutator` and `MutatorSequence`. There is **no random state setter**. `CurriculumStateMutator` is therefore mandatory, not optional.

Interface **[V]**: a `StateMutator` implements one method, `apply(state: GameState, shared_info: dict) -> None`, mutating a plain dataclass (`state.ball`, `state.cars[id].physics`, `.boost_amount`, …). `KickoffMutator` is ~30 lines and is the template.

Stage schedule (from `RESEARCH.md` §9.5), with the mixture probability driven by config so a stage change is a config diff, not a code change:

```
stage 0: kickoff                                        p=1.0
stage 1: kickoff 0.3 / random-ground 0.7
stage 2: kickoff 0.5 / random-ground 0.5
stage 3: kickoff 0.4 / random-ground 0.4 / airborne 0.2
stage 4: kickoff 0.3 / random-ground 0.3 / airborne 0.2 / defensive 0.2
stage 5: full-length matches, reduced truncation
```

**The mutator must be versioned and its stage recorded in the run metadata** — it changes the state distribution, so checkpoints from different stages are not directly comparable.

### 4.4 Termination semantics

`GoalCondition` is a **termination** (episode truly ends; no value bootstrap). The custom `IdleTouchCondition` and the shipped `TimeoutCondition` are **truncations** (time limit; the value *is* bootstrapped). `[V]` RLGym exposes these as separate channels and `MultiAgentController`'s default `choose_env_actions` resets only when all agents are terminated or truncated. Getting this wrong silently biases the critic and is not visible in any metric — treat it as a correctness invariant and test it.

**Required unit tests [E]:**
- A goal-scoring step produces `is_terminated=True, is_truncated=False`, and no reward or obs is emitted after it.
- Hitting the absolute timeout with no goal produces `is_terminated=False, is_truncated=True`, and the critic *is* bootstrapped from the final value.
- `IdleTouchCondition` fires after its timeout **following** an earlier touch in the same episode — the exact case the shipped `NoTouchTimeoutCondition` gets wrong (§4.2 (b)).
- `TouchEventReward` emits `+1` exactly once per increment of `ball_touches` and `0` on all other steps.

---

## 5. Learner layer

### 5.1 Networks

**Phase 1 [E]:** `SeparateActorCritic(DiscreteFF(92, 90, (256,256,256)), BasicCritic(92, (256,256,256)))`.

**[C]** 178 522 parameters per network (`RESEARCH.md` §8.4). fp32 weights 0.7 MB, Adam states 1.4 MB — negligible. Activation memory at a 50 k minibatch ≈ 154 MB forward / 307 MB retained. Fits 8 GB with enormous headroom, which is exactly why it is the right *starting* size: it isolates pipeline correctness from VRAM pressure.

**Scaling rule [E]:** increase width only when gate **B4** shows `batch_consumption_time` is still a small fraction of `iteration_time` at the current size *and* the learning curve has visibly flattened. Target sequence: `(256,256,256)` → `(512,512,512)` → `(1024,1024,512,512)` → `(2048,2048,1024,1024)`, with `n_minibatches` raised at each step to hold VRAM under control. At `(2048,2048,1024,1024)` and a 50 k minibatch, activations alone are 1.2–2.5 GB per network **[C]** — `n_minibatches ≥ 2` becomes mandatory.

**Changing the network size or the obs width invalidates every checkpoint.** Both are one-way doors; decide them at a phase boundary, never mid-run.

### 5.2 PPO update

**[V]** `rlgym-learn-algos` implements clipped-surrogate PPO with GAE. Specifics confirmed from source:

- Critic loss = plain `nn.MSELoss()` — **no value clipping**. Monitor `critic_loss`; expect the critic to move freely.
- `advantage_standardization = True` (default), separate from `standardize_rewards = True`.
- `standardize_rewards` divides by the standard deviation of *unstandardized, unclipped* returns computed across all trajectories being added (0.4.x semantics) **[V]**.
- `reward_clip = 10` is a separate optional field.
- `max_grad_norm = 0.5`.
- Separate `Adam` optimizers for actor and critic, configured through `optimizer_named_parameter_group_kwargs`.
- `n_epochs` passes over each batch, split into `n_minibatches`.
- `torch.backends.cudnn.benchmark = True` when `device` is CUDA.
- `DiscreteFF` ends in `Softmax` with `clamp(probs, 1e-11, 1)`.

**Recommended values** are tabulated in `RESEARCH.md` §9.1. The one that must not be forgotten: **`PPOLearnerConfigModel.device` defaults to `"cpu"`** — pass `device="cuda:0"` explicitly **[V]**.

### 5.3 Experience buffer and advantage estimation

**[V]** `NumpyExperienceBuffer` holds `observations` (numpy circular buffer), `actions` (numpy), `log_probs` / `values` / `advantages` (tensor circular buffers). It does **not** store `next_states` — that is why it is cheap. `device="cpu"` in all upstream examples, so the buffer lives in host RAM and minibatches are moved to the GPU per update.

**[C]** Memory per step = `n_obs·sizeof(dtype) + 8 + 4 + 4 + 4`. For float32 / 92-dim obs: **388 B/step** → 150 k steps = 58 MB, 1 M steps = 388 MB. Two controllers ⇒ ×2.

**[E] Sizing rule:** `max_size = 2–3 × timesteps_per_iteration`, per the RLGym tutorial's guidance **[R]** and consistent with PPO's on-policy nature (a buffer much larger than a few iterations holds badly off-policy data).

`GAETrajectoryProcessor` runs in **Rust** **[V]**, so advantage computation is not a Python bottleneck. `gamma=0.99, lmbda=0.95`.

### 5.4 Checkpointing

**[V]** Layout: `agent_controller_save_folder/<run_name><run_suffix>/<time_ns>/` containing `ppo_agent.json`, `ppo_learner/{actor_critic.pt, optimizers.pt, misc.json}`, `experience_buffer/` (zip), `metrics_logger/`, optionally `iteration_trajectories.pkl` and `iteration_shared_infos.pkl`. Resumption via `checkpoint_load_folder`.

**Two-tier retention [E]** (the pattern BotBoi uses **[R]**, and the only way to build an opponent pool):

| Tier | Path | Cadence | Retention | Purpose |
|---|---|---|---|---|
| **Resumable checkpoint** | `agent_controller_checkpoints/<run>/<ts>/` | every 1 M steps | last 5 (`n_checkpoints_to_keep`) | Crash recovery, exact resume |
| **Policy snapshot** | `snapshots/<run>/<steps>.pt` | every 25–50 M steps | **forever** | Opponent pool, evaluation, regression tests |

⚠️ **Never store anything else in `agent_controller_save_folder`** — the trainer *"MAY DELETE ANYTHING INSIDE THIS FOLDER"* **[V]**.

`run_suffix` should encode the experiment id, e.g. `"-exp012-1v1-goalonly-seed3"`, so checkpoint paths are self-describing.

### 5.5 Experiment tracking

**[V]** `WandbMetricsLogger(PPOMetricsLogger(), ppo_additional_derived_config_factory)` with `WandbMetricsLoggerConfigModel(inner_metrics_logger_config=None, group=...)`. `WANDB_MODE=offline` works for local-only logging.

**Required panels [E]:**
- *Throughput:* `overall steps/s`, `timestep_collection_time`, `batch_consumption_time`, `iteration_time`, `timesteps_collected`
- *Learning:* `actor_entropy`, `kl_divergence`, `sb3_clip_fraction`, `critic_loss`, `actor_update_magnitude`, `critic_update_magnitude`, `cumulative_model_updates`
- *Returns:* `average_reward`, `average_undiscounted_episodic_return`, `average_return`, `return_standard_deviation`
- *Reward components:* one series per term (farming detector)
- *Evaluation:* win rate + Wilson CI vs. each frozen opponent, goal differential
- *Resource:* `torch.cuda.max_memory_allocated()`, host RSS

All of the first three groups are emitted by the library already **[V]**; only reward-component and evaluation series need adding.

---

## 6. Self-play and opponent pools

### 6.1 The mechanism

**[V]** `MultiAgentController(subcontrollers: Mapping[str, MultiAgentSubcontroller])` with two abstract methods:

- `choose_subcontrollers(agent_ids: dict[env_id, list[AgentID]]) -> dict[env_id, list[str]]` — parallel list of subcontroller names.
- `choose_env_actions(env_state_info_dict) -> (n_new_envs, {env_id: EnvActionResponse})` — `STEP` / `RESET` / `DEFER` / `CLOSE`.

Upstream's own test drives blue with `"PPO1"` and orange with `"PPO2"` **[V]**. Subcontrollers need not learn, so a frozen checkpoint is a drop-in opponent.

### 6.2 Staged rollout

| Stage | Subcontrollers | Assignment | Purpose |
|---|---|---|---|
| **SP1** | `PPOAgentController` only | implicit — one controller drives every agent | Pure self-play. **The Phase 1 architecture.** No `MultiAgentController` needed. |
| **SP2** | `"current"` (learns) + `"hist_k"` (frozen, inference only) | per-env random draw from a pool | Prevents cycling, stabilises the value target |
| **SP3** | + `"baseline"` (scripted / random) | small fraction of envs | Anchors absolute skill, detects collapse |
| **SP4** | two learning controllers, blue vs. orange | team-based | Asymmetric league (upstream's tested configuration) |

**Opponent pool policy [E]:**
- Maintain **≥ 5 frozen snapshots** spanning the recent skill range, refreshed every ~100 M timesteps.
- Sample uniformly from the pool, **not** "last K checkpoints" — a pool drawn from a collapsing policy collapses with it.
- Optionally weight by skill proximity (matchmaking) once an Elo/TrueSkill estimate exists, so the current policy mostly faces opponents it can neither trivially beat nor never beat. This is the AlphaStar-league idea in its cheapest useful form.
- Frozen subcontrollers load **actor weights only**, run **greedy or fixed-temperature**, and hold **no optimizer and no experience buffer** — that keeps their memory cost to a few MB each.

### 6.3 Why a pool rather than pure self-play

Pure self-play (SP1) is the right *start* because it is the simplest thing that can work and it is what produced the reference results. But the literature is consistent that single-opponent self-play cycles: AlphaStar's league (**[V]** *Nature* 2019), open-ended learning (arXiv:2109.13286), and fictitious/prior-responding self-play (arXiv:1603.01121; PSRO, arXiv:1711.00832) all argue for averaging over historical opponents. The architecture makes that a `choose_subcontrollers` change rather than a rewrite — which is the whole point of choosing this stack.

---

## 7. Evaluation layer

### 7.1 Separation

The evaluator is a **separate process**, run while training is paused or on a held-out core budget. Rationale: it competes for the same CPU and GPU, so running it concurrently corrupts both the throughput measurements and the evaluation timings.

### 7.2 Contract

```
evaluate(candidate, opponent, n_games, seed_base) -> {
    wins, losses, draws, goals_for, goals_against,
    wilson_lower, wilson_upper, goal_diff_ci,
    per_episode: [side, gf, ga, end_reason, length_steps]
}
```

- **Greedy** actions (`beta = 1`) — evaluation measures skill, not exploration **[V]** precedent in Nexto's `bot.py`.
- **Fixed seeds** so candidate and incumbent face the same situations.
- **Side-balanced**: n/2 as blue, n/2 as orange.
- **Same episode rules** as training.
- Runs on **RocketSim**, not the game. 1000+ games in RLBot at 120 tps is hours; in RocketSim it is minutes.

### 7.3 Promotion gate

Reproduced from `RESEARCH.md` §10.4 — all six conditions must hold:

1. ≥ 1000 games vs. the incumbent.
2. Wilson 95 % **lower bound** on win rate > 0.50.
3. Goal differential > 0 with a 95 % CI excluding 0.
4. No regression beyond CI width against **any** frozen baseline.
5. No single shaping term contributing more than a defined share (~30 %) of mean episodic return.
6. ≥ 3 matches visually inspected in rlviser.

**[C]** Calibration: at n = 1000, a 53 % win rate has a Wilson lower bound of ~0.499 and **fails**; ~53.1 % is the threshold. At n = 400 the threshold is 55 %. Failures retain the candidate in the historical pool without promoting it.

---

## 8. Model lifecycle

```
                    ┌──────────────┐
        train ────► │  CANDIDATE   │  checkpoint every 1M steps
                    └──────┬───────┘
                           │ snapshot every 25–50M steps
                           ▼
                    ┌──────────────┐   fails gate   ┌──────────────────┐
      evaluate ───► │  EVALUATED   ├───────────────►│  POOL (frozen)   │
                    └──────┬───────┘                └────────┬─────────┘
                           │ passes all 6 conditions         │
                           ▼                                 │ sampled as
                    ┌──────────────┐                         │ opponents
                    │  INCUMBENT   │◄────────────────────────┘
                    └──────┬───────┘
                           │ export (torch.jit.trace actor) + equivalence test
                           ▼
                    ┌──────────────┐   parity test passes   ┌────────────────┐
                    │  EXPORTED    ├───────────────────────►│  DEPLOYED      │
                    └──────────────┘                        │  (Windows/RLBot)│
                                                            └────────────────┘
```

| State | Stored as | Can be | Cannot be |
|---|---|---|---|
| CANDIDATE | resumable checkpoint | resumed, snapshotted | promoted without passing the gate |
| POOL | actor-only snapshot, `snapshots/<run>/<steps>.pt` | used as an opponent, evaluated | used as the incumbent without re-evaluation |
| INCUMBENT | snapshot + evaluation record | exported, used as the regression baseline | silently replaced |
| EXPORTED | traced artifact + equivalence report | deployed, benchmarked for latency | trained further without re-export |
| DEPLOYED | artifact on the Windows side | played against, measured for the sim/game gap | trained against |

Every transition writes a record: run id, git commit, `config.json` hash, dependency lock hash, seed, and the evaluation statistics. That record *is* the reproducibility artifact.

---

## 9. Deployment boundary

### 9.1 The hand-off

```
WSL 2                                     Windows
──────────────────────────────────────    ─────────────────────────────────────
agent_controller_checkpoints/.../
  ppo_learner/actor_critic.pt
        │
        ▼
  exporter.py
   ├─ load actor (DiscreteFF)
   ├─ torch.jit.trace(actor, example_obs)
   ├─ assert max|traced(obs) - actor(obs)| < 1e-5   ← equivalence test
   └─ write policy.pt + obs_builder.py + manifest.json
        │
        └────── file copy to the Windows filesystem ──────►  bot/policy.pt
                                                             bot/obs_builder.py
                                                             bot/manifest.json
```

`manifest.json` records: obs width, obs normalization coefficients, action space (90), tick skip (8), `rlbot_delay` setting, network layer sizes, training config hash, source checkpoint path. **The bot refuses to start if the manifest disagrees with the obs builder it loads** — that check costs nothing and prevents the most likely silent failure.

### 9.2 The two environments

| | WSL 2 (training) | Windows (deployment) |
|---|---|---|
| Python | 3.12 venv | 3.12 venv (separate) |
| Deps | `rlgym[rl-sim]`, `rlgym-learn`, `rlgym-learn-algos`, `torch` (CUDA), `numpy`, `wandb` | **`rlbot`, `torch` (CPU), `numpy`, `rlgym-compat`** — nothing else |
| GPU | yes | no |
| RocketSim | yes | **no** |
| Rocket League | no | yes |
| RLBot | no | yes |

**[V]** Precedent: `doublesize-nexto-v5/src/requirements.txt` is `torch==2.0.1+cpu` + `numpy`, and RLBot's Python example requires Python 3.12+. Keeping the Windows side CUDA-free and RocketSim-free is what makes it stable.

### 9.3 Latency contract

**[V]** `get_output(GamePacket) -> ControllerState` is called at **120 Hz**; the bot must respond within roughly **8.3 ms** (with the ~6 ms figure from RLBot's own docs as the practical guide); missed ticks reuse the previous controller state; packets arrive ~1 ms old.

**[C]** With tick skip 8 the policy re-infers at 15 Hz, so the *network* gets 66.7 ms of budget. The real risk is the **obs builder**, which must run inside 8.3 ms on the ticks where it is invoked and produce values matching training.

**[E]** Structure copied from the working Nexto RLBot v5 bot **[V]**: cache `self.action`, recompute only when `self.ticks >= self.tick_skip`, and return the cached `ControllerState` otherwise. Add an explicit timing assertion in the integration test (gate B6).

### 9.4 Integration tests

| Test | Asserts | Gate |
|---|---|---|
| **Observation parity** | Training obs (RocketSim) vs. deployment obs (GamePacket → `rlgym_compat`) agree elementwise within tolerance on matched states | B6 |
| **Boost pad ordering** | Pad index order matches `common_values.BOOST_LOCATIONS` (RocketSim's engine carries a `# TODO check if the order is correct` comment **[V]**) | B6 |
| **Action mapping** | The 8-element control vector maps identically to `ControllerState` | B6 |
| **Numerical equivalence** | Traced artifact == training actor on fixed observations (max abs logit delta < 1e-5) | M6 |
| **Manifest guard** | The bot refuses to start when `manifest.json` disagrees with the loaded obs builder | M6 |
| **Latency** | p99 `get_output` wall time well under 8.3 ms, and 100 % bot response rate in RLBot's performance metric | B6 (step 4) → M7 |
| **Behavioural smoke** | Bot moves, touches the ball, and does not immediately own-goal in a 5-minute match | M7 |

*Gate/milestone IDs are defined in [`ROADMAP.md`](ROADMAP.md) §2 and §4. B6 covers parity, angular-velocity checks, action mapping and latency instrumentation; B7 is the Nexto-vs-ballchaser harness calibration that must pass before any external comparison is published.*

---

## 10. Key technical decisions and alternatives considered

### D1 — Learner: `rlgym-learn` + `rlgym-learn-algos`, not `rlgym-ppo`

**Decision:** adopt `rlgym-learn` 2.0.0 / `rlgym-learn-algos` 0.4.2.
**Why:** it is the only maintained RLGym v2 learner with a self-play primitive (`MultiAgentController`); it has no numpy-2 conflict; it needs no abandoned `gym` dependency; its config is a serializable pydantic artifact.
**Alternatives:** *rlgym-ppo* — frozen since 2025‑01, single policy, `numpy<2.0` vs `rlviser-py`'s `numpy>=2.0.0`, depends on `gym` 0.26.2 (unmaintained). *rocket-learn* — RLGym v1, no license declared, requires Redis. *rlgymppo_rs* — technically strongest, but **GPL‑2.0**, 5 stars, Rust-only extension points, and it targets RocketSim v3 with no PyPI wheel. *SB3* — no multi-agent-per-env policy assignment, needs a flattening wrapper.
**Cost accepted:** rlgym-learn has **no documentation** — the RLGym Learn page says *"not finished yet"* **[V]**. Plan for source-reading; `quick_start_guide.py`, `speed_test.py` and `multi_agent_controller_test.py` are the de facto docs.
**Escape hatch:** if gate B3 proves the Python per-step path is the hard ceiling, `rlgymppo_rs` is the documented path to 3–10× throughput, at the price of a license decision and a Rust rewrite of custom components.

### D2 — Start 1v1, not 2v2 or 3v3

**Decision:** 1v1 soccar, Octane, both cars driven by the current policy.
**Why:** obs is 92 floats instead of 132/172 **[C]**; no teammate-credit-assignment problem; episodes are short; and ZealanL's guidance is explicit — *"I recommend starting with the default of 1, as making bots that can play with teammates has added challenges"* **[R]**.
**Cost accepted:** a 1v1 policy does not automatically generalise to team modes. Revisit at SP4 / C-mix once 1v1 passes its gate.

### D3 — Discrete 90 actions at tick skip 8

**Decision:** `RepeatAction(LookupTableAction(), repeats=8)`.
**Why:** `[V]` 1944 → 90 with explicit pruning; identical to Necto's table, so external baselines are comparable; replay-based learning evidence says fixed action sets are necessary **[V]**; `MultiDiscreteFF` was removed from the library.
**Cost accepted:** no partial throttle/steer, which caps fine car control. Accepted deliberately for Phase 1.

### D4 — Single `PPOAgentController` before `MultiAgentController`

**Decision:** Phase 1 uses one controller driving every agent; `MultiAgentController` is introduced at SP2.
**Why:** the smallest thing that can work, and pure self-play is what the reference results were produced with. Adding a pool before the policy can score adds variance without adding signal.
**Alternative rejected:** starting with a two-controller asymmetric league (upstream's tested config) — more moving parts, two optimizers, two buffers, twice the memory, before the basics are verified.

### D5 — Frozen inference-only subcontrollers for opponents

**Decision:** historical checkpoints become inference-only `MultiAgentSubcontroller`s inside the training rollout, not external processes.
**Why:** no extra processes, no IPC, and opponents share the current obs/reward path so there is no drift. Actor-only loading keeps them a few MB each.
**Alternative rejected:** a separate evaluation daemon feeding matches back into training — much more machinery, and it breaks the clean training/evaluation separation.

### D6 — Evaluation is out-of-process and never concurrent with throughput measurement

**Decision:** evaluator runs separately; training is paused (or its numbers discarded) during evaluation.
**Why:** both contend for the same CPU and GPU. Concurrent execution invalidates the throughput measurement *and* adds noise to evaluation timings.

### D7 — Custom `CurriculumStateMutator` from the start, but starting at "kickoff only"

**Decision:** write the mutator in Phase 1, but begin with `stage=0` (kickoff only) so behaviour matches the upstream default.
**Why:** RLGym v2 has no random state setter **[V]**, and every curriculum stage past kickoff needs one. Writing it early and leaving it inert costs little; discovering you need it at stage C1 costs a rewrite mid-run.

### D8 — Two-tier checkpoint retention

**Decision:** short-retention resumable checkpoints + forever-retained policy snapshots in a **separate directory**.
**Why:** opponent pools and regression tests need old policies; the trainer may delete anything inside `agent_controller_save_folder` **[V]**. Mixing the two guarantees eventual data loss.

### D9 — Deployment artifact is a traced CPU-only actor plus a manifest

**Decision:** `torch.jit.trace` the actor, ship it with a manifest and a copied obs builder, on a CUDA-free Windows venv.
**Why:** `[V]` Nexto's own RLBot v5 requirements are CPU-only torch; the latency budget is generous for an MLP; and a manifest lets the bot refuse to start on a mismatch instead of playing badly and confusingly.
**Alternative considered:** ONNX Runtime. Viable, slightly smaller dependency, but adds a converter and a second numerical-equivalence surface. Revisit only if PyTorch CPU startup proves too slow (unlikely at this model size).

### D10 — Config-as-code that generates a serializable artifact

**Decision:** express configuration as pydantic models in Python, and always call `generate_config(..., force_overwrite=True)` so `config.json` is written per run.
**Why:** `[V]` every config model uses `extra="forbid"`, so typos are hard errors rather than silent defaults; and the JSON file is the run's ground truth. Commit the generated JSON alongside the run record.

### D11 — Two custom components replace two shipped ones (`TouchEventReward`, `IdleTouchCondition`)

**Decision:** do not use RLGym v2's `TouchReward` or `NoTouchTimeoutCondition`. Write delta-based replacements (~10 and ~15 lines).
**Why:** both shipped classes read the **cumulative** `ball_touches` counter with no previous-state tracking **[V]**. `TouchReward` therefore pays out on every step after the first touch — 3 to 44 times the value of a goal at the conventional 0.1 weight **[C]** — and `NoTouchTimeoutCondition` never fires again after the first touch, so idle episodes run to the 4 500-step absolute timeout instead of being cut at 30 s.
**Alternative rejected:** keep the shipped classes and lower the touch weight to ~0.002. That restores the goal/touch balance on paper but leaves a *dense per-step* incentive that is not a touch event at all, and still loses idle truncation entirely.
**Cost accepted:** two more custom classes to write and unit-test in Phase 1. Both are covered by the tests listed in §4.4.
**Generalisation:** any shipped component that reads a cumulative counter is suspect. Read the source before weighting it — this is now a standing rule (§4.2).

---

## 11. Data flow — one training iteration

1. **Learner** asks the pool for `timesteps_per_iteration` steps.
2. Each **env process** steps its RocketSim arena tick by tick (8 ticks per agent step), builds obs via `Float32DefaultObs`, computes rewards, evaluates termination/truncation, writes to shared memory and signals over UDP.
3. **AgentController.choose_subcontrollers** decides which policy produces each agent's action; **choose_env_actions** decides `STEP` / `RESET`.
4. The **learner** batches ≥ `min_frac_process_responses_per_collection` of pending observations, runs one GPU forward pass, samples actions, returns them.
5. Completed trajectories go to **`GAETrajectoryProcessor`** (Rust): reward standardisation, `reward_clip`, GAE(γ, λ), advantage computation.
6. Timesteps land in **`NumpyExperienceBuffer`** (host RAM); the oldest are evicted past `max_size`.
7. **`PPOLearner`** draws `n_epochs × n_minibatches` shuffled minibatches, moves each to `cuda:0`, computes the clipped surrogate loss + entropy bonus and the MSE critic loss, steps `Adam(actor)` and `Adam(critic)` with `max_grad_norm`.
8. **`MetricsLogger`** emits `PPOData` + `GAETrajectoryProcessorData` + timing to console and W&B.
9. Every `save_every_ts`, a **checkpoint** is written; every snapshot interval, a **policy snapshot** is written to the permanent directory.
10. Loop until `timestep_limit`.

**Where time goes [E]:** step 2 dominates at small networks (CPU, N processes); step 7 dominates at large networks (GPU). The library already logs the split as `timestep_collection_time` vs. `batch_consumption_time` **[V]**, so the bottleneck is observable rather than guessed.

---

## 12. Smallest viable architecture, and how it grows

### 12.1 M0 — the minimum that can train (Phase 1)

```
build_rlgym_v2_env()  →  RLGym(FixedTeamSizeMutator(1,1) + KickoffMutator,
                               Float32DefaultObs(zero_padding=1),
                               RepeatAction(LookupTableAction(), 8),
                               CombinedReward(GoalReward×10, TouchEventReward×0.1),
                               GoalCondition,
                               AnyCondition(IdleTouchCondition(30), TimeoutCondition(300)),
                               RocketSimEngine(rlbot_delay=True))

PPOAgentController(DiscreteFF(92,90,(256,256,256)), BasicCritic(92,(256,256,256)),
                   NumpyExperienceBuffer(GAETrajectoryProcessor()),
                   PPOMetricsLogger())

LearningCoordinator(env_create_function, agent_controller, config)  # n_proc=8, device="cuda:0"
```

Plus, outside the library: a snapshot-copier, a `random`-policy evaluation script, and a `config.json` committer. **That is the whole of Phase 1.** Everything else in this document is an addition, not a prerequisite.

### 12.2 Growth path — each step is additive

| Step | What is added | What it unlocks | Trigger |
|---|---|---|---|
| +1 | `Float32DefaultObs` + `OPENBLAS_NUM_THREADS=1` | ~2× less obs memory/IPC | immediately (it is nearly free) |
| +2 | Snapshot copier + out-of-process evaluator | Measurable progress instead of vibes | immediately |
| +3 | `CurriculumStateMutator` stages 1–2 | Learning from non-kickoff states | once the policy touches the ball reliably |
| +4 | Reward-component logging + annealing wrapper | Farming detection, clean late-game objective | before the first long run |
| +5 | `MultiAgentController` + frozen historical subcontrollers (SP2) | Anti-cycling, stable value target | once a 25 M-step snapshot exists |
| +6 | Scripted/random baseline subcontrollers (SP3) | Absolute skill anchor | with SP2 |
| +7 | Larger network, gated by B4 | Higher ceiling | when the learning curve flattens *and* B4 allows |
| +8 | Two-controller asymmetric league (SP4) | Teamplay groundwork, league dynamics | after 1v1 passes its gate |
| +9 | Elo/TrueSkill tracker over the pool | Principled matchmaking and skill reporting | with SP2 |
| +10 | Exporter + parity test + RLBot bot (Windows) | Real-game evaluation | after a promotable incumbent exists |
| +11 | (Optional) replay pretraining, distillation, `rlgymppo_rs` | Sample efficiency / throughput step-change | only on demonstrated need |

**The rule [E]:** never advance two steps at once, and never advance a step without the previous step's gate passing. The failure mode this prevents is a large, unattributable regression — with six changes in flight, a stalled learning curve tells you nothing.

---

## 13. Failure isolation and observability

| Failure | Detection | Containment |
|---|---|---|
| Env process crash | `EnvCloseReason.EXCEPTION` surfaced via `handle_env_closes` **[V]** | Coordinator reports it; the run continues with fewer envs. Alert if the count drops. |
| Hung env process | Missing responses ⇒ collector stalls | `min_frac_process_responses_per_collection` tolerates some stragglers; add a watchdog that logs per-process response rates |
| GPU OOM | `torch.cuda.OutOfMemoryError` | Raise `n_minibatches`; `max_grad_norm` and buffer sizes are unaffected |
| Host OOM | Process killed by the kernel | Lower `n_proc` or `max_size`; monitor RSS per process |
| Silent CPU fallback | `device` left at default | Assert `torch.cuda.is_available()` at startup and log the resolved device |
| Reward farming | One component dominating mean return | Promotion gate rule 5; periodic rlviser inspection |
| Obs drift between sim and game | Policy plays badly only in the real game | Parity test (B6) is a CI gate, not a manual check |
| Checkpoint corruption | Load failure | Two-tier retention means an older snapshot always exists |
| Throughput regression | `overall steps/s` drops between runs | Record it per run; treat a >15 % drop as a regression to investigate, since dependency drift is the usual cause |

---

## 14. What this architecture deliberately does not solve

- **Reaching Nexto's documented level.** `RESEARCH.md` §12 shows the compute gap. The architecture maximises progress per unit compute; it cannot manufacture compute.
- **Sim-to-game fidelity in general.** It provides a *measurement* (the parity test and the sim-vs-game win-rate gap), not a fix.
- **Fine continuous car control.** The discrete 90-action space caps it (D3).
- **Team coordination.** Deferred to SP4 / mixed-mode training.
- **Any form of online play.** Out of scope by policy — offline/RLBot modes only.
