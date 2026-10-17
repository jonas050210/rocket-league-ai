# Rocket League AI — Implementation Roadmap

**Companion to** [`RESEARCH.md`](RESEARCH.md) (evidence, citations, computed values) and [`ARCHITECTURE.md`](ARCHITECTURE.md) (design, decisions D1–D11).
**Status:** planning document. Nothing here is implemented yet.

**How to read this.** Every phase has an *objective*, *entry criteria*, *work items*, *acceptance criteria* (all must pass before the next phase starts) and *explicit non-goals*. Benchmark gates **B1–B7** are numbered experiments whose only purpose is to replace an estimate with a measurement; they are defined in §2 and referenced from the phases and from `RESEARCH.md` §14 (U1–U12).

**The single governing rule:** never advance two phases at once, and never advance a phase whose gate has not passed. With several changes in flight a stalled learning curve carries no information — that is the failure mode this document exists to prevent.

---

## 1. Phase map

| Phase | Name | Produces | Gate(s) | Milestone |
|---|---|---|---|---|
| **0** | Environment bring-up & platform verification | Working WSL 2 + Windows setup, dependency lock | **B1**, **B2a** | **M0** |
| **1** | Minimum viable training pipeline | First learning run in 1v1 | **B3** | **M1** ← *first implementation milestone* |
| **2** | Throughput characterization & optimization | A measured, tuned steps/s number | **B2b/c**, **B4**, **B5** | **M2** |
| **3** | Reward, curriculum & state randomization | Non-kickoff competence | — | **M3** |
| **4** | Evaluation harness, baselines & opponent pool | Promotion gate, frozen pool | — | **M4** |
| **5** | Scale-up: network, league, skill rating | Strong 1v1 policy | **B4** re-run | **M5** |
| **6** | Deployment to the original game | Working RLBot v5 bot | **B6**, **B7** | **M6**, **M7** |
| **7** | Competitive validation & reporting | Published, defensible strength claim | **B6** re-run | **M8** |

**[E] Effort ordering.** Phases 0–2 are engineering and take days. Phases 3–5 are dominated by *wall-clock training time*, which is the unknown that gate B3 resolves. Phases 6–7 are a few days of engineering plus real-game session time.

---

## 2. Benchmark gates — procedures and decision rules

Each gate is a **script with a fixed protocol and a written output**, committed to the repo alongside a `benchmarks/RESULTS.md` row. No gate may be skipped on the grounds that "it probably works".

> **No throughput figure in this document or in `RESEARCH.md` is a prediction.** Everything numeric below is either a computed quantity **[C]**, a vendor-reported quantity **[R]**, or explicitly labelled as the *output* of a gate.

### B1 — Platform verification: GPU and rendering under WSL 2

**Resolves:** U10. **Blocks:** Phase 1 (a CUDA-less run silently trains on CPU at ~50× less throughput — the `device` field defaults to `"cpu"` **[V]**).

**Protocol**
1. On the **Windows host**: confirm an NVIDIA driver is installed; record `nvidia-smi` driver version and its CUDA capability. **Do not** install a Linux driver inside WSL and **do not** install the `cuda` / `cuda-drivers` metapackages **[V]** NVIDIA WSL guide.
2. In **WSL 2**: `ls /dev/dri` (present ⇒ WSLg GPU device exists), `ls /usr/lib/wsl/lib/` (libcuda present), `nvidia-smi` from inside WSL.
3. In the project venv: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`, `torch.cuda.get_device_capability(0)`, `torch.version.cuda`.
4. Allocate a 2 GB tensor on `cuda:0`, run a matmul, report achieved TFLOP/s; compare against the 4060 Ti's published fp32 figure as a sanity bound, not a target.
5. Render a 1v1 rollout with `rlviser` (best effort). Record: works / black window / crash / needs native Windows.
6. Record `.wslconfig` (`memory`, `processors`, `swap`) and `free -h` inside WSL — WSL 2 does not see the full 32 GB by default.

**Accept / record**
- **B1 PASS** requires: `torch.cuda.is_available() == True`, device name contains `4060 Ti`, capability `(8, 9)`, matmul completes.
- rlviser is recorded as **PASS / DEGRADED / FAIL** — it is a convenience, never a gate. If FAIL, visual inspection moves to native Windows via the Phase 6 export path (gate B6) and the promotion gate's rule 6 must be satisfied there.
- Write the measured `processors`/`memory` WSL exposes into `benchmarks/RESULTS.md`; it bounds `n_proc` and buffer sizing for B2/B3.

### B2 — RocketSim and env-process characterization

Three sub-experiments, all cheap (minutes each), all prerequisite to any throughput claim.

**B2a — Pure physics ceiling (Phase 0).** Resolves: nothing directly; establishes the upper bound.
- Drive `rsim.Arena` directly (no RLGym, no Python obs builder): 1v1 and 3v3, 1 thread and N threads.
- Report **ticks/s** per thread and aggregate.
- Context value: the vendor README reports **114 481 tps** single-threaded 2v2 on an **i5‑11400** at v2.1.0 **[R]**. Divide by 8 to convert to agent steps **[C]**. Do not treat the vendor number as achievable on this machine.

**B2b — Per-process footprint and reset cost (Phase 2).** Resolves: U2, U7.
- Launch the real RLGym env-process pool at `n_proc ∈ {4, 8, 12, 16, 20}` with a trivial obs builder.
- Report **RSS per process** (from `/proc/<pid>/status` VmRSS), total host RSS, and **resets/s** separately from **steps/s** — RLGym v2's `set_state()` tears down and re-adds every car on each reset **[V]**, so short episodes may be reset-bound rather than step-bound.
- Output: the largest `n_proc` that keeps total RSS under 60 % of WSL-exposed memory.

**B2c — P-core affinity on the hybrid CPU (Phase 2).** Resolves: U3.
- i7‑12700F = 8 P-cores (16 threads) + 4 E-cores **[V]**. E-core max turbo 3.6 GHz vs P-core 4.9 GHz ⇒ a process on an E-core is ~26 % slower **[C]**, and with `min_frac_process_responses_per_collection = 0.45` **[V]** the collector waits on stragglers.
- Compare three placements at the best `n_proc` from B2b: (i) no pinning, (ii) `taskset` env processes onto P-core siblings with the learner on E-cores, (iii) `numactl --cpunodebind` if multiple NUMA nodes are exposed.
- **Decision rule:** adopt pinning only if it improves end-to-end steps/s by > 5 %; otherwise record it as tested-and-rejected and keep the simpler configuration.

**B2 gate output:** a single table in `benchmarks/RESULTS.md` — `n_proc`, placement, steps/s, resets/s, RSS/proc, total RSS — with the chosen configuration justified by a row of that table.

### B3 — End-to-end training throughput (the number that matters)

**Resolves:** U1, U6, U9. **Blocks:** any time/scale commitment in Phase 5 or beyond.

**Protocol**
1. Full Phase 1 configuration: 1v1, obs 92 (float32), `(256,256,256)` actor and critic, `timesteps_per_iteration = 50 000`, `device="cuda:0"`.
2. Sweep `n_proc ∈ {8, 12, 16, 18, 20}` × `min_frac_process_responses_per_collection ∈ {0.45, 0.7, 0.9}`, 3 repetitions of 500 k steps each.
3. Report, per configuration: **`overall steps/s`** (collection *and* learning — the honest number), `timestep_collection_time`, `batch_consumption_time`, `iteration_time`, GPU peak memory, host RSS, and the collection/consumption ratio.
4. A/B one variant with the stock float64 `DefaultObs` to quantify the float32 win (U6).

**Decision rule**
- Adopt the configuration with the highest `overall steps/s` that leaves ≥ 20 % headroom on both host RSS and GPU memory (verified over a full iteration in B5).
- **If `timestep_collection_time` > 70 % of `iteration_time`** at every configuration, the run is CPU-bound: adding GPU work (larger network, more `n_epochs`) is nearly free, and the documented escape hatch (`rlgymppo_rs`, 3–10× throughput, GPL‑2.0 + Rust — `RESEARCH.md` §5.4) becomes a real decision for Phase 5. Record the measurement either way; do not act on it before M4.

**Calibration warning:** the RLGym-PPO guide's `Collected: 8 581 steps/s / Overall: 5 977 steps/s` **[R]** is on an unspecified machine with an unspecified `n_proc` and a different learner. It is **not** a target and must not appear in any progress estimate.

### B4 — Largest network × minibatch on 8 GB VRAM

**Resolves:** U4, U5. **Blocks:** Phase 5 network scaling.

**Protocol**
1. Fix `timesteps_per_iteration = 50 000`, `batch_size = 50 000`.
2. Grid over networks `(256,256,256)`, `(512,512,512)`, `(1024,1024,512,512)`, `(2048,2048,1024,1024)` × `n_minibatches ∈ {1, 2, 4, 8}`.
3. For each cell: run 3 PPO iterations, record `torch.cuda.max_memory_allocated()`, `batch_consumption_time`, `iteration_time`, and whether it OOMs.
4. Separately, measure raw host→device transfer of a 50 000 × 92 float32 minibatch (pinned vs pageable) to test whether WSL 2's documented pinned/managed-memory limitation **[V]** is material here (U5).

**Context [C]:** activation memory alone for `(2048,2048,1024,1024)` at a 50 k minibatch is ≈ 1.2–2.5 GB per network (`RESEARCH.md` §8.4); with actor + critic, CUDA context (~300–500 MB) and allocator fragmentation, `n_minibatches ≥ 2` becomes mandatory. `(256,256,256)` uses ~154 MB forward / ~307 MB retained.

**Decision rule:** the production network is the largest cell with `batch_consumption_time ≤ 20 % of iteration_time` and peak VRAM ≤ 6.5 GB. Everything larger is documented as rejected-with-numbers.

### B5 — Steady-state resource headroom

**Resolves:** memory-creep risk over multi-day runs.
- Run the chosen configuration for ≥ 4 hours (or ≥ 20 M steps).
- Plot host RSS, GPU allocated/reserved, and `overall steps/s` against wall time.
- **Accept** if RSS growth is flat after the first hour and steps/s variance across consecutive 10-minute windows is < 15 %. Otherwise find the leak before committing to a long run — a run that dies at day 3 costs a day 3.

### B6 — Sim ↔ game observation parity and transfer gap

**Resolves:** U8. **Blocks:** Phase 7 (a strength claim without a parity test is not a strength claim).

**Protocol**
1. **Observation parity:** capture matched states from RocketSim (training obs) and from a real match via `rlgym_compat` (GamePacket → v2-shaped state). Compare elementwise: ball state, per-car state, the 34 boost pads in `common_values.BOOST_LOCATIONS` order (RocketSim's engine carries a `# TODO check if the order is correct` comment **[V]**), the 9 partial flags. Report max and mean absolute error per field.
2. **Angular-velocity check:** the peer-reviewed transfer study found the effective max angular velocity needed empirical correction (7.3 vs 5.5 rad/s) and that ablating *any single* physics adaptation was catastrophic **[V]** arXiv 2205.05061. Log the observed angular-velocity distribution in both environments.
3. **Action mapping:** assert the 8-element control vector maps identically to `ControllerState`.
4. **Latency:** instrument the bot's `get_output` wall time; report p50/p99 against the ~8.3 ms 120 Hz budget, and RLBot's own bot-response-rate metric.
5. **Transfer gap:** run the same exported policy in RocketSim and in the game against the same opponent class; report **sim win rate**, **game win rate**, and the **gap**. Never quote one as the other (`RESEARCH.md` §10.7).

**Accept rule:** parity error must be within a pre-registered tolerance per field (propose: relative error < 1 % on positions/velocities, exact on booleans, < 5 % on angular velocities pending the check above). A failed parity test is a **stop**: fix the obs builder, do not train around it.

### B7 — External baseline calibration

**Resolves:** U11. **Blocks:** any claim involving Nexto.
- Run the public Nexto weights (`Rolv-Arild/Necto` → `rlbot-support/Nexto/nexto-model.pt`) under RLBot v5 — the working reference is `JPK314/doublesize-nexto-v5` (torch 2.0.1+cpu, `beta` temperature knob) **[V]** — against the project's scripted ballchaser.
- Report win rate with a 95 % CI over ≥ 400 games.
- **Purpose:** if Nexto does not decisively beat a trivial ballchaser in this harness, the harness is broken, not Nexto. This gate validates the *measurement instrument*, and must run before any project policy is compared to Nexto.
- **License precondition (U12):** `Necto`'s license is `NOASSERTION` **[V]**. Desk-check permission to use the weights as an evaluation opponent before Phase 6. If unresolved, use it **only** as a private diagnostic and publish comparisons against own snapshots plus RLBotPack bots instead.

---

## 3. Phases

### Phase 0 — Environment bring-up & platform verification

**Objective:** prove the machine can do what the architecture assumes, before writing any training code.

**Entry criteria:** none.

**Work items**
1. Windows host: NVIDIA driver installed and current (no Linux driver in WSL, no `cuda`/`cuda-drivers` metapackages).
2. WSL 2 Ubuntu: Python **3.12** (satisfies `rlgym-learn`'s `>=3.10,<=3.14` **[V]** and RLBot's example `>=3.12` **[V]**), a venv **inside the Linux filesystem** — never `/mnt/c`, where RocketSim is ~10× slower **[R]** PILOT.
3. Install `rlgym[rl-sim]==2.0.1`, `rlgym-learn==2.0.0`, `rlgym-learn-algos==0.4.2`, torch (CUDA build matching the Windows driver's capability), `numpy`, `wandb`, `rlviser-py`. Emit a **lock file** and commit it.
4. Verify the installed tree resolves the known conflicts: numpy ≥ 2 present, `rlgym-sim` **absent**, `gym` **absent** (`RESEARCH.md` §13 compatibility matrix).
5. Windows side (separate venv, deferred until Phase 6 but checked now): `rlbot` 2.0.0b56, CPU-only torch, `rlgym-compat`.
6. **U12 desk check** on `Necto/LICENSE`.
7. Run **B1** and **B2a**.
8. Repo scaffolding: `configs/`, `benchmarks/`, `snapshots/`, `evals/`, `exports/`, `.gitignore` for checkpoints and `shmem_flinks/`.

**Acceptance criteria (all required)**
- [ ] `torch.cuda.is_available()` is `True`; `get_device_capability` returns `(8, 9)`; device name contains `4060 Ti`. **(B1 PASS)**
- [ ] `pip check` is clean in the WSL venv; the lock file is committed; numpy ≥ 2 confirmed.
- [ ] `import rocketsim` works and a bare `rsim.Arena` simulates 1v1 without error; **B2a** numbers are recorded.
- [ ] Collision-mesh resolution verified: `rlgym.rocket_league`'s `rsim.init(...)` is wrapped in a bare `except: pass` **[V]**, so a silent failure is possible. Assert explicitly that `rsim.init(collision_meshes)` succeeded rather than relying on the library's silence.
- [ ] rlviser status recorded as PASS / DEGRADED / FAIL, with the fallback decided.
- [ ] `benchmarks/RESULTS.md` exists with B1 and B2a rows; `RESEARCH.md` §14 U10 and U12 marked resolved or explicitly blocked.

**Non-goals:** no training, no reward design, no evaluation code.

**Risks:** WSL exposes less RAM/CPU than the host (**mitigation:** read `.wslconfig`, record it, size accordingly); rlviser unusable (**mitigation:** native-Windows rendering in Phase 6).

---

### Phase 1 — Minimum viable training pipeline → **M1 (first implementation milestone)**

**Objective:** the smallest thing that can train, trained. One policy, one process pool, kickoff-only states, `GoalReward`-dominated reward, 1v1.

**Entry criteria:** Phase 0 accepted.

**Work items** (exactly `ARCHITECTURE.md` §12.1)
1. `env/build_env.py` — `build_rlgym_v2_env()`:
   - `MutatorSequence(FixedTeamSizeMutator(1,1), KickoffMutator())`
   - `Float32DefaultObs(zero_padding=1)` → obs width **92** **[C]**
   - `RepeatAction(LookupTableAction(), repeats=8)` → **90** discrete actions, 15 decisions/s **[V]**
   - `CombinedReward((GoalReward(), 10.0), (TouchEventReward(), 0.1))` — **`TouchEventReward` is custom**, see item 2
   - `GoalCondition()` as **termination**; `AnyCondition(IdleTouchCondition(30), TimeoutCondition(300))` as **truncation** — **`IdleTouchCondition` is custom**, see item 2
   - `RocketSimEngine(rlbot_delay=True)` — keep the deliberate 1-tick input latency on **[V]**
2. `env/rewards.py` + `env/done_conditions.py` — the two **mandatory** custom components, replacing two shipped ones that read cumulative counters (decision D11; `RESEARCH.md` §4.4.1/§4.4.2 **[V]**):
   - **`TouchEventReward`** — `reset()` stores each agent's `ball_touches`; `get_rewards()` returns `max(0, current − previous)`. *Why:* RLGym v2's shipped `TouchReward` has a no-op `reset()` and tests `ball_touches > 0`, so it pays `+1` on **every step after the first touch** — **[C]** 3 to 44× the value of a goal at weight 0.1, i.e. a touch-once-then-stall incentive.
   - **`IdleTouchCondition`** — tracks the tick of the last *increment* in any car's `ball_touches`. *Why:* the shipped `NoTouchTimeoutCondition` tests `any(car.ball_touches > 0)`, which is cumulative, so it returns `False` forever after the first touch and idle episodes run to the 4 500-step absolute timeout.
3. `train.py` — `PPOAgentController(DiscreteFF(92, 90, (256,256,256)), BasicCritic(92, (256,256,256)), NumpyExperienceBuffer(GAETrajectoryProcessor()), PPOMetricsLogger())` + `LearningCoordinator`, with `device="cuda:0"` **set explicitly** and asserted at startup.
4. Serialization config: `obs_serde_type = NUMPY(float32, STATIC(shape=(92,)))` derived from `DefaultObs.get_obs_space()` (never hard-coded), `action_serde_type = NUMPY(int64, STATIC((1,)))`, `agent_id_serde_type = STRING`, `recalculate_agent_id_every_step = False` **[V]**, `OPENBLAS_NUM_THREADS=1` set before any numpy import **[V]**.
5. `configs/exp001_*.py` producing a serialised `config.json` via `generate_config(..., force_overwrite=True)` (all models are `extra="forbid"` **[V]** — typos become hard errors).
6. `tools/snapshot.py` — copy actor weights to `snapshots/<run>/<steps>.pt` every 25 M steps into a directory **outside** `agent_controller_save_folder`, which the trainer *"MAY DELETE ANYTHING INSIDE"* **[V]**.
7. `evals/eval_random.py` — head-to-head, greedy argmax, fixed seeds, side-balanced, records GF/GA/end-reason/length. Opponent: a random policy.
8. `tests/test_env_semantics.py` — the termination/truncation and reward-semantics tests listed in `ARCHITECTURE.md` §4.4.
9. Run **B3** (the `n_proc` / `min_frac` sweep) and record it.

**Known upstream traps to avoid while writing this** (all verified, `RESEARCH.md` §4.4/§4.6/§5.2):
- **`TouchReward` is cumulative, not per-event** — 3 to 44× a goal at weight 0.1 **[C]**. Use the custom `TouchEventReward`.
- **`NoTouchTimeoutCondition` never fires after the first touch** — use the custom `IdleTouchCondition`.
- `quick_start_guide.py` uses `team_size=2` while its `ObsType` alias still says 92 — do not copy it.
- `rlgym-ppo`'s v2 example calls `NoTouchTimeoutCondition(timeout=...)`; the v2 kwarg is `timeout_seconds=`.
- `PPOLearnerConfigModel.device` defaults to `"cpu"`.
- `DefaultObs` emits **float64**.
- `GoalReward.__init__` takes **no arguments** — there is no `aggression_bias` knob to reach for.

**Acceptance criteria — M1 (all required)**
- [ ] `tests/test_env_semantics.py` passes: goal ⇒ terminated (not truncated); timeout ⇒ truncated (not terminated); **`IdleTouchCondition` fires after its timeout even though a touch happened earlier in the same episode**; **`TouchEventReward` pays exactly once per increment of `ball_touches`**.
- [ ] A training run completes **50 M timesteps** without manual intervention and resumes correctly from a checkpoint after a forced kill.
- [ ] `torch.cuda.is_available()` asserted `True` at startup and the resolved device logged; GPU peak memory logged.
- [ ] **`overall steps/s` measured and recorded for the chosen `n_proc`** (gate B3) — this number replaces every estimate in `RESEARCH.md` §12.
- [ ] **The policy beats a random policy with a Wilson 95 % lower bound on win rate > 0.50 over ≥ 400 games**, side-balanced, greedy. *(A random opponent is a low bar on purpose: it separates "the pipeline works" from "the policy is good".)*
- [ ] Mean undiscounted episodic return and `average_reward` both trend upward across the run, **and** `actor_entropy` decreases without collapsing to ~0.
- [ ] `kl_divergence`, `sb3_clip_fraction`, `critic_loss`, `actor_update_magnitude` and `critic_update_magnitude` are logged and inspected once; no divergence.
- [ ] `config.json`, the dependency lock, the git commit hash and the seed are recorded against the run.
- [ ] At least 3 matches inspected visually (rlviser, or the Phase 6 path if B1 recorded rlviser as FAIL).

**Why these and not more:** M1 must be achievable in a day or two of wall clock. "Beats random" is a real, measurable, non-trivial outcome — a policy that scores at all in Rocket League is already non-random behaviour. Everything about *quality* belongs to later milestones.

**Non-goals:** no opponent pool, no curriculum, no shaping beyond the touch-event term, no 2v2, no network scaling, no export, no RLBot.

**Risks:** silent CPU fallback (mitigated by the startup assertion); obs-width mismatch between config and builder (mitigated by deriving from `get_obs_space()`); reset-cost domination at short episodes (detected by B2b's resets/s).

---

### Phase 2 — Throughput characterization & optimization → M2

**Objective:** convert "it trains" into "it trains as fast as this machine allows", with numbers.

**Entry criteria:** M1 accepted.

**Work items:** run **B2b**, **B2c**, **B4**, **B5**. Implement the winning configuration. Add per-process response-rate logging (straggler watchdog). Commit `benchmarks/RESULTS.md`.

**Acceptance criteria — M2**
- [ ] B2b: RSS per env process measured; the `n_proc` ceiling under WSL-exposed memory is written down with its justification row.
- [ ] B2c: P-core affinity tested and the decision (> 5 % or reject) recorded either way.
- [ ] B4: the full network × `n_minibatches` grid is recorded; the largest cell with `batch_consumption_time ≤ 20 %` of `iteration_time` and peak VRAM ≤ 6.5 GB is identified.
- [ ] B5: ≥ 4-hour run shows flat RSS after the first hour and < 15 % steps/s variance across 10-minute windows.
- [ ] `overall steps/s` at the tuned configuration is recorded and **is higher than at the M1 configuration**, or the reason it is not is documented.
- [ ] A one-page `benchmarks/RESULTS.md` exists that a future reader can use to reproduce every number.

**Non-goals:** no learning-strategy changes. This phase changes *speed*, never *behaviour*.

---

### Phase 3 — Reward, curriculum & state randomization → M3

**Objective:** a policy that is competent from arbitrary game states, not just kickoffs, with an objective that cannot be farmed.

**Entry criteria:** M2 accepted.

**Work items**
1. **`CurriculumStateMutator`** — mandatory custom code: RLGym v2 ships only `KickoffMutator`, `FixedTeamSizeMutator` and `MutatorSequence`, i.e. **no random state setter exists** **[V]**. Follow `KickoffMutator`'s ~30-line template (`apply(state, shared_info) -> None`, mutate the plain dataclass).
2. Stage schedule per `ARCHITECTURE.md` §4.3, mixture probabilities in config so a stage change is a config diff. **The stage is recorded in run metadata** — different stages mean different state distributions, so their checkpoints are not comparable.
3. **Potential-based shaping** only: `Φ(s_t) − Φ(s_{t−1})`, the form proven not to change the optimal policy (Ng et al., ICML 1999) and the form Necto actually uses **[V]**.
4. **Per-component reward logging** (`Reward/<name>` series) — the primary farming detector and the input to promotion rule 5.
5. **Annealing driven by global `cumulative_timesteps`**, never per-process step counts (a documented bug to avoid).
6. Every component gets a one-line note on *how it could be exploited*. If that line cannot be written, the term does not ship.

**Acceptance criteria — M3**
- [ ] `CurriculumStateMutator` unit-tested: ball and car states lie inside legal bounds (pitch dimensions, boost ∈ [0,100], velocities finite), and the observed stage mixture matches the configured probabilities within sampling error over ≥ 10 000 resets.
- [ ] A policy trained from **stage 2** beats the M1 kickoff-only policy from **random ground states** with a Wilson lower bound > 0.50 over ≥ 400 games.
- [ ] Reward-component dashboard shows no single term exceeding ~30 % of mean episodic return.
- [ ] Ablation recorded: removing the annealed shaping terms does **not** reduce final win rate against the frozen M2 incumbent (evidence that shaping helped learning, not that it is being farmed).
- [ ] At least 5 matches inspected; no exploit behaviour (e.g. the goal-waiting pathology Necto's own code comment flags **[V]**).

**Non-goals:** no opponent pool, no 2v2, no network change.

---

### Phase 4 — Evaluation harness, baselines & opponent pool → M4

**Objective:** make progress measurable and make collapse detectable. This is the phase that turns a training run into a *result*.

**Entry criteria:** M3 accepted.

**Work items**
1. **Evaluator** (`ARCHITECTURE.md` §7): out-of-process, greedy, fixed seeds, side-balanced, same episode rules as training, run on **RocketSim** with training paused.
   - **[C] Cost check:** at 5 000 steps/s a 1 000-game, 1 500-step-per-game evaluation is ~5 minutes sequential and under a minute across 8 envs. Even at 4 500 steps/episode it is ~15 minutes sequential. **Statistically meaningful evaluation is minutes, not hours** — there is no excuse for evaluating rarely.
2. **Baselines** (`RESEARCH.md` §10.5): random, scripted ballchaser (~50 lines), ≥ 5 own historical snapshots, best-ever snapshot.
3. **Promotion gate** — all six conditions of `RESEARCH.md` §10.4, implemented as a script that prints a verdict, not a human judgement.
4. **`MultiAgentController` + frozen inference-only subcontrollers (SP2)**, then **scripted/random baselines as subcontrollers (SP3)**. Actor-only loading, no optimizer, no experience buffer, fixed temperature.
5. **Opponent pool policy:** ≥ 5 frozen snapshots spanning the recent skill range, refreshed every ~100 M steps, sampled **uniformly from the pool** — not "last K checkpoints", since a pool drawn from a collapsing policy collapses with it.
6. **Regression suite:** every promotion runs the candidate against the *whole* frozen baseline set, so slow multi-month regression is caught.

**Acceptance criteria — M4**
- [ ] The evaluator reproduces a known result: candidate vs. an identical copy of itself yields a win rate whose 95 % CI contains 0.50 over ≥ 400 games. *(This is the harness's own null test — if it fails, every later number is suspect.)*
- [ ] Promotion script emits a machine-readable verdict implementing all six conditions, with the Wilson lower bound and goal-differential CI printed.
- [ ] SP2 running: at least one frozen historical subcontroller active, with its memory overhead measured (expected: a few MB each).
- [ ] **The SP2 policy beats the M3 policy** with a Wilson lower bound > 0.50 over ≥ 1000 games, and does not regress against any frozen baseline beyond CI width.
- [ ] No reward component exceeds ~30 % of mean episodic return at the promoted checkpoint.
- [ ] A policy-collapse detector is demonstrated: run one deliberately destabilised configuration (e.g. very high entropy coefficient) and show that win rate vs. the *frozen pool* falls while training reward stays flat — the exact failure `RESEARCH.md` §10.1 warns about.

**Non-goals:** no network scaling, no deployment.

---

### Phase 5 — Scale-up: network, league, skill rating → M5

**Objective:** spend the compute budget on capability, now that the pipeline is fast, measurable and protected against collapse.

**Entry criteria:** M4 accepted.

**Work items**
1. **Network scaling**, gated by the B4 grid — `(256,256,256)` → `(512,512,512)` → `(1024,1024,512,512)` → `(2048,2048,1024,1024)`, raising `n_minibatches` at each step. **Changing width or obs width invalidates every checkpoint** — do it only at a phase boundary.
2. **`n_epochs = 2`** (from 1), gated behind B4 since it doubles GPU work per batch.
3. **SP4: two learning controllers** (blue vs. orange) — `rlgym-learn`'s own tested configuration **[V]**.
4. **Elo / TrueSkill tracking over the pool**, with a fixed external anchor (precedent: `rlgymppo_rs` pins `nexto_mmr = 1500` **[V]**) and optional skill-proximity matchmaking.
5. **Optional γ anneal** 0.99 → 0.995 late in training (BotBoi's two-phase curriculum **[R]**).
6. **Optional mixed-mode training** (1v1 + 2v2 + 3v3) once 1v1 passes its gate.

**Acceptance criteria — M5**
- [ ] The scaled configuration is justified by a B4 row (`batch_consumption_time ≤ 20 %` of `iteration_time`, peak VRAM ≤ 6.5 GB) — not by preference.
- [ ] The scaled policy beats the M4 policy with a Wilson lower bound > 0.50 over ≥ 1000 games, **or** the null result is recorded honestly ("scaling did not help at this step budget") and the smaller network is retained.
- [ ] SP4 running with both controllers learning; memory and throughput re-measured and recorded.
- [ ] An Elo/TrueSkill series exists over the snapshot pool, anchored to a fixed reference.
- [ ] Wall-clock budget for the target step count is computed **from the measured B3 number**, not from an estimate, and written into `benchmarks/RESULTS.md`.

**Scale reality check [C]** — wall-clock days to reach various **step budgets** at various *hypothetical* throughputs. These are step-budget labels, **not** the milestone IDs of §4 — the mapping from steps to a milestone is deliberately left open until the B3 measurement exists. The real row is filled in by B3:

| Step budget | Steps | @2 k/s | @5 k/s | @10 k/s | @20 k/s |
|---|---|---|---|---|---|
| Pipeline viability (≈ M1) | 5 × 10⁷ | 0.3 d | 0.1 d | 0.1 d | 0.03 d |
| Early competence | 2.5 × 10⁸ | 1.4 d | 0.6 d | 0.3 d | 0.1 d |
| Established 1v1 policy | 1 × 10⁹ | 5.8 d | 2.3 d | 1.2 d | 0.6 d |
| Extended self-play run | 4 × 10⁹ | 23 d | 9.3 d | 4.6 d | 2.3 d |
| Nexto-scale **[R]** | 1.35 × 10¹⁰ | 78 d | 31 d | 16 d | 7.8 d |

The last row is the honest ceiling: Nexto's reported ~250 000 hours **[R]** is 1.35 × 10¹⁰ agent steps **[C]**, i.e. **8–78 continuous days on this single machine**. The project must therefore be judged against *internal* milestones, not against Nexto's rank.

---

### Phase 6 — Deployment to the original game → M6, M7

**Objective:** a policy that plays the real game within RLBot's latency budget, with the transfer gap measured rather than assumed.

**Entry criteria:** M5 accepted (or M4 if scaling was rejected) **and** U12 resolved.

**Work items**
1. **Exporter** (`ARCHITECTURE.md` §9): `torch.jit.trace` the actor, write `policy.pt` + `obs_builder.py` + `manifest.json` (obs width, normalisation, action space 90, tick skip 8, `rlbot_delay`, layer sizes, config hash, source checkpoint). **Numerical equivalence test** vs. the training actor on fixed observations.
2. **Windows bot** (separate venv: `rlbot`, CPU-only torch, `numpy`, `rlgym-compat` — **no** `rocketsim`, **no** `rlgym_learn`, **no** CUDA **[V]** precedent: `doublesize-nexto-v5` ships `torch==2.0.1+cpu`).
   - `get_output(GamePacket) -> ControllerState`, cached-action pattern with `tick_skip = 8` (re-infer at 15 Hz) **[V]** Nexto's bot.
   - `rlgym_compat` V2 `GameState` → the **same** obs builder as training.
   - **The bot refuses to start if the manifest disagrees with the obs builder it loads.**
3. **Latency instrumentation** and RLBot's bot-response-rate metric.
4. Run **B6** and **B7**.

**Acceptance criteria — M6 (sim-side export correctness)**
- [ ] Traced artifact matches the training actor to max |Δlogit| < 1e-5 on a fixed observation batch.
- [ ] The manifest/obs-builder consistency check demonstrably rejects a deliberately mismatched manifest.
- [ ] The Windows venv imports no CUDA and no RocketSim; `pip check` clean.

**Acceptance criteria — M7 (in-game operation)**
- [ ] The bot loads in RLBot v5 and completes a full 5-minute offline match without a missed-tick warning; p99 `get_output` well under 8.3 ms.
- [ ] **B6 observation parity passes** within the pre-registered per-field tolerances. A failure is a **stop**, not a note.
- [ ] **B7 passes**: public Nexto decisively beats the scripted ballchaser in this harness (≥ 400 games, CI excludes 0.50) — otherwise the harness is broken and no comparison is published.
- [ ] **The sim-vs-game gap is reported**: sim win rate, game win rate, and the difference, over the same opponent class. Never quote one as the other.
- [ ] Behavioural smoke test: the bot moves, touches the ball, and does not immediately own-goal.
- [ ] All play is in **offline / RLBot modes only** — no online play, per `RESEARCH.md` §7.5.

**Non-goals:** no training against the live game, no online play, no ranked matches.

---

### Phase 7 — Competitive validation & reporting → M8

**Objective:** a defensible, reproducible statement of how good the agent is.

**Entry criteria:** M7 accepted.

**Work items**
1. Head-to-head vs. the frozen baseline set at **≥ 2000 games per pairing** (the reporting threshold, `RESEARCH.md` §10.2).
2. Head-to-head vs. public Nexto weights under RLBot v5, subject to U12.
3. Matches vs. a sample of RLBotPack bots for a diversity check.
4. Human evaluation: the user plays the bot in offline matches; record subjective rank impressions alongside the objective numbers.
5. Final report: architecture, measured throughput, learning curves, win-rate table with CIs, sim-vs-game gap, reward-component audit, all unresolved questions and their disposition.

**Acceptance criteria — M8**
- [ ] Every win-rate claim is accompanied by a Wilson 95 % CI and a game count ≥ 2000.
- [ ] The report distinguishes **sim** and **game** results everywhere; no sim number is presented as game strength.
- [ ] Every number in the report traces to a committed benchmark row or evaluation output.
- [ ] `RESEARCH.md` §14 U1–U12 is updated: each marked **resolved** (with the answer) or **still open** (with why).
- [ ] The report states plainly which results are *measured here*, which are *reported by others*, and which are *estimates* — the same discipline as `RESEARCH.md` §0.

---

## 4. Milestone summary

| ID | Milestone | Definition of done (short form) | Primary gate |
|---|---|---|---|
| **M0** | Platform verified | CUDA visible in WSL 2, RocketSim imports and steps, lock committed, B1+B2a recorded | B1, B2a |
| **M1** | **Pipeline learns** | Env-semantics tests pass (incl. the two shipped-component traps); 50 M steps unattended and resumable; beats random with Wilson LB > 0.50 over ≥ 400 games; throughput measured | B3 |
| **M2** | Throughput tuned | B2b/c + B4 + B5 complete; steps/s improved or the reason documented; 4-h stability | B2c, B4, B5 |
| **M3** | Arbitrary-state competence | Curriculum mutator tested; stage-2 policy beats kickoff-only policy from random states | — |
| **M4** | Measurable & collapse-proof | Promotion gate implemented; harness null test passes; SP2 beats M3 over ≥ 1000 games | — |
| **M5** | Scaled & rated | Network/`n_epochs` scaling justified by B4 rows; SP4 + Elo series; wall-clock budget from measured B3 | B4 |
| **M6** | Export correct | Traced actor numerically equivalent; manifest guard works; Windows venv CUDA/RocketSim-free | B6 (part) |
| **M7** | Plays the real game | RLBot v5 match at 120 Hz with no missed ticks; obs parity passes; Nexto calibration passes; sim/game gap reported | B6, B7 |
| **M8** | Validated & reported | ≥ 2000-game comparisons with CIs; sim vs. game kept distinct; U1–U12 dispositioned | B6 re-run |

---

## 5. Decision and stop conditions

Recorded in advance so they are decisions, not reactions.

| Trigger | Pre-committed response |
|---|---|
| **B1 fails** (no CUDA in WSL 2) | Stop. Fix the Windows driver / WSL configuration. Do **not** proceed to CPU-only training — the throughput penalty is ~50× and every later estimate is void. |
| **B3 shows `timestep_collection_time` > 70 % of `iteration_time` everywhere** | Run is CPU-bound. Adding GPU work is nearly free. Revisit `rlgymppo_rs` (3–10×, but GPL‑2.0 + Rust + RocketSim v3, `RESEARCH.md` §5.4) **as a Phase 5 decision**, with the license question settled first. |
| **B6 observation parity fails** | Stop. Fix the obs builder. Never compensate in the reward or by retraining — a parity failure means training and deployment are solving different problems. |
| **B7 fails** (Nexto does not beat the ballchaser) | The harness is broken. Fix it before publishing any comparison. |
| **Win rate vs. the frozen pool falls while training reward rises** | Policy collapse or reward farming. Freeze the run, audit reward components, roll back to the last promoted snapshot. |
| **Two consecutive promotions fail** | Stop adding compute. Change one variable (reward, curriculum stage, network, `n_epochs`) and re-test. Do not stack changes. |
| **Host OOM / GPU OOM mid-run** | Lower `n_proc` or `max_size` (host) / raise `n_minibatches` (GPU). Re-run B5 before resuming a long run. |
| **Throughput drops > 15 % between runs at the same config** | Treat as a regression. Dependency drift is the usual cause — diff the lock file. |
| **U12 unresolved (Necto license)** | Use Nexto as a private diagnostic only. Publish comparisons against own snapshots and RLBotPack bots. |
| **No promotable policy after the M5 step budget** | Report the null result with its measurements. A well-measured negative is a valid Phase 1 outcome; an invented one is not. |

---

## 6. Standing engineering rules

1. **One change at a time.** Never advance two phases or two variables simultaneously.
2. **Every number has a source.** Measured, computed, reported by someone else, or an estimate — labelled, as in `RESEARCH.md` §0.
3. **Never quote a reported figure as a target.** 114 481 tps **[R]**, 8 581 steps/s **[R]** and ~250 000 hours **[R]** are context, not goals.
4. **Win rate against a frozen opponent is the only progress metric.** Reward is a training signal (`RESEARCH.md` §10.1).
5. **Config is code and is committed.** pydantic models are `extra="forbid"` **[V]**; `config.json` is the run's ground truth.
6. **Nothing valuable lives in `agent_controller_save_folder`.** The trainer may delete anything inside it **[V]**.
7. **`device` is set explicitly** and asserted. The default is `"cpu"` **[V]**.
8. **Evaluation never runs concurrently with a throughput measurement.**
9. **Termination vs. truncation is a correctness invariant**, and is unit-tested.
10. **Offline / RLBot modes only.** No online play, ever (`RESEARCH.md` §7.5).
11. **Read the source of any shipped component before weighting it.** Two of RLGym v2's shipped classes (`TouchReward`, `NoTouchTimeoutCondition`) read cumulative counters and behave nothing like their names — both failures are invisible in logs (`RESEARCH.md` §4.4.1/§4.4.2, decision D11).

---

## 7. The first thing to build

> **Milestone M1.** Build the Phase 1 pipeline (`ARCHITECTURE.md` §12.1) and run it for 50 M timesteps in 1v1.
>
> **It is done when all of these hold:**
> 1. `tests/test_env_semantics.py` passes — including the two cases the shipped RLGym v2 components get wrong: **`IdleTouchCondition` fires after its timeout even though a touch happened earlier in the same episode**, and **`TouchEventReward` pays exactly once per increment of `ball_touches`** (`RESEARCH.md` §4.4.1/§4.4.2 **[V]**).
> 2. CUDA is asserted available at startup and the resolved device is logged.
> 3. The run reaches **50 M timesteps** unattended and resumes correctly after a forced kill.
> 4. **`overall steps/s` is measured** and recorded (gate B3), replacing every estimate in `RESEARCH.md` §12.
> 5. The policy **beats a random policy** with a **Wilson 95 % lower bound on win rate > 0.50 over ≥ 400 side-balanced greedy games**.
> 6. `average_undiscounted_episodic_return` trends up while `actor_entropy` falls without collapsing to ~0.
> 7. `config.json`, the dependency lock, the git commit and the seed are recorded against the run.
> 8. At least 3 matches are inspected visually.
>
> **Nothing else is in scope for M1.** No opponent pool, no curriculum, no shaping beyond the touch-event term, no 2v2, no network scaling, no export, no RLBot.
