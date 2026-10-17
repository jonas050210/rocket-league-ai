# Rocket League AI — Phase 1 Technical Research

**Repository:** `rocket-league-ai` · **Branch:** `arena/edfccc11-rocket-league-ai`
**Target machine:** Ubuntu on WSL 2 · Intel Core i7‑12700F · NVIDIA GeForce RTX 4060 Ti 8 GB · 32 GB RAM
**Research date:** 2026‑10‑17
**Phase boundary:** research and architecture planning only. No implementation, no dependency installation, no training, no config changes.

Companion documents: [`ARCHITECTURE.md`](ARCHITECTURE.md) (system design) · [`ROADMAP.md`](ROADMAP.md) (phases, gates, acceptance criteria).

---

## 0. How to read this document

Every substantive claim carries an evidence tag. The tags are the difference between "documented", "measured", "computed" and "believed" — this project's biggest failure mode is treating someone else's reported result as a reproducible fact.

| Tag | Meaning | Trust |
|---|---|---|
| **[V]** | **Verified** — read directly from source code, package metadata or an official page during this research pass. Link given. | High |
| **[C]** | **Computed** — arithmetic performed here, inputs stated. Reproducible. | High, but it is arithmetic, not measurement |
| **[R]** | **Reported** — a third-party claim (README, paper, forum). Not independently reproduced. | Low–Medium |
| **[E]** | **Estimate / recommendation** — engineering judgement. Must be confirmed by the named benchmark gate. | Hypothesis |
| **[U]** | **Unresolved** — cannot be answered from available sources; requires a local experiment. | Unknown |

### 0.1 Repository state at start of phase

**[V]** The repository contains exactly one tracked file, `README.md`, whose entire content is the line `# rocket-league-ai`. There is a single commit (`c9a0736 Initial commit`), no code, no dependency manifest, no CI. Nothing was overwritten by this phase: the three documents below are the only additions.

---

## 1. Executive summary

**The recommended stack is: RLGym v2 + RocketSim 2.2.x + `rlgym-learn` 2.0.0 / `rlgym-learn-algos` 0.4.2 + PyTorch (CUDA) + RLBot v5 for in‑game deployment.**

Four findings dominate everything else in this document:

1. **RLGym is on v2, and almost every tutorial on the internet describes v1.** The current API is `rlgym.api.RLGym` composed of pluggable `ObsBuilder` / `ActionParser` / `RewardFunction` / `StateMutator` / `DoneCondition` / `TransitionEngine` objects — not `rlgym.make(...)`. `rlgym-sim` (the RocketSim fork of v1) is a self-described "TEMPORARY STOP-GAP" and is superseded by `rlgym.rocket_league.sim.RocketSimEngine`. **[V]** §4.1, §4.6.

2. **`RLGym-PPO` — the learner every tutorial points at — is effectively frozen and cannot self-play.** Its last push is 2025‑01‑06; its `requirements.txt` pins `numpy>1.21,<2.0`, which directly conflicts with `rlviser-py`'s `numpy>=2.0.0`; it depends on the abandoned `gym` package; and its `Learner` drives a *single* policy for every agent, so there is no mechanism for an opponent pool. **[V]** §5.1, §10.2.

3. **`rlgym-learn` is the maintained successor and it has the one feature that matters: `MultiAgentController.choose_subcontrollers()`, which assigns a named subcontroller (i.e. a distinct policy) to each agent id per environment.** That is the primitive for self-play against historical checkpoints, for a blue/orange asymmetric league, and for evaluation against a frozen baseline — all inside the existing rollout loop. **[V]** §5.2.

4. **`rlgym-learn` does not run on native Windows, and that is the correct outcome for this project anyway.** Its env-process backend is Rust over `mio` + `shared_memory`, and the community project PILOT documents `FileExistsError: entity already exists` at startup on Windows. WSL 2 is the supported path on a Windows host — which is exactly the stated environment. Rocket League itself runs on Windows; WSL 2 and the Windows host coexist, so training in WSL 2 and deploying to the game on Windows is one machine, not two. **[V][R]** §5.2, §7.4.

**The honest bottom line on strength:** Nexto's documented level is *approximately Grand Champion 1 in 1v1/2v2/3v3* **[R]**, reached with a Perceiver-style attention network trained on crowd-sourced compute from 10–40 contributing game clients over several months **[R]**, and with a reported ~250,000 hours of gameplay **[R]**. That compute is not available here, and §12 quantifies the gap. A single 20‑thread machine can absolutely produce a bot that plays competently and beats casual players; it cannot be assumed to reach GC1. The roadmap is therefore built around *measurable* milestones rather than a promised rank.

---

## 2. Ecosystem map

```
                    ┌───────────────────────────────────────────────────────────┐
                    │  PHYSICS / SIMULATION                                     │
                    │  RocketSim 2.2.1 (C++, MIT)  ── pip: rocketsim            │
                    │     └─ v3-rust branch / rocketsim-rs  (newer, not on PyPI) │
                    └───────────────────────────────────────────────────────────┘
                                          │
                    ┌───────────────────────────────────────────────────────────┐
                    │  ENVIRONMENT API                                          │
                    │  rlgym-api 2.0.0 + rlgym-rocket-league 2.0.1 (Apache-2.0) │
                    │     RocketSimEngine · DefaultObs · LookupTableAction      │
                    └───────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
   ┌────────────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐
   │ rlgym-ppo 1.3.13   │   │ rlgym-learn 2.0.0       │   │ rlgymppo_rs (Rust)   │
   │ frozen 2025-01     │   │ + algos 0.4.2           │   │ GPL-2.0, RocketSim v3│
   │ no self-play       │   │ ← RECOMMENDED           │   │ threads, Elo tracker │
   │ numpy<2 conflict   │   │ MultiAgentController    │   │ distillation         │
   └────────────────────┘   └─────────────────────────┘   └──────────────────────┘
                                          │
                    ┌───────────────────────────────────────────────────────────┐
                    │  DEPLOYMENT INTO THE REAL GAME                            │
                    │  RLBot v5 (RLBotServer / rlbot 2.0.0bxx, Python 3.12+)    │
                    │  Windows or Linux+Proton · 120 Hz · TCP + FlatBuffers     │
                    └───────────────────────────────────────────────────────────┘
```

---

## 3. RocketSim

### 3.1 What it is — verified facts

| Item | Value | Evidence |
|---|---|---|
| Repo | `ZealanL/RocketSim` | **[V]** https://github.com/ZealanL/RocketSim |
| Language / license | C++ / **MIT** | **[V]** GitHub API, `spdx_id: MIT` |
| Latest release | **v2.2.1**, published 2026‑04‑19 | **[V]** GitHub releases API |
| Last push | 2026‑10‑16 (actively maintained) | **[V]** GitHub API `pushed_at` |
| Open issues | 0 | **[V]** GitHub API |
| Python package | `rocketsim` **2.2.1** on PyPI, `requires_python >=3.9` | **[V]** https://pypi.org/pypi/rocketsim/json |
| Python wheels | `cp36-abi3` for `manylinux_2_27_x86_64`, `manylinux_2_27_aarch64`, `macosx_14_0_arm64`, `win_amd64` — abi3, so no per-Python-version rebuilds | **[V]** PyPI release files, 2026‑04‑24 |
| Runtime deps | `cmeel`, `numpy` (unpinned at runtime; `numpy>=2.0` only for the `build` extra) | **[V]** PyPI `requires_dist` |
| Game modes | Soccar, Hoops, Dropshot, Heatseeker, Snowday | **[V]** README |
| Physics tick rate | 120 Hz (`Arena::Create(GameMode, float tickRate = 120)`) | **[V]** docs.arena page |
| Arena cloning / serialization | `Arena::Clone(bool copyCallbacks)`, `WriteToFile` / `LoadFromFile`, `SerializeCar` | **[V]** RocketSim docs |
| Legal stance | "does not actually contain any code from the game whatsoever"; intended for ML environments; author is explicitly against cheating in ranked play | **[V]** README |

### 3.2 Performance — what is actually documented

The only primary-source throughput figures that exist are ZealanL's own benchmarks on an **Intel i5‑11400 @ 2.60 GHz, Windows 10, MSVC 14.16, 2v2 soccar, 1 M ticks** **[V]** README:

| RocketSim | single-thread TPS |
|---|---|
| v1.0.0 | 30,334 |
| v1.2.0 | 50,763 |
| v2.0.0 | ~50,000 |
| **v2.1.0** | **114,481** |

The README also states, in prose: *"On an average PC running a single thread of RocketSim with two cars, RocketSim can simulate around 20 minutes of game time every second. This means that with 12 threads running RocketSim, you can simulate around 10 days of game time every minute."* **[V]**

Converting **[C]**: 20 minutes of game time per second = 1 200 s/s × 120 ticks/s = **144 000 tps**, consistent with the v2.1.0 figure.

**Critical interpretation for this project [C]:** one *agent step* is 8 physics ticks by default (`RepeatAction(repeats=8)`), so 114 481 tps ≈ **14 300 env‑steps/s per RocketSim thread**. No Python observation builder, reward function and IPC path is going to keep up with that. **The physics engine is almost certainly not the bottleneck — the Python per-step path is.** This is arithmetic on a documented number, not a measurement; benchmark gate **B2** in `ROADMAP.md` measures it on this machine.

**Do not use the "12 threads → 10 days/minute" figure as a plan.** It is the raw C++ number, not RLGym throughput.

### 3.3 Accuracy and fidelity

**[V]** README, verbatim claims:

> RocketSim is not a perfectly accurate replication of Rocket League, but is close enough for most applications (such as training ML bots). RocketSim is accurate enough to: *Train machine learning bots to SSL level (and probably beyond)*, *Simulate different shots on the ball at different angles to find the best input combination*, *Simulate air control to find the optimal orientation input*, *Simulate pinches*. **However, the tiny errors will accumulate over time, so RocketSim is best suited for simulation with consistent feedback.**

That last sentence is the single most important sentence in this section: RocketSim's own author says drift accumulates and that the mitigation is *consistent feedback* — i.e. closed-loop RL with frequent resets, not long open-loop rollouts. This directly shapes the sim-to-game transfer plan (§7.6) and the episode-length policy (§4.4).

The v2.1.0 release notes list concrete accuracy work: *"Wavedash accuracy significantly improved"*, *"Flip accuracy significantly improved"*, *"Suspension accuracy slightly improved"*, plus a new custom broadphase and AABB caching that produced the ~2× speedup. v2.2.1 fixed *"Scale flip torque with tickTime"*. **[V]** GitHub releases.

There is **no published quantitative fidelity metric** (no published per-maneuver error table against the live game). **[U]** — this is why benchmark gate **B6** in `ROADMAP.md` includes an optional RocketSim-vs-RLBot state comparison rather than trusting the prose claim.

### 3.4 Collision meshes — a real deployment gotcha

**[V]** `rlgym/rocket_league/sim/rocketsim_engine.py` does:

```python
try:
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    rsim.init(os.path.join(cur_dir, 'collision_meshes'))
except Exception:
    pass
```

The `except: pass` means **a missing or unreadable mesh folder fails silently**, and the failure then surfaces later as a confusing error at `rsim.Arena(game_mode)`.

The good news **[V]**: `setup_rocket_league.py` in the RLGym repo declares `package_data` including `'sim/collision_meshes/**/*'`, and the repo tree contains `rlgym/rocket_league/sim/collision_meshes/{soccar,hoops,dropshot}/*.cmf` (16 soccar meshes). So `pip install rlgym[rl-sim]` **ships the meshes** — unlike the old `rlgym-sim` workflow, which required downloading `RLArenaCollisionDumper` and running it against a live game install (the RLGym‑PPO‑Guide still tells you to do exactly that — see §4.6).

**Action item:** the first implementation task must assert mesh presence, not assume it.

### 3.5 RocketSim v3 and the Rust bindings

**[V]** `ZealanL/RocketSim` has a `v3-rust` branch; `VirxEC/rocketsim-rs` and `azkarox-dev/rlgym_rs` target it, and `Virxec/rlgymppo_rs` is built on top. `rocketsim-rs` publishes Rust benchmarks on a Ryzen 9 5900X / 24 threads: *"Simulated 11.11 hours in 7.003 seconds — FPS: 685 459"* for a ball-only bench and *"81 253 FPS"* for its `real_bench` **[R]** (README, hardware differs from ours).

**Recommendation [E]:** stay on the PyPI `rocketsim` 2.2.x wheel for Phase 1. Rationale: (a) it is the version `rlgym-rocket-league` 2.0.1 constrains (`rocketsim >=2.0.0,<3.0.0` — **[V]** `setup_rocket_league.py`); (b) it has prebuilt manylinux wheels, so no compiler toolchain; (c) the Rust path is faster but is a different, younger stack whose API surface changes. Revisit only if benchmark **B2** proves the C++/Python boundary is the ceiling.

Note that a GPU-accelerated fork exists (`BLMChoosen/RocketSim-CUDA`, claiming 65 k+ envs at 500 k+ SPS on an RTX 5060) **[R]**. It is not evaluated further here: it is not the simulator RLGym wraps, its claims are unverified, and adopting it would mean abandoning the entire RLGym ecosystem.

---

## 4. RLGym v2

### 4.1 Package layout and versions — verified

RLGym v2 is split into independently-versioned PyPI packages **[V]**:

| Package | Version | `requires_python` | Notes |
|---|---|---|---|
| `rlgym` | 2.0.1 | `>=3.7` | Meta / install-only package. README: *"**DO NOT** specify this package as a dependency, this is an **installation only** package."* |
| `rlgym-api` | 2.0.0 | `>=3.7` | The generic API, no dependencies. |
| `rlgym-rocket-league` | 2.0.1 | `>=3.7` | The Rocket League implementation; ships collision meshes. |
| `rocketsim` | 2.2.1 | `>=3.9` | See §3. |
| `rlviser-py` | 0.6.14 | `>=3.10` | Visualizer; requires `numpy>=2.0.0` and `rocketsim>=2.2.1`. |

Install extras **[V]** README: `pip install rlgym` (api only) · `rlgym[rl]` · `rlgym[rl-sim]` · `rlgym[rl-rlviser]` · `rlgym[all]`.

**[V]** `rlgym-rocket-league` 2.0.1 declares `install_requires = ['rlgym-api >=2.0.0,<3', 'numpy >=1.19,<3.0.0']` and extras `sim: rocketsim >=2.0.0,<3.0.0`, `rlviser: rlviser-py >=0.6.0,<1.0.0`.

Maintenance status: repo last push **2026‑05‑16**, 252 stars, 20 open issues, Apache‑2.0 **[V]** GitHub API.

### 4.2 The v2 API surface — verified from source

**[V]** `rlgym/api/rlgym.py`. `RLGym` is a generic class constructed from pluggable components:

```python
RLGym(
    state_mutator: StateMutator[StateType],
    obs_builder: ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType],
    action_parser: ActionParser[AgentID, ActionType, EngineActionType, StateType, ActionSpaceType],
    reward_fn: RewardFunction[AgentID, StateType, RewardType],
    transition_engine: TransitionEngine[AgentID, StateType, EngineActionType],
    termination_cond: DoneCondition | None = None,
    truncation_cond: DoneCondition | None = None,
    shared_info_provider: SharedInfoProvider | None = None,
    renderer: Renderer | None = None,
)
```

Its contract is fully dictionary/generic — `agents: List[AgentID]`, `step(actions: Dict[AgentID, ActionType]) -> (obs, rewards, is_terminated, is_truncated)` — with no Rocket League specifics baked in. `AgentID` is conventionally a `str` like `"blue-0"`.

**[V]** `termination` and `truncation` are **separate** channels, and this is semantically load-bearing: a goal is a *termination* (no bootstrap), a timeout is a *truncation* (bootstrap the value). Getting this wrong silently biases the critic. `rlgym-learn-algos`' `MultiAgentController.choose_env_actions` default implementation resets an env when `all(terminated or truncated_dict[agent_id] ...)`, and the changelog notes `truncated[-1] = 1 if done[-1] == 0 else 0` bookkeeping.

**[V]** `shared_info_provider` is the mechanism for cross-component state (e.g. ball prediction, win probability) computed once per step and visible to obs/reward/done. `RocketSimEngine.config` is a `Dict[str, Any]` with a TODO to allow hooking RocketSim mutators through it — so **mutators must be set through the RLGym `GameConfig`, not RocketSim's `MutatorConfig`** (see §4.5).

### 4.3 Observation — exact layout, computed

**[V]** `rlgym/rocket_league/obs_builders/default_obs.py`. `DefaultObs.get_obs_space()` returns:

```python
'real', 52 + 20 * self.zero_padding * 2      # if zero_padding is not None
'real', 52 + 20 * len(self._state.cars)      # otherwise
```

Decomposing the constant 52 from `_build_obs` **[V+C]**:

| Block | Width |
|---|---|
| `ball.position` | 3 |
| `ball.linear_velocity` | 3 |
| `ball.angular_velocity` | 3 |
| `boost_pad_timers` | **34** — soccar pad count, confirmed by counting `common_values.BOOST_LOCATIONS` |
| partial observables: `is_holding_jump, handbrake, has_jumped, is_jumping, has_flipped, is_flipping, has_double_jumped, can_flip, air_time_since_jump` | 9 |
| **subtotal** | **52** |

Per car `[V]`: `position(3) + forward(3) + up(3) + linear_velocity(3) + angular_velocity(3) + [boost, demo_respawn_timer, on_ground, is_boosting, is_supersonic](5)` = **20**.

Therefore **[C]**:

| Mode | `zero_padding` | obs width |
|---|---|---|
| 1v1 | `None` or `1` | **92** |
| 2v2 | `None` or `2` | **132** |
| 3v3 | `None` or `3` | **172** |

The 1v1 figure of **92** is independently corroborated: `rlgym-learn`'s own `speed_test.py` declares `NumpySerdeConfig.STATIC(shape=(92,))` for a 1v1 env with `zero_padding=1` **[V]**.

Two important consequences:

- **`zero_padding` exists so one network can serve multiple game modes** — the obs is zero-padded to a fixed width. Cost: wasted input dimensions and an obs that no longer matches a checkpoint trained without padding. **Changing it invalidates every checkpoint.**
- **`DefaultObs` emits `float64`.** `np.concatenate` over a list of Python floats/float32 arrays widens to float64. `rlgym-learn`'s `speed_test.py` works around this with a `Float32DefaultObs(DefaultObs)` subclass overriding `_build_obs` to `.astype(np.float32)` **[V]**. Doing the same halves IPC payload and experience-buffer memory (§8.4).
- Default normalization coefficients **[V]**: `pos_coef=1/2300`, `ang_coef=1/π`, `lin_vel_coef=1/2300`, `ang_vel_coef=1/π`, `pad_timer_coef=1/10`, `boost_coef=1/100`. Every RLGym tutorial overrides `pos_coef`/`lin_vel_coef` to per-axis values (`1/SIDE_WALL_X`, `1/BACK_NET_Y`, `1/CEILING_Z`) — that is a *convention*, not a requirement, and it changes the numeric scale, so it too is a checkpoint-breaking change.

**Known gaps in `DefaultObs`** **[V]** (absent from the source): no ball prediction, no car-relative (egocentric) positions/velocities, no time-remaining, no score. The ZealanL guide recommends adding car-relative terms but flags obs changes as *"very very high-risk"* **[R]**. Nexto does **not** use `DefaultObs` — it uses a custom `NextoObsBuilder` **[V]** (`Necto/rlbot-support/Nexto/nexto_obs.py`, `JPK314/doublesize-nexto-v5/src/nexto_obs.py`).

### 4.4 Actions, control rate, done conditions

**[V]** `LookupTableAction` docstring: *"World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from **1944 to 90**."* `get_action_space` returns `('discrete', 90)`. The 8 control channels are `THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE` (`common_values.NUM_ACTIONS = 8`).

The table is built in two blocks **[V]**: 36 ground combinations (throttle × steer × boost × handbrake, pruned where `boost==1 and throttle!=1`) plus aerial combinations (pitch × yaw × roll × jump × boost, pruned where `jump==1 and yaw!=0`, where `pitch==roll==jump==0` duplicates a ground action, with handbrake enabled on jump+orientation to allow wavedashes). This is *byte-identical in construction* to Necto's `NectoAction.make_lookup_table()` **[V]** — the 90-action space is the de facto community standard, and reusing it means a Nexto-trained baseline is comparable.

**[V]** `RepeatAction(parser, repeats=8)` emulates tick skip by repeating each parsed action across 8 rows; `RocketSimEngine.step` then iterates those rows one tick at a time. 120 / 8 = **15 decisions per second**, i.e. a minimum reaction latency of 66.7 ms. The ZealanL guide notes this makes bots far more reactive than humans (~200–300 ms), and that this is *why* dribbling and late flicks emerge naturally **[R]**.

**[V]** Done conditions available: `GoalCondition`, `TimeoutCondition(timeout_seconds)`, `NoTouchTimeoutCondition(timeout_seconds)`, `NoopCondition`, plus `AnyCondition(*conditions)` / `AllCondition` combinators. Note the parameter is `timeout_seconds` in v2 (`rlgym_v2_example.py` in rlgym-ppo uses the older `timeout=` spelling and would fail against current RLGym — a live example of tutorial drift, §4.6).

#### ⚠ 4.4.1 `NoTouchTimeoutCondition` does not do what its name says — verified from source

`rlgym/rocket_league/done_conditions/no_touch_condition.py` **[V]**:

```python
def is_done(self, agents, state, shared_info):
    if any(car.ball_touches > 0 for car in state.cars.values()):
        self.last_touch_tick = state.tick_count
        done = False
    else:
        time_elapsed = (state.tick_count - self.last_touch_tick) / TICKS_PER_SECOND
        done = time_elapsed >= self.timeout_seconds
```

`ball_touches` is the **cumulative** touch count for the episode, and `reset()` only records the starting tick. So the predicate is *"has **any** car **ever** touched the ball this episode"* — not *"has it been `timeout_seconds` since the last touch"*. **Once any car touches the ball, this condition returns `False` forever** and never truncates again. Its effective meaning is "30 s of no contact *from kickoff*", which is a reasonable but different rule.

**Consequence [C]:** with `TimeoutCondition(300)` as the only other truncation, the maximum episode length is 300 s × 120 tps / 8 ticks-per-decision = **4 500 agent steps**, and idle episodes are *not* cut short after the first touch. If genuine idle truncation is wanted, a custom `DoneCondition` that tracks each car's previous `ball_touches` and compares deltas is required (~15 lines, same pattern as `KickoffMutator`).

**[V]** `rlgym.rocket_league.reward_functions` ships exactly three: `GoalReward`, `TouchReward`, `CombinedReward`. Everything else is user-written.

#### ⚠ 4.4.2 `TouchReward` is a dense cumulative signal, not a one-shot — verified from source

`rlgym/rocket_league/reward_functions/touch_reward.py` **[V]**:

```python
def reset(self, agents, initial_state, shared_info) -> None:
    pass
def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
    return {agent: 1. if state.cars[agent].ball_touches > 0 else 0. for agent in agents}
```

Because `reset()` is a no-op and the predicate is on the **cumulative** count, `TouchReward` returns `+1` on **every agent step after the first touch**, for the rest of the episode. It is not "reward once per touch".

**[C] Magnitude at the commonly copied weight 0.1** (compared against `GoalReward × 10` = ±10):

| Episode length (agent steps) | First touch at step 50 | step 100 | step 300 | Ratio to one goal |
|---|---|---|---|---|
| 600 (40 s) | +55.0 | +50.0 | +30.0 | 3.0 – 5.5 × |
| 1 500 (100 s) | +145.0 | +140.0 | +120.0 | 12 – 14.5 × |
| 4 500 (300 s, `TimeoutCondition(300)` max) | +445.0 | +440.0 | +420.0 | 42 – 44.5 × |

**Conclusion [E]:** the widely copied `CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.1))` is **not** "goal-dominant with a tiny nudge" — the touch term outweighs the objective by one to two orders of magnitude and directly incentivises touching the ball once and then stalling. PILOT's documented `touch 0.1` constant **[R]** belongs to its own reward implementation and must not be assumed to have the same semantics.

**Required fix [E]:** implement a delta-based `TouchEventReward` that stores the previous per-agent `ball_touches` in `reset()` and returns `max(0, current − previous)` — the one-shot-per-touch semantics the name implies. The `RewardFunction` interface needed is `reset(agents, initial_state, shared_info) -> None` and `get_rewards(agents, state, is_terminated, is_truncated, shared_info) -> Dict[AgentID, float]` **[V]**, both visible in `goal_reward.py`. Note that `GoalReward` itself *is* correctly one-shot (it reads `state.goal_scored`, which is only true on the scoring step, and returns `+1` for the scoring team / `−1` for the conceding team — a verified zero-sum form).

### 4.5 State setting — the biggest functional gap in RLGym v2

**[V]** `rlgym/rocket_league/state_mutators/` contains exactly three files: `fixed_team_size_mutator.py`, `kickoff_mutator.py`, `mutator_sequence.py`.

**There is no random / general state setter in RLGym v2.** RLGym v1 had `RandomState`; the ZealanL guide explicitly recommends it for early training (*"For beginner bots, kickoff is usually not the best state setter. I recommend using the `RandomState` state setter, especially in the early stages"* — **[R]**) and it does not exist in v2. `rlgymppo_rs` has "Several state setters (kickoff, random positions) weighted by probability" **[V]**; RLGym v2 has none.

**Consequence [E]:** a custom `StateMutator` for randomized mid-field / aerial / defensive situations is **required work, not optional**, for both curriculum learning (§9.5) and avoiding kickoff over-fitting. It is small (a `StateMutator` is one `apply(state, shared_info)` method that writes into a plain dataclass) but it must be written, tested and *versioned*, because it changes the state distribution the policy sees.

**[V]** `KickoffMutator` details worth knowing: 5 fixed spawn positions per team with a shuffled assignment, `car.boost_amount = 33.3`, ball at `[0, 0, BALL_RESTING_HEIGHT]` with zero velocity. `FixedTeamSizeMutator` hard-codes `car.hitbox_type = OCTANE` for every car and asserts `len(state.cars) == 0`, so it cannot be composed with another team-size mutator. **No hitbox randomization exists**, which is a known domain gap versus the real game (§7.6).

**[V]** `GameConfig` exposes only `gravity`, `boost_consumption`, `dodge_deadzone`. `RocketSimEngine.set_state` maps these onto `rsim.MutatorConfig`. RocketSim's own `MutatorConfig` has 20+ tunables (gravity, boost rate, bump force scale, …) per the v2.1.0 notes — **most are not reachable through RLGym v2** without bypassing the engine.

**[V]** `RocketSimEngine.__init__(rlbot_delay=True, game_mode=rsim.GameMode.SOCCAR)`. `rlbot_delay=True` steps the arena once *before* applying controls, which "forces the first action of the episode to no-op" — an intentional match to RLBot's one-tick input latency. **Keep it on**; turning it off would give the policy a reaction advantage it will not have in the real game.

**[V]** `set_state()` removes **all** cars from the arena and re-adds them on every reset (`for car in self._arena.get_cars(): self._arena.remove_car(car)`), with a `# TODO reuse cars?` comment. Episode resets are therefore more expensive than they look; short episodes amplify this. Benchmark **B2** should report resets/s alongside steps/s.

### 4.6 v1 → v2 migration: what the tutorials get wrong

This table exists because the majority of discoverable Rocket League RL material predates v2.

| Topic | Old (v1 / `rlgym-sim`) | Current (v2) | Evidence |
|---|---|---|---|
| Env construction | `rlgym.make(...)` / `rlgym_sim.make(...)` with ~15 kwargs | `rlgym.api.RLGym(state_mutator=…, obs_builder=…, action_parser=…, reward_fn=…, termination_cond=…, truncation_cond=…, transition_engine=…)` | **[V]** §4.2 |
| RocketSim backend | `rlgym-sim` (`AechPro/rocket-league-gym-sim`), self-described *"a TEMPORARY STOP-GAP to use RocketSim while RLGym 2.0 is in development"* | `rlgym.rocket_league.sim.RocketSimEngine` | **[V]** rlgym-sim README |
| Collision meshes | Download `RLArenaCollisionDumper`, run against a game install, copy `collision_meshes/` next to your script | Shipped inside `rlgym-rocket-league` as `package_data` | **[V]** §3.4 |
| Terminal conditions | `terminal_conditions=[NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]`, tick-based | Separate `termination_cond` / `truncation_cond`, **seconds**-based (`timeout_seconds=`). ⚠ `NoTouchTimeoutCondition` tests *"has any car ever touched"* rather than *"time since last touch"* (§4.4.1) | **[V]** §4.4 |
| Action parsers | `ContinuousAction` default; `DiscreteAction` is actually MultiDiscrete; fully-discrete `LookupAction` lives in `rlgym-tools` | `LookupTableAction` (90) and `ContinuousAction` ship in core. **`MultiDiscreteFF` was explicitly removed** from `rlgym-learn-algos` | **[V]** rlgym tree + algos CHANGELOG |
| Reward functions | `EventReward`, `VelocityPlayerToBallReward`, `VelocityBallToGoalReward`, `FaceBallReward`, `SaveBoostReward`, `LiuDistanceReward`, … | Only `GoalReward`, `TouchReward`, `CombinedReward` — and `TouchReward` has cumulative rather than per-event semantics (§4.4.2). **Everything else must be re-implemented, including a usable touch reward.** | **[V]** §4.4 |
| State setters | `DefaultState`, `RandomState`, `KickoffState` | `KickoffMutator`, `FixedTeamSizeMutator`, `MutatorSequence` only | **[V]** §4.5 |
| Obs builders | `DefaultObs`, `AdvancedObs` | `DefaultObs` only (`AdvancedObs` survives in `rlgym-compat` and `rlgymppo_rs`, not in RLGym v2) | **[V]** rlgym tree |
| Game state | `rlgym.utils.gamestates.GameState`, `PlayerData`, `state.players` list | `rlgym.rocket_league.api.GameState`, `state.cars: Dict[AgentID, Car]`, `state.ball`, `state.inverted_ball` | **[V]** §4.2 |
| Constants | `rlgym.utils.common_values` | `rlgym.rocket_league.common_values` | **[V]** |
| Deployment shim | `rlgym_compat` importing `rlgym` | `rlgym_compat` importing `rlgym.rocket_league` (JPK314 fork is the maintained one) | **[V]** §7.3 |

**[V]** Concrete live example of drift: `AechPro/rlgym-ppo/rlgym_v2_example.py` calls `NoTouchTimeoutCondition(timeout=timeout_seconds)`, but current RLGym's signature is `timeout_seconds=`. Copying that example verbatim today fails.

---

## 5. Learners — the decisive choice

### 5.1 RLGym-PPO — do not build on it

| Item | Value | Evidence |
|---|---|---|
| Repo | `AechPro/rlgym-ppo`, Apache‑2.0, 17 stars | **[V]** GitHub API |
| Version | 1.3.13 | **[V]** `setup.py` |
| **Last push** | **2025‑01‑06** — ~21 months before this research | **[V]** GitHub API |
| Python | `>=3.7` | **[V]** `setup.py` |
| Deps | `cloudpickle>2.0, filelock>3.0, gym>0.23, gym-notices==0.0.8, Jinja2>3.0, mpmath>1.0, numpy>1.21, sympy>1.10, torch>1.13, wandb>0.15, typing_extensions>4.6` | **[V]** `setup.py` |
| `requirements.txt` | additionally pins **`numpy>1.21, <2.0`** | **[V]** |

**Process model [V]** (`batched_agent_manager.py`, `batched_agent.py`): `n_proc` OS processes, one RLGym env each; each child binds a UDP socket on `127.0.0.1:0` and talks to the parent; bulk arrays move through a shared-memory buffer (`shm_buffer_size`, default 8192). The **parent process holds the policy and does all inference**, batching up to `min_inference_size` observations per forward pass. Collection and learning are *sequential* — `collect_timesteps(n)` then `ppo_learner.learn(batch)` — so GPU and CPU are never overlapped.

**Full `Learner` signature [V]** — the tuning surface: `env_create_function, metrics_logger, n_proc=8, min_inference_size=80, render=False, render_delay=0, timestep_limit=5e9, exp_buffer_size=100000, ts_per_iteration=50000, standardize_returns=True, standardize_obs=True, max_returns_per_stats_increment=150, steps_per_obs_stats_increment=5, policy_layer_sizes=(256,256,256), critic_layer_sizes=(256,256,256), continuous_var_range=(0.1,1.0), ppo_epochs=10, ppo_batch_size=50000, ppo_minibatch_size=None, ppo_ent_coef=0.005, ppo_clip_range=0.2, gae_lambda=0.95, gae_gamma=0.99, policy_lr=3e-4, critic_lr=3e-4, log_to_wandb, checkpoints_save_folder, add_unix_timestamp=True, checkpoint_load_folder="latest", save_every_ts=1e6, instance_launch_delay, random_seed=123, n_checkpoints_to_keep=5, shm_buffer_size=8192, device="auto"`.

**Why it is the wrong foundation [V]:**

1. **No self-play.** There is exactly one `self.policy`, shared by `self.agent` and `self.ppo_learner`. Every agent id in every environment is driven by the current policy. There is no opponent pool, no league, no historical-checkpoint evaluation inside training. You would have to fork it.
2. **Hard dependency conflict.** `rlviser-py` 0.6.14 requires `numpy>=2.0.0`; `rlgym-ppo/requirements.txt` requires `numpy<2.0`. Installing `rlgym[rl-rlviser]` and `rlgym-ppo` in one environment is unsatisfiable **[V]** PyPI metadata. (Its `setup.py` `install_requires` is looser — `numpy>1.21` — so the conflict depends on which file the installer honours. That ambiguity is itself a reason to avoid it.)
3. **`gym>0.23`.** The `gym` package's final release is **0.26.2** and it is unmaintained (superseded by Gymnasium in 2022) **[V]** PyPI. It is imported inside `batched_agent.py`. On Python 3.12+ this needs `setuptools` present and is a permanent maintenance liability.
4. **Frozen.** 21 months without a push, 5 open issues, while `rlgym-learn` shipped 2.0.0 in August 2026.

**What to keep from it:** the *iteration report* vocabulary it established — `Policy Reward`, `Policy Entropy`, `Value Function Loss`, `Mean KL Divergence`, `SB3 Clip Fraction`, `Policy Update Magnitude`, `Collected Steps per Second`, `Overall Steps per Second`, `Timestep Collection Time`, `Timestep Consumption Time` **[V]** RLGym-PPO-Guide `intro.md`. `rlgym-learn-algos` logs the same quantities (§5.2), so the guide's diagnostic advice transfers even though its install instructions do not. Its reported example values (`Collected: 8 581 steps/s`, `Overall: 5 977 steps/s`) are on an **unspecified machine with an unspecified `n_proc`** and must not be used as a target **[R]**.

### 5.2 rlgym-learn + rlgym-learn-algos — recommended

| Item | `rlgym-learn` | `rlgym-learn-algos` |
|---|---|---|
| Repo | `JPK314/rlgym-learn`, Apache‑2.0 | `JPK314/rlgym-learn-algos`, Apache‑2.0 |
| Version | **2.0.0** (2026‑08‑29) | **0.4.2** (2026‑09‑06) |
| Last push | 2026‑08‑29 | 2026‑09‑06 |
| Python | `>=3.10, <=3.14` | `>=3.10, <=3.14` |
| Deps | `rlgym>=2.0.0`, `pydantic>=2.13.0`, `typing-extensions>=4.15.0`; extras `rl: rlgym[rl], numpy>1.21`, `tensor: torch>1.13` | `pydantic>=2.8.2, numpy>1.21, torch>1.13, typing-extensions>4.6, rlgym>=2.0.0, rlgym-learn>=2.0.0`; extra `wandb` |
| Wheels | manylinux (i686, x86_64, aarch64, armv7l, ppc64le), macOS, Windows | same |
| Build | `maturin` (Rust/PyO3); building from source needs a Rust toolchain | same |

All values **[V]** from PyPI JSON + `pyproject.toml`.

**Note the absence of an upper numpy pin** — unlike rlgym-ppo, this stack is compatible with numpy 2.x, so `rlgym[rl-rlviser]` works alongside it. **[V]**

Self-described feature list **[V]** README: *"Full support for all generics of the RLGym v2 API · Fast parallelization of environments using Rust and shared memory · Support for metrics gathering from environments · Detailed checkpointing system · File-based configuration · Provided optimized PPO implementation · **Allows multiple learning algorithms to provide actions for agents within an environment** · Multi-platform (Windows, Linux)"*.

It is an explicit successor: *"This project was built using Matthew Allen's wonderful RLGym-PPO as a starting point… All of his files which remain similar have been refactored out to rlgym-learn-algos."* **[V]**

#### 5.2.1 Process architecture — verified

**[V]** `python/rlgym_learn/env_processing/env_process.py` delegates immediately to `rust_env_process_fn(proc_id, parent_addr_str, build_env_fn, flinks_folder, serde_types, render_this_proc, render_delay, recalculate_agent_id_every_step)`. `src/env_process.rs` uses `mio::net::UdpSocket`, `mio::Poll`, `shared_memory::ShmemConf` and `raw_sync` events. So: **N OS processes, one RLGym env each, UDP control channel + shared-memory data channel, handshake coordinated through files in `flinks_folder` (default `"shmem_flinks"`).**

This is why the Windows failure mode is what it is: `mio`'s epoll backend on Linux vs. IOCP on Windows. PILOT documents `FileExistsError: entity already exists` at startup on native Windows, unfixed in 2.0.0 **[R]** §5.2.4.

#### 5.2.2 The self-play primitive — verified

**[V]** `rlgym_learn_algos/agent_controller/multi_agent/multi_agent_controller.py`:

```python
class MultiAgentController(AgentController[...], ABC):
    def __init__(self, subcontrollers: Mapping[str, MultiAgentSubcontroller[...]]): ...

    @abstractmethod
    def choose_subcontrollers(self, agent_ids: dict[int, list[AgentID]]) -> dict[int, list[str]] | None:
        """For each env_id, a list of subcontroller names parallel to the AgentID list."""

    @abstractmethod
    def choose_env_actions(self, env_state_info_dict) -> tuple[int, dict[int, EnvActionResponse]]:
        """Decide RESET / STEP / DEFER / CLOSE per env; may also spawn new envs."""
```

`rlgym-learn`'s own `multi_agent_controller_test.py` **[V]** shows the canonical use — two independently-trained PPO controllers split by team:

```python
agent_subcontrollers = {"PPO1": PPOAgentController(...), "PPO2": PPOAgentController(...)}

class SimpleMultiAgentController(MultiAgentController):
    def choose_subcontrollers(self, agent_ids):
        return {env_id: ["PPO1" if "blue" in agent_id else "PPO2" for agent_id in ids]
                for env_id, ids in agent_ids.items()}
```

**This one method is the entire reason to choose this stack.** Every advanced technique in §9 — opponent pools, league play, in-training evaluation against a frozen baseline, asymmetric blue/orange policies — is a different implementation of `choose_subcontrollers`, with zero changes to the rollout machinery.

Note that a **subcontroller does not have to learn.** `PPOAgentController` is a `MultiAgentSubcontroller`; so is anything that implements the interface. A frozen-checkpoint inference-only subcontroller is therefore a first-class citizen — that is the evaluation-opponent and opponent-pool mechanism.

#### 5.2.3 Configuration and PPO knobs — verified

Config is **file-based and pydantic-validated** (`generate_config(..., config_location="config.json")`, `LearningCoordinator(config=...)` or `config_location=...`), which is a significant reproducibility win over rlgym-ppo's constructor kwargs: the exact run configuration is a serializable artifact.

**[V]** `ProcessConfigModel`: `n_proc=8, min_frac_process_responses_per_collection=0.45, render=False, render_delay=None, launch_delay=None, recalculate_agent_id_every_step=False`.
**[V]** `BaseConfigModel`: `serde_types, random_seed=123, shm_buffer_size=16384, flinks_folder="shmem_flinks", timestep_limit=5_000_000_000`.
**[V]** `PPOAgentControllerConfigModel`: `timesteps_per_iteration=50000, save_every_ts=1_000_000, run_suffix=<time_ns>, checkpoint_load_folder=None, n_checkpoints_to_keep=5, random_seed=123, save_mid_iteration_data_in_checkpoint=True, learner_config, experience_buffer_config, run_name="rlgym-learn-run", metrics_logger_config=None`.
**[V]** `PPOLearnerConfigModel`: `dtype=torch.float32, n_epochs=1, batch_size=50000, n_minibatches=1, ent_coef=0.005, clip_range=0.2, max_grad_norm=0.5, optimizer_named_parameter_group_kwargs={"actor":{"lr":3e-4},"critic":{"lr":3e-4}}, advantage_standardization=True, device="cpu", cudnn_benchmark_mode=True`.
**[V]** `GAETrajectoryProcessorConfigModel`: `gamma=0.99, lmbda=0.95, standardize_rewards=True, max_returns_per_stats_increment=None, reward_clip=10`.
**[V]** `ExperienceBufferConfigModel`: `max_size`, `trajectory_processor_config`, `device`.

⚠️ **`PPOLearnerConfigModel.device` defaults to `"cpu"`** and `validate_default=True` — you must explicitly pass `device="cuda:0"`. Silently training on CPU is the most likely first-time failure and the hardest to notice if you are only watching that "something is running".

Algorithmic details worth recording **[V]**:
- Critic loss is **plain `nn.MSELoss()`** — there is **no value-function clipping** (`ppo_learner.py`: `self.critic_loss_fn = torch.nn.MSELoss()`). This is the "no clip" PPO variant; expect the critic to be freer to move and monitor `critic_loss`.
- Advantage standardization is on by default and is a *separate* flag from reward standardization.
- Reward standardization changed semantics in 0.4.x **[V]** CHANGELOG: it now computes true returns over all trajectories and divides by the standard deviation of unstandardized/unclipped returns, rather than the OpenAI running-return-std wrapper. `reward_clip` is now a separate optional field instead of a hard-coded `[-10, 10]`.
- `max_returns_per_stats_increment` now defaults to `None` (use all returns) rather than 150.
- `ContinuousActor` was reworked to tanh-squashing with an affine transform for arbitrary finite ranges; `MultiDiscreteFF` was **removed**.
- `standardize_returns` was **renamed** to `standardize_rewards`. Copying old configs will fail validation (`extra="forbid"` on every config model — typos are hard errors, which is good).
- The experience-buffer checkpoint format changed from `.pkl` to `.zip` in 0.4.x.

`DiscreteFF` **[V]** is `Linear→ReLU` stacks ending in `Linear→Softmax`, with `torch.clamp(probs, 1e-11, 1)`. `BasicCritic` is the value head. `SeparateActorCritic(actor, critic)` is the default composition, with independent optimizers built by your `optimizers_factory`.

**Logged diagnostics [V]** — `PPOData`: `batch_consumption_time, cumulative_model_updates, actor_entropy, kl_divergence, critic_loss, sb3_clip_fraction, actor_update_magnitude, critic_update_magnitude`. `PPOAgentControllerData`: `ppo_data, trajectory_processor_data, cumulative_timesteps, iteration_time, timesteps_collected, timestep_collection_time`. `GAETrajectoryProcessorData`: `average_reward, average_undiscounted_episodic_return, average_return, return_standard_deviation`.

**That set is exactly what is needed to diagnose stalled learning** (§9.7): the collection/consumption time split locates the bottleneck, `kl_divergence` + `sb3_clip_fraction` detect too-large updates, `actor_entropy` collapse detects premature convergence, and `return_standard_deviation` detects reward-scale drift.

**Checkpoint layout [V]**: `agent_controller_save_folder/<run_name><run_suffix>/<time_ns>/` containing `ppo_agent.json`, `ppo_learner/{actor_critic.pt, optimizers.pt, misc.json}`, `experience_buffer/` (zip), `metrics_logger/`, and optionally `iteration_trajectories.pkl` / `iteration_shared_infos.pkl` (governed by `save_mid_iteration_data_in_checkpoint`). ⚠️ README warning, verbatim: *"WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER."* — so the checkpoint folder must never contain anything else, including your own artifacts.

`rlgym_learn_algos/conversion/convert_rlgym_ppo_checkpoint.py` **[V]** exists to migrate RLGym-PPO checkpoints forward — useful if you ever need to inherit a public run, not needed here.

#### 5.2.4 Known problems with rlgym-learn — all documented by third parties

1. **Native Windows does not work.** PILOT: *"`rlgym-learn` 2.0.0's env-process backend crashes on native Windows (`FileExistsError: entity already exists` — an epoll/IOCP mismatch in its Rust socket registration, unfixed upstream). On Windows, run everything below inside WSL2."* **[R]** — consistent with the verified Rust/`mio` implementation **[V]**. **For this project WSL 2 is already the environment, so this is a non-issue — but it means the Windows-side evaluation harness cannot share the training code path.**
2. **Documentation is missing.** `https://rlgym.org/RLGym%20Learn/introduction/` reads verbatim: *"This is an upcoming project that is not finished yet. It will be a generic framework for reinforcement learning algorithms that can be used with RLGym. When available, this webpage will be updated."* **[V]** The rlgym-learn README itself says *"See the RLGym website for complete documentation… **[COMING SOON]**"*. The de facto documentation is `quick_start_guide.py`, `speed_test.py`, `speed_test_heavy.py`, `multi_agent_controller_test.py` and `CHANGELOG.md`. **Plan for source-reading, not doc-reading.**
3. **`OPENBLAS_NUM_THREADS` must be pinned.** Both `quick_start_guide.py` and `speed_test.py` open with `os.environ["OPENBLAS_NUM_THREADS"] = "1"` and the comment *"needed to prevent numpy from using a ton of memory in env processes and causing them to throttle each other"* **[V]**. Omitting this is a silent, large throughput and memory regression.
4. **`recalculate_agent_id_every_step` must stay `False`** — the CHANGELOG states `PPOAgentController` *"now enforces that `recalculate_agent_id_every_step` is false in `ProcessConfig` as it breaks how the controller builds up trajectories"* **[V]**.
5. **The upstream example has a latent bug worth not copying.** `quick_start_guide.py` sets `team_size = 2` and `zero_padding=team_size` (⇒ obs width 132) while its `ObsType` alias and the surrounding comments still say 92 **[V]**. Since the serde shape must match the actual obs, this is a trap. Always derive the serde shape from `DefaultObs.get_obs_space()`, never hard-code it.

### 5.3 rocket-learn (Necto's trainer) — historical, not a starting point

**[V]** `Rolv-Arild/rocket-learn`, 29 stars, last push 2025‑11‑06, no license file declared (`license: None`). README: *"a machine learning framework specifically designed for Rocket League Reinforcement Learning. It works in conjunction with Rocket League, RLGym, and Bakkesmod."* Features: PPO, *"Distributed compute from multiple computers"*, *"Automatic saving of and training against previous agent versions"*, *"Trueskill progress tracking"*, *"Training against Hardcoded/Pretrained Agents"*, *"Training against Humans"*, wandb logging.

Its own README is refreshingly blunt about when *not* to use it **[V]**: *"You should use Stable Baselines3 (SB3) to make your bot at first… If you don't yet have these, rocket-learn will add a large amount of complexity for no added benefit."* Setup requires **Redis**, and the README warns *"Improper Redis setup can leave your computer extremely vulnerable to Bad Guys."*

**Necto's pinned environment [V]** (`Necto/training/requirements.txt`): `rlgym==1.2.0`, `torch==1.11.0`, `numpy==1.22.4`, `gym==0.24.0`, `EARL-pytorch==0.5.1`, `redis==4.3.3`, `trueskill==0.4.5`, `wandb==0.12.18`. This is a 2022-era RLGym v1 stack. **Not usable as-is today** — but the *ideas* (TrueSkill against historical versions, pretrained/scripted opponents, distributed workers) are exactly what `MultiAgentController` now provides natively, minus Redis.

### 5.4 rlgymppo_rs — the high-performance alternative, with a licensing caveat

**[V]** `Virxec/rlgymppo_rs`, **GPL‑2.0**, 5 stars, last push 2026‑10‑15, Rust. Built on **RocketSim v3** (`ZealanL/RocketSim` branch `v3-rust`) + `VirxEC/rlgym_rs`.

Its README describes capabilities that `rlgym-learn` does not have **[V]**:
- **Threads instead of processes**: `num_threads: 4, num_games_per_thread: 64` ⇒ 256 concurrent games, avoiding per-process Python interpreter and memory overhead entirely.
- `SelfPlayConfig { save_policy_versions: true, ts_per_version: 100_000_000 }`.
- `SkillTrackerConfig` — **Elo ratings evaluated in-loop** against randomly-selected saved policy versions and against a fixed Nexto baseline (`nexto_mmr: Some(1500.0)`). Two-sided Elo; each saved version keeps its own rating; Nexto's MMR is fixed.
- Several state setters (kickoff, random positions) weighted by probability — the thing RLGym v2 lacks.
- **Policy distillation** (`transfer_learn`, `TeacherConfig`, `TransferLearnConfig`): distil a large teacher into a small student for deployment, with `Transfer/loss` and `Transfer/accuracy` (argmax agreement) metrics. This is precisely the deployment-size problem in §7.
- Seven ML backends via Burn (`torch`/LibTorch is "the most mature and fastest"), `cuda`, `wgpu`, `metal`, `rocm`, `flex`, `candle`.

**Assessment [E]:** technically the most capable option and almost certainly faster per core, but: (a) **GPL‑2.0** is copyleft and would apply to the whole training system if statically linked, which is a policy decision, not a technical one; (b) 5 stars and months old, so the bus factor is one maintainer; (c) it targets RocketSim v3, which has no PyPI wheel; (d) it is Rust, so every custom reward/obs/state-setter is Rust. **Recommendation: not for Phase 1.** Keep it as the documented escape hatch if benchmark **B3** shows the Python per-step path is the hard ceiling and the project needs 3–10× more throughput. Its design (thread-per-N-games, in-loop Elo, distillation) is worth copying conceptually into the Python architecture regardless.

### 5.5 Stable Baselines3 / RL-Zoo3 — not suitable

**[V]** `stable-baselines3` is actively maintained (v2.9.x, Dec 2025) and `rl-zoo3` 2.7.0 exists. But SB3's `PPO` expects a `gymnasium.Env`. RLGym v2 is deliberately *not* a gymnasium env — it is a multi-agent, per-agent-id dictionary API. `rlgym-ppo` provides `RLGymV2GymWrapper` to flatten it, at the cost of losing per-agent identity. SB3 has no notion of assigning different policies to different agents inside one env, no shared-memory multi-process RocketSim collector, and no opponent-pool concept. Necto's own author recommends SB3 *for learning the fundamentals*, then moving off it **[V]** §5.3. **Not recommended** beyond an optional sanity check that "PPO on this obs/reward learns anything at all".

### 5.6 Learner comparison

| | **rlgym-ppo 1.3.13** | **rlgym-learn 2.0.0** | rocket-learn | rlgymppo_rs |
|---|---|---|---|---|
| RLGym v2 | via wrapper | native | **no (v1.2.0)** | own Rust env |
| Maintained | **frozen 2025‑01** | **yes (2026‑08)** | dormant | yes (2026‑10) |
| License | Apache‑2.0 | Apache‑2.0 | **none declared** | **GPL‑2.0** |
| Self-play / opponent pool | **no** | **yes — `MultiAgentController`** | yes (previous versions) | yes (`SelfPlayConfig`) |
| In-training skill rating | no | no (must build) | TrueSkill | **Elo, built in** |
| Parallelism | N processes, Python | N processes, Rust-backed | Redis workers | **N threads × M games** |
| Config artifact | kwargs | **JSON file, pydantic** | code | Rust struct |
| Checkpointing | basic | detailed, resumable | yes | yes |
| Docs | guide (v1-era) | **none yet** | README | good README |
| Python / numpy 2 | `gym` + numpy<2 conflict | ✅ 3.10–3.14, numpy 2 OK | 3.x pinned 2022 | n/a |
| **Verdict** | reference only | **✅ adopt** | mine for ideas | escape hatch |

---

## 6. Nexto, Necto and other reference agents

### 6.1 Necto / Nexto — documented facts vs. claims

**[V]** `Rolv-Arild/Necto`, 302 stars, last push **2024‑09‑17**, license **`NOASSERTION`** (a non-standard `LICENSE` file — see §11). README, verbatim:

> So far, we've made 3 versions: **V1: Necto** — Around Diamond level. **V2: Nexto** — Approximately Grand Champion 1 level in 1v1, 2v2 and 3v3 (top 0.12%, 0.95%, 0.46% of the playerbase respectively). **V3: Tecko** — Canceled due to lack of improvement. **The project is not being worked on anymore.**

Everything in that paragraph is a **self-reported [R]** skill claim measured informally, not a peer-reviewed or independently reproduced result. Treat "GC1" as a directional statement about what the architecture achieved, not a target this project can be held to.

**Architecture — verified from source [V]** (`Necto/training/agent.py`):

```python
def get_critic():
    return Necto(EARLPerceiver(256, 4, 8, 1, query_features=36, key_value_features=25 + 30),
                 Linear(256, 1))
def get_actor():
    split = (90,)
    return DiscretePolicy(Necto(EARLPerceiver(256, 4, 8, 1, query_features=36, key_value_features=55),
                                ControlsPredictorDot(256)), split)
```

So Nexto's actor/critic is an **EARL Perceiver** — a Perceiver-style cross-attention model (`EARL-pytorch`, inspired by arXiv:2103.03206) with 256 latent dimensions, 4 latent vectors, 8 heads, 1 block, 36 query features, 55 key/value features — feeding a `ControlsPredictorDot` head that scores all **90** discrete actions by dot product against a learned action embedding (`mlp(8, 256, 1, 32)`; the `8` is the raw control-vector width). Initialization is `xavier_uniform_` for all `dim > 1` parameters. Adam with separate actor/critic parameter groups.

`EARL-pytorch` README **[V]** documents the entity-based input: ~21 features per entity (ball / boost / player), with `pos` and `vel` divided by 2300, `ang_vel` by 5.5, boost by 100, plus a `CLS` token used to summarize the game state. This is why Nexto generalizes across 1s/2s/3s with one network: **the input is a variable-length set of entities, not a fixed-width vector.** `DefaultObs` cannot do this.

**Reward — verified from source [V]** (`Necto/training/reward.py`). `NectoRewardFunction` default weights:

| Term | Weight | Type |
|---|---|---|
| `goal_w` | 10 | event |
| `win_prob_w` | 10 | potential (scoreboard-derived win probability) |
| `goal_dist_w` | 10 | potential (exp distance ball→goals) |
| `goal_speed_bonus_w` | 2.5 | event |
| `goal_dist_bonus_w` | 2.5 | event |
| `demo_w` | 8 | event (split ±4 between demoer/demoed) |
| `dist_w` | 0.25 | potential (`exp(-dist/1410)`, 1410 = max driving speed) |
| `align_w` | 0.25 | potential (cosine similarity ball/goal alignment) |
| `boost_gain_w` / `boost_lose_w` | 1.5 / 0.8 | potential on `sqrt(boost)` |
| `ang_vel_w` | 0.005 | dense |
| `touch_grass_w` | 0.005 | dense penalty |
| `touch_height_w` | 3 | dense (cubic-root height activation) |
| `touch_accel_w` | 0.5 | event, scales with ball Δvelocity |
| `flip_reset_w` | 10 | event |
| `opponent_punish_w` | 1 | zero-sum coupling |
| **`team_spirit`** | **0.6** | mixes individual reward with team mean |

Structurally notable and worth copying conceptually **[V]**:
- **Potential-style shaping.** `dist_w`, `align_w`, `goal_dist_w`, `win_prob_w`, `boost` are all *state potentials* applied as `player_rewards += player_qualities - self.player_qualities` — i.e. the **difference of a potential function**, which is the potential-based form that preserves optimal-policy invariance (Ng, Harada & Russell, ICML 1999). This is why Necto's dense rewards do not obviously distort the objective.
- **Zero-sum team coupling.** `player_rewards[:mid] = (1-spirit)*blue + spirit*mean(blue) - opponent_punish_w*mean(orange)`. The reward is genuinely adversarial, which is what makes self-play produce skill rather than reward farming.
- **The author's own caveat, in a code comment [V]:** `# TODO Want to find something better, this could promote waiting to score when losing` next to the goal-importance term. Even the reference implementation flags its reward as exploitable.

**Action space [V]**: `NectoAction` — the same 90-entry lookup table as `LookupTableAction`.

**Training scale — reported [R]:**
- *"Nexto was trained by crowd-sourcing compute power from our community. We had anywhere between 10 and 40 game clients contributing data to a computer running the learning algorithm at a time. It took several months of starting, stopping, adjusting, and tweaking things before Nexto finally came about."* — a developer on r/RocketLeague, Jan 2023.
- *"Nexto learned by playing against itself in approximately 250,000 hours (roughly 29 years 24/7) worth of gameplay"* — deeplearning.ai's summary of the same discussion.
- *"trained using PPO with workers run by people all around the world"* — `bot.cfg`, verbatim **[V]**.
- `maximum_tick_rate_preference = 120` in `bot.cfg` **[V]**.

**Available artifacts [V]:** `rlbot-support/Nexto/nexto-model.pt` and `nexto_obs.py` (RLBot v4), and `JPK314/doublesize-nexto-v5/src/{agent.py,bot.py,nexto_obs.py,nexto-model.pt}` + `collision_meshes/soccar/*.cmf` (RLBot v5). So **a working Nexto inference path on RLBot v5 exists and is public** — this is the single most valuable external asset for this project, because it gives a fixed, strong, *reproducible* evaluation opponent (§10.5) without needing to train one. Its `src/requirements.txt` uses `torch==2.0.1+cpu` **[V]** — Nexto inference is comfortable on CPU.

**[V]** `bot.py` also shows two deployment tricks worth noting: `hardcoded_kickoffs` replays a fixed 46-step control sequence for kickoffs (bypassing the network entirely), and `stochastic_kickoffs` sets `beta = 0.5` during `MatchPhase.Kickoff` to sample rather than argmax. `beta` is a general temperature knob: *"1=best action, 0.5=sampling from probability, 0=random, -1=worst action"* **[V]**.

### 6.2 Lucy-SKG — the strongest published academic result

**[V]** arXiv:2305.15801, *"Lucy-SKG: Learning to Play Rocket League Efficiently Using Deep Reinforcement Learning"* (Moschopoulos, Kyriakidis, Lazaridis, Vlahavas; submitted 2023‑05‑25). Abstract claims it *"learned how to play Rocket League in a sample-efficient manner, **outperforming by a notable margin the two highest-ranking bots in this game, namely Necto (2022 bot champion) and its successor Nexto**, thus becoming a state-of-the-art agent"*, via (a) a reward analysis/visualization library, (b) parameterizable reward shapes with a "Kinesthetic Reward Combination" technique, and (c) auxiliary networks trained on reward-prediction and state-representation tasks in an on-policy fashion.

**Categorization:** this is a **peer-reviewable but author-reported [R]** claim. No third party has reproduced it, no weights or training code are known to be public, and it was evaluated against the *public* Nexto rather than Nexto+/Tecko. Its transferable, credible ideas are the *techniques* — auxiliary reward-prediction and state-representation heads for sample efficiency, and explicit reward-component visualization as a debugging tool — not the ranking claim.

### 6.3 Other relevant work

| Project | What it contributes | Evidence |
|---|---|---|
| **`Rolv-Arild/replay-pretraining`** (RL‑RPT), GPL‑3.0, 37★ | Imitation learning from replay files via an **Inverse Dynamics Model** (VPT, arXiv:2206.11795). README explains *why* naive behaviour cloning from replays fails (inputs not in replays, lossy reconstruction, compounding error, human inconsistency, non-independent buttons) and how an IDM over ±20 frames fixes it. Produced **Ripple**, "a high level bot with many human-like traits". | **[V]** README |
| **`Rolv-Arild/EARL-pytorch`**, 17★ | The attention architecture behind Necto/Nexto. `pip install EARL-pytorch`. | **[V]** README |
| **`Virxec/rlgymppo_rs`** | Rust PPO with Elo skill tracking and Nexto baseline. §5.4. | **[V]** |
| **`CPollreis/rocket-league-bot` (PILOT)** | The most complete public **WSL 2 + rlgym-learn** reference. Documents the Windows failure, `/mnt/c` slowness, WSLg rendering breakage, and a working reward/hyperparameter set. | **[V][R]** §7.4 |
| **`RafeSymonds/rocket-league-bot` (BotBoi)** | **WSL 2 training → Windows RLBot v5 deployment** reference. Exports a policy into a bot folder on the Windows drive; has a pytest suite *"including sim-vs-RLBot observation parity"*; two-phase curriculum (early dense shaping → main winning-only, γ 0.99→0.995); policy snapshots every 50 M steps kept forever; `bin/eval` for checkpoint-vs-checkpoint matches. | **[V][R]** §7.4, §9.5 |
| **Neville Walo, "Seer: Reinforcement Learning in Rocket League"** (MSc thesis) | Independent academic treatment of RL in Rocket League; describes Necto/Nexto as "currently the best Rocket League bots". | **[R]** |
| **arXiv:2205.05061** (IEEE CoG 2022) | **The only peer-reviewed sim-to-game transfer result for Rocket League.** Trains goalie/striker agents in a *Unity* simulation with PPO and transfers to the real game via RLBot: goalkeeper saves ~100 % of shots, striker scores ~75 %. Reports ~950 steps/s of simulation on a GTX 1080 + Ryzen 7 3700X vs. RLBot's real-time 120 steps/s. Also documents that their sim needed an empirically-corrected max angular velocity (7.3 rad/s vs. the 5.5 rad/s reference) and that **ablating any single physical adaptation was catastrophic**. | **[V]** arXiv |
| **`SaltieRL/Saltie`**, `roboserg/RoboLeague` | Historical (2019) distributed RL and a RoboLeague-derived car-soccer env. Context only. | **[V]** GitHub |

**On imitation learning as a head start [E]:** the evidence says *maybe, but not first*. RL‑RPT demonstrates it works and produced a competent bot, but its author frames it as "a jumping off point", and it required training an IDM, curating SSL replays and handling replay reconstruction artefacts. That is a substantial sub-project. **Recommendation: defer to a later phase; revisit only if pure self-play stalls below a defined skill floor.** If revisited, RL‑RPT's GPL‑3.0 license is a constraint.

---

## 7. Deployment into the original game — RLBot v5

### 7.1 Architecture — verified

**[V]** RLBot v5 wiki, *Architecture*: *"Matches are typically started by RLBotGUI, CleoPetra, or some other match orchestrator… Once the RLBotServer receives a MatchConfiguration, it will start up Rocket League with the `-rlbot` flag (if needed) and then the bot and script processes. RLBotServer communicates with Rocket League through the official Psyonix API. Since the API is under NDA, a closed-source `Bridge.dll` hides the details of the API. On startup bots and scripts connect to the RLBotServer and are briefed about the match, their index, and more. Once the match is loaded, RLBotServer will send live game data to all connected clients… Each bot client controls their car(s) by responding to each GamePacket with PlayerInput… **All client connections use TCP sockets.**"*

**[V]** v5 overview: *"No more DLLs… the transition to web sockets is finally complete"*; *"All bots connect to core directly via the sockets flatbuffer spec"*; *"Any language can start/stop/manage a match"*; *"All config files now use the standard toml format"*; *"First-class Windows & Linux support"*; *"The bot configuration's `python_file` has been replaced with `run_command` and `run_command_linux` — On Linux, if `run_command_linux` isn't specified, v5 will attempt to run `run_command` under Wine"*; *"Performance metrics (missed responses by bots). Displayed automatically only when there are problems."*

**[V]** Repos and versions: `RLBot/core` (C#, MIT, pushed 2026‑09‑09, builds `RLBotServer.exe`, requires **.NET 10 SDK**); `RLBot/gui` (Go + wails); `RLBot/flatbuffers-schema`; `RLBot/launcher`. Python client package: **`rlbot` 2.0.0b56** on PyPI (2026‑10‑02), i.e. **still in beta** **[V]**. `RLBot/python-example` README: *"Install **Python 3.12 or later**"* and `requirements.txt` is `rlbot>=2.0.0.beta` **[V]**.

### 7.2 Operating-system support — the pivotal constraint

**[V]** RLBot v5 wiki, *Operating system support*: **Windows** — RLBot launches Rocket League through Steam or Epic. **Linux** — *"RLBot currently supports either Rocket League running under Proton on Steam or Rocket League installed from the Epic Games Store via Legendary and Heroic."* **Mac** — *"Native MacOS support has been dropped."*

**What this means for a WSL 2 machine [E]:**
- Rocket League **cannot run inside WSL 2** (no GPU-accelerated DirectX/Vulkan game runtime in the WSL VM for this title; EAC and the launcher are not supported).
- Therefore evaluation in the real game happens **on the Windows host**, while training happens **in WSL 2**. Both run on the same physical machine, concurrently, sharing the one GPU.
- This is a **hard architectural boundary**: two Python environments, two dependency sets, and a one-way artifact hand-off (weights). It is not a nuisance — it is the correct separation, because the training environment must never depend on a game client. BotBoi does exactly this **[R]**.
- Practical corollary: never run in-game evaluation while a training run needs the GPU, or the training run's throughput measurements become meaningless.

### 7.3 Interfaces, protocols and action controls

**[V]** From `RLBot/python-example` and `JPK314/doublesize-nexto-v5/src/bot.py`:

```python
from rlbot.flat import Color, ControllerState, DesiredCarState, DesiredPhysics, GamePacket, \
                       MatchPhase, Rotator, RotatorPartial, Vector3, Vector3Partial
from rlbot.managers import Bot

class Nexto(Bot):
    def initialize(self): ...                       # field_info, match_config, index, team available
    def get_output(self, packet: GamePacket) -> ControllerState: ...
    # self.renderer.begin_rendering(...) / draw_string_3d / draw_line_3d / end_rendering
Nexto("agent-id").run()
```

`ControllerState` fields **[V]**: `throttle, steer, pitch, yaw, roll, jump, boost, handbrake` — the same 8 channels as `LookupTableAction`'s output, so **the action representation transfers 1:1 with no remapping**. Mapping code from `bot.py`: `controls.throttle = action[0] … controls.jump = action[5] > 0 …`.

Also available: `DesiredCarState` / `DesiredPhysics` (state setting for scenario testing), rendering, quick chat, ball prediction.

### 7.4 Latency and tick rate — the acceptance criterion for deployment

**[V]** RLBot v5 wiki, *Tick rate*: *"the tick rate is `min(rocketLeagueFPS, 120)`"* — Rocket League's physics runs at 120 Hz, and *"All bots are expected to run at 120hz. If you need more time, engineer your bot around this (e.g. an extra asynchronous thread for complex algorithms)."* Non-standard tick rates are explicitly **not** a v5 feature.

**[V]** *Game data*: *"**GamePacket:** Contains live game data… It is sent to all clients at 120 Hz. **Controller:** … The bots send this to the RLBot server as often as possible in response to the GamePacket. **If a tick is missed, the controller from previous tick is used.**"*

**[V]** v4-era wiki (still the clearest statement of packet freshness): *"In most languages, your choice of tick rate will not affect the 'freshness' of packets delivered. They should all be very fresh, **about 1 ms old**, when they are delivered to you"* and *"If your bot is slow to compute a frame, you will fall behind and potentially miss a frame, since bots generally loop on a single thread."*

**[V]** `samuelpmish/RLBot` (historical): *"Each time RLBot receives a packet from Rocket League, it starts a timer for about 6 milliseconds (for 120 Hz)… Bots that take longer than the allotted time will not receive new messages from RLBot until a response is received."*

**Derived requirement [C]:** the bot must produce a `ControllerState` within roughly **8.3 ms** of receiving a `GamePacket` (1/120 s), with practical headroom suggested by the ~6 ms figure above. Because the policy only re-infers every 8 ticks (tick skip 8), the *policy* need only run at 15 Hz — but `get_output` is called at 120 Hz and must return immediately on the 7 out of 8 ticks where it reuses the cached action. That is exactly the structure `doublesize-nexto-v5/src/bot.py` implements with its `self.ticks` / `self.update_action` bookkeeping **[V]**.

**Therefore [E]:** a ~178 k-parameter MLP running on CPU is comfortably inside budget; even a 2 M-parameter MLP is fine. **Inference latency is not the deployment risk — observation construction is**, because the obs builder must run inside that same 8.3 ms and must produce *bit-comparable* values to training (§7.6).

### 7.5 Licensing and game integrity — verified

- **[V]** `RLBot/core` README: *"The `Bridge.dll` file in `RLBotCS/lib` is built from a **closed-source** repository due to legal reasons. It is maintained by RLBot developers who have signed an agreement with Psyonix to keep it private."* The dll is platform-independent and works for building on Windows and Linux.
- **[V]** `RLBot/RLBot` README: *"This repository is currently incomplete to meet the legal needs of the Psyonix API. It is missing the source code behind `RLBot.exe`."* And: *"The RLBot framework helps people create bots for use in Rocket League's **offline modes**, just for fun."*
- **[V]** RLBot v5 overview: *"The framework uses an official Psyonix API making our bots safe to play with and against."*
- **[V]** RocketSim README: the author is *"strongly against cheating/botting in Rocket League"* and notes ML bots are being used to cheat in ranked matches.
- **[R]** The Nexto cheating incident (Jan 2023): a third party removed the offline-only restriction from the public Nexto weights and used it in ranked play.

**Binding constraints for this project [E], stated plainly:**
1. **Offline / RLBot modes only.** Never connect a trained policy to online, ranked or competitive matchmaking. This is a design constraint, not a preference.
2. **Do not modify, inject into, or hook the game client.** Use RLBot v5's sanctioned API exclusively.
3. **Do not redistribute trained weights without an explicit offline-only restriction**, and record the restriction in the repository.
4. **The `Bridge.dll` is closed source.** Any deployment path depends on a binary whose source cannot be audited. Accept and document.
5. **Do not train against the live game** (i.e. no RLBot-in-the-loop RL). It is real-time, 120 tps, and would be ~1000× slower than RocketSim. RLBot is for *evaluation* only.

### 7.6 RocketSim ↔ real game differences that affect transfer

| Gap | Detail | Evidence | Mitigation |
|---|---|---|---|
| **Physics drift** | *"the tiny errors will accumulate over time"* | **[V]** RocketSim README | Short episodes, frequent resets, no-touch timeouts; never long open-loop rollouts |
| **One-tick input latency** | `RocketSimEngine(rlbot_delay=True)` injects it deliberately | **[V]** source | Keep it enabled; verify it stays enabled in configs |
| **Hitbox** | `FixedTeamSizeMutator` hard-codes `OCTANE` for every car; the real game has 6 hitboxes (`OCTANE, DOMINUS, PLANK, BREAKOUT, HYBRID, MERC` in `common_values`) | **[V]** source | Train Octane-only for Phase 1; add hitbox randomization as a later mutator; **evaluate against Octane opponents only** |
| **Observation source** | Training obs comes from RocketSim state; deployment obs must be reconstructed from a `GamePacket`. Any mismatch is silent and catastrophic. | **[E]** | Port `DefaultObs` onto `rlgym_compat`'s v2 `GameState`, and add a **sim-vs-RLBot observation parity test** (BotBoi does this **[R]**) |
| **Ball prediction** | RLBot supplies a 720-slice / 6-second prediction; RocketSim does not, and `DefaultObs` uses neither | **[V]** | Do not use ball prediction in the obs — keep the training and deployment obs identical |
| **Tick rate** | Training is unthrottled; the game is capped at `min(FPS, 120)` | **[V]** | Cap game FPS at 120 with VSync off for all evaluation (Nexto's README says the same **[V]**) |
| **Boost pad order** | `RocketSimEngine._get_state` has a `# TODO check if the order is correct, I think mtheall's bindings handle it internally` comment | **[V]** source | Assert pad ordering matches `common_values.BOOST_LOCATIONS` in the parity test |
| **Mutators** | RLGym v2 exposes only `gravity`, `boost_consumption`, `dodge_deadzone`; RocketSim has 20+ | **[V]** | Keep default match settings in evaluation so training and eval physics agree |

**The single highest-value, lowest-cost mitigation is an observation parity test**: run identical game states through the training obs builder (RocketSim) and the deployment obs builder (GamePacket → `rlgym_compat`), and assert elementwise agreement to a tolerance. Without it, a deployment failure is indistinguishable from a weak policy.

### 7.7 Model export for deployment

**[E] Recommendation:** export the trained actor to a **self-contained, CPU-only artifact** and load it with a *separate, minimal* runtime on the Windows side.

- `doublesize-nexto-v5/src/requirements.txt` uses `torch==2.0.1+cpu` **[V]** — precedent that CPU PyTorch is the norm for RLBot ML bots, and Nexto (a Perceiver with attention) runs fine on it.
- The deployment environment should contain **only** `rlbot`, `torch` (CPU), `numpy`, `rlgym-compat` and the obs builder — never `rocketsim`, never `rlgym-learn`, never CUDA.
- Because WSL 2 and Windows have separate filesystems, the hand-off is a file copy to the Windows drive (BotBoi's `bin/export` does exactly this **[R]**).
- TorchScript (`torch.jit.trace`) is the lowest-friction option for a plain `DiscreteFF` MLP. ONNX is an alternative if a non-Python bot runtime is wanted later. Either way, **the export must be covered by a numerical equivalence test against the training-time actor** on a fixed set of observations.

---

## 8. Hardware, WSL 2, CUDA and throughput planning

### 8.1 The machine — verified specifications

**CPU: Intel Core i7‑12700F** **[V]** Intel ARK:
- 12 cores total = **8 Performance-cores + 4 Efficient-cores**; **20 threads** (P-cores are HyperThreaded, E-cores are not).
- P-core base 2.10 GHz, P-core max turbo 4.80 GHz, Turbo Boost Max 3.0 4.90 GHz; E-core base 1.60 GHz, E-core max turbo 3.60 GHz.
- 25 MB Intel Smart Cache (L3), 12 MB total L2. Base power 65 W, max turbo power 180 W. **No integrated graphics** (F SKU).

**GPU: NVIDIA GeForce RTX 4060 Ti 8 GB** **[V]** NVIDIA product page + tech press:
- Ada Lovelace, AD106, **34 SMs / 4352 CUDA cores**, 136 Tensor cores, boost 2.54 GHz.
- **8 GB GDDR6, 128-bit bus, 288 GB/s**, 32 MB L2, 160 W TGP, PCIe 4.0 ×8, **compute capability 8.9**.

**RAM: 32 GB.** **OS: Ubuntu on WSL 2.**

### 8.2 CUDA on WSL 2 — documented requirements and limitations

**[V]** NVIDIA *CUDA on WSL User Guide* (release 13.4, Sept 2026):

- *"Install NVIDIA GeForce Game Ready or NVIDIA RTX Quadro Windows 11 display driver… **This is the only driver you need to install. Do not install any Linux display driver in WSL.** Once a Windows NVIDIA GPU driver is installed on the system, CUDA becomes available within WSL 2. The CUDA driver installed on Windows host will be stubbed inside the WSL 2 as `libcuda.so`, therefore **users must not install any NVIDIA GPU Linux driver within WSL 2**."*
- If a toolkit is needed: *"DO NOT choose the 'cuda', 'cuda-12-x', or 'cuda-drivers' meta-packages under WSL 2 as these packages will result in an attempt to install the Linux NVIDIA driver under WSL 2. Install the `cuda-toolkit-12-x` metapackage only."*
- *"WSL 2 GPU acceleration will be available on **Pascal and later** GPU architecture on both GeForce and Quadro product SKUs in **WDDM** mode."* → RTX 4060 Ti (Ada, sm_89) is well within scope.
- *"We recommend [WSL kernel] 5.10.16.3 or later for better performance and functional fixes."*

**Documented limitations that matter [V]:**

| Limitation | Impact on this project |
|---|---|
| **Unified/managed memory not fully supported**; *"Concurrent CPU/GPU access is not supported"* | PyTorch does not rely on UVM for ordinary training. Low impact, but rules out exotic zero-copy designs. |
| **Pinned (page-locked) system memory has limitations**, affecting *"some deep learning training workloads"* | Directly relevant: the plan is to stream minibatches from a **CPU** experience buffer to the GPU every iteration. `pin_memory` / `non_blocking=True` transfers may underperform vs. native Linux. Measure; do not assume. |
| **NVML (`nvidia-smi`) does not support all queries** — *"GPU utilization, active compute process are some queries that are not yet supported"* | `nvidia-smi`-based monitoring will be incomplete. Use `torch.cuda.max_memory_allocated()` / `memory_reserved()` for VRAM instead. |
| **OpenGL–CUDA interop not supported** | Irrelevant to headless training; relevant to rlviser (see below). |

**[V]** Ubuntu's official WSL guide states the same rule: *"On WSL 2, the CUDA driver used is part of the Windows driver installed on the system, and, therefore, care must be taken not to install this Linux driver."*

**PyTorch [V]:** PyTorch wheels bundle their own CUDA runtime; only the Windows driver is required. Current PyPI latest is **torch 2.14.1** (2026‑09‑30), `requires_python >=3.10`. `rlgym-learn-algos` requires only `torch>1.13`. **Recommendation [E]:** pick the CUDA-build wheel matching the installed Windows driver's CUDA capability, pin it exactly, and verify with `torch.cuda.is_available()` + `torch.cuda.get_device_name(0)` before any training (gate **B1**). Do not install a CUDA toolkit inside WSL unless something must be compiled — nothing in this stack needs it.

**numpy [V]:** latest is 2.5.4, `requires_python >=3.12`. `rlgym-learn`/`algos` accept `numpy>1.21` with no upper bound. **Recommendation [E]:** Python **3.12** — satisfies rlgym-learn (`<=3.14`), numpy 2.5, torch 2.14, and matches RLBot v5's "Python 3.12 or later". One Python version across training and deployment halves the surface area for drift.

### 8.3 WSL 2 platform traps — all from primary sources

| Trap | Source | Action |
|---|---|---|
| **`/mnt/c` is drastically slower.** *"Clone into the Linux filesystem (NOT `/mnt/c` — RocketSim is ~10× slower there)"* / *"RocketSim hammers the disk and the `/mnt/c` Windows bridge is an order of magnitude slower."* | **[R]** PILOT, BotBoi | Repo, venv, checkpoints and `shmem_flinks` all under `~`, never `/mnt/c` |
| **`OPENBLAS_NUM_THREADS` must be 1** or env processes *"use a ton of memory… and throttle each other"* | **[V]** rlgym-learn `quick_start_guide.py`, `speed_test.py` | Set before any numpy import |
| **WSLg graphics for rlviser is unreliable**: no GPU acceleration, `libEGL` falls back to `dri2` then `zink` and both fail (`ZINK: failed to choose pdev`), yielding a crash or a black window behind WSLg's `[WARN:COPY MODE]` RDP fallback | **[R]** PILOT, in detail | Treat in-WSL rendering as best-effort. Check `ls /dev/dri`. If absent, run rlviser natively on Windows and point training at it over UDP, or skip rendering entirely |
| **WSL 2 → Windows UDP loopback forwarding can silently break after a reboot** | **[R]** PILOT | Only affects the visualizer. A user-space relay fixes it |
| **Default WSL memory allocation** is a fraction of host RAM | **[E]** — verify with `free -h` and `.wslconfig` | Confirm ≥ 24 GB visible to WSL before scaling `n_proc`; §8.5 shows why |
| **rlgym-learn fails on native Windows** (`mio` epoll vs IOCP) | **[R]** PILOT + **[V]** Rust source | Irrelevant here — WSL 2 is the plan. But it means the **Windows-side eval bot must not import `rlgym_learn`** |

### 8.4 Memory budget — computed

**VRAM.** Parameter counts for `DiscreteFF(obs=92 → 90)` / `BasicCritic(obs=92 → 1)` **[C]**:

| Layers | Params | fp32 weights | Adam (2 states) |
|---|---|---|---|
| `(256,256,256)` | 178 522 | 0.7 MB | 1.4 MB |
| `(512,512,512)` | 619 098 | 2.5 MB | 5.0 MB |
| `(1024,1024,512,512)` | 1 978 458 | 7.9 MB | 15.8 MB |
| `(2048,2048,1024,1024)` | 7 626 842 | 30.5 MB | 61.0 MB |

Weights and optimizer state are trivial on 8 GB. **Activations dominate** — for a minibatch of *M* through widths *wᵢ*, forward layer outputs total `M · Σwᵢ · 4 bytes`, and autograd must retain them for backward **[C]**:

| Layers | MB=25 000 | MB=50 000 | MB=100 000 |
|---|---|---|---|
| `(256,256,256)` | 77 / 154 MB | 154 / 307 MB | 307 / 614 MB |
| `(1024,1024,512,512)` | 307 / 614 MB | 614 / 1 229 MB | 1 229 / 2 458 MB |
| `(2048,2048,1024,1024)` | 614 / 1 229 MB | 1 229 / 2 458 MB | 2 458 / **4 915 MB** |

*(first number = layer outputs, second = ×2 retained for backward)*

**[E] Conclusion:** the ZealanL-recommended `[2048, 2048, 1024, 1024]` **[R]** at a 50 000-row minibatch needs on the order of 2.5–5 GB of activation memory for *one* network — and with two controllers (self-play) both actor and critic pass through, plus CUDA context (~300–500 MB) and allocator fragmentation, **8 GB is genuinely tight**. That is precisely why `n_minibatches` exists. **`[2048,2048,1024,1024]` is not a starting configuration on this GPU**; `[256,256,256]` or `[512,512,512]` is. Gate **B4** finds the largest network that keeps `batch_consumption_time` a small fraction of `iteration_time`.

**Host RAM.** `NumpyExperienceBuffer` stores `observations` (numpy), `actions` (numpy), `log_probs` / `values` / `advantages` (tensors) **[V]** — it does **not** store `next_states`, which is what makes the buffer cheap. Per step **[C]**: `n_obs·sizeof(dtype) + 8 (int64 action) + 4 + 4 + 4`.

| obs dtype / width | cap 150 k | cap 300 k | cap 1 M |
|---|---|---|---|
| float32 / 92 (1v1) | 58 MB | 116 MB | 388 MB |
| float32 / 132 (2v2) | 82 MB | 164 MB | 548 MB |
| float64 / 92 (1v1) | 113 MB | 227 MB | 756 MB |
| float64 / 132 (2v2) | 161 MB | 323 MB | 1 076 MB |

×2 for two self-play controllers. Plus per-env-process RSS (Python + numpy + RocketSim arena), which is the **dominant and currently unknown** term — **`[U]`**, to be measured in gate **B2**. Using a `Float32DefaultObs` roughly halves the buffer term **[C]**.

**[E]** With 32 GB total and WSL 2 typically seeing ~50 % by default, budget ≈ 16 GB: at ~350–600 MB per env process, **`n_proc` in the range 12–20 is the plausible operating region**, leaving room for a 300 k–1 M step buffer. This is an estimate; `n_proc` must be found by measurement, not assertion.

### 8.5 CPU-bound vs GPU-bound — the reasoning

**[C]** From §3.2, one RocketSim thread at 114 481 tps ⇒ ~14 300 env-steps/s of *pure physics* for 2v2. 1v1 is cheaper. The 12700F has 12 physical cores. If the Python path could keep up, physics would allow a very large number of environments.

The counterweight **[V]**: the per-step Python path in each env process is `obs_builder.build_obs` (numpy concatenations over 34 pads + up to 6 cars) → `reward_fn.get_rewards` → `termination/truncation` → serde → shared-memory write → UDP signal. That is dozens of small numpy allocations per agent per step. It is the classic Python-per-step bottleneck, and it is why `rlgymppo_rs` exists at all (Rust, threads, no per-step Python).

**[E] Expectation to be tested:** on this machine the run will be **CPU-bound on the env processes** at small-to-medium networks, and become **GPU-bound only at large networks with large minibatches**. The observable signature is in the already-logged metrics: `timestep_collection_time` vs `batch_consumption_time` within `iteration_time`.

**[E] Practical levers, in order of expected value:**
1. **`OPENBLAS_NUM_THREADS=1`** — verified as necessary by upstream.
2. **float32 observations** (`Float32DefaultObs`) — halves IPC payload and buffer RAM; also halves the serde cost.
3. **`n_proc` sweep** — the primary knob. Start at 8; sweep 8/12/16/18/20; keep the value where `timestep_collection_time` stops improving.
4. **`min_frac_process_responses_per_collection`** (default 0.45) — controls how many process responses are batched per inference pass. Higher = bigger GPU batches, more straggler wait. This is the rlgym-learn analogue of rlgym-ppo's `min_inference_size` and is a real throughput knob.
5. **CPU affinity / E-core awareness** — the 12700F is heterogeneous: 8 P-cores at up to 4.9 GHz and 4 E-cores at up to 3.6 GHz. Under a symmetric `n_proc` pool, an env process scheduled on an E-core becomes a straggler that gates the whole inference batch (the collector waits for a *fraction* of responses). **`[U]`** whether `taskset`/`numactl` pinning of env processes to P-cores, with the learner on the E-cores or a P-core pair, helps. **This is a concrete, cheap experiment** (gate **B2c**) and a plausible 10–30 % win. Do not assume the number — measure it.
6. **Network size and `n_minibatches`** — trade GPU time against VRAM (§8.4).

**Do not invent a throughput target.** The only defensible statement is: *measure `overall steps/s` with the project's own `speed_test.py`-style harness, then `steps/day = overall_sps × 86 400`.* §12 shows what that implies for reaching Nexto-class compute at plausible values, without asserting a value we have not measured.

---

## 9. Learning strategy — evidence-based recommendations

### 9.1 PPO configuration

**[V]** Original PPO (arXiv:1707.06347) MuJoCo-1M hyperparameters: horizon 2048, Adam 3×10⁻⁴, 10 epochs, minibatch 64, γ=0.99, λ=0.95. Atari: horizon 512, 15 epochs, minibatch 4096, 32–128 actors. GAE from arXiv:1506.02438.

**[V]** `rlgym-learn-algos` defaults: `n_epochs=1, batch_size=50000, n_minibatches=1, ent_coef=0.005, clip_range=0.2, max_grad_norm=0.5, actor/critic lr=3e-4, advantage_standardization=True, gamma=0.99, lmbda=0.95, standardize_rewards=True, reward_clip=10`. Critic loss is plain MSE — **no value clipping**.

**[V]** rlgym.org's own recommended 1v1/2v2 settings (adapted from ZealanL): `ppo_batch_size=100 000`, `policy_layer_sizes=critic_layer_sizes=[2048,2048,1024,1024]`, `ts_per_iteration=100 000`, `exp_buffer_size=300 000` (*"keep this 2–3× the batch size"*), `ppo_minibatch_size=50 000`, `ent_coef=0.01`, `policy_lr=critic_lr=1e-4`, `ppo_epochs=2`, `standardize_returns=True`, `standardize_obs=False`, `save_every_ts=1e6`, `timestep_limit=1e9`.

**[V]** ZealanL's staged guidance (RLGym-PPO-Guide `learner_settings.md`, `making_a_good_bot.md`):
- `ts_per_iteration`: *"values of 50 000 are good for early learning, but once the bot is actually hitting the ball, it should be increased to 100 000. Once the bot is actually shooting and scoring, increase it to 200 000 or even 300 000."*
- `exp_buffer_size`: *"2× or 3× `ts_per_iteration`"*.
- `ent_coef`: *"The golden value for this seems to be about 0.01."* Reduce only to refine, and *"Don't do this if you plan on continuing to train your bot after."*
- `policy_lr` staging: *"Bot that can't score yet: 2e-4. Bot that is actually trying to score on its opponent: 1e-4. Bot that is learning outplay mechanics: 0.8e-4 or lower."* And: *"If your bot seems stuck, try decreasing LR."*
- `ppo_epochs`: *"I recommend 2 or 3… When testing, compare the increase in rewards from a common starting point, not SPS."*
- Network: *"increase both policy and critic size until you start losing a ton of SPS… My computer (with a RTX 3060 ti) seems to run best on `[2048, 2048, 1024, 1024]`. More than that, and my SPS tanks."*
- `standardize_obs`: marked *"Don't touch these"* for `standardize_returns=True, standardize_obs=False` in the official tutorial.

**Recommended starting configuration for this project [E]** — deliberately conservative, all values traceable:

| Knob | Start | Source / rationale |
|---|---|---|
| `n_epochs` | **2** | ZealanL "2 or 3"; doubles GPU work per batch, so gate it behind B4 |
| `batch_size` = `timesteps_per_iteration` | **50 000** | rlgym-learn default; raise to 100 k once the policy reliably touches the ball |
| `n_minibatches` | **1** (⇒ minibatch 50 k) at `(256,256,256)`; increase if VRAM-bound | §8.4 arithmetic |
| `ent_coef` | **0.01** | "golden value" **[R]**; library default 0.005 is lower |
| `clip_range` | 0.2 | PPO paper default |
| `max_grad_norm` | 0.5 | library default |
| `actor_lr` / `critic_lr` | **2e-4** early → 1e-4 → 8e-5 | staged schedule **[R]**; library default 3e-4 is on the hot side for RL |
| `gamma` | 0.99 → **0.995** in the "main" phase | BotBoi's documented phase change **[R]**; longer horizon for winning-oriented play |
| `lmbda` | 0.95 | GAE default |
| `standardize_rewards` | True | library default |
| `reward_clip` | 10 | library default; keep until a reward exceeds it legitimately |
| `advantage_standardization` | True | library default |
| `device` | **`"cuda:0"` explicitly** | default is `"cpu"` — §5.2.3 |
| network | `(256,256,256)` for actor and critic | §8.4; scale only after B4 |

### 9.2 Reward design — winning first, shaping second

**Principle [E], supported by [V][R]:** the reward must be dominated by the actual objective. Every dense term is a hypothesis about what helps, and each is a potential exploit. The ZealanL guide states the failure mode plainly **[R]**: *"the learning algorithm is trying to maximize and explore your rewards, not winning. So, if your rewards are farmable in a way that does not improve your bot's skill, your bot will likely get worse at the game."*

**Recommended structure:**

```
R = 10.0 * GoalReward()                              # the objective, never annealed
  + 0.1  * TouchEventReward()                        # custom, delta-based — NOT RLGym's TouchReward (§4.4.2)
  + Σ_i  w_i(t) * ShapingReward_i()                  # annealed toward 0
```

> ⚠ **Do not use RLGym v2's shipped `TouchReward` here.** It returns `+1` on every step after the first touch (§4.4.2 **[V]**), so at weight 0.1 it contributes 3–45× the value of a goal over a normal episode and rewards touching-then-stalling. Write the ~10-line delta-based `TouchEventReward` instead.

Justification for each auxiliary term, and its known exploit:

| Term | Why | Known exploit | Weight guidance |
|---|---|---|---|
| `GoalReward` (+score / −concede) | The actual objective. **[V]** Verified zero-sum: `+1` if `state.scoring_team == car.team_num` else `−1`, on the goal step only | — | Dominant, constant |
| **Asymmetric concede penalty** | ZealanL: *"just decrease the concede penalty… At `aggression_bias = 0.25`, the penalty for conceding is 25 % less than the reward for scoring"*, used at ~0.2 to counter bot passiveness **[R]**. ⚠ **[V]** RLGym v2's `GoalReward.__init__` takes **no arguments** — this knob does not exist in the shipped class and requires a custom `RewardFunction` | Can produce reckless play; watch concede rate, not just goals | Introduce **only after** the policy can score; start symmetric |
| **`TouchEventReward`** (custom, delta-based) | Prevents doing nothing, without the stalling exploit of the shipped class | Ball-pushing / repeated soft taps | Small (0.1) |
| **Touch strength** (∝ ‖Δball velocity‖) | ZealanL: *"a slight push that barely changes the velocity of the ball will give almost no reward, but a strong shot will give lots of reward"* **[R]**. Necto does this via `touch_accel_w` **[V]** | Weak | Replace plain touch once scoring emerges |
| Velocity toward ball / distance to ball | Early exploration | Ballchasing that ignores positioning | **Anneal to 0** |
| Ball velocity toward opponent goal | Shooting | Wall-shot farming | **Anneal down, not to 0** |
| Air time × ball height | ZealanL's `min(air_time_frac, height_frac)` to get aerials without wall-shot farming **[R]**; Necto's `touch_height_w` with a wall-distance factor **[V]** | Wall shots | Small, anneal |
| `sqrt(boost)` conservation | Necto's `boost_gain_w`/`boost_lose_w` on `sqrt` **[V]**; *"sqrt() effectively makes boost more important the less you have"* **[R]** | Boost hoarding | Small |
| Flip reset | Necto `flip_reset_w = 10` **[V]** | Reward farming without scoring | Only if flip resets never emerge; defer |

**Two structural lessons worth adopting from Necto [V]:**
1. **Prefer potential-based shaping.** Necto's distance/alignment/win-probability/boost terms are all applied as `potential(s_t) − potential(s_{t−1})`. Potential-based shaping is the one form proven not to change the optimal policy (Ng, Harada & Russell, ICML 1999). Dense terms written as raw per-step values are not policy-invariant.
2. **Anneal shaping to zero.** PILOT does exactly this and documents the schedule: `dist_to_ball 0.05 → 0.0 @ 200 k steps`, `vel_to_ball 0.1 → 0.0 @ 200 k`, `ball_vel_to_goal 0.1 → 0.02 @ 500 k`, with `goal 10.0` and `touch 0.1` constant **[R]**. This is the cleanest way to have dense guidance early and a clean winning objective late. **Caveat [R]**, also from PILOT: its annealing counts *per-process* env steps, so each of `n_proc` workers anneals independently — a subtle bug to avoid by driving the schedule off *global* `cumulative_timesteps`.

### 9.3 Self-play, opponent pools and policy collapse

**Literature basis:**
- **OpenAI Five** (arXiv:1912.06680) **[V]**: *"self-play reinforcement learning can achieve superhuman performance on a difficult task"*; a single 10-month run, 770±50 PFlops/s·days, 480–1536 optimizer GPUs, 80 000–172 800 rollout CPUs, total batch up to 3 145 728 timesteps, frameskip 4 (7.5 decisions/s). **Also documents the team-spirit trick**: reward = `(1−spirit)·own + spirit·team_mean`, the same form as Necto's `team_spirit = 0.6`.
- **AlphaStar** (Vinyals et al., *Nature* 575, 350–354, 2019): league training with main agents, exploiters and a historical pool — the canonical argument that a pool, not a single opponent, prevents cycling.
- **Open-Ended Learning Leads to Generally Capable Agents** (arXiv:2109.13286): populations and open-ended task/opponent generation produce general capability that single-opponent training does not.
- **Fictitious / prior-responding self-play** (Heinrich & Silver, arXiv:1603.01121; PSRO, Lanctot et al., NeurIPS 2017): averaging over historical opponents approximates a game-theoretic solution and dampens cycles.
- **Population Based Training** (arXiv:1711.09846): online hyperparameter adaptation; relevant later, not for Phase 1.

**Recommendation for this project [E], staged:**

| Stage | Opponent set | Mechanism |
|---|---|---|
| **S1** | Current policy drives **both** cars (pure self-play) | Single `PPOAgentController`, no `MultiAgentController` needed |
| **S2** | Current policy + **uniformly sampled historical checkpoint** | `MultiAgentController.choose_subcontrollers` returns `"current"` or `"hist_<k>"`; inference-only subcontrollers loaded from `agent_controller_checkpoints` |
| **S3** | Current policy + historical pool + **fixed external baseline** (Nexto, or a scripted/random opponent) | Same mechanism; the baseline subcontroller never learns |
| **S4** | Blue/orange **asymmetric** league (two learning controllers, as in `multi_agent_controller_test.py`) | Two `PPOAgentController`s with different seeds/LRs |

**Anti-collapse measures [E]:**
- Keep a **fixed pool** of ≥5 historical checkpoints spanning the recent skill range, refreshed every N timesteps, rather than "last K checkpoints" — this prevents the pool from drifting with a collapsing current policy.
- **Track skill, not reward.** Reward rises in self-play without any improvement (the ZealanL guide makes this exact point **[R]**). Use win rate against a *frozen* reference set (§10).
- **Monitor entropy and KL.** `actor_entropy` collapsing toward 0 with rising reward is the signature of premature convergence. `kl_divergence` and `sb3_clip_fraction` spiking means updates are too large — reduce LR or `n_epochs`.
- **Do not reduce `ent_coef` to "refine" mid-run** unless the run is being retired; ZealanL is explicit about this **[R]**.
- **Exploit prevention:** audit reward components individually. `rlgymppo_rs` auto-tracks any component named `Reward/*` **[V]** and Lucy-SKG's headline contribution is a reward analysis/visualisation library **[V]** — the pattern to copy. Log each reward term separately and periodically watch matches in rlviser.

### 9.4 Discrete vs. continuous actions

**[E] Recommendation: discrete, 90 actions, tick skip 8.** Reasons, all evidenced:
- `LookupTableAction` reduces 1944 raw combinations to **90** with explicit pruning **[V]** — the exploration burden is far smaller than an 8-dimensional continuous box.
- It is the **community standard**: identical to Necto's table **[V]**, used by PILOT (*"one of 90 discrete controller combos (Necto's lookup table), held for 8 physics ticks, so the bot makes 15 decisions per second"* **[R]**) and by the CoG 2022 sim-to-sim paper's multi-discrete design **[V]**.
- RL‑RPT's analysis of why replay-based learning fails includes: *"Buttons are not fully independent, e.g. jump+pitch is very different from just pitch or just jump, so a fixed set of possible actions is required for the best possible performance."* **[V]**
- ZealanL: *"Continuous actions means the bot can use any partial input, which allows for more precise input. However, this is more difficult to use, and I do not recommend it as your first action parser."* **[R]**
- `MultiDiscreteFF` was **removed** from rlgym-learn-algos, so multi-discrete is not an off-the-shelf option **[V]**.
- **Cost, stated honestly:** discrete actions cannot express partial throttle/steer. That caps fine car control (e.g. precise dribble touches, delicate aerials). This is a real ceiling, accepted deliberately for Phase 1.

**Tick skip 8 (15 Hz)** is the documented default everywhere **[V]** and matches the frameskip philosophy in OpenAI Five (frameskip 4 at 30 fps = 7.5 Hz) **[V]**. A smaller tick skip increases decisions/s and cost per game-second without an established benefit; changing it invalidates checkpoints. Keep 8.

### 9.5 Curriculum

**RLGym v2 provides only kickoff state setting [V]**, so a curriculum is *implementation work*, not configuration. Recommended progression, each stage gated on measured behaviour:

| Stage | State setter | Goal | Exit criterion |
|---|---|---|---|
| C0 | `KickoffMutator` only | Touch the ball at all | Touch rate per episode above a threshold |
| C1 | **Random ground states** (custom mutator: random positions/orientations, ball on ground, zero-ish velocities) | Reach and hit the ball from anywhere | Goal events occur without kickoff |
| C2 | Kickoff + random ground, 50/50 (BotBoi's documented split **[R]**) | Score from kickoff | Positive goal differential vs. own older checkpoint |
| C3 | + airborne / random-velocity states | Recoveries and aerial challenges | Air-touch rate above threshold |
| C4 | + defensive states (ball near own goal, opponent in possession) | Defending | Concede rate drops vs. fixed opponent |
| C5 | Full match length, no truncation shortcuts | Winning | Win rate vs. frozen pool |

Also worth doing **[E]**: mix **1v1 and 2v2** episodes (BotBoi does 50/50 **[R]**) once 1v1 is stable — it forces positioning rather than pure ballchasing. Do this *after* 1v1 works, never instead of it. ZealanL: *"I recommend starting with the default of 1, as making bots that can play with teammates has added challenges."* **[R]**

### 9.6 Checkpoint selection and reproducibility

**[V]** `PPOAgentController` provides `save_every_ts`, `n_checkpoints_to_keep`, `checkpoint_load_folder`, `save_mid_iteration_data_in_checkpoint`, and `run_suffix` (a string factory appended to the run name — use it to encode the experiment id). Checkpoints contain actor+critic, optimizers, `misc.json`, the experience buffer (zip) and metrics-logger state, so **resumption is exact, not approximate**.

**Reproducibility requirements [E]:**
- Pin every dependency to an exact version, including the `+cuXXX` local version of torch, and record it in the run metadata.
- Persist the generated `config.json` per run (rlgym-learn already writes it — commit a copy into the run folder).
- `random_seed` is available on `BaseConfigModel` and `PPOAgentControllerConfigModel`. Set both. Note that with `n_proc` env processes, **full determinism is not achievable** — process scheduling affects the interleaving of trajectories. Reproducibility here means *"same config ⇒ statistically indistinguishable learning curves"*, verified by running 2–3 seeds per experiment.
- **Retain policy snapshots forever, separate from resumable checkpoints.** BotBoi's scheme is the right one **[R]**: `policies/<steps>.pt` every 50 M steps kept forever; `checkpoints/` every 10 M steps with only the last 5 kept. This is what makes an opponent pool and regression testing possible.
- ⚠️ Never store anything else in `agent_controller_save_folder` — the trainer *"MAY DELETE ANYTHING INSIDE THIS FOLDER"* **[V]**.

### 9.7 Diagnosing stalled learning

The metrics already emitted **[V]** map directly to causes:

| Symptom | Likely cause | Action |
|---|---|---|
| `actor_entropy` → 0, reward flat | Premature convergence / exploration exhausted | Raise `ent_coef`, lower LR, verify reward is not saturated |
| `kl_divergence` and `sb3_clip_fraction` high | Updates too large | Lower LR, reduce `n_epochs`, check `max_grad_norm` |
| `critic_loss` large and not falling | Critic cannot fit the return scale | Check `standardize_rewards`; reduce `reward_clip` mismatch; raise critic capacity |
| `return_standard_deviation` exploding | A reward term is unbounded | Audit components; enforce `reward_clip` |
| `average_reward` rises, win rate vs. frozen pool flat | **Reward farming, not improvement** | Re-audit the densest term; anneal it to 0 |
| `timestep_collection_time` ≫ `batch_consumption_time` | CPU/env-bound | `n_proc`, float32 obs, affinity, cheaper obs builder |
| `batch_consumption_time` ≫ collection | GPU-bound | More `n_minibatches`, smaller net, `n_epochs=1` |
| Everything flat from the start | Reward scale or obs normalization broken | Check obs range (should be O(1)), check reward magnitude, verify `device` is not `"cpu"` |
| Learning works, deployment is bad | **Obs mismatch between sim and game** | Run the parity test (§7.6) |

### 9.8 Imitation learning, demonstrations, scripted opponents, pretrained policies

| Option | Evidence | Verdict |
|---|---|---|
| **Replay pretraining (RL‑RPT / VPT IDM)** | Works; produced Ripple. Requires training an IDM over ±20 frames, curating SSL replays, handling replay reconstruction loss. Author calls it *"a jumping off point"* **[V]** | **Defer.** Real but expensive; GPL‑3.0 |
| **Distil from a strong teacher** | `rlgymppo_rs` implements exactly this, with `Transfer/loss` and `Transfer/accuracy` **[V]** | **Defer** to the deployment-sizing phase — this is about making a *small* deployable model, not about learning to play |
| **Load Nexto weights as a starting policy** | Weights are public **[V]**, but the architecture is EARL Perceiver with a 36/55-feature entity obs, **not** `DefaultObs` with 92 floats | **Not feasible** without also adopting Nexto's obs and architecture. It is also a licensing question (§11) |
| **Nexto as a fixed evaluation opponent** | Working RLBot v5 inference exists **[V]**; `rlgymppo_rs` treats it as a fixed MMR-1500 baseline **[V]** | **Yes — high value.** Best available external yardstick |
| **Scripted / random opponents in the training pool** | rocket-learn supports "Training against Hardcoded/Pretrained Agents" **[V]**; `MultiAgentController` makes an inference-only subcontroller trivial | **Yes, cheap.** A random-policy subcontroller is a useful floor for verifying the pipeline learns anything at all |
| **Behaviour cloning of a rule-based bot** | RL‑RPT notes TensorBot and Levi are trained on bot inputs/outputs and *"it's doubtful that method would ever outperform the bot it's trained on"* **[V]** | **No** — hard ceiling at the teacher's skill |

### 9.9 Recommended initial objective

**[E] 1v1, soccar, Octane, discrete 90 actions, tick skip 8, `GoalReward` dominant with annealed shaping, pure self-play (current policy on both cars), `KickoffMutator` → then add a random ground state setter, `(256,256,256)` actor and critic, GPU learner, checkpoints every 1 M steps, policy snapshots retained forever.**

Progression, each step requiring the previous to pass its gate:
1. **Beats a random-policy opponent decisively** (pipeline sanity).
2. **Beats its own 10 M-step checkpoint** (learning is happening).
3. **Beats a simple scripted ballchaser** (game competence, not just reward).
4. **Positive goal differential over 400+ games vs. a frozen pool** (real strength).
5. **Then** consider 2v2, larger networks, opponent pools, curriculum C3–C5, and only then Nexto as a yardstick.

Do not attempt 2v2, 3v3, EARL-style attention, replay pretraining or multi-machine distribution in the first iteration. Every one of those is a separate research project.

---

## 10. Evaluation and competitive validation

### 10.1 Why win rate, not reward

**[R]** ZealanL, verbatim: *"Since the bot is playing against itself, an increase in reward, goals, or most other metrics do imply a change in gameplay, but not improvement."* And: *"For my first rlgym-ppo bot I wrote a little Python script that used rlgym-sim to run an infinite game between two versions of my bot. This could run far faster than real-time (unlike RLBot) so I was quickly able to see if the bot was actually improving."*

**[R]** BotBoi: *"Training reward alone does not show improvement in self-play, so check against older snapshots."*

**Conclusion [E]:** reward is a *training* signal. **Win rate against a fixed opponent is the only valid progress metric.** Everything in this section follows from that.

### 10.2 Match counts and confidence — computed

Win rate is a binomial proportion. 95 % CI half-width at p = 0.5 **[C]**:

| n games | ±half-width |
|---|---|
| 100 | ±0.098 |
| 200 | ±0.069 |
| **400** | **±0.049** |
| 1000 | ±0.031 |
| 2000 | ±0.022 |

Wilson 95 % **lower bounds** on the observed win rate **[C]** — this is what a promotion gate should threshold:

| n | 52 % | 55 % | 58 % | 60 % |
|---|---|---|---|---|
| 200 | 0.451 | 0.481 | 0.511 | 0.531 |
| **400** | 0.471 | **0.501** | 0.531 | 0.551 |
| 600 | 0.480 | 0.510 | 0.540 | 0.560 |
| 1000 | 0.489 | 0.519 | 0.549 | 0.569 |

**Read the table:** at n = 400, a 52.5 % win rate has a lower bound *below* 0.50 and must not be promoted; 55 % just clears it. At n = 1000 you need ~53.1 %.

Games needed to *detect* a difference from 50 % with α = 0.05 two-sided and 80 % power **[C]**:

| True win rate | n per side |
|---|---|
| 55 % | 1 562 |
| 58 % | 606 |
| **60 %** | **385** |
| 65 % | 167 |

**[E] Operating rules:**
- **Routine regression checks:** ≥ **400** games per pairing.
- **Promotion decisions:** ≥ **1000** games, and require the Wilson lower bound > 0.50 *and* goal differential's CI to exclude 0.
- **Reporting a skill claim:** ≥ **2000** games.
- Always report the interval, never the point estimate alone.
- Alternate sides (blue/orange) across games — kickoff assignment is shuffled per episode **[V]**, but side-specific asymmetries can still exist.

### 10.3 Evaluation harness requirements [E]

- **Out of process**, separate from training, so it never perturbs throughput measurements.
- **Greedy (argmax) actions**, not sampled — `beta = 1` in Nexto's terms **[V]**. Stochastic evaluation measures exploration, not skill.
- **Fixed seeds** for the state setter, so two policies face the same sequence of situations.
- **Same episode rules as training** (goal termination, no-touch and absolute timeouts) so results are comparable across checkpoints.
- **Side-balanced**: run N/2 as blue and N/2 as orange.
- Record **goals for, goals against, goal differential, episode length, and the reason each episode ended**.
- Run on **RocketSim**, not the game — RocketSim runs far faster than real time and 400–2000 games in RLBot at 120 tps would take many hours. Use the game only for the final transfer check.

### 10.4 Model-promotion policy [E]

A candidate checkpoint is promoted to `incumbent` **only if all** hold:

1. **≥ 1000 games** vs. the current incumbent.
2. **Wilson 95 % lower bound on win rate > 0.50.**
3. **Goal differential > 0** with a 95 % CI excluding 0.
4. **No regression on the fixed baseline set**: win rate vs. *each* frozen baseline (random, scripted ballchaser, oldest retained snapshot, best-ever snapshot) must not drop by more than the CI width.
5. **No reward-component anomaly**: no single shaping term contributes more than a defined share (e.g. 30 %) of mean episodic return — a cheap farming detector.
6. **Visual spot-check**: watch ≥ 3 matches in rlviser. Cheap, and the only defence against a policy that wins by exploiting a simulator quirk.

On failure, the candidate is **retained in the historical pool** (it may still be a useful opponent) but is not promoted. This is exactly the distinction the AlphaStar and open-ended-learning literature draws between *the agent being trained* and *the agents used as opponents*.

### 10.5 Baselines

| Baseline | Purpose | Cost |
|---|---|---|
| **Random policy** | Pipeline sanity — proves learning happens at all | Free |
| **Scripted ballchaser** (drive toward ball, hit it) | Minimum game competence | ~50 lines |
| **Own historical snapshots** (≥5, spanning the run) | Detects collapse and cycling | Free |
| **Best-ever snapshot** | Prevents slow regression over a long run | Free |
| **Nexto (public weights)** | The only strong external yardstick | Requires the RLBot v5 path or `rlgymppo_rs`'s Rust port |
| **Other RLBotPack bots** | Diversity check in the real game | Windows + RLBot |

`rlgymppo_rs` sets `nexto_mmr: Some(1500.0)` as its fixed reference **[V]** — a useful precedent for anchoring an Elo scale.

### 10.6 Throughput, sample efficiency, utilisation

Track per run: **`overall steps/s`** (the honest number — collection *and* learning), `timestep_collection_time` / `batch_consumption_time` split, GPU peak memory via `torch.cuda.max_memory_allocated()`, host RSS per env process, and steps/day. Note that `nvidia-smi` GPU utilisation is **not reliably available under WSL 2** **[V]** — plan monitoring accordingly.

### 10.7 Sim performance vs. real-game performance

**[E] These are different quantities and must be reported separately.** A policy can be strong in RocketSim and weaker in the game because of the transfer gaps in §7.6. Report:
- **Sim win rate** — the training-time metric, cheap, high-N.
- **Game win rate** — the transfer metric, expensive, low-N, on Windows.
- **The gap between them** — the actual sim-to-game fidelity measurement for *this* policy. The CoG 2022 paper's method (train in sim, evaluate the same weights in the game, report both) **[V]** is the template.

Never quote a sim win rate as if it were game strength.

---

## 11. Licensing

| Component | License | Implication |
|---|---|---|
| RocketSim | **MIT** **[V]** | Permissive. Note the author's stated anti-cheating position and that it contains no game code **[V]** |
| rlgym-api, rlgym-rocket-league | **Apache‑2.0** **[V]** | Permissive, with patent grant |
| rlgym-learn, rlgym-learn-algos | **Apache‑2.0** **[V]** | Permissive |
| rlviser / rlviser-py | **MIT** **[V]** | Permissive |
| RLBot (framework), RLBot/core, RLBotGUI | **MIT** **[V]** | Permissive, **but** `Bridge.dll` is closed-source under a Psyonix agreement **[V]** |
| `rlbot` Python client | see PyPI/repo | Beta (`2.0.0b56`) **[V]** |
| **Necto / Nexto** | **`NOASSERTION`** — a custom `LICENSE` file GitHub cannot classify **[V]** | **Treat as restrictive until read.** Do not copy code or redistribute weights without reading that file |
| **rocket-learn** | **No license file declared** **[V]** | Legally ambiguous — read for ideas, do not copy code |
| **rlgymppo_rs** | **GPL‑2.0** **[V]** | **Copyleft.** Using it would propagate to the training system. Decision required before adoption |
| **replay-pretraining (RL‑RPT)** | **GPL‑3.0** **[V]** | Copyleft, stronger |
| EARL-pytorch | see repo | Needed only if adopting the Nexto architecture |
| PyTorch | BSD | Permissive |

**[E] Recommended posture:** keep the entire training system on the Apache‑2.0/MIT stack (RLGym v2 + rlgym-learn + rocketsim + PyTorch), which is permissive and self-consistent. Treat `rlgymppo_rs` (GPL‑2.0) and `replay-pretraining` (GPL‑3.0) as *design references* rather than dependencies. Read `Necto/LICENSE` before using anything from that repository, and do not redistribute Nexto weights.

---

## 12. Compute reality check

This section exists so that expectations are set by arithmetic rather than optimism.

**Known reference point [R]:** Nexto ≈ 250 000 hours of gameplay, collected from 10–40 contributing clients over several months.

**Conversion [C]:** 250 000 hours = 9.0 × 10⁸ game-seconds. At 15 agent-decisions/s (tick skip 8), that is **1.35 × 10¹⁰ ≈ 13.5 billion agent steps**.

**What that means for this machine.** Let `S` = measured `overall steps/s` (unknown until gate **B3**). Time to reach 13.5 B steps:

| measured `S` | wall-clock to 13.5 B steps |
|---|---|
| 2 000 | 78 days |
| 5 000 | 31 days |
| 10 000 | 15.6 days |
| 20 000 | 7.8 days |

**[C] Note the units carefully:** these are *continuous, single-machine* days. Nexto's 250 000 hours were collected by many contributors in parallel over months; the equivalent single-machine figure above is the honest comparison, and it is why the roadmap targets *milestones*, not "reach GC1".

**Also note [V]:** the RLGym tutorial's own framing — *"If you plan on making a good bot, expect to see numbers in the tens of billions someday"* and *"1 billion steps is ~18 500 hours of Rocket League"* — is consistent with this scale.

**Implication for the plan [E]:**
- The binding constraint is **steps/day**, so throughput work (§8.5) has a direct, multiplicative payoff on every downstream milestone.
- **Sample efficiency matters more than raw scale here.** That is the argument for: dominated-by-winning rewards, annealed shaping, potential-based terms, a sensible curriculum, and *not* wasting compute on 3v3 or attention architectures early.
- Set expectations as **"a bot that is measurably better than its previous version every week, validated against frozen opponents"** — not a rank.

---

## 13. Compatibility matrix — verified

| | Constraint | Value |
|---|---|---|
| Python | rlgym-learn / algos | **>=3.10, <=3.14** **[V]** |
| Python | rlgym-api / rlgym-rocket-league | >=3.7 **[V]** |
| Python | rocketsim | >=3.9 **[V]** |
| Python | rlviser-py | >=3.10 **[V]** |
| Python | torch 2.14.1 | >=3.10 **[V]** |
| Python | numpy 2.5.4 | **>=3.12** **[V]** |
| Python | RLBot v5 client | "3.12 or later" **[V]** |
| **⇒ chosen** | | **Python 3.12** |
| numpy | rlgym-rocket-league | >=1.19, <3.0.0 **[V]** |
| numpy | rlviser-py | **>=2.0.0** **[V]** |
| numpy | rlgym-learn / algos | >1.21 (no upper bound) **[V]** |
| numpy | **rlgym-ppo `requirements.txt`** | **>1.21, <2.0** — conflicts with rlviser-py **[V]** |
| rocketsim | required by rlgym-rocket-league[sim] | >=2.0.0, **<3.0.0** **[V]** |
| rocketsim | required by rlviser-py | >=2.2.1 **[V]** |
| torch | rlgym-learn-algos | >1.13 **[V]** |
| pydantic | rlgym-learn | >=2.13.0 **[V]** |
| CUDA | WSL 2 | Pascal+ in WDDM; RTX 4060 Ti = sm_89 ✅ **[V]** |
| OS | rlgym-learn env backend | **Linux (or WSL 2). Native Windows fails.** **[R]+[V]** |
| OS | Rocket League + RLBot v5 | **Windows** (or Linux + Proton) **[V]** |
| .NET | RLBot/core build | .NET 10 SDK **[V]** |

**Conflicts to avoid:** do not install `rlgym-ppo` in the same environment as `rlgym[rl-rlviser]` **[V]**. Do not install any Linux NVIDIA driver inside WSL 2 **[V]**. Do not install the `cuda` / `cuda-drivers` metapackages inside WSL 2 **[V]**.

---

## 14. Unresolved questions

Each of these is answerable only by a local experiment. They are turned into named benchmark gates in `ROADMAP.md`.

| # | Question | Gate |
|---|---|---|
| U1 | What is `overall steps/s` on this machine at `(256,256,256)`, 1v1, and at what `n_proc` does it peak? | **B2, B3** |
| U2 | What is the per-env-process RSS, and therefore the real `n_proc` ceiling under 32 GB (and whatever fraction WSL 2 exposes)? | **B2** |
| U3 | Does pinning env processes to P-cores (and the learner elsewhere) improve throughput on the 12700F's hybrid topology? By how much? | **B2c** |
| U4 | What is the largest network × minibatch that fits 8 GB VRAM while keeping `batch_consumption_time` ≤ ~20 % of `iteration_time`? | **B4** |
| U5 | Does WSL 2's pinned-memory limitation measurably slow CPU→GPU minibatch transfer vs. expectations? | **B4** |
| U6 | How much does `Float32DefaultObs` + `min_frac_process_responses_per_collection` tuning actually buy? | **B3** |
| U7 | Is RocketSim's `set_state` (full car teardown per reset) a material cost at short episode lengths? | **B2** |
| U8 | What is the actual sim-vs-game performance gap for a trained policy, and what is the observation parity error? | **B6** |
| U9 | How many env processes can run before `min_frac_process_responses_per_collection` straggler-wait dominates? | **B3** |
| U10 | Is rlviser usable at all under WSLg on this machine (`/dev/dri` present?), or is native-Windows rendering required? | **B1** |
| U11 | What win rate does the public Nexto achieve against a simple scripted ballchaser under RLBot v5 on this setup? (Needed to calibrate the external baseline.) | **B7** |
| U12 | Does `Necto/LICENSE` permit using Nexto weights as an evaluation opponent? | **Phase 0 desk check** |

---

## 15. Sources

**RocketSim**
- Repo: https://github.com/ZealanL/RocketSim · Releases: https://github.com/ZealanL/RocketSim/releases
- Docs: https://zealanl.github.io/RocketSimDocs/ · Arena: https://zealanl.github.io/RocketSimDocs/arena.html
- PyPI: https://pypi.org/project/rocketsim/
- Rust bindings: https://github.com/Virxec/rocketsim-rs · v3 branch: https://github.com/ZealanL/RocketSim/tree/v3-rust

**RLGym**
- Docs: https://rlgym.org/ · Quick start: https://rlgym.org/Getting%20Started/quickstart/ · Training an agent: https://rlgym.org/Rocket%20League/training_an_agent/ · RLGym Learn: https://rlgym.org/RLGym%20Learn/introduction/
- API reference: https://captainglac1er.github.io/rocket-league-gym/
- Repo: https://github.com/RLGym/rlgym · PyPI: https://pypi.org/project/rlgym/ , https://pypi.org/project/rlgym-api/ , https://pypi.org/project/rlgym-rocket-league/ , https://pypi.org/project/rlviser-py/
- Superseded v1 sim fork: https://github.com/AechPro/rocket-league-gym-sim
- Source files read: `rlgym/api/rlgym.py`, `rlgym/rocket_league/obs_builders/default_obs.py`, `action_parsers/lookup_table_action.py`, `action_parsers/repeat_action.py`, `sim/rocketsim_engine.py`, `state_mutators/*.py`, `api/game_config.py`, `common_values.py`, `setup_rocket_league.py`

**Learners**
- RLGym-PPO: https://github.com/AechPro/rlgym-ppo (`learner.py`, `batched_agents/*`, `setup.py`, `requirements.txt`, `rlgym_v2_example.py`)
- rlgym-learn: https://github.com/JPK314/rlgym-learn (`CHANGELOG.md`, `basic_config.py`, `learning_coordinator*.py`, `env_processing/env_process.py`, `src/env_process.rs`, `pyproject.toml`, `quick_start_guide.py`, `speed_test.py`, `multi_agent_controller_test.py`) · https://pypi.org/project/rlgym-learn/
- rlgym-learn-algos: https://github.com/JPK314/rlgym-learn-algos (`agent_controller/multi_agent/*`, `ppo/ppo_agent_controller.py`, `ppo/ppo_learner.py`, `ppo/gae_trajectory_processor.py`, `ppo/experience_buffer_numpy.py`, `ppo/actor_critic/discrete_actor.py`, `CHANGELOG.md`) · https://pypi.org/project/rlgym-learn-algos/
- rocket-learn: https://github.com/Rolv-Arild/rocket-learn
- rlgymppo_rs: https://github.com/Virxec/rlgymppo_rs
- Community guide (v1-era): https://github.com/ZealanL/RLGym-PPO-Guide (`intro.md`, `learner_settings.md`, `making_a_good_bot.md`, `rewards.md`)

**Reference agents**
- Necto/Nexto: https://github.com/Rolv-Arild/Necto (`training/agent.py`, `training/reward.py`, `training/parser.py`, `training/requirements.txt`, `rlbot-support/Nexto/*`)
- Nexto on RLBot v5: https://github.com/JPK314/doublesize-nexto-v5 (`src/bot.py`, `src/nexto_obs.py`)
- rlgym-compat: https://github.com/JPK314/rlgym-compat
- EARL: https://github.com/Rolv-Arild/EARL-pytorch
- Replay pretraining: https://github.com/Rolv-Arild/replay-pretraining
- WSL 2 references: https://github.com/CPollreis/rocket-league-bot (PILOT) · https://github.com/RafeSymonds/rocket-league-bot (BotBoi)

**RLBot v5**
- https://rlbot.org/ · https://wiki.rlbot.org/v5/ · https://wiki.rlbot.org/v5/framework/v5/ · https://wiki.rlbot.org/v5/framework/architecture/ · https://wiki.rlbot.org/v5/framework/operating-system-support/ · https://wiki.rlbot.org/v5/botmaking/tick-rate/ · https://wiki.rlbot.org/v5/botmaking/game-data/ · https://wiki.rlbot.org/v5/botmaking/ball-path-prediction/ · https://wiki.rlbot.org/v5/botmaking/config-files/
- https://github.com/RLBot/core · https://github.com/RLBot/python-example · https://github.com/RLBot/gui · https://github.com/RLBot/flatbuffers-schema · https://pypi.org/project/rlbot/
- Historical: https://github.com/RLBot/RLBot/wiki/Tick-Rate · https://github.com/samuelpmish/RLBot

**Hardware / platform**
- Intel ARK i7‑12700F: https://www.intel.com/content/www/us/en/products/sku/134592/
- NVIDIA RTX 4060 Ti: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
- CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html · https://developer.nvidia.com/cuda/wsl
- Microsoft: https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
- Ubuntu on WSL GPU: https://ubuntu.com/wsl/docs/stable/howto/gpu-cuda/
- PyTorch install: https://pytorch.org/get-started/locally/ · https://pypi.org/project/torch/

**Papers**
- PPO — Schulman, Wolski, Dhariwal, Radford, Klimov (2017). https://arxiv.org/abs/1707.06347
- GAE — Schulman, Moritz, Levine, Jordan, Abbeel (ICLR 2016). https://arxiv.org/abs/1506.02438
- OpenAI Five — Berner et al. (2019). https://arxiv.org/abs/1912.06680
- AlphaStar — Vinyals et al., *Nature* 575, 350–354 (2019). https://doi.org/10.1038/s41586-019-1724-z
- Open-Ended Learning — Team et al. (2021). https://arxiv.org/abs/2109.13286
- PBT — Jaderberg et al. (2017). https://arxiv.org/abs/1711.09846
- Fictitious self-play — Heinrich & Silver (2016). https://arxiv.org/abs/1603.01121
- PSRO — Lanctot et al. (NeurIPS 2017). https://arxiv.org/abs/1711.00832
- Rocket League sim-to-sim — Pleines et al., IEEE CoG 2022. https://arxiv.org/abs/2205.05061 · https://doi.org/10.1109/CoG51982.2022.9893628
- Lucy-SKG — Moschopoulos, Kyriakidis, Lazaridis, Vlahavas (2023). https://arxiv.org/abs/2305.15801
- Perceiver — Jaegle et al. (2021). https://arxiv.org/abs/2103.03206
- VPT — Baker et al. (2022). https://arxiv.org/abs/2206.11795
- Potential-based reward shaping — Ng, Harada, Russell (ICML 1999)
