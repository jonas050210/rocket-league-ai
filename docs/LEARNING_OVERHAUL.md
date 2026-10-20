# Learning overhaul — root causes and changes

Date: 2026-10-19 · Trigger: run `1e2bad3aedb7` (2,001,920 steps, final
checkpoint `m-20261019-143324-0002001920-final`)

Measured facts of that run: 6,816 episodes, 6,811 truncated, **5 goals**
against `idle`, mean reward 7.37 (alignment 4.21, progress 3.15, approach
0.69, touch 0.015, goal 0.007), curriculum stuck at `ball-control`,
evaluation vs `chase` 0W-4L-2D with 0:10 goals. PPO losses were stable;
playing strength did not move. This document records *why*, with code
evidence, and what changed. **No improvement is claimed as proven** — that
requires the next training run plus a comparable evaluation (see
"Expected evidence" below). No training or long benchmark was started for
this overhaul; existing runs and checkpoints are untouched.

## Root causes (in impact order)

### RC1 — The reward paid proxies, not football (`src/rlai/rewards.py`)

* `alignment` (0.02/step) and `approach` (0.03/step) were per-decision
  **state** bonuses. A car parked facing the ball earned
  225 × 0.02 = **4.50** per 15 s episode; the run's mean alignment of 4.21
  is 94 % of that maximum — the policy had essentially converged to the
  farming optimum.
* `progress` measured **car→ball** distance, i.e. chasing, anywhere on the
  field. The single most important shaping signal — ball→opponent-goal
  progress — did not exist.
* Scoring ends the episode (`GoalCondition`, `src/rlai/envs.py`), so a goal
  at decision 50 (+10) forfeited up to ~175 × (0.02+0.03) ≈ **8.75** of
  remaining state bonuses. Delaying or avoiding goals was near-rational.
* `touch` was direction-blind and tiny next to the state bonuses; the
  implied rate is ~0.06 touches/episode. `defense` paid for *being behind
  the ball while it travelled toward the own goal* (and used the opponent
  goal's distance as its scale) — another farmable state term.
* Goal reward contributed 0.007 of the 7.37 mean return (0.1 %). The real
  objective was noise in its own return.

### RC2 — Reward accounting mixed both teams (`rewards.py`, `envs.py`)

`shared_info["reward_stats"]` summed **all** agents. In 1v1 the reported
component means therefore mixed the policy's and the idle opponent's
shaping; components did not sum to the policy return (verified live:
component sum 3.34 vs return 2.69 on a probe episode). Diagnostics
described nobody's reward.

### RC3 — Curriculum stage 0 tested scoring, not ball control (`src/rlai/curriculum.py`)

`ball-control` required a 60 % episode **win** rate vs `idle`. Episodes end
on the first goal, so a "win" means scoring within the horizon: the stage
demanded reliable scoring from a cold start (actual: 5/6816 ≈ 0.07 %).
`observe()` returned `None` silently forever; there was no blocked-reason,
no touch-based criterion, no demotion, and the evidence window was dropped
on every resume.

### RC4 — Kickoff-only starts against a passive opponent (`envs.py`, configs)

Every episode began from a kickoff vs `idle`. A weak policy almost never
visits shooting states, so goal credit has nothing to attach to, and the
learned timing does not transfer to a contesting `chase` opponent (0:10).

### RC5 — Horizon restored against the request; unit confusion (`resume.py`, `workers.py`)

`episode_seconds` was an *inherited* field, so the checkpoint's 15 s
replaced the requested 60 s. The "~1,800 steps" truncation length was
physics **ticks** (15 s × 120 Hz) stored in a field everyone reads as
decision steps (true length: 225 decisions at `tick_skip=8`). With
γ=0.995 at 15 Hz the effective horizon is ~200 decisions — a goal at the
end of a 225-decision episode is worth 32 % at its start.

### RC6 — Own speed unobservable (`src/rlai/obs.py`)

v1/v2 contain only *relative* velocities; the single-frame MLP has no
memory, so the car's own linear/angular velocity cannot be recovered.
Braking, shot power and recovery all depend on it.

### What was *not* broken

PPO truncation semantics were already correct and are unchanged:
time-limit ends bootstrap from V(final state), goal terminations do not,
and the GAE recursion is cut at every episode boundary (`src/rlai/ppo.py`,
`src/rlai/workers.py`, pinned by `tests/test_gae_truncation.py`). Secondary
issue kept in view: `clip_value=0.2` is an *absolute* value clip against
returns of magnitude ~10, which slows value fitting; the recommended
config uses 0.5 and `losses/explained_variance` is already logged to judge
it with data.

## Changes

| Area | Change |
| --- | --- |
| Reward | New `ball_progress` potential (task), weaker car→ball `progress`, directional `touch`, event `shot` and `defense`, legacy `alignment`/`approach`/`boost`/`bump` default **0** (old configs still reproduce old behaviour exactly). Per-agent components **and event counts**; env reports the learning team only, so components sum to the return. |
| Curriculum | Stages test their own skill: `first-touch` (touches/ep ≥ 1.5, drill resets, 20 s) → `shooting` (goals/ep ≥ 0.2 vs idle, mixed resets, 30 s) → `chase` (win ≥ 30 %) → `mixed` (≥ 40 %) → self-play. Exact `blocked_reason()`, per-criterion status, demotion, stagnation warnings, evidence window persisted in checkpoints, stages without criteria are config errors. |
| Resets | `reset_mode`: `kickoff` \| `drill` \| `mixed`. Drill starts put every car behind a randomly placed (sometimes rolling) ball so the last seconds of an attack are practiced directly. Chosen per curriculum stage or by config. |
| Workers | Episode length in **decision steps** (+ `length_ticks`); episodes carry touches/shots; envs continue across rollout collections instead of discarding one partial episode per worker per iteration; per-stage reset mode/horizon applied by rebuilding the env on transitions. |
| Config/resume | `episode_seconds`/`rollout_steps` now follow the request. Inherited fields can be changed deliberately via `--override-config <fields>` (CLI/headless/resume-plan/API `config_overrides`) — a recorded *phase change*. Plans now contain real `overrides`, `effective_config`, `phase_change` and an LR-schedule warning when `total_steps` changes. |
| Observation | **v3** (204): + own linear/angular velocity in car frame. v1/v2 unchanged; module default stays v2 so existing configs/checkpoints keep their meaning. Resume validation still refuses layout mixing. First-frame clock now uses the configured horizon. |
| Evaluation/diagnostics | Episodes store touches/shots (additive DB migration); report gains `learning_evidence` (first vs last 200 episodes: outcomes, touches, shots, conversion — plus a "reward up, play flat ⇒ proxy farming" warning) and a `curriculum` status block. `best` additionally needs ≥ 2 decisive games. Recommended eval: 10 games vs `idle` **and** `chase`, both sides. |
| Configs | All examples updated to the new reward defaults. `1v1-curriculum.json` = recommended fresh v3 run; new `1v1-continue-v2.json` = continue the existing v2 checkpoint with the fixed reward. |

## Recommended next runs (approval required — not started)

**A (recommended): fresh policy, obs v3.** The 2M-step checkpoint mainly
learned the farming optimum on an observation without own velocity;
keeping it would keep the handicap to preserve little transferable skill.

```bash
python -m rlai.cli train --config-file configs/1v1-curriculum.json
```

**B (alternative): keep the v2 checkpoint**, continue it under the fixed
reward as an explicit phase change (latest suitable checkpoint is still
selected automatically):

```bash
python -m rlai.cli resume-plan --config-file configs/1v1-continue-v2.json --override-config reward
python -m rlai.cli train --config-file configs/1v1-continue-v2.json --override-config reward
```

Key settings (A): `obs_version=3`, `curriculum=true` (stage horizons
20→30→45 s override `episode_seconds=30`), `rollout_steps=512 × 6 workers`,
`gamma=0.995`, `gae_lambda=0.95`, `clip_value=0.5`, eval 10 games vs
`idle`+`chase` on both sides every 25 iterations, `total_steps=5,000,000`.

## Expected evidence of success (falsifiable)

Within the first ~500k steps of run A, the report/metrics should show —
and if they do not, the overhaul has failed and the numbers say where:

1. `charts/touches` rising from ~0 toward **≥ 1.5/episode**; curriculum
   leaves `first-touch` (log line with criteria), then `shooting`.
2. `learning_evidence.delta.goals_per_episode > 0` with
   `touches_per_episode > 0` — reward and play moving **together** (the
   proxy-farming warning must not fire).
3. Reward composition dominated by `ball_progress`/`touch`/`shot`/`goal`,
   with `alignment`/`approach` exactly 0.
4. Periodic evaluation vs `idle` reaching wins (goals actually scored),
   then non-zero goals vs `chase`; `losses/explained_variance` clearly
   positive.
5. No silent stage: every iteration logs stage + blocked reason in
   metrics; a stalled stage produces an explicit stagnation warning
   instead of 2M silent steps.

## Known limitations

* Potential differences are undiscounted by default (γΦ′−Φ would be
  exact); at γ=0.995 the residual bias is small and favours moving the
  ball upfield early. Round 2 added the opt-in `discount_shaping` flag
  for the exact form (see below); default stays off for readable
  diagnostics.
* `best` selection still compares win rates inside one protocol; with
  10-game suites it remains a coarse pointer, not a ranking proof.
* Old episode rows in an existing database keep tick-based lengths;
  new rows are decision steps. Averages spanning the overhaul boundary
  mix units.
* Stage thresholds (1.5 touches, 0.2 goals/ep, 30/40 % win rates) are
  reasoned starting points, **not measured optima**; the diagnostics now
  exist to tune them from the next run's window statistics.
* Throughput was not re-architected in round 1; round 2 profiled it and
  doubled collection speed by removing per-step distribution/dict
  overhead (see below). Physics remains the floor.

---

# Round 2 — measured follow-ups (2026-10-19, no training started)

Profiling on the 2-vCPU/3 GB sandbox (no CUDA) showed where time and
learning signal were actually going; every change below is tied to a
measurement or a demonstrated defect.

## Throughput: inference was the bottleneck, not the learner

A real-size PPO update costs ~0.18 s on this machine, but rollout
collection ran at 913 steps/s/worker with `TorchPolicy.act` taking 44 % of
wall time — of which ~40 % was `torch.distributions.Categorical` ceremony
(dict juggling, per-agent Python loops). `act_batch` now samples via
`log_softmax` + `multinomial` over plain arrays, and `collect_rollout`
keeps everything in preallocated numpy buffers. Result: **1,811
steps/s/worker** on the same probe (~2×), remaining time split between
RocketSim and observation building. `act` keeps its exact old semantics
(hit by evaluation/RLBot/tests); a regression test pins batch/dict parity
and the true log-probabilities.

## Accuracy & anti-farming details

* `shot` now requires the ball's trajectory (with gravity) to cross the
  goal plane *inside* the posts and under the bar; hard misses are counted
  (`shots_off_target`) instead of paid. "Shoot accurately", not "shoot
  hard-ish upfield".
* Touch/shot/defense payments are rate-limited per car (one paid event
  per 4 decisions ≈ 0.53 s); contacts are still all counted, so skill
  gates stay honest while pin/tap farming is worthless.
* `discount_shaping` (default off) makes both potentials discount-exact
  (`Φ(s) − γΦ(s′)`) — the known round-1 limitation, now available as an
  explicit opt-in because it makes "reward at rest" non-zero and the
  diagnostics harder to read.

## Curriculum transfer & sample diversity

Drill resets were attack-only: the agent never practised defending in
training but conceded 0:10 in evaluation. Drills are now 45 % attack /
20 % shot (ball in the attacking third, a keeper in the goal mouth) /
20 % defense (ball rolling at the learning team's goal with speed) /
15 % midfield race; the `chase`/`mixed` stages train on `mixed` resets
(kickoff + drills) instead of pure kickoffs. Every worker also draws its
**own** opponent (mixed/self-play stages), with league results recorded
per snapshot — one batch is no longer 100 % correlated against a single
opponent. Custom stages can use `difficulty:<level>` opponents.

## Observation v4 (222)

The last orientation blind spots are closed: own up-vector (pitch/roll vs
gravity; needed for recoveries and wall/air play) and every other car's
forward vector in the own frame (where it can drive next — its velocity
fails to say this at low speed). v4 is a strict superset of v3's prefix;
all recommended configs moved to v4 before any v3 run exists. Mirror
rules are extended and tested (world-frame up x/y flip; car-frame
forwards invariant).

## Evaluation & "best" selection had a silent comparability bug

The periodic-evaluation seed included the iteration number, so **every
evaluation was a different protocol**: a new `best-<hash>.json` pointer
file each time, the primary best pointer frozen at the first protocol,
and the incumbent comparison seeded from a possibly incomparable
protocol's score. The seed is now fixed (one protocol, comparable
kickoffs), the incumbent is loaded *per protocol* from disk, and the
default suite grew from 4 to 8 games (4-game win rates were mostly
noise). Three consecutive periodic evaluations without a win or any
goal-difference/touch improvement now raise an explicit
`evaluation stall` warning (+ `eval/stalled` metric) instead of being
discovered in the final report.

## Checked and deliberately NOT changed

* **Value clipping**: a bounded synthetic experiment (goal-spike returns,
  25 repeated updates) showed clip 0.2 / 0.5 / disabled converging
  identically (MSE 1.247, explained variance 0.784) — it is not the
  bottleneck, so PPO's value loss and the default are untouched.
* Advantage normalization (per-minibatch), LR anneal tied to the run's
  step budget, GAE/truncation semantics and the 15 Hz decision rate all
  re-verified as sound; changing them without a failing measurement
  would be superstition.
* No new dependencies; no checkpoint/data deleted; no training started.

Suite after round 2: **490 fast + 6 slow tests passing**, ruff clean.
