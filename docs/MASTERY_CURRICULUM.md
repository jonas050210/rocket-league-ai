# Mastery-based 1v1 curriculum (2026-10-19)

Why the previous run never left `first-touch`, what was changed about it, and
what the new curriculum measures.  Everything asserted here is either code or
a measurement from a bounded experiment; the experiments are listed with their
numbers in [Experiments](#experiments) and their scripts are described in
[Reproduction](#reproduction).

Nothing in this document (or in the change it describes) starts training.
No existing checkpoint, run row or log was modified.

---

## 1. The bottleneck, measured

The previous analysis (`docs/STALL_ANALYSIS.md`) established that the policy
stays near-uniform and that its policy gradient is indistinguishable from
noise.  That is a *description*.  The question that decides what to change is
**why** a fresh policy collects so few ball contacts, so the candidate causes
were separated with bounded rollouts (no training, ~17 s each):

| policy | resets 700–2600 uu | resets 250–900 uu |
| --- | --- | --- |
| uniform over all 90 actions | 0.63 touches/ep, 30 % of episodes | 0.83, 30 % |
| uniform over the 24 non-aerial actions | 0.33, 13 % | 0.77, 23 % |
| always full throttle, no steering | 2.93, 73 % | 5.23, 87 % |
| scripted chase driver | 10.83, 100 % | 16.00, 100 % |

(30 episodes per cell, 20 s episodes, idle opponent, `MatchEnv` + obs v4.)

Three conclusions, one of them negative:

1. **The action space is not the problem.** 66 of the 90 actions (73 %)
   contain a jump, pitch or roll, so a uniform policy spends most of its
   decision steps in the air - but restricting sampling to the 24 non-aerial
   actions made contact *worse* (0.63 → 0.33 touches/episode), because the
   aerial part of the table also holds most of the boost combinations.  An
   action mask was therefore **not** implemented.  Measured, rejected.
2. **Forward motion is what produces contacts, and distance shortens the
   chain.** A car that only ever drives straight ahead touches the ball in
   73 % of episodes from 700–2600 uu and in 87 % from 250–900 uu, and its
   first contact moves from 36.1 to 15.7 decision steps (for the uniform
   policy: 61.4 → 54.2).  A uniform policy
   improves only from 0.63 to 0.83, because it does not drive *at* the ball -
   but the same shortening applies to every policy that starts to.
3. **The reward was not the limiting factor.** See the A/B below: re-weighting
   the reward without changing the starting distribution changed nothing.

The gradient signal-to-noise was measured directly, by computing the PPO
policy gradient of two *independent* 3072-sample batches from the same fresh
policy and taking their cosine (0 = pure noise, 1 = noise-free):

| setup | cosine (mean of 5) | touches/ep | return std sources |
| --- | --- | --- | --- |
| default reward, drill resets | 0.14 (min −0.10) | 0.40 | `ball_progress` σ 2.22, `progress` σ 0.70, `touch` σ 0.08 |
| default reward, close resets | 0.15 (min 0.00) | 0.76 | `ball_progress` σ 1.84, `progress` σ 0.71, `touch` σ 0.12 |
| touch-only reward, close resets | 0.23 (min 0.09) | 0.76 | `touch` σ 0.91 |

Two things follow.  The informative event (a touch, σ ≈ 0.1) is buried under
terms a near-uniform policy cannot control (where an accidental contact sends
the ball, σ ≈ 2) - a ~20× variance ratio.  And even so, one batch is *not*
pure noise: the cosine is 0.1–0.2, i.e. a weak but non-zero consistent
direction, which over ~16 000 updates can accumulate.  The curriculum was
therefore stuck because the useful events per update were too few (≈3 ball
contacts per 3072-sample batch), not because no gradient ever arrived.

The decisive test was an actual A/B ([experiment 7](#7-ab-does-the-new-starting-distribution-learn-faster)):

| variant | touches/ep in the last quarter (n = 3 seeds) | vs baseline |
| --- | --- | --- |
| A `drill` resets, default reward (baseline) | 0.460 ± 0.113 | — |
| B **`reach` resets**, default reward | **1.338 ± 0.102** | +0.878, se 0.088, t = 10.0, df = 4.0 |
| C `drill` resets, touch-dominant reward | 0.327 ± 0.022 | −0.132, t = −2.0 |
| D `reach` + touch-dominant reward | 1.176 ± 0.134 | +0.716, t = 7.1 |
| E `reach` + reward + 10 s episodes | 1.463 ± 0.128 | +1.003, t = 10.2 |

81 920 steps, 1 worker, obs v4, 128×128 tanh, idle opponent, isolated
`RLAI_DATA_DIR` per run.  All three seeds of B beat all three seeds of A
(per-seed q4: A 0.373/0.419/0.587, B 1.366/1.423/1.225), and B's curve rises
across the run (0.91 → 1.14 → 1.36 → 1.34) while A's is flat
(0.37 → 0.27 → 0.32 → 0.46).

**Conclusion: the starting-state distribution was the bottleneck.**  So the
only early-learning change in this revision is a new `reach` reset
distribution.  The reward weights are *unchanged* - variant C shows the
re-weighting buys nothing on its own and variant D shows it adds nothing on
top of `reach`, so there was no evidence for it and no reward-farming risk is
introduced.

---

## 2. What was changed

| file | change |
| --- | --- |
| `src/rlai/scenarios.py` | new: the `reach` starting distribution and exact scenario placement (both mirrored for the orange side) |
| `src/rlai/envs.py` | `reset_mode="reach"`; `shared_info["scenario"]` overrides the distribution; `MatchEnv(scenario=…)` |
| `src/rlai/skills.py` | new: independent skill probes (`approach`, `progress`, `shooting`, `defense`, `reengage`) and the match-gate adapter |
| `src/rlai/curriculum.py` | mastery stages, `Gate`, confirmation counts, retention baselines, diagnostics; the old stages remain as the `legacy` preset |
| `src/rlai/trainer.py` | gate evaluations on their own cadence, gate metrics/logs, regression warnings, curriculum restore by stage name |
| `src/rlai/config.py` | `curriculum_preset`, `curriculum_gate_interval`, `curriculum_gate_trials`; `reset_mode="reach"` |
| `configs/1v1-mastery-1v1.json` | the recommended 50 M-step configuration |
| `docs/TRAINING.md` | curriculum, gates and resume sections |

Reward weights, observation layout/version, action table, PPO
hyperparameters, the evaluation protocol and the checkpoint policy are
**unchanged**.

---

## 3. The stages and their mastery criteria

Promotion is decided by the probes in `src/rlai/skills.py`, never by the
training reward.  A probe plays scripted, seeded starting situations on
**both sides** of the field (the orange side through `MirroredPolicy`, exactly
like the match evaluation) and measures the **simulator state** afterwards.

Reference rates, 32 trials per probe, both sides (experiment 5):

| probe | idle | random | fresh network | scripted shooter | scripted chase |
| --- | --- | --- | --- | --- | --- |
| `approach` | 0.062 | 0.125 | 0.094 | 0.750 | **1.000** |
| `progress` | 0.000 | 0.094 | 0.156 | 0.219 | **1.000** |
| `shooting` | 0.031 | 0.031 | 0.031 | 0.344 | **0.438** |
| `defense` | 0.188 | 0.406 | 0.125 | 0.562 | **0.688** |
| `reengage` | 0.000 | 0.000 | 0.000 | **0.938** | 0.000 |

Every threshold below sits far above the do-nothing/random/fresh references
and at or below what a competent scripted driver reaches, so no stage is free
and none is unreachable by construction.  Two probes were *rejected* during
calibration because they could not discriminate - see
[Rejected probes](#rejected-probes).

| # | stage | opponent | resets | gates (point estimate + 95 % CI lower bound) |
| --- | --- | --- | --- | --- |
| 0 | `approach` | idle | `reach`, 20 s | `approach` ≥ 0.80, CI-low ≥ 0.60 |
| 1 | `control` | idle | `reach`, 20 s | `approach` ≥ 0.80, `progress` ≥ 0.70 (CI-low ≥ 0.50) |
| 2 | `shooting` | idle | `mixed`, 30 s | `progress` ≥ 0.70, `shooting` ≥ 0.30 (CI-low ≥ 0.15) |
| 3 | `defense` | chase | `mixed`, 30 s | `defense` ≥ 0.60 (CI-low ≥ 0.40), `reengage` ≥ 0.70 (CI-low ≥ 0.50) |
| 4 | `pressure` | chase | `mixed`, 30 s | `shooting` ≥ 0.30, `defense` ≥ 0.60, **match vs chase** win rate ≥ 0.50 (CI-low ≥ 0.30) |
| 5 | `integrated` | mixed | `mixed`, 45 s | `shooting` ≥ 0.30, `defense` ≥ 0.60, **match vs mixed** ≥ 0.40 (CI-low ≥ 0.20) |
| 6 | `self-play` | self | kickoff, 45 s | terminal (no gate) |

What each probe measures:

* **`approach`** - ball at rest 800/1300/1800/2400 uu away with a lateral
  offset of up to ±1200 uu, 6 s.  Success = at least one contact.
* **`progress`** - ball at rest in the own half, the car starting *beside* it
  (900 uu to the side), 10 s.  Driving straight therefore pushes the ball
  sideways: success needs contact **and** ≥ 1500 uu of movement toward the
  target goal.
* **`shooting`** - ball in the attacking third, car behind it with a ±300/600
  uu offset, idle keeper on the line, 10 s.  Success = a goal.
* **`defense`** - ball rolled **at the goal centre** from 1500–3500 uu at
  700–1300 uu/s, the car 700 uu goal-side and 300 uu to the side, 8 s.
  Success = no goal conceded.  (Aiming the ball at the goal is what makes
  this measurable: with a straight roll ~50 % of trials miss on their own and
  idle/random/chase all scored ~0.5 - no signal.)
* **`reengage`** - the car starts 700–1500 uu from the ball **facing away**
  from it, 8 s.  Success = getting back to the ball.  This is the measurable
  part of "recovery".
* **`match_*`** - a real match series through `rlai.evaluation`, side
  balanced, with the run's `eval_games`/`eval_match_seconds`.

### Promotion requires demonstrated mastery

A stage promotes only when **all** of these hold:

1. every gate passes on its point estimate **and** on its 95 % confidence
   lower bound (a lucky 4-trial probe cannot promote anything);
2. the same measurement passed `confirmations = 2` times **in a row** - one
   unusually good evaluation is not mastery;
3. the stage has run for `min_iterations` and seen `min_episodes`;
4. no skill mastered in an earlier stage regressed by more than
   `RETENTION_TOLERANCE = 0.15`.

Gate evaluations run every `curriculum_gate_interval` iterations (default 25)
with `curriculum_gate_trials` trials per probe (default 24, i.e. 12 per
side).  At 24 trials the 95 % interval is roughly ±0.2 around a 50 % rate,
which is why the CI bounds sit 0.15–0.20 below the point estimates.

### When a stage does not promote

Training **continues on that stage**; nothing is skipped and nothing is
faked.  The reason is reported exactly, for example:

```
gate[approach] approach=0.31[n=24] in 1.8s -> 0/2 confirmations; blocked by:
  stage_iterations 12 < required 20; approach/rate: 0.310 < required 0.800;
  gate_confirmations 0 < required 2
```

`Curriculum.diagnostics()` adds, per failing criterion, what to look at next
(probe details, per-episode touch rate, entropy, more evaluation games).  The
same information is in the run report under `curriculum.diagnostics` and in
the GUI.  After `stagnation_iterations` (250) the trainer additionally logs a
WARNING naming the blocking criterion.

If a *retention* check fails, the run logs a WARNING
(`skill regression: retention:approach/rate 0.42 dropped below 0.65 …`),
resets the confirmation counter and does **not** count the stage as mastered.

---

## 4. Retention: one integrated policy, not a set of specialists

There is exactly one network.  No runtime selector exists, so a stage that
"passes" by forgetting an earlier skill would produce a policy that cannot
play a match.  Therefore:

* every gate evaluation measures the active stage's skills **and** the skills
  of all earlier stages (`Curriculum.required_skills()`), under one protocol
  and in one call, so the comparison is apples to apples;
* when a stage is mastered, the measured value of each of its gates is stored
  as the retention baseline (`curriculum.gates.baselines`, persisted in the
  checkpoint);
* on every later evaluation, any earlier skill more than 0.15 below its
  baseline blocks promotion and is logged as a regression;
* the later stages keep the earlier skills in their own gate list
  (`pressure` and `integrated` both re-require `shooting` and `defense`), so
  the final stage cannot be reached with a policy that only defends or only
  shoots.

The metrics are exported as `curriculum/retention/<skill>` and
`curriculum/gate/<gate>`, so a regression is visible in the charts and not
only in the log.

---

## 5. Reward improvements vs real gameplay improvements

* **Gates never read the reward.**  They read the ball position, the touch
  counters, the goal events and the car's up vector from the simulator state.
  A policy that farms the shaping terms gains nothing at a gate.
* **Reward components stay in the report** (`reward/<component>` per episode)
  so a rising return can be attributed: if `touch` rises but the `approach`
  probe does not, the increase is not contact skill.
* **The existing anti-farming guards are untouched**: touches are paid at
  most once per 4 decision steps, every dense term is a potential difference
  (so parking and oscillation net ~0), and `ball_progress` cannot be farmed
  because it telescopes.
* **No reward weight was changed** in this revision, precisely because the
  A/B gave no evidence for it (variant C/D above).

---

## 6. Checkpoints, resuming and the 50 M-step target

Unchanged from the previous revision: no routine periodic checkpoints, one
`final` checkpoint at the configured target (written before the closing
evaluation), one `interrupt` checkpoint on stop / Ctrl+C / SIGTERM, never
overwriting or pruning existing checkpoints, and optimizer / RNG /
curriculum / opponent-pool state preserved.

Two additions:

* the **gate state is persisted** (`extra.curriculum.gate_state`): measured
  values, the confirmation counter and the retention baselines survive a
  resume, so an interrupted run does not get a second chance at a gate it
  already failed (or passed);
* the curriculum is restored **by stage name**, not by index.  A checkpoint
  written under a different preset would otherwise continue at whatever stage
  happens to have that position; if the name is unknown the run says so in a
  WARNING instead of guessing silently.

**`total_steps` and the restored step.**  `total_steps` is a *request* field:
on resume the requested value wins over the checkpoint's.  A configuration
whose target is at or below the restored step therefore trains **zero**
steps.  That is not silent: `rlai.resume.plan_resume` warns
(`checkpoint step N already reached total_steps M; raise total_steps to keep
training`), the trainer logs the same as a run WARNING, `steps_completed` is
0 in the result, and no checkpoint is written (`tests/test_trainer_gates.py::
test_restored_step_beyond_total_steps_is_reported_not_silent`).  When
continuing the stopped 50 M-step run, keep `total_steps` at 50 000 000 - it
is above every step that run reached.

The observation version stays at 4 and the resume validation still refuses to
mix layouts; no migration or reset is performed.

---

## 7. Experiments

All experiments ran in isolated `RLAI_DATA_DIR` directories under `/tmp`,
single process, 2 vCPU, no GPU.  The user's data directory, checkpoints and
runs were never touched, and no training run was started outside these
bounded experiments.

| # | question | result |
| --- | --- | --- |
| 1 | how discoverable is a contact? | table in §1: forward motion and short distance decide it; masking to non-aerial actions makes it worse |
| 2 | is one PPO batch above the noise floor? | gradient cosine 0.04–0.11 at 3072 samples, 0.08–0.11 at 8192; `advantage(forward) − advantage(other)` ≈ −0.03…+0.006 (σ = 1) |
| 3 | which reward composition gives a usable signal? | cosine 0.14 (default) vs 0.23 (touch-only); the informative event is ~20× smaller in σ than the uncontrolled terms |
| 4/5 | do the probes discriminate? | first design failed (a fresh network passed 4 of 6); the calibrated design separates idle/random/fresh from the scripted drivers (§3) |
| 6 | probe tuning | shooting and defense re-tuned until a competent scripted driver passes and a random one does not |
| 7 | does the new starting distribution learn faster? | yes: 1.338 vs 0.460 touches/episode, t = 10.0 on 3 seeds (§1) |

### Rejected probes

Two probes were implemented, measured and removed because they cannot gate
anything.  Both are recorded here so they are not re-added by accident:

* **upside-down recovery** - in RocketSim a car that does nothing rights
  itself after the drop: idle 1.00, random 0.63, fresh network 0.50,
  scripted chase 0.00.  The passive reference passes and the competent one
  fails.  Replaced by `reengage`.
* **"stay goal-side" positioning** - the scripted chase driver scores 0.00
  (it goes for the ball and ends up ahead of it) while a random policy scores
  0.46.  The probe measures *not going for the ball*, which is not a skill we
  want to require.  Boost and positioning are reported as diagnostics
  (`goal_side_share`, final boost) inside the other probes instead.

### Negative results and limits of this evidence

* 81 920 steps × 3 seeds is enough to show that `reach` multiplies early
  ball contact; it is **not** enough to show that any stage of the mastery
  curriculum will be reached, and it says nothing about match strength.
* The A/B measured *training-window* touches, not gate outcomes.  No
  experiment here has passed a gate.
* Variant E (`reach` + reward + 10 s episodes) has the highest touch *rate*
  but a different episode length, so its per-episode numbers are not
  comparable; it was not adopted.
* The gradient-cosine probe uses a 64×64 network and one full-batch gradient;
  it is a signal-to-noise indicator, not a prediction of learning speed.
* Thresholds were calibrated against *scripted* references.  A scripted
  driver is not a strong 1v1 player: `shooting` at 0.44 and `defense` at 0.69
  are what a simple script reaches, and a gate set there may still be hard
  for a learned policy.  The first honest measurement of that is a run that
  reaches stage 2.
* The `match_*` gates reuse `eval_games`/`eval_match_seconds`; with the
  shipped 10 games × 60 s the 95 % interval is about ±0.3, so the CI bounds
  (0.30 / 0.20) will require several evaluations before they can be met.
  That is intentional, but it also means the last two stages promote slowly.

---

## 8. Running it

```bash
# the recommended configuration (does not start by itself)
rlai train --config-file configs/1v1-mastery-1v1.json --run-id mastery-1

# compare with the pre-mastery curriculum
rlai train --config-file configs/1v1-curriculum.json \
  --override-config curriculum_preset=legacy --run-id legacy-1
```

`curriculum_gate_interval=0` with a gated preset is refused at configuration
time: without gate evaluations no mastery stage could ever be left.
