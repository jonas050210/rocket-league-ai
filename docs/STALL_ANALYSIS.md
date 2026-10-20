# Training-stall analysis (2026-10-19)

Reproduction and root-cause report for the reported stall: resumed run,
`obs_version=2`, evaluation stuck at `0W 9L 1D` (goals ~0:25), curriculum
stuck on `first-touch`, very low touch rate, `evaluation stall` warning,
PPO KL ~0.001–0.01, training return swinging without evaluation progress.

Nothing in this document changes reward weights, curriculum thresholds,
observation versions or PPO hyperparameters.  No existing checkpoint, run
row or training log was modified or deleted.

## Method

The live training process was not touched, stopped or restarted, and no
experiment ran against the user's data directory.  Every measurement below
comes from an isolated `RLAI_DATA_DIR` under `/tmp`:

| run | config | result |
| --- | --- | --- |
| `diag1` | 40k steps, rollout 512, 1 worker, 20 s episodes, obs v2, curriculum, drill resets, idle opponent, eval every 10 iterations × 2 games | reproduced the stall signature exactly (see below), 9 checkpoints (old behaviour) |
| `diag2` | identical to `diag1`, run again after the fixes | same stall signature, 1 checkpoint (new behaviour); reported touches 41 → 67 (not a controlled comparison: the reward change alters the updates) |
| `learn1` | 400k steps, 1 worker, same reward, no evaluation | policy still nearly uniform |
| `term1` | 200k steps, hidden 32×32, 2 epochs, no curriculum | policy still nearly uniform |
| probes | direct `MatchEnv` rollouts, `ppo_update`, `evaluate_policy` | used for the mechanism measurements |

Machine: 2 vCPU, no GPU, Torch CPU.  Throughput ~1,900 steps/s per worker.

## Confirmed code defects and fixes

### 1. `ball_touches` was treated as a cumulative counter (3 places)

RLGym defines `Car.ball_touches` as *"number of ball touches since last state
was sent"* (`rlgym/rocket_league/api/car.py`), and the simulator engine
assigns it to the car and immediately zeroes its own counter
(`rlgym/rocket_league/sim/rocketsim_engine.py`, `_get_state`).  It is
therefore already a **per-step delta**.  Measured directly: a contact step
reports `2`, every following step reports `0`.

The project subtracted the previous step's value anyway:

* `src/rlai/rewards.py` (`SoccerReward._reward_full`):
  `new_touches = max(touches - mem.prev_touches, 0)`.
  Consequence: if two decision steps in a row report a contact, the second
  one is dropped (`1 - 1 = 0`).  The same value feeds
  `StepInfo.reward_events["touches"]`, the episode row, `charts/touches`,
  the curriculum window and the run report.
* `src/rlai/evaluation.py`: the same "add the difference when the counter
  grew" logic, with an extra reset heuristic.  Evaluation touch totals were
  therefore undercounted in exactly the same situations.
* `src/rlai/envs.py` (`MatchEnv._touch_cache`): cached per-car values were
  only refreshed when the **sum** over cars changed.  With two cars, a blue
  touch followed by an orange touch has the same total, so the old per-car
  dict was kept and the second touch was attributed to the first car.
  `StepInfo.touches` is consumed by `benchmarks.py` (opponent touch rate).

Measured undercount (30 drill episodes, per-step touch events counted
independently):

| policy | raw contacts | reported before fix | undercount | episodes affected |
| --- | --- | --- | --- | --- |
| scripted chase | 197 | 184 | 6.6 % | 13/30 |
| uniform random | 13 | 11 | 15.4 % | 2/30 |

Fix:

* `rewards.py`: the per-step value is used as-is; `_AgentMemory.prev_touches`
  is gone.  The anti-tap-farming cooldown is unchanged and still pays at most
  once per `TOUCH_COOLDOWN_STEPS` decision steps.
* `evaluation.py`: new `count_touches(env, agents)` helper that adds the
  per-step values, used by the evaluation loop.
* `envs.py`: new `touch_snapshot(state)` helper; the cache compares per car
  and returns a copy, so a stored `StepInfo` cannot change later.

Verification after the fix, same scripted chase episode: raw sum `20`,
`reward_events["touches"] == 20` (before: 184 vs 197 over 30 episodes).

### 2. Checkpoint policy (task 7)

Before: one checkpoint every `checkpoint_interval` iterations, and one more
after every periodic evaluation.  `diag1` wrote 9 checkpoint directories for
40k steps; a 50M-step run would write ~10,000 directories (and `keep` only
prunes within the run).

After:

* no routine periodic checkpoints in the training loop;
* one `final` checkpoint when `total_steps` is reached (written **before**
  the closing evaluation, so the weights are on disk even if the evaluation
  is interrupted);
* one `interrupt` checkpoint on a stop request (`control.json`), Ctrl+C
  (`SIGINT`) **and now also `SIGTERM`**;
* a stop request no longer runs the closing evaluation first — the
  supervisor waits a bounded time for the process and a multi-minute
  evaluation could otherwise be killed before the save;
* periodic evaluations keep their metrics but are recorded without a model
  id; only the closing evaluation of a completed run is linked to the
  `final` checkpoint (and only that one can update `best.json`);
* `_save` still de-duplicates a save at the same step by relabelling the
  existing manifest instead of writing a second identical model, and writes
  remain atomic (`model.pt` + `fsync` + `os.replace` + manifest).  Existing
  checkpoints are never overwritten or removed; `prune_candidates` still
  protects stable/best/latest/final/interrupt and only considers the same
  run.

`checkpoint_interval` remains in the configuration (old config files and
checkpoint payloads still validate) but no longer has an effect; this is
documented in `config.py` and `docs/TRAINING.md`.

End-to-end checks (all in `/tmp`):

* 4,096-step run with `checkpoint_interval: 1` and eval every 4 iterations
  → exactly one checkpoint, `…-final`, step 4096, written before
  `eval[after]`;
* auto-resume at `total_steps` → no new checkpoint, no duplicate;
* stop request during a 200k-step run → exactly one `…-interrupt`
  checkpoint at the current step, status `stopped`, `resumable: true`;
* `SIGTERM` on a running process → exit 0, `interrupt` checkpoint
  (`terminated request (SIGTERM)`), resumable; resuming from that
  checkpoint continued at the right step and later wrote the `final`
  checkpoint.

## Audits that came back healthy

### Touch detection chain (task 1)

`RocketSim` touch callback → `RocketSimEngine._ball_touch_callback`
(`_touches[car] += 1`) → `_get_state` copies to `Car.ball_touches` and
zeroes the counter → `SoccerReward` counts events → `MatchEnv.reward_events`
sums the learning team only → `workers.collect_rollout` stores them in the
episode row → `Store.log_episode` → `charts/touches`, the curriculum window
and the report.  The custom `ModeRocketSimEngine` re-registers the callback
on its own arena, and the runtime probe confirmed callbacks fire.  Only the
counting arithmetic was wrong.

The curriculum window uses the same episode rows, so it inherited the same
undercount — fixed with the reward change, without touching the thresholds.

### Reward components (task 2)

`diag1`/`diag2` reward components are dominated by `progress`,
`ball_progress` and `concede`, exactly as documented.  Both are
potential-difference terms (`gamma = 1` by default), so they telescope:
a parked car earns 0, oscillation nets ~0
(`test_parked_car_facing_ball_earns_nothing`,
`test_ball_oscillation_nets_zero_ball_progress`).  `concede` is negative and
only fires on a goal.  No component was found that pays a positive return
without ball interaction in an unbounded way.  `ball_progress` does pay for
ball motion the policy did not cause (including the opponent's), but because
it is a potential this cannot be farmed and it does not change the optimum.

### Evaluation (task 4)

* The trainer evaluates the **live model** (`TorchPolicy(self.model)`); the
  weights are the ones just trained.  `obs_version` is the run's version and
  `MatchEnv` builds the same observation layout as training.
* Checkpoint/observation compatibility is validated before resuming
  (`validate_checkpoint` compares the checkpoint `PolicyConfig` with the
  requested architecture **and** `saved_config.obs_size` with
  `obs_size(requested obs_version)`); the resume merge inherits
  `obs_version` from the checkpoint.
* A hand-written chase policy routed through `evaluate_policy` won 1, drew 1
  and scored 2:0 with 23 touches in 2 games — resets, goal detection, side
  balancing, touch counting and aggregation all work.
* The observed `0W … 0:0` evaluations against `idle` are real: with a fresh
  (near-uniform) policy, 20 kickoff games produce 0–2 touches and no goals.
  A random policy therefore reports `eval/touches = 0` for small suites; it
  is not a counting failure.
* Only defect found here: the per-step touch arithmetic above.

### PPO diagnostics (task 5)

`ppo_update` itself is healthy.  A synthetic batch with a clear signal moved
every parameter (max |Δ| ≈ 2 × 10⁻³) and drove the probability of the
rewarded action from 0.011 to 0.999 in 100 updates.  In the real runs the
losses are finite, the critic learns (`explained_variance` −0.16 → 0.5–0.9),
`clipfrac = 0`, gradient norm 0.1–0.6 and KL is small.  KL is *not* evidence
of a dead optimiser; it is small because the effective step is small.

## Why the curriculum is stuck (confirmed mechanism)

This is the important finding, and it is **not** a code bug:

1. On the first-touch stage the policy is still essentially uniform
   (`entropy 4.49981 → 4.49852` after **200,192** steps; `ln 90 = 4.49981`).
2. It touches the ball in 35 of 668 episodes (0.11–0.29 touches/episode in
   the 40k runs).  A scripted chase policy at the same resets gets
   7.9 touches/episode, so the gate (1.5) is reachable — just not by this
   policy.
3. At this policy the policy gradient is indistinguishable from noise:
   two independent 1,536-sample batches give gradient cosine **−0.008**;
   two 3,072-sample batches (≈ one full multi-worker iteration) give
   **−0.030**.  Over 10,240 samples, the mean GAE advantage of
   "throttle + boost" is +0.021 (no-op +0.051, standard error 0.094); the
   apparent "best" actions are within the noise band of 90 samples.
   In other words, at a uniform policy one decision barely changes the
   20 s return: the per-step reward is ~0.02 while the return is dominated
   by ball motion the policy did not cause.
4. Curriculum promotion requires 1.5 touches/episode over ≥30 episodes and
   ≥3 iterations.  With the measured 0.11–0.8 this is blocked, and it stays
   blocked because PPO has no consistent gradient with which to leave the
   uniform policy.  Lowering the threshold would only hide that.

The measurement bug made the reported touch rate smaller than reality, but
even the corrected rate (6–15 % higher) is far below 1.5, so it was not the
cause of the stall — only a bias that worked in the same direction.

The same mechanism explains the evaluation picture: the model never learns
to reach the ball, so it neither scores nor concedes against an idle
opponent and the stall warning fires on every window.

## Measured learning and evaluation after the fixes

A fresh end-to-end run with the repository's own smoke configuration
(`configs/learning-smoke-1v1.json`: 150,528 steps, obs v4, 2 workers,
tanh 256×256, drill resets, idle opponent, run `ls1`, data directory
`/tmp/rlai-ls`, 131 s) gives:

| metric | value |
| --- | --- |
| episodes | 540 |
| episodes with at least one touch | 114 (21.1 %) |
| touches per episode | **0.548** (uniform random 0.433, idle 0.067) |
| touches per episode by quarter | 0.400 → 0.541 → 0.793 → 0.459 |
| training goals for:against | 13:51 |
| wins | 13 |
| policy entropy | 4.34 → 4.15 |
| explained variance | 0.54–0.91 |
| evaluation before / after | 0W 0L 4D, 0:0 / 0W 0L 4D, 0:0 |
| 19 periodic evaluations | **2W 0L 74D, 2:0 goals** |
| checkpoints written | 1 (`final`) |

The periodic evaluations are the first non-zero evaluation goals measured
after the fixes (every earlier reproduction reported `0:0`).  This is
credible evidence that the initial policy *can* acquire useful ball
interaction, but it is weak evidence: 2 goals in 76 evaluation games has a
wide confidence interval, the training goal rate is close to the historical
pre-fix smoke run, the run is short and single-seeded, and the curriculum
still does not promote.  It shows the fixed measurement is no longer blind
to progress; it does not show that 50M steps will solve the task.

## Harness validation probes

To separate "the trainer cannot learn" from "this task is a slow, noisy
exploration problem", the same `MatchEnv` → `collect_rollout` → `ppo_update`
path was run with rewards that are still driven by real simulator steps but
carry an unambiguous learning signal:

| probe | reward | result |
| --- | --- | --- |
| pure bandit | +1 for action 18, independent of state | sampled P(18) 0.012 → 0.991 in 150k steps; entropy 4.50 → 0.10 |
| contextual bandit | +1 for action 18 when `ball.y > 0`, action 8 otherwise | success rate 0.023 → 0.869; entropy 4.50 → 0.40 |
| dense proximity | +1 per decision step within 600 units of the ball | no trend (12.2 vs 13.1 reward in the first quarter) |
| `tick_skip=16` | unchanged soccer reward | 0.353 touches/episode vs 0.42–0.55 at tick 8; no improvement |

The two bandit probes show that rollout storage, the observation path,
advantage computation, `ppo_update` and the action parser all transmit a
learnable signal.  The soccer task is not stalled because gradients never
reach the network; at a near-uniform policy the advantage of any single
action is small relative to the episode-level variance (independent-batch
gradient cosine −0.01 to −0.03; even 61,440 samples only separate
no-op/throttle at z ≈ 3).

The dense-proximity probe is inconclusive rather than negative: with drill
resets the car already starts 700–2600 units from the ball, so the reward
mostly reflects the reset distribution rather than the policy's behaviour.
Raising the learning rate (3 × 10⁻³) did not produce a trend either.  None
of these probes is a substitute for a controlled A/B run of a real reward
change on the user's 50M-step budget.

## What was deliberately not changed

* reward weights (`goal`, `touch`, `ball_progress`, `progress`, …),
* curriculum stages, thresholds and promotion windows,
* PPO hyperparameters (CLIP, KL target, entropy coefficient, learning rate),
* observation layout/version, action table, evaluation protocol,
* existing checkpoints, run rows, logs or the user's data directory.

## Remaining uncertainties

* The positive post-fix evidence above is a single 150k-step smoke run.  It
  is enough to show that ball interaction is measurable and can improve, but
  not enough to estimate a learning curve or time-to-first-touch-gate.
  A longer run on the same configuration is the next honest measurement;
  it was not started here because it would compete with the user's training.
* The user's run reports KL 0.001–0.01, i.e. a policy that moves more than
  the fresh-policy reproductions here (KL ≈ 2 × 10⁻⁵).  That is consistent
  with a checkpoint that is not uniform, but it could not be inspected
  without the user's data directory.  If that policy still never wins, the
  next measurement to take is the gradient cosine on *that* checkpoint
  (same probe) — it separates "no signal" from "signal but wrong objective".
* The closed-loop signal-to-noise problem above is only mitigated by
  design changes that were out of scope here (e.g. a stronger early
  approach potential, shorter first-touch episodes, or a smaller action
  space for the first stage).  Such a change needs a controlled A/B run,
  not a speculative edit.
* A hard crash (uncaught exception, SIGKILL) leaves no checkpoint by design
  under the new policy; graceful stop, Ctrl+C, SIGTERM and normal
  completion are covered.  If crash protection is wanted again, the
  smallest safe option is an opt-in interval that is explicitly documented
  as a recovery net.

## Files changed

| file | change |
| --- | --- |
| `src/rlai/rewards.py` | per-step touch counting; removed `prev_touches` |
| `src/rlai/evaluation.py` | `count_touches()` per-step helper, used by the eval loop |
| `src/rlai/envs.py` | `touch_snapshot()`; per-car touch cache, snapshot copy |
| `src/rlai/trainer.py` | no periodic/eval checkpoints; final at target; interrupt on stop/Ctrl+C/SIGTERM; eval/model linkage |
| `src/rlai/config.py` | `checkpoint_interval` documented as compatibility-only |
| `docs/TRAINING.md` | checkpoint/resume section rewritten for the new policy |
| `docs/TESTING.md` | test counts updated |
| `tests/test_rewards.py`, `tests/test_overhaul.py` | per-step touch contract; consecutive-contact regression |
| `tests/test_evaluation_v2.py` | `count_touches` regression |
| `tests/test_env_modes.py` | per-car touch snapshot regression |
| `tests/test_trainer.py` | no periodic checkpoints, stop → interrupt, SIGTERM reason, Ctrl+C during final save |

## Tests and lint

```
ruff check src tests rlbot        # All checks passed
pytest -q                         # 612 passed, 1 skipped (jsdom) — 171 s
```

(The skip is `tests/test_gui_smoke.py`, which needs `jsdom`; it is skipped by
design when node/jsdom are unavailable.)

## Nachtrag: größere Batches und No-Touch-Timeout (150k Schritte)

Dieselbe Smoke-Konfiguration, aber mit `rollout_steps = 1024` (Batch 2048
statt 1024), `no_touch_timeout_seconds = 10` und `minibatch_size = 512`,
sonst unverändert (obs v4, 2 Worker, 20 s Horizont, drill resets, Gegner
`idle`, keine Zwischen-Evaluationen, run `91fc2eff1fbb`, 135 s,
`/tmp/rlai-learn`):

| Metrik | vorher (`ls1`, 150 528 Schritte) | nachher (151 552 Schritte) |
| --- | --- | --- |
| Episoden | 540 | 942 |
| Entscheidungen pro Episode | 278 (18,5 s) | 161 (10,7 s) |
| Touches pro Episode | 0,548 | **1,355** |
| Touches pro 1 000 Schritte | 1,97 | **8,4** |
| Tore pro 1 000 Schritte | 0,086 | **0,475** |
| Tore (Training) | 13:51 | **72:1** |
| Truncations | – | 869 von 942 (92 %) |
| Entropie | 4,34 → 4,15 | 4,21 (bei ln 90 = 4,50) |
| Explained variance | 0,54–0,91 | 0,66 (letzte Iteration 0,64) |
| approx KL / Clipfrac | – | 0,0005 / 0,001 |
| Schritte/s | ~1 900 | ~1 116 |

Der Vergleich ist **nicht** sauber kausal: beides sind kurze, einfach
gesäte Läufe, und der No-Touch-Timeout verändert die Episodenlänge, also
auch Touches-pro-Episode. Deshalb die zeitraumbezogenen Zeilen: pro 1 000
Umgebungsschritte interagiert die Policy etwa viermal häufiger mit dem Ball
und erzielt etwa fünfmal so viele Tore. Der Preis ist Durchsatz
(größere Batches und der zusätzliche Done-Check kosten ~40 % Schritte/s),
und die Entropie bleibt nahe am Maximum – nach 150k Schritten ist die
Policy noch weit von „entschieden“ entfernt. Der Run belegt, dass die
Pipeline mit den neuen Einstellungen ein dichteres Lernsignal liefert als
der alte Smoke-Lauf; er belegt **keine** Spielstärke.

Auffällig: 72:1 Tore statt 13:51. Der alte Lauf kassierte gegen einen
`idle`-Gegner 51 Tore – fast ausschließlich Eigentore aus den
`defense`-Drills, in denen der Ball auf das eigene Tor rollt und eine
zufällige Policy ihn hineinschiebt. Der 10-s-Timeout beendet diese
Situationen früher, bevor die Policy sich selbst besiegt.

### Curriculum-Stufen verzerren den First/Last-Vergleich

Ein 300k-Fortsetzungslauf vom BC-Checkpoint (`f6d11b8f8ae6`: 1 669 Episoden,
154:56 Tore, 1,59 Kontakte/Episode, 1 026 Schritte/s) zeigte im Report
zunächst einen Rückgang von 4,54 auf 1,64 mittlerem Reward (−2,9) – der Lauf war
im Curriculum aber von `approach` über `control` nach `shooting` gewandert.
Jede Stufe ändert Gegner, Startzustände (reset_mode) und Horizont; die Zahlen
beschreiben also verschiedene Aufgaben.

Der Report schränkt beide Fenster jetzt auf die **aktuelle Stufe** ein
(`learning_evidence.scope` mit `since_step`, `stage_changed`, `stage_start_step`
und einer Notiz). Derselbe Lauf ergibt innerhalb der `shooting`-Stufe ab
Schritt 102 400: **+0,99 Reward, +3,5 Prozentpunkte Siegquote, +0,035 Tore pro
Episode, +0,125 Kontakte pro Episode** gegenüber dem Stufenbeginn. Der
Fortschritt war also da – der globale Vergleich hat ihn nur von einem
Stufenwechsel überlagert.

### Der `shooting`-Engpass: 55 % der Episoden ohne Ballkontakt

Dieselben 294 Curriculum-Iterationen des Laufs `f6d11b8f8ae6`, gruppiert nach
der zum Zeitpunkt der Episode aktiven Stufe:

| Stufe | Episoden | Kontakte/Episode | ohne Kontakt | Truncations | Ø Länge | Ø Reward |
| --- | --- | --- | --- | --- | --- | --- |
| 0 `approach` (`reach`) | 112 | 3,37 | 8 % | 79 % | 173 | 4,34 |
| 1 `control` (`reach`) | 298 | 4,86 | 4 % | 76 % | 172 | 4,95 |
| 2 `shooting` (`mixed`) | 1 259 | 1,39 | **55 %** | 91 % | 183 | 0,85 |

`mixed` ist zur Hälfte Kickoff: der Ball liegt dann 2 500+ uu entfernt, und der
laufweite 10-s-No-Touch-Timeout beendete die Episode, bevor eine schwache
Policy ankommen konnte. Mehr als die Hälfte der Stichproben trug damit gar
kein Reward-Signal – der `progress`-Gate blieb bei 0,29 (gefordert 0,70), und
der Lauf stand 224 Stufen-Iterationen still.

Zwei Änderungen (Commit `ec305d0`): `shooting` startet aus `drill`-Zuständen
(Attacke/Verteidigung/Race, 700–2 600 uu; Kickoffs kommen mit `defense`
zurück), und `Stage.no_touch_timeout` erlaubt jeder Stufe einen eigenen
Timeout – `shooting` … `integrated` nutzen 25 s, die Nahbereichsstufen
behalten den Laufwert. Der Timeout reist mit dem Collect-Kommando und baut
die Worker-Arena neu (wie `episode_seconds`).

Wirkung im A/B (gleicher Checkpoint, je 80 000 Schritte, `/tmp/rlai-shootA`
gegen `/tmp/rlai-shootB`): Episoden ohne Kontakt 30 % → 17 % in der ersten
Hälfte, Kontakte/Episode 2,24 → 3,24.

### Der zweite Teil des `shooting`-Engpasses: die Probe war im Training nicht enthalten

Der Lauf mit 148:60 Toren meldete `shooting` **0,00**. Die Probe schießt aus
`y = 3000–4000` mit einem Torwart auf der Linie; die `attack`-Drills zogen
Ballpositionen aber uniform über das ganze Feld (`y ∈ −3500…3500`, also nur
~7 % oberhalb von 3 000) und erzeugten **nie** einen Torwart. Der Fehler lag
nicht in der Policy, sondern in der Verteilung.

Seit dem vierten Drill-Szenario **`shot`** (20 %: Ball im Angriffsdrittel,
Torwart in der Toröffnung, lernendes Auto 600–900 uu dahinter) enthält das
Training genau diese Situation. Validierungslauf (`1v1-mastery-1v1`,
`--steps 300000`, frisch, 476 s, 199:30 Tore):

| Lauf | Schritte | Stufe am Ende | `progress` | `shooting` |
| --- | --- | --- | --- | --- |
| vorher (`f6d11b8f8ae6`) | 405 504 | `shooting` | 0,58 | **0,00** |
| nachher | 301 056 | `shooting` | 1,00 [0,93; 1,00] | **0,54 [0,40; 0,67]** |

Das Tor-Gate (≥ 0,30, CI-low ≥ 0,15) ist damit erstmals auf dem Punktwert
**und** der unteren Konfidenzgrenze erfüllt; der Lauf scheiterte nur noch an
der zweiten Bestätigung (`1/2`).

### `reengage`: Abdeckung allein reicht nicht – die Policy sättigt im No-op

Nach der `shooting`-Stufe ist `reengage` das nächste Gate, das nie erfüllt
wird: `reengage = 0,00 [0,00; 0,07]` über alle Auswertungen des
`defense`-Laufs (700 416 Schritte). Dieselbe Vermutung wie bei `shooting` –
die Situation fehlt im Training – führte zu zwei Änderungen, die **beide
wieder verworfen wurden**, weil sie das Gate nicht bewegen:

| Lauf | Änderung | `approach`-Stufe | `reengage` (48 Trials) | Training |
| --- | --- | --- | --- | --- |
| 300k | keine | `control` bei Iteration 200 | – | 199:30 Tore |
| 300k | `reach` mit 30 % Blick-weg-Starts | bei 280k **noch Stufe 0** (0,25) | 0,17 deterministisch / 0,27 stochastisch | – |
| 300k | `reach` mit 10 % Blick-weg-Starts | `control` bei Iteration 260 | 0,10 / **0,63** | – |
| 700k | `drill` + 10 % `recover`-Szenario | – | **0,00** | 315:420 Tore, 2,92 Kontakte/Ep. |
| 700k | zusätzlich `reach` 10 % | – | **0,00** | 241:542 Tore, 2,15 Kontakte/Ep. |

Der entscheidende Befund steckt in der Aktionsverteilung des Checkpoints im
`reengage`-Startzustand:

```
Entropie 0,076 (Vergleich `approach`-Start: 0,815)
Aktion 50: 99,07 %  ->  throttle 0, steer -1, air-roll  (das Auto steht)
Aktion 16:  0,39 %     throttle 1
```

Die Policy gibt in diesem Zustand praktisch ihre ganze Masse auf eine
No-op-Aktion; auch **stochastisches** Sampling berührte den Ball in 0/30
Versuchen. Erst die frühe Konfrontation (10 % Blick-weg-Starts ab Stufe 0)
erzeugt die Fähigkeit – dann aber nur stochastisch (0,63), während das Gate
mit `argmax` misst (0,10). Die Fähigkeit ist also **erlernbar, aber die
deterministische Policy committet sich nicht** auf die ~1–2 s dauernde
Drehung; dafür fehlt ein dichtes Kreditsignal, nicht Abdeckung.

Konsequenz: `drill/recover` und `REACH_FACING_AWAY_SHARE` wurden nicht
ausgeliefert (Kosten: +30 % Stufe-0-Iterationen bzw. schlechtere
Trainingsergebnisse bei 700k), das gemessene `shot`-Szenario bleibt. Der
nächste Schritt ist ein potential-basiertes Heading-Shaping (Differenz der
Ausrichtung Auto→Ball, nicht der Zustand selbst) oder eine eigene
Recovery-Stufe – beides braucht einen eigenen A/B gegen genau diese Zahlen.

### Warum 80k-Schritte-A/Bs nichts beweisen (Rauschkalibrierung)

Drei Läufe mit **identischer** effektiver Config und demselben Checkpoint
(`rlai-shootB/C/D`, je 81 920 Schritte) ergaben: Kontakte/Episode in der
zweiten Hälfte 2,31 / 1,91 / 3,84 und `progress`-Gate 0,25 / 0,71 / 0,13.
Damals war der Lauf nicht bit-deterministisch: die Worker zogen ihre
Aktions-Stichproben aus eigenen, ungeseedeten Torch-RNGs. **Das ist behoben**
– `Trainer.__init__` seedet die globalen RNGs aus `config.seed`, und
`collect_rollout` sampelt aus einem aus dem Rollout-Seed geseedeten Generator
(siehe `docs/TRAINING.md`, „Reproduzierbarkeit"). Zwei identische 4 096-
Schritte-Läufe liefern jetzt identische Rollout-Serien; die damaligen 2,31 /
1,91 / 3,84 Kontakte waren also tatsächlich Sampling-Rauschen und nicht
Policy-Drift. Was bleibt: 24 Gate-Trials haben ±18 Prozentpunkte
Konfidenzband. **Policy-Vergleiche brauchen die geseedete
Match-Evaluation (`eval_interval`, feste Seeds), keine Trainings-Rollouts** –
und selbst dort sind 10 Spiele nur eine grobe Auflösung.

Nebenbefund: `learning_rate` und `entropy_coef` stehen in `INHERITED_FIELDS`.
Ein `--config-file` mit anderen Werten wird beim Fortsetzen **nicht**
angewandt (Log: `source=inherited`); dafür braucht es `--override-config`.
Das ist Absicht (Reproduzierbarkeit), hat aber drei Testläufe gekostet.

### 24 Gate-Trials sind ein Münzwurf

Zwei Beobachtungen aus identischen Konfigurationen:

* Frischer Lauf, 150k Schritte: `approach` 0,125 (25k) → **1,00** (150k),
  Stufe 1 erreicht.
* Frischer Lauf derselben Config, 400k Schritte: `approach` 0,00 (25k und
  120k) → **1,00** (145k), dann wieder 0,00 in der Folgeauswertung.

Ein Probe-Trial ist 1–1,5 s, 24 Trials ergeben bei einer Rate von 0,5 ein
95-%-Band von ±0,2. Nahe der Schwelle (0,80 mit CI-Untergrenze 0,60)
entscheidet damit der Zufall, ob eine Stufe weitergeht. Deshalb ist
`curriculum_gate_trials` jetzt **48** (Standard und Mastery-Config): das Band
halbiert sich auf ±0,14, die Probe kostet weiter nur Sekunden. Die Schwellen
selbst bleiben unverändert – mehr Trials machen die Entscheidung präziser,
nicht leichter.

### 20-s-Evaluationen überschätzen eine passive Policy

Der BC-Checkpoint galt als „5W/7L/8D gegen idle+chase" – gemessen über 20 s
Episoden. Mit 60-s-Spielen und denselben Gegnern steht dieselbe Policy bei
**0W 0L 10D gegen `idle` (0:0)** und **0W 10L 0D gegen `chase` (0:37)**:
Bei 20 s reicht die Zeit meist nur zum 0:0, bei 60 s kassiert sie. Für
Fortschrittsaussagen sind volle Spiele nötig; alle Zahlen in diesem Abschnitt
nutzen `eval_match_seconds = 60`, 10 Spiele je Gegner, beide Seiten.

## Reproduction commands (isolated data)

```bash
RLAI_DATA_DIR=/tmp/rlai-diag2 python -m rlai.cli headless \
  --config-file /tmp/diag.json --run-id diag2
RLAI_DATA_DIR=/tmp/rlai-ls python -m rlai.cli headless \
  --config-file configs/learning-smoke-1v1.json --run-id ls1
```

`/tmp/diag.json` is the 40k-step diagnostic config (2 workers are reduced to
1 on a 2-core machine, 20 s episodes, obs v2, curriculum, drill resets, idle
opponent, eval every 10 iterations).
