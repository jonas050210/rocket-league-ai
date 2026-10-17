# Target-host runbook — RTX 4060 Ti / Windows WSL 2

Everything on this page is **prepared but not executed**. The development
sandbox has no GPU, no WSLg display and 2 CPU cores; none of the numbers from
that environment say anything about the target host (i7-12700F, RTX 4060 Ti
8 GB, 32 GB). Each step that starts training, a benchmark or a long
evaluation requires the operator's explicit approval in the UI. Nothing here
runs automatically.

## 0. Preflight (no training)

```powershell
wsl -d Ubuntu-24.04
nvidia-smi        # Windows driver visible inside WSL
```

```bash
cd ~/rocket-league-ai
uv sync --frozen --all-extras --python 3.12
uv run --no-sync rocket-ai doctor --probe
```

Pass criteria: exit 0, CUDA available, device name `RTX 4060 Ti`, CUDA matmul
passed, 16 soccar meshes, 8-tick simulation passed. Record OS/WSL/kernel,
driver, lock hash and visible CPU/RAM next to the result. A CPU fallback is
never a pass.

## 1. N0 — native desktop + bounded CUDA run

1. `uv run --no-sync rocket-ai desktop` opens in WSLg; all six views load.
2. In the Training plan panel, approve (proposal + checkbox + digest) a bounded CUDA
   plan, e.g. total 8,192 agent steps, batch 512, 1 env, hidden [256,256,256].
3. Start via the Training proposal; verify real steps/episodes/checkpoints,
   then pause → resume → stop; each must acknowledge at an update boundary.
4. Close and reopen the window mid-run: same worker, no duplicate run.
5. Record peak CUDA allocated/reserved (Dashboard GPU peak / run metrics) and
   host RSS. No OOM, no non-finite loss.
6. Recover the newest checkpoint into a linked run (proposal), confirm the
   simulator-reset disclosure, stop it.

## 2. N1 — 30-minute supervised soak

Separately initiated bounded runs only. At least three GUI reconnects and
three pause/resume cycles; one explicit stop; corrupt a *disposable copy* of
a checkpoint and show older-valid fallback; interrupt a disposable backend
and show no auto-restart. Record API latency during training, RSS/VRAM
peaks, checkpoint sizes. No orphaned process groups afterwards.

## 3. Benchmark sweeps (B2/B3) — prepared matrix

In Benchmark, for the first target-host session:

- Devices: CPU and CUDA · n_proc: 1, 2, 4, 8 · batch 512 · warm-up 1 ·
  measurement window 20 s · max 1900 measured iterations · min 2 ·
  repeats 3 · CPU 100% · RAM 95% · VRAM 95% · auto-adopt checked.
- This runs 8 candidates × 3 repeats of genuine training runs, each stopped
  after 20 s of measured updates (plus startup), with a hard timeout. Expect
  roughly 8–15 minutes on this host; it is a benchmark session, not training.
- The iteration ceiling is a best-effort stop: the worker acknowledges it at
  the next update boundary, so a hair more than 20 s (typically well under a
  second) can be measured. Compare published numbers with that in mind.
- Afterwards compare batch 1024 and 2048 in separate sessions. Leave the
  iteration ceiling at 1,900: it is clamped automatically (975 at batch 1,024,
  487 at 2,048) so the per-run step cap still holds. The 20 s window remains
  the normal stop in every case.
- Automatic adoption is limited to **device and n_proc** and requires three
  repeats with CV ≤ 0.10 and GPU telemetry for a CUDA winner. The previous
  plan is stored, and "Revert automatic default" creates a reviewed proposal.
  Strategy fields are never adopted from a throughput session.

Record per candidate: agent/sim/PPO steps/s (median, CV), episodes/min,
CPU/RAM/VRAM peaks, GPU utilization, errors. The fastest *stable* candidate
wins; maximum utilization alone never does.

If CPU stays near 10% of 20 threads, check Monitoring's "threads busy" value:
one `n_proc` environment is one RocketSim process, so `n_proc=1` is roughly
one busy core. The benchmark exists to find the largest stable value, not to
assume 8 is best.

## 3b. Why training showed ~10% CPU, and what to do

On the i7-12700F, 10% of 20 logical threads is about two busy threads. With
`n_proc=1` the run has exactly one RocketSim environment process, and each
worker/training process pins BLAS (`OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) so a small PPO network is not
oversubscribed by math libraries. That is the intended, correct default for a
GPU-bound learner; it is not a stuck run. The missing piece was visibility.

Before the sweep, confirm in Monitoring: "threads busy" near 1–2, one
`rocket-sim`/worker process, and GPU utilization above 0 for CUDA. If CUDA is
selected but utilization stays at 0%, stop and run `doctor --probe`; do not
"fix" it by raising `n_proc`.

Then use Benchmark exactly as in §3 with `auto-adopt` checked. Expect CPU
utilization to grow roughly linearly with `n_proc` until P-core threads are
saturated; GPUs usually peak around 2–4 concurrent 1v1 simulators on this
class of card. Pick the fastest **stable** candidate (CV ≤ 0.10 across three
20-second repeats), not the maximum percentage. Adoption changes only
`device` and `n_proc`; strategy fields and the step budget stay as approved.

Record: candidate, n_proc, steps/s median and CV, GPU %, VRAM peak, and
whether the value was auto-adopted. If the recommendation is excluded, keep
the current default and attach the reason to the runbook.

## 4. Gameplay baseline (P2-1) — prepared, not run

1. Pick the newest valid checkpoint of the longest completed run.
2. Models → Evaluation: opponent `random-v1`, 400 games, seed 1000,
   parallel workers 4–8, max game 60 s. Record the replay.
3. Repeat identically against `idle-v1` and against the oldest checkpoint
   (checkpoint opponent).
4. The baseline row (win rate + 95% CI, goals, durations) goes into
   `docs/VERIFICATION.md`. Only differences whose intervals separate count
   as improvement.

## 5. First shaping experiment (P2-2) — prepared, not run

One variable only, same budget/seed/evaluation as the baseline:

- New plan: `reward_version = goal10-touch0.1-shaped-v2`,
  `reward_velocity_to_ball_weight = 0.3`, `reward_ball_to_goal_weight = 0`.
- Approve via proposal, train, evaluate identically (step 4).
- Accept only if the lower Wilson bound improves; otherwise record the
  regression and revert the plan. Never stack a second shaping change on an
  unmeasured first one.

## 6. Raising the 1M-step cap

The one-million agent-step cap is a safety limit, not a target. Raising it
is a configuration decision for the operator after N0+N1 pass: propose the
new `total_timesteps` in the Training plan panel with rationale (measured steps/s →
wall-clock estimate from the benchmark) and confirm explicitly. League
self-play (snapshot opponents, PFSP) builds on checkpoint-opponent
evaluation, which is implemented; scheduling a league is an experiment
decision, not an automatic background process.
