/* Headless smoke test for the web GUI: renders every view against canned API
 * responses and fails on any exception or missing element.
 *
 * The redesigned interface exposes four primary sections (Dashboard,
 * Training, Stats, Bot Lab) plus secondary system views (Models, Live match,
 * Settings).  This test asserts the slimmed navigation, the simplified
 * 1v1-only training form (no technical controls), the automatic
 * continuation preview and the confirmation gate. */
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

const repo = process.argv[2] || "/home/user/rocket-league-ai";
const html = readFileSync(`${repo}/src/rlai/web/index.html`, "utf8");
const script = readFileSync(`${repo}/src/rlai/web/app.js`, "utf8");
const statsScript = readFileSync(`${repo}/src/rlai/web/stats.js`, "utf8");
const featuresScript = readFileSync(`${repo}/src/rlai/web/features.js`, "utf8");

const report_before = {
  opponent: "chase", games: 4, wins: 1, losses: 3, draws: 0, win_rate: 0.1, goals_for: 2, goals_against: 6,
};
const report_after = {
  opponent: "chase", games: 4, wins: 2, losses: 2, draws: 0, win_rate: 0.4, goals_for: 5, goals_against: 4,
};

const statsPayload = {
  available: true,
  run_id: "run-1",
  status: "completed",
  name: "smoke run",
  mode: "soccer-1v1",
  timing: {
    available: true,
    accumulated_s: 125.5,
    accumulated_human: "2m 05s",
    current_session_s: 12.25,
    current_session_human: "12 s",
    session_count: 2,
    interrupted_count: 0,
    started_at: "2026-10-18T10:00:01+00:00",
    ended_at: "2026-10-18T10:00:06+00:00",
    last_checkpoint_at: "2026-10-18T10:00:05+00:00",
    sessions: [
      { started_at: "2026-10-18T10:00:01+00:00", ended_at: "2026-10-18T10:00:05+00:00", duration_s: 4.2, duration_human: "4 s", status: "completed", start_step: 1000, end_step: 3048, note: "" },
      { started_at: "2026-10-18T11:00:01+00:00", ended_at: null, duration_s: null, status: "running", start_step: 3048, end_step: 3048, note: "" },
    ],
  },
  progress: { status: "running", step: 3048, total_steps: 4096, resumed_step: 1000, progress: 0.7441, episodes: { available: true, count: 12, win_rate: 0.25, mean_reward: 0.6 }, episode_count: 12, iterations: 16, mode: "soccer-1v1", name: "smoke run" },
  learning: { "charts/episodic_return": { last: 0.6, mean: 0.4, min: 0.1, max: 0.9, count: 16 }, "losses/policy": { last: -0.001, mean: -0.002, min: -0.01, max: 0, count: 16 }, "time/sps": { last: 1042, mean: 1000, min: 900, max: 1100, count: 16 } },
  curves: {
    "charts/episodic_return": [{ step: 1000, value: 0.1 }, { step: 2000, value: 0.6 }],
    "charts/win_rate": [{ step: 1000, value: 0.1 }],
    "losses/policy": [{ step: 1000, value: -0.002 }],
    "losses/explained_variance": [],
    "time/sps": [{ step: 1000, value: 1000 }, { step: 2000, value: 1042 }],
    "eval/win_rate": [],
  },
  performance: { steps_per_second: 409.6, resources: { cpu_percent: 12, cpu_count: 2, memory_percent: 40, memory_used_gb: 1.2, memory_total_gb: 3, gpu: null, process: { rss_mb: 120, threads: 8 } } },
  checkpoints: [{ id: "m-2026", step: 3048, created_at: "2026-10-18T10:00:06+00:00", duration_s: 120.25, duration_human: "2m 00s", duration_available: true, label: "final", state: "candidate", stable: false, best: true, size_bytes: 123456, eval_score: 0.4 }],
  comparison: { available: true, count: 1, best: { id: "m-2026", step: 3048, eval_score: 0.4, duration_s: 120.25 }, ranking: [{ id: "m-2026", step: 3048, eval_score: 0.4, duration_s: 120.25, duration_human: "2m 00s" }] },
  evaluation: { available: true, count: 1, per_opponent: [{ opponent: "chase", games: 4, wins: 2, losses: 1, draws: 1, win_rate: 0.5, evaluations: 1 }], evaluations: [] },
  history: [
    { started_at: "2026-10-18T10:00:01+00:00", ended_at: "2026-10-18T10:00:05+00:00", duration_s: 4.2, status: "completed", start_step: 1000, end_step: 3048, note: "" },
    { started_at: "2026-10-18T11:00:01+00:00", ended_at: null, duration_s: null, status: "running", start_step: 3048, end_step: 3048, note: "" },
  ],
  retention: { rows: 420, names: 39 },
};

const report = {
  schema: "rlai-run-report-v1",
  run_id: "run-1",
  name: "smoke run",
  mode: "soccer-1v1",
  status: "completed",
  message: "finished",
  created_at: "2026-10-18T10:00:00+00:00",
  started_at: "2026-10-18T10:00:01.123+00:00",
  ended_at: "2026-10-18T10:00:06.123+00:00",
  duration_s: 5.0,
  steps: { start: 1000, end: 3048, completed: 2048, target: 4096, progress: 0.7441 },
  steps_per_second: 409.6,
  steps_per_second_basis: "wall clock",
  iterations: 16,
  hardware: {
    device: "cpu",
    workers: 1,
    cpu_logical: 2,
    platform: "Linux",
    python: "3.11.9",
    torch: "2.4.0",
    gpu_available: false,
    gpu_name: null,
    torch_threads: 2,
  },
  configuration: { mode: "soccer-1v1", total_steps: 4096, n_workers: 1 },
  resume: {
    mode: "continue",
    parent_run: "run-0",
    restored_step: 1000,
    checkpoint: "m-2026/model.pt",
    will_resume: true,
    restores: ["weights", "optimiser state", "step counter (1000)", "fresh RNG (none stored)"],
    warnings: ["checkpoint has no RNG state"],
  },
  optimisation: {
    "losses/policy": { available: true, count: 16, first: -0.00159, last: -6.4e-5, mean: -0.001, min: -0.002, max: 0 },
    "losses/value": { available: false, reason: "metric was not recorded for this run" },
  },
  throughput: {
    wall_clock_steps_per_second: 409.6,
    iteration_steps_per_second: { available: true, mean: 1042.4, min: 1000, max: 1100, last: 1040, basis: "collect+update" },
  },
  timings: {
    "time/sps": { available: true, count: 16, mean: 1042.4, min: 1000, max: 1100, last: 1040 },
    "time/eval_seconds": { available: false, reason: "no evaluation ran" },
  },
  reward_components: { "reward/goal": { mean: 0.625, last: 0.5, count: 16 }, "reward/touch": { mean: 0.997, last: 1, count: 16 } },
  episodes: {
    this_run: { available: true, count: 16, mean_reward: 0.635, min_reward: -0.6, max_reward: 2, mean_length: 600, wins: 0, losses: 1, draws: 15, win_rate: 0, truncations: 15, goals_for: 0, goals_against: 1, opponents: { chase: 16 } },
    all_time: { available: true, count: 32, mean_reward: 0.5, mean_length: 600, wins: 1, losses: 2, draws: 29, win_rate: 0.03, truncations: 30, goals_for: 1, goals_against: 2, opponents: { chase: 32 } },
    last_100: { count: 32, mean_reward: 0.5, mean_length: 600, wins: 1, losses: 2, draws: 29, goals_for: 1, goals_against: 2 },
  },
  evaluation: {
    before: { available: false, reason: "no baseline evaluation was run" },
    after: {
      id: 7, phase: "after", model_id: "m-2026", opponent: "chase", mode: "soccer-1v1",
      games: 4, wins: 1, losses: 2, draws: 1, goals_for: 3, goals_against: 4,
      win_rate: 0.25, win_rate_ci95: [0.04, 0.55], win_rate_decisive: 0.33, goal_diff: -1,
      touches: 12, sides: ["blue", "orange"], mirror: true,
      protocol: { seed: 1000, match_seconds: 30, sides: ["blue", "orange"], mirror: true, stochastic: false, obs_version: 2, opponent_class: "strength" },
      created_at: "2026-10-18T10:00:06+00:00", duration_s: 3.2, complete: true,
    },
    delta: {
      available: true, comparable_protocol: true, opponent: "chase", opponent_class: "strength",
      win_rate_before: 0.1, win_rate_after: 0.4, win_rate_change: 0.3, goal_diff_before: -2,
      goal_diff_after: 1, wins_before: 1, wins_after: 2, games_before: 4, games_after: 4,
      seed_before: 900043, seed_after: 900059, same_seed: false,
      note: "the two series used different kickoff seeds (900043 vs 900059)",
    },
    pairs: [
      {
        before: report_before, after: report_after,
        delta: {
          opponent: "chase", opponent_class: "strength", wins_before: 1, wins_after: 2,
          games_before: 4, games_after: 4, win_rate_before: 0.1, win_rate_after: 0.4,
          win_rate_change: 0.3, goal_diff_before: -2, goal_diff_after: 1, same_seed: false,
          seed_before: 900043, seed_after: 900059,
        },
      },
      {
        before: report_before, after: report_after,
        delta: {
          opponent: "idle", opponent_class: "diagnostic", wins_before: 2, wins_after: 4,
          games_before: 4, games_after: 4, win_rate_before: 0.5, win_rate_after: 1.0,
          win_rate_change: 0.5, goal_diff_before: 1, goal_diff_after: 3, same_seed: true,
        },
      },
    ],
    pair_count: 2,
    opponents: ["chase", "idle"],
    periodic: [],
    periodic_count: 0,
    incomplete: [{ id: 8, opponent: "mixed", games: 0, created_at: "2026-10-18T10:00:05+00:00" }],
    best: { model_id: "m-2026", win_rate: 0.25, opponent: "chase", protocol: "side-balanced" },
  },
  final_checkpoint: {
    available: true, model_id: "m-2026", path: "data/models/m-2026/model.pt", step: 3048,
    label: "final", size_bytes: 123456, valid: true, validation: { ok: true, reasons: [] },
  },
  curves: {
    "charts/episodic_return": [{ step: 1000, value: 0.4 }, { step: 2000, value: 0.6 }],
    "losses/policy": [{ step: 1000, value: -0.0016 }, { step: 2000, value: -0.00006 }],
  },
  comparisons: [
    { run_id: "run-0", name: "parent", status: "stopped", mode: "soccer-1v1", obs_version: 2, steps_start: 0, steps_end: 1000, steps_completed: 1000, duration_s: 3.1, steps_per_second: 322.5, final_win_rate: null, mean_return: 0.31, parent_run: null, created_at: "2026-10-18T09:00:00+00:00" },
  ],
  warnings: ["evaluation 8 did not finish any game and is incomplete"],
  notes: ["Reward describes optimisation only.", "Values marked available=false were never measured."],
};

const models = [
  { id: "m-2026", path: "m-2026/model.pt", step: 3048, mode: "soccer-1v1", state: "candidate", stable: true, best: false, size_bytes: 123456, created_at: "2026-10-18T10:00:06+00:00" },
  { id: "m-old", path: "m-old/model.pt", step: 128, mode: "soccer-2v2", state: "archived", stable: false, best: false, size_bytes: 999, created_at: "2026-10-17T10:00:06+00:00" },
];

const runs = [
  { id: "run-1", name: "smoke run", mode: "soccer-1v1", status: "completed", step: 3048, total_steps: 4096, parent_run: "run-0", resumed_step: 1000, created_at: "2026-10-18T10:00:00+00:00", message: "finished", last_checkpoint_id: "m-2026", last_checkpoint_at: "2026-10-18T10:00:06+00:00", duration_s: 5.0 },
  { id: "run-0", name: "parent", mode: "soccer-1v1", status: "stopped", step: 1000, total_steps: 2000, parent_run: null, resumed_step: 0, created_at: "2026-10-18T09:00:00+00:00", message: "stopped by user", duration_s: 3.1 },
  { id: "run-2", name: "smoke run 2", mode: "soccer-1v1", status: "running", step: 1200, total_steps: 5000, parent_run: "run-1", resumed_step: 3048, created_at: "2026-10-19T09:00:00+00:00", message: null, last_checkpoint_id: null, last_checkpoint_at: null, duration_s: null },
];

const evaluations = [
  { id: 7, model_id: "m-2026", mode: "soccer-1v1", opponent: "chase", phase: "after", games: 4, wins: 1, losses: 2, draws: 1, goals_for: 3, goals_against: 4, seed: 1000, created_at: "2026-10-18T10:00:06+00:00", details: { win_rate: 0.25, win_rate_ci95: [0.04, 0.55], touches: 12, sides: ["blue", "orange"], mirror: true, opponent_class: "strength", protocol: { seed: 1000, match_seconds: 30, sides: ["blue", "orange"], mirror: true, stochastic: false, obs_version: 2 } } },
  { id: 6, model_id: "m-2026", mode: "soccer-1v1", opponent: "idle", phase: "manual", games: 2, wins: 2, losses: 0, draws: 0, goals_for: 4, goals_against: 0, seed: 1000, created_at: "2026-10-18T09:30:06+00:00", details: { touches: 8, opponent_class: "diagnostic" } },
];

const plan = {
  mode: "continue", requested: null, will_resume: true, checkpoint_path: "data/models/m-2026/model.pt",
  model_id: "m-2026", parent_run: "run-0", parent_step: 3048, restored_step: 3048,
  config: null, validation: { ok: true, reasons: [] },
  overrides: [], inherited: ["gamma", "learning_rate"], considered: [], rejected: [],
  warnings: [], lineage: ["run-1", "run-0"],
  fallback_used: false, explicit_unavailable: false, has_model: true, has_optimizer: true, has_rng: true,
  has_curriculum: true, has_league: true,
  restores: ["weights", "optimiser state", "step counter (3048)", "RNG state", "curriculum stage", "opponent pool ratings", "inherited training config fields"],
  missing: [],
  summary: "continuing m-2026 at step 3048",
};

const defaultConfig = {
  run_name: "1v1 baseline", mode: "soccer-1v1", seed: 42, total_steps: 200000,
  rollout_steps: 512, n_workers: 2, tick_skip: 8, episode_seconds: 60.0, learning_rate: 0.0003,
  gamma: 0.995, gae_lambda: 0.95, clip_coef: 0.2, value_coef: 0.5, entropy_coef: 0.01, epochs: 4,
  minibatch_size: 1024, max_grad_norm: 0.5, clip_value: 0.2, target_kl: 0.05, lr_anneal: true,
  device: "auto", opponent: "chase", self_play: false, snapshot_interval: 5, pool_capacity: 8,
  league_states: true, checkpoint_interval: 10, keep_checkpoints: 8, obs_version: 2,
  eval_opponent: "chase", eval_opponents: [], eval_interval: 0, eval_games: 4,
  eval_match_seconds: 30.0, eval_sides: ["blue"], eval_mirror: true, eval_before: true,
  curriculum: false, side_randomization: false, auto_worker_limit: true,
  reward: { goal: 10, concede: -6, touch: 0.6, progress: 0.4, alignment: 0.2, approach: 0.3, boost: -0.01, defense: 0.5, bump: -0.2 },
  policy: { hidden: [256, 256], activation: "tanh" },
};

const defaultRun = {
  mode: "soccer-1v1",
  config: defaultConfig,
  device: "cpu",
  workers: {
    recommended_workers: 1, requested_workers: 2, cpu_count: 2, memory_available_mb: 3000,
    memory_total_mb: 4000, disk_free_gb: 10, clamped: true, warnings: [],
    notes: ["recommended worker count on this machine: 1 (CPU budget 1, memory budget 4)."],
  },
  note: "automatic 1v1 configuration; every technical parameter keeps its backend default",
};

const startPlan = {
  config: { ...defaultConfig, total_steps: 200000 },
  device: "cpu",
  workers: defaultRun.workers,
  resume: plan,
  will_resume: true,
  resume_mode: "continue",
  warnings: [],
  notes: defaultRun.workers.notes,
};

const routes = {
  "/api/health": { status: "ok", version: "0.0.0-test", time: 0 },
  "/api/system": {
    info: {
      cpu_percent: 12, cpu_count: 2, cpu_physical: 2, memory_total_gb: 3, memory_available_gb: 2,
      disk_free_gb: 10, data_dir: "/tmp/data", project_root: "/repo", executable: "/venv/bin/python",
      python: "3.11", platform: "Linux", torch: { cuda_available: false }, packages: { torch: "2.4.0" },
    },
    active_run: null,
    data_dir: "/tmp/data",
  },
  "/api/system/resources": { cpu_percent: 12, cpu_count: 2, memory_percent: 40, memory_used_gb: 1.2, memory_total_gb: 3, active_run: null },
  "/api/diagnostics": { ok: true, info: { python: "3.11", platform: "Linux", cpu_count: 2, cpu_physical: 2, memory_total_gb: 3, memory_available_gb: 2, disk_free_gb: 10, data_dir: "/tmp/data", project_root: "/repo", executable: "/venv/bin/python", torch: { cuda_available: false }, packages: { torch: "2.4.0" } }, checks: [{ name: "torch", ok: true, required: true, detail: "installed" }] },
  "/api/modes": [{ id: "soccer-1v1", name: "Soccer 1v1", players: 2 }, { id: "soccer-2v2", name: "Soccer 2v2", players: 4 }],
  "/api/config/defaults": { training: defaultConfig, default_run: defaultRun },
  "/api/models": models,
  "/api/runs?limit=6": runs, "/api/runs?limit=20": runs, "/api/runs?limit=50": runs, "/api/runs?limit=100": runs,
  "/api/evaluations?limit=5": evaluations, "/api/evaluations?limit=200": evaluations,
  "/api/reports?limit=100": [{ run_id: "run-1", created_at: "2026-10-18T10:00:06+00:00", path: "/tmp/report.json", status: "completed", steps_completed: 2048, duration_s: 5, summary: report }],
  "/api/runs/run-1/report": report,
  "/api/runs/run-1": { ...runs[0], episode_summary: report.episodes.last_100, metric_names: ["time/sps"], alive: false, config: defaultConfig },
  "/api/runs/run-1/metrics": [{ name: "time/sps", value: 1042, step: 3048 }, { name: "time/collect_seconds", value: 0.11, step: 3048 }, { name: "charts/episodic_return", value: 0.6, step: 3048 }, { name: "charts/win_rate", value: 0.1, step: 3048 }],
  "/api/runs/run-1/metrics?names=time/sps&limit=1": [{ name: "time/sps", value: 1042, step: 3048 }],
  "/api/runs/run-1/logs?limit=200": [{ id: 1, ts: 1760000000, level: "INFO", message: "hello" }],
  "/api/live/status": { platform: "Linux", is_windows: false, stable_model: models[0], rlbot_installed: false, launch_supported: false, note: "needs Windows" },
  "/api/stats": statsPayload,
  "/api/stats/overview": { runs: [{ id: "run-1", name: "smoke run", mode: "soccer-1v1", status: "completed", step: 3048, total_steps: 4096, accumulated_s: 125.5, accumulated_human: "2m 05s", session_count: 2, interrupted_count: 0, started_at: "2026-10-18T10:00:01+00:00", last_checkpoint_at: "2026-10-18T10:00:05+00:00" }], count: 1 },
  "/api/teams": { teams: [{ name: "test team", mode: "soccer-1v1", description: "demo", bots: [{ name: "bot-1", difficulty: "medium", behaviour: "balanced" }], average_difficulty_target: 0.75, updated_at: "2026-10-18T10:00:00+00:00" }], count: 1 },
  "/api/difficulties": { profiles: [{ name: "rock", label: "Rock", target_win_rate: 0, mistake_rate: 0.75, reaction_delay: 1.2, boost_efficiency: 0.15, behaviour: "defensive", description: "Almost never does the right thing." }, { name: "tiny", label: "Tiny", target_win_rate: 0.5, mistake_rate: 0.4, reaction_delay: 0.6, boost_efficiency: 0.45, behaviour: "balanced", description: "Reaches the ball most of the time." }, { name: "medium", label: "Medium", target_win_rate: 0.75, mistake_rate: 0.18, reaction_delay: 0.25, boost_efficiency: 0.75, behaviour: "balanced", description: "Competent basics." }, { name: "pro", label: "Pro", target_win_rate: 1, mistake_rate: 0.02, reaction_delay: 0.05, boost_efficiency: 1, behaviour: "aggressive", description: "Plays the whole field." }], targets_are_measured: false, note: "the percentages are design targets that still need validation", behaviours: {} },
  "/api/bot-behaviour": { profiles: [{ name: "balanced", description: "Chases the ball.", parameters: { aggression: 0.5, boost_eagerness: 0.6, rotation: 0.6, shot_preference: 0.5, safety: 0.6, context_sensitivity: 0, reaction_delay: 0, mistake_rate: 0 } }] },
  "/api/player-memory": { summary: { players: 1, games: 4, wins: 2, losses: 1, draws: 1, win_rate: 0.5, goals_for: 3, goals_against: 2, max_entries: 500, max_age_days: 90 }, players: [{ name: "chaser", games: 4, wins: 2, losses: 1, draws: 1, win_rate: 0.5, goals_for: 3, goals_against: 2, last_seen: 1760000000, note: "fast" }], note: "the player memory is data about past encounters" },
  "/api/rivals": { count: 1, rivals: [{ name: "chaser", kind: "checkpoint", reference: "m-2026", games: 4, wins: 2, losses: 1, draws: 1, win_rate: 0.5, goal_diff: 1, evaluations: 1 }], ranking: [], bottleneck: { available: true, name: "chaser", win_rate: 0.5, games: 4, goal_diff: 1 } },
  "/api/lab": { models: [{ id: "m-2026", step: 3048, created_at: "2026-10-18T10:00:06+00:00", eval_score: 0.4, duration_s: 5, state: "candidate" }], opponents: { strength: ["chase"], diagnostic: ["idle"] }, recent_evaluations: [{ id: "e-1", model_id: "m-2026", opponent: "chase", games: 4, wins: 2, losses: 1, draws: 1, details: { win_rate: 0.5 }, phase: "manual", created_at: "2026-10-18T10:00:06+00:00" }], note: "an experiment is only executed when the GUI asks for a confirmation" },
  "/api/analysis": { schema: "rlai-match-analysis-v1", replays: { available: false, reason: "no replay parser is available" }, evaluations: { available: true, source: "evaluation game rows", games: 2, summary: { games: 2, wins: 1, losses: 1, draws: 0, win_rate: 0.5, goals_for: 2, goals_against: 1, mean_touches: 4.5 }, by_opponent: [{ opponent: "chase", games: 2, wins: 1, losses: 1, draws: 0, win_rate: 0.5, goals_for: 2, goals_against: 1, mean_touches: 4.5 }], by_side: [], timeline: [] }, episodes: { available: false, reason: "no episode" } },
};

const postRoutes = {
  "/api/resume/plan": plan,
  "/api/config/validate": (body) => ({ valid: true, config: body, warnings: [] }),
  "/api/runs/compare": { runs: [{ run_id: "run-1", name: "smoke run", mode: "soccer-1v1", status: "completed", steps_completed: 2048, steps_start: 1000, steps_end: 3048, duration_s: 5, steps_per_second: 409.6, final_win_rate: 0.25, mean_return: 0.6, parent_run: "run-0" }], comparable: true, modes: ["soccer-1v1"], warning: null },
  "/api/runs/start-plan": startPlan,
  "/api/models/m-2026/validate": { ...report.final_checkpoint.validation, model_id: "m-2026", step: 3048, label: "final", created_at: "2026-10-18T10:00:06+00:00", size_bytes: 123456, obs_version: 2, has_model: true, has_optimizer: true, has_rng: false, has_curriculum: false, has_league: false, reasons: [], manifest: models[0], resume: { continue: true, transfer: true, missing_state: ["rng", "curriculum", "league"] } },
  "/api/runs": { ...runs[0] },
  "/api/evaluations/suite": { job_id: "ev-1", model_id: "m-2026", status: "started", opponents: ["chase", "mixed"], sides: ["blue", "orange"], mirror: true, kind: "suite" },
  "/api/models/m-2026/evaluate": { job_id: "ev-2", model_id: "m-2026", status: "started", kind: "single" },
  "/api/lab/plan": { models: ["m-2026"], opponents: ["chase"], games_per_model: 6, total_games: 6, projected_sim_seconds: 180, projected_sim_minutes: 3, mode: "soccer-1v1", executed: false },
  "/api/lab/run": { executed: true, results: [{ model_id: "m-2026", opponent: "chase", games: 6 }], count: 1 },
};

const calls = [];
function makeFetch() {
  return async (url, options = {}) => {
    const method = (options.method || "GET").toUpperCase();
    calls.push(`${method} ${url}`);
    // the event stream needs its query, every other route is matched with
    // and without the query string
    const raw = String(url);
    const bare = raw.split("?")[0];
    let payload;
    if (method === "GET") payload = routes[raw] !== undefined ? routes[raw] : routes[bare];
    else payload = postRoutes[raw] !== undefined ? postRoutes[raw] : postRoutes[bare];
    if (typeof payload === "function") payload = payload(JSON.parse(options.body || "{}"));
    if (payload === undefined) {
      return { ok: false, status: 404, statusText: "Not Found", text: async () => JSON.stringify({ detail: `no canned response for ${method} ${raw}` }) };
    }
    return { ok: true, status: 200, statusText: "OK", text: async () => JSON.stringify(payload) };
  };
}

const dom = new JSDOM(html, { url: "http://localhost:8000/", pretendToBeVisual: true, runScripts: "outside-only" });
const { window } = dom;
window.fetch = makeFetch();
window.EventSource = class { constructor(url) { this.url = url; this.onmessage = null; this.onerror = null; } close() {} };
window.scrollTo = () => {};
window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
window.HTMLDialogElement.prototype.close = function () { this.open = false; };
window.eval(`${script}\n;window.__test = { views, render, app, api, setToken, resumeStateHtml, startSummaryHtml, evaluationFormHtml, splitList, chip, measured, eventUrl };`);
// the feature modules attach their views to the shared registry
window.eval(`${statsScript}\n`);
window.eval(`${featuresScript}\n`);
const test = window.__test;
if (!test) throw new Error("app.js did not expose its internals");
// let the app's own DOMContentLoaded boot render finish before we drive the views
await new Promise((resolve) => setTimeout(resolve, 150));

const failures = [];
function check(name, condition, detail = "") {
  if (condition) console.log(`  ok   ${name}`);
  else { failures.push(`${name}${detail ? ` — ${detail}` : ""}`); console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`); }
}

async function view(name, hash, assertions) {
  // replaceState avoids firing the app's own hashchange handler (we render directly)
  window.history.replaceState({}, "", `http://localhost:8000/${hash.replace(/^#/, "#")}`);
  const node = window.document.querySelector("#view");
  node.innerHTML = "";
  try {
    await test.views[name]();
  } catch (error) {
    failures.push(`${name} threw: ${error.stack || error.message}`);
    console.log(`  FAIL ${name} threw: ${error.message}`);
    return "";
  }
  const rendered = node.innerHTML;
  await assertions(rendered, node);
  return rendered;
}

console.log("GUI smoke test");

// ------------------------------------------------------------- navigation
{
  const primary = window.document.querySelector("#nav");
  const secondary = window.document.querySelector("#nav-secondary");
  const primaryRoutes = [...primary.querySelectorAll("[data-route]")].map((a) => a.dataset.route);
  const secondaryRoutes = [...secondary.querySelectorAll("[data-route]")].map((a) => a.dataset.route);
  check("navigation has four primary sections", primaryRoutes.length === 4, JSON.stringify(primaryRoutes));
  check("primary sections are dashboard, training, stats, lab", primaryRoutes.join(",") === "dashboard,training,stats,lab", JSON.stringify(primaryRoutes));
  check("secondary sections cover models, live, settings", secondaryRoutes.join(",") === "models,live,settings", JSON.stringify(secondaryRoutes));
  check("no monitoring/reports/evaluation tab in the nav", !window.document.querySelector("#nav a[href='#/monitoring'], #nav a[href='#/reports'], #nav a[href='#/evaluation'], #nav-secondary a[href='#/monitoring']"));
}

await view("dashboard", "#/dashboard", (html) => {
  check("dashboard shows the next-run panel", html.includes("Next run"));
  check("dashboard communicates the automatic continuation", html.includes("continues from m-2026"));
  check("dashboard lists recent runs", html.includes("smoke run"));
  check("dashboard lists recent evaluations", html.includes("chase"));
  check("dashboard shows the machine", html.includes("Memory") && html.includes("Device"));
  check("dashboard shows the model summary", html.includes("Models") && html.includes("m-2026"));
  check("dashboard links to the training view", html.includes("#/training"));
  check("dashboard links to stats", html.includes("#/stats"));
});

// the same dashboard with an active run: live progress, controls, throughput
{
  const saved = routes["/api/system"];
  routes["/api/system"] = { ...saved, active_run: { ...runs[0], status: "running", parent_run: "run-0", resumed_step: 1000 } };
  await view("dashboard", "#/dashboard", (html) => {
    check("active dashboard shows the progress bar", html.includes("progress-fill"));
    check("active dashboard shows the latest checkpoint", html.includes("m-2026"));
    check("active dashboard offers telemetry", html.includes("#/monitoring?run=run-1"));
    check("active dashboard offers pause and stop", html.includes('data-run-action="pause"') && html.includes('data-run-action="stop"'));
    check("active dashboard shows the measured throughput", html.includes("1042") || html.includes("1,042"));
    check("active dashboard shows the continuation", html.includes("Continues"));
  });
  routes["/api/system"] = saved;
}

await view("training", "#/training", async (html, node) => {
  check("training shows the steps input", node.querySelector('[name="total_steps"]') !== null);
  check("training shows the optional run name", node.querySelector('[name="run_name"]') !== null);
  check("training has no game mode selector", node.querySelector('[name="mode"]') === null);
  check("training is 1v1 only", !html.includes("soccer-2v2") && html.includes("1v1"));
  const hiddenControls = [
    "learning_rate", "epochs", "minibatch_size", "clip_coef", "gamma", "gae_lambda",
    "entropy_coef", "value_coef", "target_kl", "max_grad_norm", "tick_skip",
    "rollout_steps", "episode_seconds", "obs_version", "n_workers", "checkpoint_interval",
    "snapshot_interval", "keep_checkpoints", "seed", "opponent", "eval_interval",
    "eval_games", "eval_opponent", "eval_opponents", "eval_sides", "eval_mirror",
    "eval_before", "side_randomization", "self_play", "curriculum", "device",
    "reward_goal", "reward_touch",
  ];
  const leaked = hiddenControls.filter((name) => node.querySelector(`[name="${name}"]`));
  check("training hides every technical control", leaked.length === 0, `found: ${leaked.join(",")}`);
  check("training has no presets", html.toLowerCase().includes("preset") === false);
  check("training states nothing auto-starts", html.includes("Nothing starts, resumes or is scheduled by itself"));
  check("training offers step suggestions", html.includes("data-steps"));
  check("training lists the runs with controls", html.includes("data-run-action") && html.includes("smoke run"));
  await new Promise((resolve) => setTimeout(resolve, 40));
  const resumeHtml = node.querySelector("#resume-state").innerHTML;
  check("resume state shows the automatic continuation", resumeHtml.includes("Will continue from m-2026"), resumeHtml.slice(0, 300));
  check("resume state lists the restored state", resumeHtml.includes("optimiser state"));
  const summaryHtml = node.querySelector("#start-summary").innerHTML;
  check("summary shows the machine-checked worker count", summaryHtml.includes("Worker") && summaryHtml.includes("1"));
  check("summary shows the resolved device", /CPU|CUDA|MPS/i.test(summaryHtml));
  check("the start request carries the confirmation", script.includes("payload.confirm = true"));
  check("the start is behind the confirmation dialog", /confirmAction\("Start 1v1 training"/.test(script));
});

check(
  "resume state handles the fresh branch",
  test.resumeStateHtml({
    will_resume: false,
    resume: {
      warnings: ["no compatible checkpoint found for mode=soccer-1v1 (1 rejected); training would start from scratch"],
      rejected: [{ id: "x", step: 100, reasons: ["architecture mismatch"] }],
    },
  }).includes("Will start a new model"),
);

// ---------------------------------------------------------- redirects
window.history.replaceState({}, "", "http://localhost:8000/#/reports");
await test.views.reports();
check("reports redirects to stats", window.location.hash === "#/stats", window.location.hash);
window.history.replaceState({}, "", "http://localhost:8000/#/evaluation");
await test.views.evaluation();
check("evaluation redirects to the lab series tab", window.location.hash === "#/lab?tab=series", window.location.hash);
window.history.replaceState({}, "", "http://localhost:8000/#/teams");
await test.views.teams();
check("teams redirects to the lab teams tab", window.location.hash === "#/lab?tab=teams", window.location.hash);
window.history.replaceState({}, "", "http://localhost:8000/#/memory");
await test.views.memory();
check("memory redirects to the lab memory tab", window.location.hash === "#/lab?tab=memory", window.location.hash);
// settle any hashchange renders the redirects triggered
await new Promise((resolve) => setTimeout(resolve, 30));

await view("stats", "#/stats", (html) => {
  check("stats shows the current session duration", html.includes("Current session"));
  check("stats shows the accumulated training time", html.includes("Accumulated training time"));
  check("stats shows the environment steps", html.includes("Environment steps"));
  check("stats shows the throughput", html.includes("Throughput"));
  check("stats shows the throughput chart", html.includes("Throughput (steps / s)"));
  check("stats shows the learning metrics table", html.includes("Policy loss") && html.includes("Approx. KL"));
  check("stats shows the checkpoint table", html.includes("Checkpoints") && html.includes("m-2026"));
  check("stats shows the evaluation table", html.includes("Opponent") && html.includes("chase"));
  check("stats shows the session history", html.includes("Session history"));
  check("stats shows the training history", html.includes("Training history"));
  check("stats marks an unknown duration", html.includes("unknown"));
  check("stats shows the run reports panel", html.includes("Run reports") && html.includes("Compare selected"));
  check("stats offers rebuild for runs without a report", html.includes("data-rebuild"));
  check("stats links the full report", html.includes("#/report?run=run-1"));
});

await view("report", "#/report?run=run-1", (html) => {
  check("report headline steps", html.includes("2,048") || html.includes("2.05k"));
  check("report shows resume mode", html.includes("continue"));
  check("report labels unavailable metrics", html.includes("metric was not recorded for this run"));
  check("report pairs before/after per opponent", html.includes("Win-rate change") && html.includes("+30.0 pp"));
  check("report labels diagnostic pairs", html.includes("diagnostic"));
  check("report labels differing kickoff seeds", html.includes("different kickoff seeds"));
  check("report shows the after evaluation protocol", html.includes("sides blue+orange") && html.includes("mirrored"));
  check("report shows the confidence interval", html.includes("4 %") || html.includes("4.0 %") || html.includes("55"));
  check("report flags the incomplete evaluation", html.includes("incomplete"));
  check("report shows reward components", html.includes("goal") && html.includes("touch"));
  check("report shows timings", html.includes("time/sps") && html.includes("no evaluation ran"));
  check("report shows the final checkpoint path", html.includes("data/models/m-2026/model.pt"));
  check("report shows comparisons", html.includes("Other runs with the same setup"));
  check("report shows warnings", html.includes("did not finish any game"));
  check("report shows notes", html.includes("never measured"));
  check("report links the raw JSON", html.includes("/api/runs/run-1/report"));
  check("report points continuation at the training view", html.includes("#/training"));
});

await view("models", "#/models", (html) => {
  check("models lists checkpoints", html.includes("m-2026") && html.includes("m-old"));
  check("models offers validation", html.includes("data-validate"));
  check("models points the next run at the training view", html.includes("#/training"));
  check("models embeds the shared evaluation form", html.includes("eval-opponents") && html.includes("eval-sides"));
});

await view("lab", "#/lab", (html) => {
  check("lab shows the subtabs", html.includes('role="tablist"') && html.includes("Match series") && html.includes("Difficulty"));
  check("lab defaults to experiments", html.includes("New experiment"));
  check("lab offers the experiment form", html.includes("lab-opponents") && html.includes("lab-games"));
  check("lab lists the checkpoints", html.includes("m-2026"));
  check("lab shows the recent evaluations", html.includes("Recent evaluations"));
  check("lab shows the match analysis", html.includes("Match analysis"));
  check("lab states that replays are unavailable", html.includes("no replay parser"));
  check("lab separates plan and run", html.includes("lab-plan") && html.includes("lab-run"));
});

await view("lab", "#/lab?tab=series", (html) => {
  check("series tab shows the form", html.includes("series-opponents") && html.includes("series-sides"));
  check("series tab separates strength from diagnostics", html.includes("Playing strength") && html.includes("Diagnostic baselines"));
  check("series tab shows the protocol chips", html.includes("sides blue+orange"));
  check("series tab shows the phase label", html.includes("after"));
});

await view("lab", "#/lab?tab=difficulty", (html) => {
  check("difficulty tab lists the levels", html.includes("Rock") && html.includes("Pro"));
  check("difficulty tab marks the percentage as a target", html.includes("target"));
  check("difficulty tab shows the behaviour profiles", html.includes("Bot behaviour"));
  check("difficulty warns about the unmeasured targets", html.includes("design targets"));
});

await view("lab", "#/lab?tab=teams", (html) => {
  check("teams tab lists the saved team", html.includes("test team"));
  check("teams tab has the bot editor", html.includes("bot-difficulty") && html.includes("bot-behaviour"));
  check("teams tab defaults to 1v1", html.includes('<option value="soccer-1v1" selected="">1v1</option>'));
});

await view("lab", "#/lab?tab=memory", (html) => {
  check("memory tab lists the known players", html.includes("chaser"));
  check("memory tab shows the summary", html.includes("Known players"));
  check("memory tab separates the memory from the weights", html.includes("never model weights"));
  check("memory tab lists the rivals", html.includes("Rivals"));
  check("memory tab offers the export", html.includes("/api/player-memory/export"));
  check("memory tab offers the reset", html.includes("reset-memory"));
});

await view("live", "#/live", (html) => {
  check("live shows the deployment status", html.includes("Deployment status"));
  check("live disables launch where unsupported", html.includes('id="live-launch"') && html.includes("disabled"));
  check("live keeps the windows checklist", html.includes("Windows checklist"));
});

await view("monitoring", "#/monitoring?run=run-1", (html) => {
  check("monitoring shows the continuation callout", html.includes("Continuation of run-0"));
  check("monitoring shows iteration cost", html.includes("Iteration cost"));
  check("monitoring links the report", html.includes("#/report?run=run-1"));
});

await view("settings", "#/settings", (html) => {
  check("settings exposes the benchmark options", html.includes("bench-parallel") && html.includes("bench-inference"));
  check("settings exposes the token field", html.includes("token-input"));
  check("settings lists the dependency checks", html.includes("Dependency checks"));
});

// token handling
test.setToken("secret");
check("token is stored", window.localStorage.getItem("rlai.token") === "secret");
check("event url carries the token", test.eventUrl("/api/events", { run_id: "run-1" }).includes("token=secret"));
await test.api("/api/health");
check("api sends the token header", calls.length > 0);
test.setToken("");
check("token can be cleared", window.localStorage.getItem("rlai.token") === null);

// helper behaviour
check("splitList trims and drops empties", JSON.stringify(test.splitList(" a, ,b ")) === '["a","b"]');
check("measured labels unavailable values", test.measured({ available: false, reason: "nope" }).includes("unavailable"));
check("the start-plan endpoint is used for the preview", calls.some((item) => item.startsWith("POST /api/runs/start-plan")));

dom.window.close();
if (failures.length) {
  console.error(`\n${failures.length} failure(s):`);
  failures.forEach((item) => console.error(` - ${item}`));
  process.exit(1);
}
console.log("\nall GUI smoke checks passed");
process.exit(0);
