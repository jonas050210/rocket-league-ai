/* One offline client shared by Qt and the local diagnostic browser. No simulated metrics. */
const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const icons = {
  dashboard: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  training: "m9 5 10 7-10 7V5Z M4 5v14",
  monitoring: "M3 12h4l3-8 4 16 3-8h4",
  models: "m12 3 9 5-9 5-9-5 9-5Z M3 12l9 5 9-5 M3 16l9 5 9-5",
  configuration: "M4 7h16 M4 17h16 M8 4v6 M16 14v6",
  visualization: "M3 6h18v12H3z M8 21h8 M12 18v3 m-2-8 4 2-4 2v-4Z",
  plus: "M12 5v14 M5 12h14",
  arrow: "M5 12h14 M14 7l5 5-5 5",
  check: "m5 12 4 4 10-10",
  clock: "M12 8v5l3 2 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
  bolt: "m13 2-9 12h7l-1 8 10-12h-7l1-8Z",
  trophy:
    "M8 3h8v7a4 4 0 0 1-8 0V3Z M8 5H4v3a4 4 0 0 0 4 4 M16 5h4v3a4 4 0 0 1-4 4 M12 14v6 M8 21h8",
  cpu: "M6 6h12v12H6z M9 9h6v6H9z M9 3v3 M15 3v3 M9 18v3 M15 18v3 M3 9h3 M3 15h3 M18 9h3 M18 15h3",
  shield: "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z m-4 9 3 3 5-5",
  info: "M12 11v6 M12 7v.1 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
  pause: "M8 5v14 M16 5v14",
  play: "m8 5 11 7-11 7V5Z",
  stop: "M6 6h12v12H6z",
  download: "M12 3v12 m-5-5 5 5 5-5 M4 17v4h16v-4",
  logs: "M5 3h14v18H5z M8 8h8 M8 12h8 M8 16h5",
  refresh:
    "M20 7v5h-5 M4 17v-5h5 M6 7a7 7 0 0 1 12-1l2 3 M4 15l2 3a7 7 0 0 0 12-1",
  delete: "M3 6h18 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7",
  lock: "M5 10h14v11H5z M8 10V7a4 4 0 0 1 8 0v3",
  chart: "M4 4v16h16 M8 14l4-4 4 2 4-7",
  file: "M14 3H5v18h14V8l-5-5Z M14 3v5h5 M8 13h8 M8 17h5",
};
const icon = (name) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${icons[name] || icons.info}"/></svg>`;
const names = {
  dashboard: "Dashboard",
  training: "Training",
  monitoring: "Monitoring",
  models: "Models",
  configuration: "Configuration",
  visualization: "Visualization",
  benchmark: "Benchmark",
};
const activeStatuses = new Set([
  "starting",
  "running",
  "pausing",
  "paused",
  "resuming",
  "stopping",
]);
const terminalStatuses = new Set([
  "stopped",
  "failed",
  "interrupted",
  "completed",
]);
let state = null,
  events = [],
  cursor = 0,
  connected = false,
  readOnly = false,
  health = null;
let route = "dashboard",
  chartKind = "reward",
  selectedRun = null,
  pendingProposal = null;
let configDraft = null,
  configRaw = null,
  logLevel = "all",
  logSearch = "",
  refreshPromise = null;
let benchmarkForm = {
  devices: ["cpu"],
  n_proc_values: [1],
  batch_size: 512,
  warmup_iterations: 1,
  measure_iterations: 3,
  repeats: 2,
  max_cpu_percent: 100,
  max_memory_percent: 95,
  timeout_seconds: 600,
};
let evaluationForm = { opponent_id: "random-v1", games: 20, seed: 1000, parallel_workers: 1 };
let replay = { id: null, data: null, frame: 0, playing: false, speed: 1, follow: true, timer: null };
let busy = false,
  toastTimer,
  lastConnectedAt = null;
let token = sessionStorage.getItem("rlai-capability") || "";
function consumeCapabilityFragment() {
  const fragment = new URLSearchParams(location.hash.slice(1));
  if (!fragment.has("token")) return false;
  token = fragment.get("token") || "";
  sessionStorage.setItem("rlai-capability", token);
  history.replaceState(null, "", location.pathname + "#dashboard");
  return true;
}
consumeCapabilityFragment();
const number = (value, precision = 0) =>
  value == null
    ? "—"
    : Number(value).toLocaleString(undefined, {
        maximumFractionDigits: precision,
        minimumFractionDigits: precision,
      });
const short = (value, width = 8) => String(value || "").slice(0, width);
const time = (value) =>
  new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
const date = (value) =>
  new Date(value * 1000).toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
const bytes = (value) =>
  value == null
    ? "—"
    : value >= 1024 ** 3
      ? `${number(value / 1024 ** 3, 1)} GB`
      : `${number(value / 1024 ** 2, 1)} MB`;
const authorized = () => connected && !!token;
const writable = () => authorized() && !busy;
const idle = () =>
  !state?.active_run && !state?.worker_health.orphan_lease_held;
const disabled = (condition) => (condition ? " disabled" : "");
const statusPill = (status) =>
  `<span class="pill status-pill ${esc(status || "idle")}">${esc((status || "idle").toUpperCase())}</span>`;
const button = (
  action,
  label,
  ico = "arrow",
  style = "secondary",
  off = false,
  extra = "",
) =>
  `<button class="button ${style}" data-action="${action}"${disabled(off)} ${extra}>${icon(ico)}${esc(label)}</button>`;
const heading = (title, subtitle, actions = "") =>
  `<div class="page-heading"><div><span class="eyebrow">YOUR TRAINING WORKSPACE</span><h1>${esc(title)}</h1><p>${esc(subtitle)}</p></div><div class="heading-actions">${actions}</div></div>`;
const empty = (ico, title, description, action = "") =>
  `<div class="empty-state"><div class="empty-symbol">${icon(ico)}</div><h2>${esc(title)}</h2><p>${esc(description)}</p>${action}</div>`;
const stat = (label, value, unit, foot, ico) =>
  `<div class="stat-card"><div class="stat-label">${esc(label)}${icon(ico)}</div><div class="stat-value">${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}</div><div class="stat-foot"><span class="mini-dot"></span>${esc(foot)}</div></div>`;
const metadataRow = (key, value) =>
  `<div class="metadata-row"><span>${esc(key)}</span><strong>${esc(value)}</strong></div>`;
const fieldArt = () =>
  `<svg class="field-art" viewBox="0 0 360 250" fill="none" aria-hidden="true"><g stroke="currentColor" stroke-width=".65"><path d="m20 176 170-101 154 88-170 80-154-67Z M34 170l144 63 151-71 M190 75l-12 158 M103 127l156 77 M277 124l-158 83" opacity=".42"/><path d="m58 173 130-77 116 66-130 62-116-51Z M188 96l-14 128"/><ellipse cx="181" cy="160" rx="31" ry="16" transform="rotate(-3 181 160)"/><path d="m74 163 24 11-19 11 M287 153l-26 13 21 10 M58 173v-12l130-78 116 67v12 M188 83v13 M98 174v-12l-24-11 M261 166v-12l26-13 M154 208l-3-14 43-21 3 14 M151 194l-15-6 M197 187l-15-6"/><circle cx="181" cy="160" r="2" fill="currentColor"/><path d="M22 190v16 M24 206h20 M316 177h20v-15 M174 58v-9h15" opacity=".3"/></g><g fill="currentColor" opacity=".45"><circle cx="63" cy="111" r="1"/><circle cx="288" cy="87" r="1"/><circle cx="322" cy="208" r="1"/></g></svg>`;

async function api(path, body, method = body === undefined ? "GET" : "POST") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 6000);
  try {
    const headers = { Accept: "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(`/api/${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 401 && !readOnly) {
        connected = false;
        if (!$("#connect-dialog").open) $("#connect-dialog").showModal();
      }
      const detail = Array.isArray(data.detail)
        ? data.detail
            .map((item) => `${(item.loc || []).join(".")}: ${item.msg}`)
            .join("\n")
        : data.detail;
      throw new Error(detail || `Backend returned HTTP ${response.status}`);
    }
    return data;
  } finally {
    clearTimeout(timeout);
  }
}
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").className = error ? "error" : "";
  $("#toast").hidden = false;
  toastTimer = setTimeout(
    () => {
      $("#toast").hidden = true;
    },
    error ? 10000 : 5000,
  );
}
function connectionUI() {
  $("#connection").className = `connection ${connected ? "online" : "offline"}`;
  $("#connection").innerHTML =
    `<i></i>${connected ? "Backend connected" : "Backend disconnected"}`;
  $("#preview-badge").hidden = !readOnly || !!token;
  $("#confirm-action").disabled =
    !pendingProposal || !$("#approval-checkbox").checked || !writable();
  const banner = $("#connection-banner");
  banner.hidden = connected || !state;
  if (!banner.hidden)
    banner.textContent = `Connection lost. Displaying the last received state${lastConnectedAt ? ` from ${time(lastConnectedAt)}` : ""}. Controls are disabled. Reconnecting automatically; the independent backend may still be training.`;
}
function navigation() {
  $("#navigation").innerHTML = Object.entries(names)
    .map(
      ([key, name]) =>
        `<a href="#${key}" data-nav="${key}" class="nav-item ${route === key ? "active" : ""}" ${route === key ? 'aria-current="page"' : ""} title="${name}">${icon(key)}<span class="nav-text">${name}</span>${key === "models" && state?.checkpoints.length ? `<span class="nav-count">${state.checkpoints.length}</span>` : ""}</a>`,
    )
    .join("");
  $("#breadcrumb-view").textContent = names[route];
}
function viewedRun() {
  return (
    state?.runs.find((run) => run.id === selectedRun) ||
    state?.active_run ||
    state?.runs[0] ||
    null
  );
}
function runMetrics(run) {
  return run?.metrics || {};
}
function runError(run) {
  if (!run || !["failed", "interrupted"].includes(run.status)) return "";
  const recoverable = state.latest_valid[run.id] && idle();
  return `<div class="callout error">${icon("shield")}<div class="callout-copy"><strong>${run.status === "interrupted" ? "Run interrupted — recovery is your decision" : "Training stopped safely after a failure"}</strong><p>${esc(run.error || "Inspect the logs before restarting. The last committed checkpoint is preserved.")}</p></div>${button("recover", "Review recovery", "refresh", "secondary", !writable() || !recoverable, `data-run-id="${run.id}"`)}</div>`;
}
function metricsGrid(run) {
  const m = runMetrics(run);
  return `<div class="stat-grid">${stat("Agent steps", number(m.steps), "", run ? `${number(run.config.total_timesteps)} planned steps` : "No run started", "bolt")}${stat("Completed episodes", number(m.episodes), "", "Real simulator episodes", "training")}${stat("Mean reward", number(m.reward, 3), "", m.reward == null ? "Available after first PPO update" : "Per-agent transition reward", "chart")}${stat("Training speed", number(m.steps_per_second), "steps/s", m.steps_per_second == null ? "Waiting for a PPO update" : "Collection + PPO update timing", "clock")}</div>`;
}
function chart(run) {
  const samples = events
    .filter(
      (event) =>
        event.kind === "training_metrics" &&
        event.run_id === run?.id &&
        Number.isFinite(
          event.data[chartKind === "reward" ? "reward" : "steps_per_second"],
        ),
    )
    .slice(-80);
  let graph = "";
  if (samples.length) {
    const values = samples.map(
      (event) =>
        event.data[chartKind === "reward" ? "reward" : "steps_per_second"],
    );
    const low = Math.min(...values),
      high = Math.max(...values),
      range = high - low || Math.max(Math.abs(high), 1);
    const points = values
      .map(
        (value, index) =>
          `${samples.length === 1 ? 300 : 10 + (index / (samples.length - 1)) * 580},${155 - ((value - low) / range) * 125}`,
      )
      .join(" ");
    graph = `<svg viewBox="0 0 600 180" preserveAspectRatio="none" role="img" aria-label="Measured ${chartKind} history"><polyline fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke" points="${points}"/>${samples.length === 1 ? `<circle cx="300" cy="155" r="3" fill="var(--accent)"/>` : ""}</svg>`;
  } else
    graph = `<div class="chart-empty">${icon("chart")}<span>Signals appear after the first PPO update</span></div>`;
  return `<section class="panel chart-panel"><div class="panel-header"><div><h2>Learning signals</h2><p>Measured updates, never a projected curve</p></div><div class="chart-tabs"><button data-chart="reward" class="${chartKind === "reward" ? "selected" : ""}">Reward</button><button data-chart="speed" class="${chartKind === "speed" ? "selected" : ""}">Throughput</button></div></div><div class="chart"><div class="chart-grid"><span></span><span></span><span></span><span></span></div>${graph}</div><div class="chart-legend"><span class="legend-line"></span>${chartKind === "reward" ? "Mean transition reward" : "Overall agent steps / second"}<span style="margin-left:auto">${samples.length ? `${samples.length} measured updates` : "No data yet"}</span></div></section>`;
}
function resources() {
  const s = state.system,
    gpu = s.gpu_percent;
  return `<section class="panel"><div class="panel-header"><div><h2>System resources</h2><p>Measured on the backend host</p></div>${icon("cpu")}</div><div class="panel-body"><div class="resource-item"><div class="resource-label"><span>CPU utilization</span><strong>${number(s.cpu_percent, 1)}%</strong></div><div class="meter-track"><div class="meter-fill" style="width:${Math.min(s.cpu_percent || 0, 100)}%"></div></div></div><div class="resource-item"><div class="resource-label"><span>Host memory</span><strong>${bytes(s.memory_used_bytes)} <span>/ ${bytes(s.memory_total_bytes)}</span></strong></div><div class="meter-track"><div class="meter-fill" style="width:${Math.min(s.memory_percent || 0, 100)}%"></div></div></div><div class="resource-item"><div class="resource-label"><span>GPU utilization</span><strong>${gpu == null ? "Not available" : `${number(gpu, 1)}%`}</strong></div><div class="meter-track"><div class="meter-fill green" style="width:${Math.min(gpu || 0, 100)}%"></div></div></div><p class="resource-note">${esc(s.gpu_name || s.gpu_reason || "GPU telemetry has not been sampled.")}<br>${s.sampled_at ? `Last sampled ${time(s.sampled_at)}` : "Sampling host telemetry…"}</p></div></section>`;
}
function readiness() {
  const installed = Object.values(state.training_packages).every(Boolean),
    gpu = state.system.gpu_name;
  const rows = [
    [
      "Backend service",
      "Independent of the desktop window",
      connected,
      "CONNECTED",
    ],
    [
      "Training stack",
      installed
        ? "Locked packages are installed"
        : "Install the training extra",
      installed,
      installed ? "INSTALLED" : "MISSING",
    ],
    [
      "Accelerator telemetry",
      gpu || "CUDA must be verified on the training host",
      !!gpu,
      gpu ? "MEASURED" : "UNVERIFIED",
    ],
  ];
  return `<section class="panel"><div class="panel-header"><div><h2>Workspace readiness</h2><p>A real stack. Clear boundaries.</p></div>${icon("shield")}</div><div class="readiness-list">${rows.map(([name, description, ready, tag]) => `<div class="readiness-row"><span class="readiness-icon ${ready ? "" : "unknown"}">${icon(ready ? "check" : "info")}</span><div><strong>${name}</strong><small>${esc(description)}</small></div><span class="ready-tag" ${ready ? "" : 'style="color:var(--faint)"'}>${tag}</span></div>`).join("")}</div></section>`;
}
function recentActivity() {
  const activity = events
    .filter(
      (event) =>
        !["worker_output", "training_metrics", "action_proposed"].includes(
          event.kind,
        ),
    )
    .slice(-4)
    .reverse();
  return `<section class="panel"><div class="panel-header"><div><h2>Workspace activity</h2><p>Every decision leaves a trace</p></div><a class="text-button" href="#monitoring">All logs ${icon("arrow")}</a></div><div class="activity-list">${activity.length ? activity.map((event) => `<div class="activity-row"><span class="activity-dot ${event.level}"></span><div class="activity-copy"><strong>${esc(event.message)}</strong><small>${esc(event.kind.replaceAll("_", " "))}</small></div><time class="activity-time">${time(event.ts)}</time></div>`).join("") : '<p class="subtle">No events received yet.</p>'}</div></section>`;
}
function advisoryPanel() {
  return `<section class="panel"><div class="panel-header"><div><h2>Workspace insights</h2><p>Recommendations, not automatic changes</p></div><span class="pill muted-pill">ADVISORY ONLY</span></div><div class="panel-body">${state.next_experiment ? `<div class="recommendation"><strong>Next experiment: ${esc(state.next_experiment.title)}</strong><p>${esc(state.next_experiment.rationale)}</p><p>${esc(state.next_experiment.action)}</p><small>Evidence: ${esc(state.next_experiment.evidence)} · Requires your explicit approval; nothing runs automatically.</small></div>` : ""}${state.recommendations.length ? state.recommendations.map((item) => `<div class="recommendation"><strong>${esc(item.title)}</strong><p>${esc(item.rationale)}</p><small>Expected impact: ${esc(item.impact)}</small></div>`).join("") : '<p class="subtle">No health warnings in the latest sample.</p>'}</div></section>`;
}
function dashboard() {
  const run = state.active_run || state.runs[0],
    m = runMetrics(run);
  const system = state.system || {};
  return (
    heading(
      "Training, under control.",
      "One workspace for experiments, policies, and the signals that matter.",
      button("report", "Export report", "download", "ghost", !connected) +
        button(
          "start",
          "New training run",
          "plus",
          "primary",
          !writable() || !idle(),
        ),
    ) +
    runError(run) +
    metricsGrid(run) +
    `<div class="two-col"><section class="panel active-run"><div class="panel-header"><h2>Active experiment</h2>${statusPill(state.active_run?.status)}</div><div class="run-stage"><div class="run-copy"><h2>${state.active_run ? esc(run.name) : "Your next run<br>starts here."}</h2><p>${state.active_run ? "Real 1v1 simulation, one shared policy, and an explicit plan. Your run continues independently of this window." : "A focused foundation for a self-trained agent. Start small, observe clearly, and build on verified results."}</p><a class="button secondary" href="#training">${icon(state.active_run ? "monitoring" : "training")}${state.active_run ? "View training run" : "Open training controls"}${icon("arrow")}</a></div>${fieldArt()}</div><div class="run-footer"><span>${icon("cpu")}<b>HEADLESS</b></span><span>1v1 Soccar <b>·</b> Tick skip 8</span><span>${icon("lock")}User-initiated only</span></div></section>${readiness()}</div><div class="two-col">${chart(run)}${resources(system)}</div><div class="two-col balanced">${recentActivity()}<section class="panel"><div class="panel-header"><div><h2>Policy health</h2><p>Learning is not the same as winning</p></div>${icon("trophy")}</div><div class="metric-pairs"><div class="metric-pair"><span>Policy entropy</span><strong>${number(m.entropy, 3)}</strong></div><div class="metric-pair"><span>Evaluation win rate</span><strong>${(state.evaluation_comparison || [])[0] ? `${number(state.evaluation_comparison[0].win_rate * 100, 0)}% vs ${esc(state.evaluation_comparison[0].opponent_id)}` : "Not evaluated"}</strong></div></div><div class="panel-body"><div class="system-note">${icon("info")}<span>${(state.evaluation_comparison || [])[0] ? `Best measured checkpoint by lower 95% bound over ${state.evaluation_comparison[0].games} games. See Models for intervals and opponents.` : "No win rate is inferred from self-play reward. Evaluate a checkpoint in Models against a frozen scripted baseline."}</span></div></div></section></div>`
  );
}
function training() {
  const run = viewedRun(),
    m = runMetrics(run),
    live = run?.id === state.active_run?.id;
  const progress = run
    ? Math.min(((m.steps || 0) / run.config.total_timesteps) * 100, 100)
    : 0;
  const estimatedRemaining = run?.metrics?.steps_per_second
    ? number(Math.max(0, (run.config.total_timesteps - (m.steps || 0)) / run.metrics.steps_per_second), 1)
    : "—";
  let body = run
    ? `<div class="panel-header"><div><h2>Training lifecycle</h2><p>Update-boundary controls · atomic checkpoints</p></div>${statusPill(run.status)}</div><div class="panel-body"><div class="run-title"><div><h2>${esc(run.name)}</h2><span class="run-id">RUN / ${short(run.id, 12)}</span></div><span class="pill muted-pill">${esc(run.config.device)}</span></div><div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:0">${stat("PPO iterations", number(m.iterations), "", "Completed update boundaries", "refresh")}${stat("Optimizer updates", number(m.updates), "", "Upstream learner counter", "chart")}${stat("Episodes", number(m.episodes), "", "Completed, not estimated", "training")}${stat("Step throughput", number(m.steps_per_second), "steps/s", "Collection + PPO update timing", "clock")}</div></div><div class="progress-wrapper"><div class="progress-label"><span>Step budget</span><strong>${number(m.steps || 0)} / ${number(run.config.total_timesteps)} <span class="muted">(${number(progress, 1)}%)</span></strong></div><div class="progress-track" role="progressbar" aria-label="Training step budget" aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100"><div class="progress-fill" style="width:${progress}%"></div></div></div>${run.status === "running" && estimatedRemaining !== "—" ? `<div class="estimated-remaining"><span>Est. remaining: ${estimatedRemaining}s</span></div>` : ""}<div class="control-buttons">${button("pause", "Pause", "pause", "secondary", !writable() || !live || run.status !== "running", `data-run-id="${run.id}"`)}${button("resume", "Resume", "play", "secondary", !writable() || !live || run.status !== "paused", `data-run-id="${run.id}"`)}${button("stop", "Stop run", "stop", "danger", !writable() || !live || !activeStatuses.has(run.status) || run.status === "stopping", `data-run-id="${run.id}"`)}<span class="control-note">${run.status === "paused" ? "Worker is alive; simulation and updates are paused." : ["pausing", "stopping"].includes(run.status) ? "Finishing the current update before acknowledging." : "Closing the GUI does not stop your run."}</span></div>`
    : empty(
        "training",
        "Ready when you are.",
        "Approve a small, explicit training plan. The backend will initialize the real headless environment only after you confirm.",
        button(
          "start",
          "Review training run",
          "play",
          "primary",
          !writable() || !idle(),
        ),
      );
  return (
    heading(
      "Training",
      "A deliberate lifecycle. From first rollout to a recoverable checkpoint.",
      button(
        "start",
        "New training run",
        "plus",
        "primary",
        !writable() || !idle(),
      ),
    ) +
    runError(run) +
    `<div class="two-col"><section class="panel">${body}</section><section class="panel"><div class="panel-header"><div><h2>${run ? "Experiment metadata" : "Approved plan"}</h2><p>Immutable for every started run</p></div>${icon("file")}</div><div class="metadata-list">${metadataRow("Device", (run?.config || state.plan).device)}${metadataRow("Environment", "1v1 · kickoff-v1")}${metadataRow("Observation / actions", "92 float32 / 90 discrete")}${metadataRow("Seed", (run?.config || state.plan).seed)}${metadataRow("Workers", (run?.config || state.plan).n_proc)}${metadataRow("Source commit", run?.metadata.git.commit ? short(run.metadata.git.commit, 10) + (run.metadata.git.dirty ? " (dirty)" : "") : "Recorded when run starts")}${metadataRow("Config hash", run ? short(run.config_hash, 16) : "Locked at run creation")}${metadataRow("Recovery", "Learner restored · episodes reset")}</div><div class="panel-body"><a class="text-button" href="#configuration">Review approved configuration ${icon("arrow")}</a></div></section></div><section class="panel"><div class="panel-header"><div><h2>Experiment history</h2><p>Original records stay intact when you recover a run</p></div><span class="pill muted-pill">${state.runs.length} RUNS</span></div>${state.runs.length ? `<div class="table-wrap"><table><thead><tr><th>EXPERIMENT</th><th>CREATED</th><th>STEPS</th><th>STATUS</th><th></th></tr></thead><tbody>${state.runs.map((item) => `<tr><td><strong>${esc(item.name)}</strong><small>${short(item.id, 12)}${item.parent ? " · recovered lineage" : ""}</small></td><td>${date(item.created)}</td><td>${number(item.metrics.steps || 0)}</td><td>${statusPill(item.status)}</td><td>${button("select-run", "View run", "arrow", "ghost", false, `data-run-id="${item.id}"`)}</td></tr>`).join("")}</tbody></table></div>` : empty("file", "A clean experiment history", "No training has been started. Opening this application never launches a run.")}</section>`
  );
}
function monitoring() {
  const health = state.worker_health;
  const filtered = events
    .filter(
      (event) =>
        (logLevel === "all" || event.level === logLevel) &&
        (!logSearch ||
          `${event.message} ${event.kind}`
            .toLowerCase()
            .includes(logSearch.toLowerCase())),
    )
    .slice(-250);
  return (
    heading(
      "Monitoring",
      "The live signal behind every run. Logs, resources, and backend health.",
      button("report", "Export report", "download", "secondary", !connected),
    ) +
    `<div class="stat-grid">${stat("Backend health", connected ? "Connected" : "Disconnected", "", "Independent control-plane process", "shield")}${stat("Worker heartbeat", health.heartbeat_age_seconds == null ? "—" : number(health.heartbeat_age_seconds, 1), health.heartbeat_age_seconds == null ? "" : "s", health.connected ? "Time since last worker message" : "No active training worker", "monitoring")}${stat("Visible CPU threads", number(state.system.cpu_count), "", "Actual backend host allocation", "cpu")}${stat("GPU peak allocation", bytes(runMetrics(state.active_run).gpu_peak_bytes), "", "PyTorch measurement, when available", "bolt")}</div><section class="panel"><div class="panel-header"><div><h2>Live event stream</h2><p>Structured, durable, and resumed by sequence cursor</p></div><span class="live-indicator"><i></i>${connected ? "LIVE" : "RECONNECTING"}</span></div><div class="log-toolbar"><div class="log-filters">${["all", "warning", "error"].map((level) => `<button data-log-level="${level}" class="${logLevel === level ? "selected" : ""}">${level === "all" ? "All events" : level === "warning" ? "Warnings" : "Errors"}</button>`).join("")}</div><input class="log-search" id="log-search" aria-label="Filter logs" placeholder="Filter event messages…" value="${esc(logSearch)}"></div><div class="log-stream" id="log-stream" aria-label="Live logs">${filtered.length ? filtered.map((event) => `<div class="log-line"><time class="log-time">${time(event.ts)}</time><span class="log-level ${esc(event.level)}">${esc(event.level)}</span><span class="log-message">${esc(event.message)}</span></div>`).join("") : '<span class="muted">No matching events.</span>'}</div><div class="log-footer"><span>Cursor ${cursor} · Showing ${filtered.length} events</span><span>No messages are synthesized</span></div></section><div class="two-col" style="margin-top:20px">${resources()}${advisoryPanel()}</div>`
  );
}
function latestEvaluation(checkpointId) {
  return (state.evaluations || []).find(
    (item) => item.status === "completed" && item.result?.candidate_checkpoint_id === checkpointId,
  );
}
function evaluationSection(valid) {
  const evaluations = state.evaluations || [];
  const running = evaluations.some((item) => item.status === "running");
  const options = valid
    .map((item) => `<option value="${item.run_id}|${item.id}">${esc(item.id.slice(0, 24))} · ${esc(item.run_name)} · step ${number(item.step)}</option>`)
    .join("");
  const rows = evaluations
    .map((item) => {
      const r = item.result;
      return `<tr><td>${r ? esc(r.candidate_checkpoint_id.slice(0, 24)) : esc(item.spec.checkpoint_id.slice(0, 24))}<small>${item.created ? date(item.created) : ""}</small></td><td>${esc(item.spec.opponent_id)}</td><td>${esc(item.status)}${item.status === "running" ? ` · ${number((item.progress || 0) * 100, 0)}% ${button("stop-evaluation", "Stop", "stop", "ghost", !writable(), `data-evaluation-id="${item.id}"`)}` : ""}</td><td>${r ? `${r.wins}W · ${r.losses}L · ${r.draws}D` : "—"}</td><td>${r ? `${r.goals_for} – ${r.goals_against}` : "—"}</td><td>${r ? `${number(r.win_rate * 100, 0)}% [${number(r.win_rate_ci95[0] * 100, 0)}–${number(r.win_rate_ci95[1] * 100, 0)}]` : "—"}</td><td>${item.trajectory ? `<a class="text-button" href="#visualization" data-action="open-replay" data-evaluation-id="${item.id}">${icon("visualization")}Replay</a>` : ""}${item.error ? esc(item.error) : ""}</td></tr>`;
    })
    .join("");
  const comparison = (state.evaluation_comparison || [])
    .map((entry, index) => `<div class="recommendation"><strong>#${index + 1} · ${esc(entry.candidate_checkpoint_id.slice(0, 24))}</strong><p>vs ${esc(entry.opponent_id)}: win rate ${number(entry.win_rate * 100, 0)}% over ${entry.games} games; ranked by lower 95% bound (${number(entry.score_lower_bound * 100, 0)}%). Goals ${entry.goals_for}–${entry.goals_against}.</p></div>`)
    .join("");
  return `<section class="panel" style="margin-top:20px"><div class="panel-header"><div><h2>Checkpoint evaluation</h2><p>Frozen weights vs scripted baselines · side-balanced · seeded</p></div><span class="pill muted-pill">${running ? "EVALUATING" : "REPRODUCIBLE"}</span></div><div class="panel-body"><div class="form-grid"><label class="field">Checkpoint<select id="eval-checkpoint" ${valid.length ? "" : "disabled"}>${options || '<option value="">No valid checkpoint</option>'}</select></label><label class="field">Opponent<select id="eval-opponent"><option value="random-v1" ${evaluationForm.opponent_id === "random-v1" ? "selected" : ""}>Random actions (seeded)</option><option value="idle-v1" ${evaluationForm.opponent_id === "idle-v1" ? "selected" : ""}>Idle (no input)</option><option value="checkpoint">Frozen checkpoint (select below)</option></select></label><label class="field">Opponent checkpoint (only for checkpoint opponent)<select id="eval-opponent-checkpoint">${options || '<option value="">No valid checkpoint</option>'}</select></label><label class="field">Games (even, both sides)<input id="eval-games" type="number" min="2" max="400" step="2" value="${evaluationForm.games}"></label><label class="field">Seed<input id="eval-seed" type="number" min="0" value="${evaluationForm.seed}"></label><label class="field">Parallel game workers<input id="eval-workers" type="number" min="1" max="8" value="${evaluationForm.parallel_workers}"><small>Same seeds/sides; only scheduling changes.</small></label></div><div class="config-actions">${button("start-evaluation", "Start evaluation", "trophy", "primary", !writable() || !idle() || !valid.length || running)}</div><p class="subtle">Evaluation runs in a separate process with inference-only frozen weights; it never trains and never changes the designated best. Win rates include 95% Wilson intervals — small samples stay visibly uncertain.</p></div>${rows ? `<div class="table-wrap"><table><thead><tr><th>CHECKPOINT</th><th>OPPONENT</th><th>STATUS</th><th>RECORD</th><th>GOALS</th><th>WIN RATE · 95% CI</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}${comparison ? `<div class="panel-body"><h3>Comparison (by lower confidence bound)</h3>${comparison}</div>` : ""}</section>`;
}
function models() {
  const best = state.designated_best;
  const valid = state.checkpoints.filter((item) => item.integrity === "valid");
  return (
    heading(
      "Models",
      "Recovery-ready checkpoints. Preserved by default, never automatically pruned.",
      button("report", "Export inventory", "download", "secondary", !connected),
    ) +
    `<div class="stat-grid">${stat("Saved checkpoints", number(state.checkpoints.length), "", "Published, not pending writes", "models")}${stat("Integrity verified", number(valid.length), "", "SHA-256 file manifests", "shield")}${stat("Designated best", best ? "Manually set" : "Not set", "", "Explicit user designation only", "trophy")}${stat("Evaluation results", number((state.evaluations || []).filter((item) => item.status === "completed").length), "", "Frozen-opponent evaluations only", "chart")}</div><section class="panel"><div class="panel-header"><div><h2>Checkpoint inventory</h2><p>Click a checkpoint to inspect its real metadata and recovery contract</p></div><span class="pill muted-pill">LOCAL STORAGE</span></div>${
      state.checkpoints.length
        ? `<div class="table-wrap"><table><thead><tr><th>CHECKPOINT</th><th>AGENT STEPS</th><th>INTEGRITY</th><th>EVALUATION</th><th>ACTIONS</th></tr></thead><tbody>${state.checkpoints
            .map((item) => {
              const target = `data-run-id="${item.run_id}" data-checkpoint-id="${item.id}"`;
              const designated =
                best?.run_id === item.run_id && best?.checkpoint_id === item.id;
              const run = state.runs.find((run) => run.id === item.run_id);
              const canRecover =
                run &&
                ["failed", "interrupted", "stopped"].includes(run.status) &&
                item.step < run.config.total_timesteps;
              const last =
                state.latest_valid[item.run_id] === item.id &&
                valid.filter((value) => value.run_id === item.run_id).length <=
                  1;
              return `<tr><td><button class="text-button" data-action="inspect-model" ${target}>${icon("file")}${esc(item.id.slice(0, 17))}</button><small>${esc(item.run_name)} · ${item.created ? date(item.created) : "Unknown date"}</small>${designated ? '<span class="model-designation">DESIGNATED BEST · MANUAL</span>' : ""}</td><td>${number(item.step)}</td><td><span class="${item.integrity === "valid" ? "model-check" : "bad-text"}">${icon(item.integrity === "valid" ? "check" : "info")}${esc(item.integrity)}</span></td><td>${latestEvaluation(item.id) ? `<small style="margin:0">${number(latestEvaluation(item.id).result.win_rate * 100, 0)}% vs ${esc(latestEvaluation(item.id).result.opponent_id)} (n=${latestEvaluation(item.id).result.games})</small>` : '<small style="margin:0">Not evaluated</small>'}</td><td class="actions">${button("recover-checkpoint", "Recover", "refresh", "secondary", !writable() || !idle() || item.integrity !== "valid" || !canRecover, target)}${button("designate", "Designate", "trophy", "ghost", !writable() || !idle() || item.integrity !== "valid" || designated, target)}${button("delete-checkpoint", "Delete", "delete", "ghost", !writable() || !idle() || item.integrity !== "valid" || designated || last, target)}</td></tr>`;
            })
            .join("")}</tbody></table></div>`
        : empty(
            "models",
            "Your first checkpoint is ahead.",
            "A checkpoint is published at initialization, at the approved cadence, and on pause or stop. Incomplete writes are never offered for recovery.",
            `<a class="button secondary" href="#training">${icon("training")}Open training</a>`,
          )
    }</section><div class="two-col balanced" style="margin-top:20px"><section class="panel"><div class="panel-header"><div><h2>Recommended candidates</h2><p>Evidence before promotion</p></div><span class="pill muted-pill">EVALUATION-BACKED</span></div><div class="panel-body">${(state.evaluation_comparison || []).length ? `<div class="system-note">${icon("info")}<span>Ranked by lower 95% Wilson bound in the evaluation table below. Designation remains manual and requires your confirmation.</span></div>` : `<div class="system-note">${icon("info")}<span>No candidates are recommended without measured, frozen-opponent evaluation results. A manual designation is not proof of strength.</span></div>`}</div></section><section class="panel"><div class="panel-header"><div><h2>Recovery contract</h2><p>Clear about what is restored</p></div>${icon("shield")}</div><div class="panel-body"><p class="resource-note">Actor, critic, optimizers, experience buffer, GAE statistics, RNG state, and counters are restored at a completed update. New simulator episodes start after recovery. The original run and checkpoints remain untouched.</p></div></section></div>` +
    evaluationSection(valid) + compareSection(valid)
  );
}
let compareSelection = { a: "", b: "" };
function compareSection(valid) {
  const options = valid.map((item) => `<option value="${item.run_id}|${item.id}">${esc(item.id.slice(0, 24))} · step ${number(item.step)}</option>`).join("");
  let panel = "";
  const a = state.checkpoints.find((item) => `${item.run_id}|${item.id}` === compareSelection.a);
  const b = state.checkpoints.find((item) => `${item.run_id}|${item.id}` === compareSelection.b);
  if (a && b) {
    const runOf = (item) => state.runs.find((run) => run.id === item.run_id);
    const fields = ["device", "n_proc", "batch_size", "learning_rate", "n_epochs", "clip_range", "reward_version", "environment_version", "hidden_sizes", "seed"];
    const rows = fields.map((field) => {
      const va = JSON.stringify(runOf(a)?.config[field]);
      const vb = JSON.stringify(runOf(b)?.config[field]);
      return `<tr><td>${esc(field)}</td><td>${esc(va)}</td><td>${esc(vb)}</td><td>${va === vb ? "same" : "DIFFERS"}</td></tr>`;
    }).join("");
    const ea = latestEvaluation(a.id), eb = latestEvaluation(b.id);
    const verdict = ea && eb
      ? (ea.result.win_rate_ci95[0] > eb.result.win_rate_ci95[1] ? "A's interval is entirely above B's — measured advantage for A under its evaluation conditions."
        : eb.result.win_rate_ci95[0] > ea.result.win_rate_ci95[1] ? "B's interval is entirely above A's — measured advantage for B under its evaluation conditions."
        : "Confidence intervals overlap — no measured advantage. Evaluate both against the same opponent with more games.")
      : "At least one checkpoint is unevaluated; no strength comparison is possible.";
    panel = `<div class="table-wrap"><table><thead><tr><th>FIELD</th><th>A · ${esc(a.id.slice(0, 20))} (step ${number(a.step)})</th><th>B · ${esc(b.id.slice(0, 20))} (step ${number(b.step)})</th><th></th></tr></thead><tbody>${rows}<tr><td>evaluation</td><td>${ea ? `${number(ea.result.win_rate * 100, 0)}% vs ${esc(ea.result.opponent_id)} (n=${ea.result.games})` : "not evaluated"}</td><td>${eb ? `${number(eb.result.win_rate * 100, 0)}% vs ${esc(eb.result.opponent_id)} (n=${eb.result.games})` : "not evaluated"}</td><td></td></tr></tbody></table></div><div class="panel-body"><p class="subtle">${esc(verdict)}</p></div>`;
  }
  return `<section class="panel" style="margin-top:20px"><div class="panel-header"><div><h2>Compare checkpoints</h2><p>Configuration differences and measured evaluation — never age or reward alone</p></div>${icon("chart")}</div><div class="panel-body"><div class="form-grid"><label class="field">Checkpoint A<select id="compare-a">${options}</select></label><label class="field">Checkpoint B<select id="compare-b">${options}</select></label></div><div class="config-actions">${button("compare-models", "Compare", "chart", "secondary", !valid.length)}</div></div>${panel}</section>`;
}
function configInput(
  label,
  key,
  type = "number",
  description = "",
  options = null,
) {
  const value = configDraft[key];
  const input = options
    ? `<select id="cfg-${key}" data-config="${key}">${options.map(([v, label]) => `<option value="${v}" ${value === v ? "selected" : ""}>${label}</option>`).join("")}</select>`
    : `<input id="cfg-${key}" data-config="${key}" type="${type}" value="${esc(value)}" ${type === "number" ? 'step="any"' : ""}>`;
  return `<label class="field">${label}${input}${description ? `<small>${description}</small>` : ""}</label>`;
}
function configuration() {
  if (!configDraft) {
    configDraft = structuredClone(state.plan);
    configRaw = JSON.stringify(configDraft, null, 2);
  }
  return (
    heading(
      "Configuration",
      "An explicit plan, validated before it reaches the learner. No hot strategy changes.",
      `<span class="plan-badge"><i></i>Approved plan · revision <span id="plan-revision">${state.plan_revision}</span></span>`,
    ) +
    `<div class="config-layout"><section class="panel"><div class="panel-header"><div><h2>Experiment & training parameters</h2><p>Draft changes do not affect existing runs</p></div>${icon("configuration")}</div><div class="form-grid">${configInput("Experiment name", "name", "text")}${configInput(
      "Learner device",
      "device",
      "text",
      "CUDA is never silently replaced with CPU.",
      [
        ["cuda:0", "CUDA · device 0"],
        ["cpu", "CPU · bounded correctness test"],
      ],
    )}${configInput("Total agent steps", "total_timesteps", "number", "Must be a multiple of iteration steps.")}${configInput("Steps per PPO iteration", "batch_size", "number", "Also sets collection size and 3× buffer capacity.")}${configInput("Environment processes", "n_proc", "number", "Fixed 1v1, kickoff-only, headless.")}${configInput("Minibatches", "n_minibatches", "number", "Must divide the batch; at least 16 samples each.")}${configInput("Learning rate", "learning_rate", "number")}${configInput("Entropy coefficient", "entropy_coefficient", "number")}${configInput("Random seed", "seed", "number")}${configInput("Checkpoint every N iterations", "checkpoint_every_iterations", "number", "Pause and stop also save at a safe boundary.")}</div><details class="advanced-config"><summary>Advanced · complete validated configuration</summary><textarea id="config-json" class="json-editor" aria-label="Complete training configuration" spellcheck="false">${esc(configRaw)}</textarea></details><div class="validation-message" id="validation-message" hidden role="status"></div><div class="config-actions">${button("reset-draft", "Reset draft", "refresh", "ghost", false)}${button("validate-config", "Validate draft", "check", "secondary", !writable())}${button("apply-config", "Review & apply", "arrow", "primary", !writable() || !idle(), 'id="apply-config"')}</div></section><div><section class="panel"><div class="panel-header"><div><h2>Guardrails, not guesswork.</h2><p>Changes require your approval</p></div>${icon("shield")}</div><div class="panel-body"><div class="system-note">${icon("lock")}<span>Every consequential action explains its rationale and expected impact before you confirm it. Configuration is immutable once a run starts.</span></div><div class="recommendation"><strong>Start with a small vertical slice</strong><p>One shared policy, 1v1 Soccar, 92 float32 observations, 90 actions, and one PPO epoch. No game client or match rendering.</p></div><div class="recommendation"><strong>Verify the actual target host</strong><p>Use <code>rocket-ai doctor --probe</code> in WSL 2. A CPU test verifies correctness, not RTX 4060 Ti throughput.</p></div></div></section><section class="panel" style="margin-top:20px"><div class="panel-header"><div><h2>Installed training stack</h2><p>Resolved packages, not assumptions</p></div>${icon("cpu")}</div><div class="panel-body">${Object.entries(
      state.training_packages,
    )
      .map(
        ([name, version]) =>
          `<div class="stack-row"><span>${esc(name)}</span><code>${esc(version || "not installed")}</code></div>`,
      )
      .join("")}</div></section></div></div>`
  );
}
function visualization() {
  const recorded = (state.evaluations || []).filter(
    (item) => item.status === "completed" && item.trajectory,
  );
  const options = recorded
    .map(
      (item) =>
        `<option value="${item.id}" ${replay.id === item.id ? "selected" : ""}>${esc(item.result.candidate_checkpoint_id.slice(0, 24))} vs ${esc(item.result.opponent_id)} · ${item.result.games} games</option>`,
    )
    .join("");
  return (
    heading(
      "Visualization",
      "Recorded evaluation replays, rendered locally. Training stays headless.",
      `<span class="pill muted-pill">RECORDED REPLAY · TOP-DOWN</span>`,
    ) +
    `<section class="panel"><div class="panel-header"><div><h2>Replay viewer</h2><p>Real RocketSim trajectories recorded during evaluation — a schematic top-down view, not a 3D renderer</p></div>${statusPill(replay.data ? "paused" : "idle")}</div><div class="panel-body"><div class="replay-controls"><label class="field">Recorded evaluation<select id="replay-select" ${recorded.length ? "" : "disabled"}>${options || '<option value="">No replay recorded yet</option>'}</select></label><button class="button secondary" data-action="load-replay" ${recorded.length && writable() ? "" : "disabled"}>${icon("play")}Load replay</button><button class="button secondary" data-action="replay-toggle" ${replay.data ? "" : "disabled"}>${icon(replay.playing ? "pause" : "play")}${replay.playing ? "Pause" : "Play"}</button><label class="field">Speed<select id="replay-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label><label class="approval-check"><input type="checkbox" id="replay-follow" ${replay.follow ? "checked" : ""}> Follow ball</label></div><canvas id="replay-canvas" width="920" height="620" aria-label="Top-down replay of a recorded evaluation game"></canvas><input id="replay-scrub" type="range" min="0" max="${replay.data ? replay.data.frames.length - 1 : 0}" value="${replay.frame}" ${replay.data ? "" : "disabled"} aria-label="Scrub replay">${replay.data ? `<div class="replay-markers">${replayGoalMarkers()}</div>` : ""}<div class="replay-readout" id="replay-readout">${replay.data ? esc(replayReadout()) : "Run an evaluation from Models to record a replay. Opening this viewer never affects a training run."}</div></div><div class="viz-specs"><span>SOURCE<strong>Evaluation trajectory artifact</strong></span><span>FRAME RATE<strong>Decision rate · 15 Hz per car</strong></span><span>TRAINING MODE<strong>Headless · renderer = None</strong></span><span>BOUNDARY<strong>No live 3D renderer (NumPy/renderer conflict documented)</strong></span></div></section>`
  );
}
function replayReadout() {
  if (!replay.data) return "";
  const frame = replay.data.frames[replay.frame];
  if (!frame) return "";
  const ballSpeed = frame.ball_velocity
    ? Math.hypot(...frame.ball_velocity).toFixed(1)
    : "—";
  return `Game ${frame.game + 1} · t=${frame.t.toFixed(2)}s · Score blue ${frame.score[0]} – orange ${frame.score[1]} · Ball speed ${ballSpeed} uu/s · Frame ${replay.frame + 1}/${replay.data.frames.length} · Candidate: ${esc(replay.data.candidate_side_by_game[frame.game] || "?")}`;
}
function drawReplayFrame() {
  const canvas = document.getElementById("replay-canvas");
  if (!canvas || !replay.data) return;
  const frame = replay.data.frames[replay.frame];
  if (!frame) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  // Soccar field: x in [-4096, 4096], y in [-5120, 5120]; map y to canvas x.
  let centreX = 0, centreY = 0;
  if (replay.follow) { centreX = frame.ball[1]; centreY = frame.ball[0]; }
  const scale = replay.follow ? 0.11 : Math.min(W / 10600, H / 8800);
  const px = (wx, wy) => [W / 2 + (wy - centreX) * scale, H / 2 - (wx - centreY) * scale];
  ctx.fillStyle = "#0b3d2e";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 2;
  const [fx0, fy0] = px(-5120, -4096), [fx1, fy1] = px(5120, 4096);
  ctx.strokeRect(fx0, fy0, fx1 - fx0, fy1 - fy0);
  ctx.beginPath();
  const [cx, cy] = px(0, 0);
  ctx.moveTo(cx, fy0); ctx.lineTo(cx, fy1); ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 500 * scale, 0, Math.PI * 2); ctx.stroke();
  for (const goalY of [-5120, 5120]) {
    const [gx0, gy0] = px(goalY - 0, -893), [gx1, gy1] = px(goalY, 893);
    ctx.strokeRect(Math.min(gx0, gx1), Math.min(gy0, gy1), Math.abs(gx1 - gx0) || 4, Math.abs(gy1 - gy0));
  }
  for (const car of frame.cars) {
    const [x, y] = px(car.position[1], car.position[0]);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-(car.yaw || 0));
    ctx.fillStyle = car.team === "blue" ? "#3b82f6" : "#f97316";
    ctx.fillRect(-14, -8, 28, 16);
    ctx.fillStyle = "#fff";
    ctx.fillRect(8, -5, 6, 10);
    ctx.restore();
  }
  const [bx, by] = px(frame.ball[1], frame.ball[0]);
  ctx.beginPath();
  ctx.arc(bx, by, 7, 0, Math.PI * 2);
  ctx.fillStyle = "#f8fafc";
  ctx.fill();
  ctx.strokeStyle = "#111827";
  ctx.stroke();
  const readout = document.getElementById("replay-readout");
  if (readout) readout.textContent = replayReadout();
  const scrub = document.getElementById("replay-scrub");
  if (scrub && document.activeElement !== scrub) scrub.value = replay.frame;
}
function replayGoalMarkers() {
  if (!replay.data) return "";
  const markers = [];
  let previous = [0, 0];
  replay.data.frames.forEach((frame, index) => {
    if (frame.score[0] !== previous[0] || frame.score[1] !== previous[1]) {
      markers.push(`<button class="marker" data-action="replay-jump" data-frame="${index}" title="Goal at frame ${index + 1}">Goal ${frame.score[0]}-${frame.score[1]}</button>`);
      previous = frame.score;
    }
  });
  return markers.length ? markers.join("") : '<span class="subtle">No goals in the recorded games.</span>';
}
function stopReplayTimer() {
  if (replay.timer) { clearInterval(replay.timer); replay.timer = null; }
  replay.playing = false;
}


function benchmark() {
  const running = state.benchmark?.running;
  const check = (list, value) => (list.includes(value) ? "checked" : "");
  return (
    heading(
      "Benchmark",
      "Real bounded training runs that find a fast, stable configuration for this hardware.",
      running
        ? button("stop-benchmark", "Stop benchmark", "stop", "secondary", !writable())
        : button("start-benchmark", "Start benchmark", "play", "primary", !writable() || !idle()),
    ) +
    `<section class="panel"><div class="panel-header"><div><h2>Benchmark configuration</h2><p>Each candidate is measured with genuine warm-up + measured PPO iterations; nothing is simulated</p></div>${icon("cpu")}</div><div class="panel-body"><div class="form-grid"><fieldset class="field"><legend>Learner devices</legend><label class="approval-check"><input type="checkbox" data-benchmark-list="devices" value="cpu" ${check(benchmarkForm.devices, "cpu")}> CPU simulation + CPU PPO</label><label class="approval-check"><input type="checkbox" data-benchmark-list="devices" value="cuda:0" ${check(benchmarkForm.devices, "cuda:0")}> CPU simulation + CUDA PPO</label></fieldset><fieldset class="field"><legend>Parallel environments</legend>${[1, 2, 4, 8].map((value) => `<label class="approval-check"><input type="checkbox" data-benchmark-list="n_proc_values" value="${value}" ${check(benchmarkForm.n_proc_values, value)}> ${value} env process${value > 1 ? "es" : ""}</label>`).join("")}</fieldset><label class="field">Batch / iteration agent steps<input type="number" data-benchmark="batch_size" value="${benchmarkForm.batch_size}" min="128" step="128"><small>Also the collection size, as in the training adapter.</small></label><label class="field">Warm-up iterations<input type="number" data-benchmark="warmup_iterations" value="${benchmarkForm.warmup_iterations}" min="0" max="10"><small>Excluded from all statistics.</small></label><label class="field">Measured iterations<input type="number" data-benchmark="measure_iterations" value="${benchmarkForm.measure_iterations}" min="1" max="20"></label><label class="field">Repeats per candidate<input type="number" data-benchmark="repeats" value="${benchmarkForm.repeats}" min="1" max="5"><small>Medians across repeats; instability excludes a candidate.</small></label><label class="field">CPU limit %<input type="number" data-benchmark="max_cpu_percent" value="${benchmarkForm.max_cpu_percent}" min="1" max="100"></label><label class="field">Memory limit %<input type="number" data-benchmark="max_memory_percent" value="${benchmarkForm.max_memory_percent}" min="1" max="100"></label></div><p class="subtle">A measurement run is capped at 16,384 agent steps and uses the approved plan's strategy fields. Benchmark runs appear in Training history. Adopting a recommendation always requires a separate reviewed configuration proposal — throughput is not learning quality.</p></div></section><div id="benchmark-live">${benchmarkLive()}</div><div class="two-col balanced" style="margin-top:20px"><section class="panel"><div class="panel-header"><div><h2>Stability before speed</h2><p>How a recommendation is earned</p></div>${icon("shield")}</div><div class="panel-body"><p class="subtle">A candidate must complete every repeat, keep throughput variation (CV) at or below 0.25, and stay within the resource limits. The fastest stable candidate wins; maximum utilization alone never does. Failed candidates keep their real error.</p></div></section><section class="panel"><div class="panel-header"><div><h2>Throughput ≠ learning</h2><p>Kept deliberately separate</p></div>${icon("info")}</div><div class="panel-body"><p class="subtle">Simulation steps/s, PPO steps/s and episodes/min are measured independently. Whether a configuration actually plays better is decided only by checkpoint evaluation in Models.</p></div></section></div>`
  );
}
function benchmarkLive() {
  const bench = state.benchmark || { session: null, results: [] };
  if (!bench.session)
    return `<section class="panel" style="margin-top:20px"><div class="panel-body">${empty("chart", "No benchmark has been run yet.", "Start a benchmark to measure genuine candidate configurations on this host. Results persist here and in the status report.")}</div></section>`;
  const session = bench.session;
  const rows = bench.results
    .map((item) => {
      const a = item.aggregate;
      return `<tr><td>${esc(item.candidate.device)} · ${item.candidate.n_proc} env</td><td>${esc(item.status)}${item.status === "running" ? ` · ${number((item.progress || 0) * 100, 0)}% ${button("stop-evaluation", "Stop", "stop", "ghost", !writable(), `data-evaluation-id="${item.id}"`)}` : ""}</td><td>${a ? number(a.agent_steps_per_second.median, 1) : "—"}</td><td>${a ? number(a.simulation_steps_per_second.median, 1) : "—"}</td><td>${a ? number(a.ppo_steps_per_second.median, 1) : "—"}</td><td>${a ? number(a.episodes_per_minute, 1) : "—"}</td><td>${a ? number(a.agent_steps_per_second.cv, 2) : "—"}</td><td>${item.resources ? `${number(item.resources.cpu_percent_peak, 0)}% CPU · ${number(item.resources.memory_percent_peak, 0)}% RAM` : "—"}</td><td>${item.error ? esc(item.error) : ""}</td></tr>`;
    })
    .join("");
  const rec = session.recommendation;
  return `<section class="panel" style="margin-top:20px"><div class="panel-header"><div><h2>Benchmark results</h2><p>Session ${esc(session.id.slice(0, 12))} · ${esc(session.status)} · ${date(session.created)}</p></div>${statusPill(session.status === "running" ? "running" : "idle")}</div>${rows ? `<div class="table-wrap"><table><thead><tr><th>CANDIDATE</th><th>STATUS</th><th>AGENT STEPS/S</th><th>SIM STEPS/S</th><th>PPO STEPS/S</th><th>EPISODES/MIN</th><th>CV</th><th>RESOURCE PEAKS</th><th>ERROR</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<div class="panel-body"><p class="subtle">Measurements are running. Each candidate executes real bounded training runs; results appear as candidates complete.</p></div>'}${rec ? `<div class="panel-body"><div class="recommendation"><strong>Recommendation</strong><p>${esc(rec.explanation)}</p>${rec.candidate ? button("apply-benchmark-recommendation", "Review as configuration proposal", "arrow", "primary", !writable() || !idle()) : ""}${(rec.excluded || []).map((item) => `<p class="subtle">Excluded ${esc(item.candidate.device)} · ${item.candidate.n_proc} env: ${esc(item.reasons.join("; "))}</p>`).join("")}</div></div>` : ""}${session.error ? `<div class="panel-body"><p class="bad-text">${esc(session.error)}</p></div>` : ""}</section>`;
}
const views = {
  dashboard,
  training,
  monitoring,
  models,
  configuration,
  visualization,
  benchmark,
};
function render(force = false) {
  navigation();
  connectionUI();
  if (!state) return;
  if (
    route === "configuration" &&
    $("#content").dataset.page === "configuration" &&
    !force
  ) {
    $("#plan-revision").textContent = state.plan_revision;
    $("#apply-config").disabled = !writable() || !idle();
    $('[data-action="validate-config"]').disabled = !writable();
    return;
  }
  if (route === "benchmark" && $("#content").dataset.page === "benchmark" && !force) {
    const live = $("#benchmark-live");
    if (live) live.innerHTML = benchmarkLive();
    return;
  }
  if (route === "visualization" && $("#content").dataset.page === "visualization" && !force) {
    drawReplayFrame();
    return;
  }
  const focus = document.activeElement?.id;
  const selection = focus && document.activeElement.selectionStart;
  const previousStream = $("#log-stream"),
    scroll = previousStream?.scrollTop;
  const pinned = previousStream
    ? previousStream.scrollHeight -
        previousStream.scrollTop -
        previousStream.clientHeight <
      30
    : true;
  $("#content").innerHTML = views[route]();
  $("#content").dataset.page = route;
  if (focus && document.getElementById(focus)) {
    const element = document.getElementById(focus);
    element.focus({ preventScroll: true });
    if (selection != null && element.setSelectionRange) {
      try {
        element.setSelectionRange(selection, selection);
      } catch {
        /* number input */
      }
    }
  }
  if ($("#log-stream"))
    $("#log-stream").scrollTop = pinned
      ? $("#log-stream").scrollHeight
      : scroll || 0;
  if (route === "visualization") drawReplayFrame();
}
async function refresh() {
  if (refreshPromise) return refreshPromise;
  refreshPromise = (async () => {
    try {
      if (!health || !connected) {
        health = await api("health");
        readOnly = health.preview_readonly;
      }
      if (!token && !readOnly) {
        if (!$("#connect-dialog").open) $("#connect-dialog").showModal();
        connectionUI();
        return;
      }
      const [snapshot, batch] = await Promise.all([
        api("snapshot"),
        api(`events?after=${cursor}&limit=200`),
      ]);
      state = snapshot;
      events.push(...batch.events);
      events = events.slice(-1200);
      cursor = batch.cursor;
      connected = true;
      lastConnectedAt = Date.now() / 1000;
      window.__RLAI_READY__ = true;
      if (token && $("#connect-dialog").open) $("#connect-dialog").close();
      render();
    } catch (error) {
      connected = false;
      window.__RLAI_READY__ = false;
      render();
      if (!state)
        $("#content").innerHTML = empty(
          "monitoring",
          "Waiting for the backend",
          error.name === "AbortError"
            ? "The service did not respond in time. Reconnecting automatically."
            : error.message,
        );
    } finally {
      refreshPromise = null;
    }
  })();
  return refreshPromise;
}
async function review(kind, payload = {}) {
  if (!authorized()) {
    toast(
      readOnly
        ? "This diagnostic preview is read-only. Open the native desktop to control training."
        : "Connect before changing the workspace.",
      true,
    );
    return;
  }
  if (pendingProposal) return;
  const proposal = await api("proposals", { kind, payload });
  pendingProposal = proposal;
  const titles = {
    start_run: "Start this training run?",
    apply_config: "Apply this training plan?",
    recover_run: "Recover from this checkpoint?",
    designate_checkpoint: "Change the designated best?",
    delete_checkpoint: "Permanently delete this checkpoint?",
  };
  $("#confirmation-icon").innerHTML = icon(
    {
      delete_checkpoint: "delete",
      recover_run: "refresh",
      designate_checkpoint: "trophy",
      apply_config: "configuration",
      start_run: "play",
    }[kind],
  );
  $("#confirmation-title").textContent = titles[kind];
  $("#proposal-rationale").textContent = proposal.rationale;
  $("#proposal-impact").textContent = proposal.impact;
  const config = proposal.payload.config;
  $("#proposal-target").textContent = config
    ? `${config.name}\nDevice: ${config.device} · Seed: ${config.seed} · Envs: ${config.n_proc}\nSteps: ${number(config.total_timesteps)} · Batch: ${number(config.batch_size)} · Network: ${config.hidden_sizes.join(" × ")}`
    : `Run: ${proposal.payload.run_id}\nCheckpoint: ${proposal.payload.checkpoint_id}`;

  $("#proposal-diff").innerHTML =
    proposal.payload.changes && Object.keys(proposal.payload.changes).length
      ? `<table class="diff-table"><thead><tr><th>FIELD</th><th>APPROVED</th><th>PROPOSED</th></tr></thead><tbody>${Object.entries(
          proposal.payload.changes,
        )
          .map(
            ([key, value]) =>
              `<tr><td>${esc(key)}</td><td>${esc(JSON.stringify(value.before))}</td><td>${esc(JSON.stringify(value.after))}</td></tr>`,
          )
          .join("")}</tbody></table>`
      : "";
  $("#approval-checkbox").checked = false;
  $("#confirm-action").disabled = true;
  $("#confirmation-dialog").showModal();
}
function draft() {
  try {
    return JSON.parse(configRaw);
  } catch (error) {
    throw new Error(`Configuration JSON is invalid: ${error.message}`);
  }
}
function validation(message, valid = true) {
  $("#validation-message").hidden = false;
  $("#validation-message").textContent = message;
  $("#validation-message").className =
    `validation-message${valid ? "" : " invalid"}`;
}
async function action(action, element) {
  if (action === "help") {
    $("#guide-dialog").showModal();
    return;
  }
  if (action === "close-guide") {
    $("#guide-dialog").close();
    return;
  }
  if (action === "connect") {
    token = $("#access-key").value.trim();
    sessionStorage.setItem("rlai-capability", token);
    $("#access-key").value = "";
    $("#connect-dialog").close();
    await refresh();
    return;
  }
  if (action === "cancel-proposal") {
    const proposal = pendingProposal;
    pendingProposal = null;
    $("#confirmation-dialog").close();
    if (proposal) await api(`proposals/${proposal.id}/reject`, {});
    return;
  }
  if (action === "confirm-proposal") {
    if (!pendingProposal || !$("#approval-checkbox").checked || !authorized())
      return;
    $("#confirm-action").disabled = true;
    const proposal = pendingProposal;
    const result = await api(`proposals/${proposal.id}/confirm`, {
      confirmed: true,
      expected_digest: proposal.digest,
    });
    pendingProposal = null;
    $("#confirmation-dialog").close();
    if (result.run_id && ["start_run", "recover_run"].includes(proposal.kind)) {
      selectedRun = result.run_id;
      location.hash = "training";
    }
    if (proposal.kind === "apply_config") {
      configDraft = null;
      configRaw = null;
      $("#content").dataset.page = "";
    }
    toast(
      proposal.kind === "delete_checkpoint"
        ? "Checkpoint deleted after your confirmation."
        : "Action confirmed. The workspace audit log has been updated.",
    );
    await refresh();
    return;
  }
  if (action === "select-run") {
    selectedRun = element.dataset.runId;
    render(true);
    return;
  }
  if (action === "reset-draft") {
    configDraft = null;
    configRaw = null;
    render(true);
    return;
  }
  if (action === "report") {
    const report = await api("report");
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = "rocket-ai-foundation-report.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(
      "Status report exported. It contains measured state, not an evaluation result.",
    );
    return;
  }
  if (action === "inspect-model") {
    const item = state.checkpoints.find(
      (item) =>
        item.id === element.dataset.checkpointId &&
        item.run_id === element.dataset.runId,
    );
    const run = state.runs.find((run) => run.id === item.run_id);
    const dialog = document.createElement("dialog");
    dialog.setAttribute("aria-label", "Checkpoint metadata");
    dialog.innerHTML = `<div class="dialog-top"><span class="eyebrow">CHECKPOINT METADATA</span><button class="icon-button" aria-label="Close metadata">×</button></div><h2>Saved, verified, recoverable.</h2><p class="subtle">No evaluation result is available.</p><pre class="json-editor" style="white-space:pre-wrap;max-height:450px;overflow:auto">${esc(JSON.stringify({ checkpoint: item, seed: run.config.seed, network: run.config.hidden_sizes, environment: run.config.environment_version, config_hash: run.config_hash, source: run.metadata.git }, null, 2))}</pre>`;
    document.body.append(dialog);
    dialog.querySelector("button").onclick = () => dialog.close();
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.showModal();
    return;
  }
  if (["pause", "resume", "stop"].includes(action)) {
    if (!authorized())
      throw new Error(
        "Connect to the authenticated backend before controlling a run.",
      );
    await api(`runs/${element.dataset.runId}/control`, { command: action });
    toast(
      `${action[0].toUpperCase() + action.slice(1)} requested. Waiting for a safe update-boundary acknowledgement.`,
    );
    await refresh();
    return;
  }
  if (action === "start") return review("start_run");
  if (action === "recover")
    return review("recover_run", {
      run_id: element.dataset.runId,
      checkpoint_id: state.latest_valid[element.dataset.runId],
    });
  if (action === "recover-checkpoint")
    return review("recover_run", {
      run_id: element.dataset.runId,
      checkpoint_id: element.dataset.checkpointId,
    });
  if (action === "designate")
    return review("designate_checkpoint", {
      run_id: element.dataset.runId,
      checkpoint_id: element.dataset.checkpointId,
    });
  if (action === "delete-checkpoint")
    return review("delete_checkpoint", {
      run_id: element.dataset.runId,
      checkpoint_id: element.dataset.checkpointId,
    });
  if (action === "start-benchmark") {
    const session = await api("benchmarks", benchmarkForm);
    toast(`Benchmark started (${session.id.slice(0, 12)}). Real bounded runs are measuring candidates.`);
    await refresh();
    return;
  }
  if (action === "stop-benchmark") {
    await api("benchmarks/stop", {});
    toast("Benchmark stop requested. The current measurement run is stopping safely.");
    await refresh();
    return;
  }
  if (action === "apply-benchmark-recommendation") {
    const candidate = state.benchmark?.session?.recommendation?.candidate;
    if (!candidate) throw new Error("No stable benchmark recommendation is available.");
    const config = structuredClone(state.plan);
    config.device = candidate.device;
    config.n_proc = candidate.n_proc;
    config.batch_size = state.benchmark.session.spec.batch_size;
    config.timesteps_per_iteration = config.batch_size;
    config.buffer_size = config.batch_size * 3;
    return review("apply_config", { config });
  }
  if (action === "start-evaluation") {
    const selected = $("#eval-checkpoint")?.value || "";
    const [run_id, checkpoint_id] = selected.split("|");
    if (!run_id || !checkpoint_id) throw new Error("Select a valid checkpoint first.");
    const opponentChoice = $("#eval-opponent").value;
    evaluationForm.opponent_id = opponentChoice === "checkpoint" ? "random-v1" : opponentChoice;
    evaluationForm.games = Number($("#eval-games").value);
    evaluationForm.seed = Number($("#eval-seed").value);
    evaluationForm.parallel_workers = Number($("#eval-workers").value);
    const payload = { run_id, checkpoint_id, ...evaluationForm };
    if (opponentChoice === "checkpoint") {
      const [opponent_run_id, opponent_checkpoint_id] = ($("#eval-opponent-checkpoint").value || "").split("|");
      if (!opponent_run_id || !opponent_checkpoint_id) throw new Error("Select an opponent checkpoint first.");
      payload.opponent_run_id = opponent_run_id;
      payload.opponent_checkpoint_id = opponent_checkpoint_id;
    }
    await api("evaluations", payload);
    toast("Evaluation started in a separate frozen-inference process.");
    await refresh();
    return;
  }
  if (action === "load-replay" || action === "open-replay") {
    const id = element.dataset.evaluationId || $("#replay-select")?.value;
    if (!id) throw new Error("Select a recorded evaluation first.");
    const data = await api(`evaluations/${id}/trajectory`);
    stopReplayTimer();
    replay = { id, data, frame: 0, playing: false, speed: Number($("#replay-speed")?.value || 1), follow: true, timer: null };
    location.hash = "visualization";
    render(true);
    return;
  }
  if (action === "compare-models") {
    compareSelection = { a: $("#compare-a").value, b: $("#compare-b").value };
    render(true);
    return;
  }
  if (action === "replay-jump") {
    replay.frame = Number(element.dataset.frame);
    drawReplayFrame();
    render(true);
    return;
  }
  if (action === "replay-toggle") {
    if (!replay.data) return;
    replay.playing = !replay.playing;
    if (replay.playing) {
      replay.timer = setInterval(() => {
        replay.frame += 1;
        if (replay.frame >= replay.data.frames.length) {
          replay.frame = replay.data.frames.length - 1;
          stopReplayTimer();
          render(true);
          return;
        }
        drawReplayFrame();
      }, 1000 / (15 * replay.speed));
    } else {
      stopReplayTimer();
    }
    render(true);
    return;
  }
  if (action === "stop-evaluation") {
    await api(`evaluations/${element.dataset.evaluationId}/stop`, {});
    toast("Evaluation stop requested. No partial result will be reported.");
    await refresh();
    return;
  }
  if (action === "validate-config") {
    try {
      const result = await api("config/validate", draft());
      validation(result.message + "\nConfig hash: " + result.config_hash);
    } catch (error) {
      validation(error.message, false);
    }
    return;
  }
  if (action === "apply-config")
    return review("apply_config", { config: draft() });
}
document.addEventListener("click", async (event) => {
  const chartButton = event.target.closest("[data-chart]");
  if (chartButton) {
    chartKind = chartButton.dataset.chart;
    render(true);
    return;
  }
  const levelButton = event.target.closest("[data-log-level]");
  if (levelButton) {
    logLevel = levelButton.dataset.logLevel;
    render(true);
    return;
  }
  const element = event.target.closest("[data-action]");
  if (!element || element.disabled || busy) return;
  const value = element.dataset.action;
  // Keep the review checkbox independent of polling and never auto-approve a proposal.
  busy = true;
  render();
  try {
    await action(value, element);
  } catch (error) {
    toast(error.message, true);
  } finally {
    busy = false;
    render();
  }
});
document.addEventListener("input", (event) => {
  if (event.target.id === "approval-checkbox") {
    $("#confirm-action").disabled = !event.target.checked || !writable();
    return;
  }
  if (event.target.id === "log-search") {
    logSearch = event.target.value;
    render(true);
    return;
  }
  if (event.target.id === "replay-scrub") {
    replay.frame = Number(event.target.value);
    drawReplayFrame();
    return;
  }
  if (event.target.id === "replay-speed") {
    replay.speed = Number(event.target.value);
    return;
  }
  if (event.target.id === "replay-follow") {
    replay.follow = event.target.checked;
    drawReplayFrame();
    return;
  }
  if (event.target.dataset.benchmarkList) {
    const key = event.target.dataset.benchmarkList;
    const value = key === "n_proc_values" ? Number(event.target.value) : event.target.value;
    const list = new Set(benchmarkForm[key]);
    if (event.target.checked) list.add(value); else list.delete(value);
    benchmarkForm[key] = [...list];
    return;
  }
  if (event.target.dataset.benchmark) {
    benchmarkForm[event.target.dataset.benchmark] = Number(event.target.value);
    return;
  }
  if (event.target.id === "config-json") {
    configRaw = event.target.value;
    $("#validation-message").hidden = true;
    try {
      const parsed = JSON.parse(configRaw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        configDraft = parsed;
        document.querySelectorAll("[data-config]").forEach((input) => {
          input.value = parsed[input.dataset.config] ?? "";
        });
      }
    } catch {
      /* Keep invalid text intact until the user explicitly validates it. */
    }
    return;
  }
  const key = event.target.dataset.config;
  if (key) {
    const value =
      event.target.type === "number"
        ? Number(event.target.value)
        : event.target.value;
    configDraft = { ...configDraft, [key]: value };
    if (key === "batch_size") {
      configDraft.timesteps_per_iteration = value;
      configDraft.buffer_size = value * 3;
    }
    configRaw = JSON.stringify(configDraft, null, 2);
    $("#config-json").value = configRaw;
    $("#validation-message").hidden = true;
  }
});
$("#confirmation-dialog").addEventListener("cancel", (event) => {
  event.preventDefault();
  action("cancel-proposal").catch((error) => toast(error.message, true));
});
$("#connect-dialog").addEventListener("cancel", (event) =>
  event.preventDefault(),
);
window.addEventListener("hashchange", () => {
  const changedKey = consumeCapabilityFragment();
  route = names[location.hash.slice(1)] ? location.hash.slice(1) : "dashboard";
  if (changedKey) {
    connected = false;
    refresh();
  }
  render(true);
});
route = names[location.hash.slice(1)] ? location.hash.slice(1) : "dashboard";
navigation();
refresh();
setInterval(refresh, 1000);
