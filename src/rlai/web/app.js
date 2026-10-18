/* Rocket League AI — local control room. Every value comes from the API. */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const icon = (name, cls = "icon") =>
  `<svg class="${cls}"><use href="/theme/icons.svg#icon-${name}"></use></svg>`;

const app = { route: "dashboard", cleanup: null, eventSource: null, timers: [], cache: {} };

// ---------------------------------------------------------------- utilities
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : response.statusText;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

function toast(message, type = "info") {
  const node = $("#toast");
  node.textContent = message;
  node.className = type === "error" ? "toast error" : "toast";
  node.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => (node.hidden = true), type === "error" ? 9000 : 4500);
}

function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function pct(value, digits = 0) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)} %` : "—";
}

function statusClass(status) {
  const map = {
    running: "running",
    starting: "starting",
    paused: "paused",
    pausing: "pausing",
    stopping: "stopping",
    stopped: "stopped",
    completed: "stable",
    failed: "failed",
    interrupted: "interrupted",
    queued: "idle",
  };
  return map[status] || "idle";
}

function pill(status) {
  return `<span class="status-pill ${statusClass(status)}">${esc(status)}</span>`;
}

function modelState(state, stable) {
  if (stable) return `<span class="state-chip stable">stable</span>`;
  if (state === "rejected") return `<span class="state-chip rejected">rejected</span>`;
  if (state === "archived") return `<span class="state-chip">archived</span>`;
  return `<span class="state-chip candidate">candidate</span>`;
}

function emptyState(title, body, iconName = "info") {
  return `<div class="empty-state"><span class="empty-symbol">${icon(iconName)}</span><h2>${esc(title)}</h2><p>${esc(body)}</p></div>`;
}

function lineChart(points, { color = "var(--accent)", height = 150 } = {}) {
  if (!points || points.length < 1) {
    return `<div class="chart-empty">${icon("chart")}<span>No data recorded yet</span></div>`;
  }
  const w = 620;
  const h = height;
  const pad = 8;
  const values = points.map((p) => Number(p.value));
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const x = (i) => pad + (i / Math.max(points.length - 1, 1)) * (w - pad * 2);
  const y = (v) => h - pad - ((v - min) / (max - min)) * (h - pad * 2);
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  const area = `${path} L${x(points.length - 1).toFixed(1)},${h - pad} L${x(0).toFixed(1)},${h - pad} Z`;
  return `
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="metric chart">
      <path d="${area}" fill="var(--accent-softer)" stroke="none"></path>
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"></path>
    </svg>
    <div class="chart-legend"><span>min ${fmtNumber(min, 3)}</span><span>max ${fmtNumber(max, 3)}</span><span>${points.length} points</span></div>`;
}

function downsample(points, limit = 240) {
  if (points.length <= limit) return points;
  const step = Math.ceil(points.length / limit);
  return points.filter((_, i) => i % step === 0);
}

// ---------------------------------------------------------------- navigation
function currentRoute() {
  const hash = location.hash.replace(/^#\/?/, "").split("?")[0];
  return hash || "dashboard";
}

function setActiveNav(route) {
  $$(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.route === route));
  const label = { dashboard: "Dashboard", training: "Training", monitoring: "Monitoring", models: "Models", evaluation: "Evaluation", live: "Live match", settings: "Settings" }[route] || "Dashboard";
  $("#breadcrumb").textContent = label;
  document.title = `${label} · Rocket League AI`;
}

function cleanup() {
  if (app.eventSource) {
    app.eventSource.close();
    app.eventSource = null;
  }
  app.timers.forEach((t) => clearInterval(t));
  app.timers = [];
  if (app.cleanup) {
    app.cleanup();
    app.cleanup = null;
  }
}

async function render() {
  cleanup();
  const route = currentRoute();
  app.route = route;
  setActiveNav(route);
  const view = $("#view");
  view.innerHTML = `<div class="loading-state"><span class="spinner"></span> Loading ${esc(route)}…</div>`;
  try {
    await views[route]();
  } catch (error) {
    view.innerHTML = `${emptyState("Could not load this view", error.message, "warning")}
      <div class="callout error"><div class="callout-copy"><strong>Backend error</strong><p>${esc(error.message)}</p></div></div>`;
    toast(error.message, "error");
  }
}

// ---------------------------------------------------------------- dashboard
const views = {};

views.dashboard = async () => {
  const [system, runs, models, evaluations, diagnostics] = await Promise.all([
    api("/api/system"),
    api("/api/runs?limit=6"),
    api("/api/models"),
    api("/api/evaluations?limit=5"),
    api("/api/diagnostics"),
  ]);
  const info = system.info;
  const active = system.active_run;
  const totalSteps = active ? Number(active.total_steps || 0) : 0;
  const step = active ? Number(active.step || 0) : 0;
  const progress = totalSteps ? Math.min(100, (step / totalSteps) * 100) : 0;
  $("#model-count").textContent = models.length;
  $("#workspace-mode").textContent = active ? `${active.mode} · ${active.status}` : "no active run";
  $("#app-version").textContent = `RLAI ${system.version || ""}`.trim() || "RLAI";

  const runRows = runs.length
    ? runs
        .map(
          (run) => `<tr>
            <td><strong>${esc(run.name)}</strong><small>${esc(run.mode)} · ${esc(run.id)}</small></td>
            <td>${pill(run.status)}</td>
            <td class="table-strong">${fmtNumber(run.step)} / ${fmtNumber(run.total_steps)}</td>
            <td>${esc(run.created_at.replace("T", " ").slice(0, 19))}</td>
            <td><a class="text-button" href="#/monitoring?run=${encodeURIComponent(run.id)}">Inspect ${icon("arrow")}</a></td>
          </tr>`
        )
        .join("")
    : "";

  const readiness = (diagnostics.checks || [])
    .filter((check) => check.required || !check.ok)
    .slice(0, 6)
    .map(
      (check) => `<div class="readiness-row">
        <span class="readiness-icon ${check.ok ? "" : "unknown"}">${icon(check.ok ? "check" : "warning")}</span>
        <div><strong>${esc(check.name.replace("package:", ""))}</strong><small>${esc(check.detail || "")}</small></div>
        <span class="ready-tag ${check.ok ? "" : "muted"}">${check.ok ? "OK" : check.required ? "MISSING" : "OPTIONAL"}</span>
      </div>`
    )
    .join("");

  const evaluationsHtml = evaluations.length
    ? evaluations
        .map(
          (item) => `<tr><td><strong>${esc(item.model_id || "—")}</strong><small>${esc(item.mode)} vs ${esc(item.opponent)}</small></td>
          <td class="table-strong">${item.wins}W · ${item.losses}L · ${item.draws}D</td>
          <td>${pct(item.wins / Math.max(item.games, 1))}</td>
          <td>${item.goals_for} : ${item.goals_against}</td></tr>`
        )
        .join("")
    : "";

  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Local control room</span>
        <h1>Dashboard</h1>
        <p>Live state of the trainer, the simulator and the model registry.</p>
      </div>
      <div class="heading-actions">
        <a class="button primary" href="#/training">${icon("plus")} New training run</a>
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-card accent">
        <div class="stat-label"><span>Active run</span>${icon("activity")}</div>
        <div class="stat-value">${active ? fmtNumber(step) : "—"}${active ? `<small>/ ${fmtNumber(totalSteps)}</small>` : ""}</div>
        <div class="stat-foot">${active ? pill(active.status) : "no run in progress"}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><span>CPU</span>${icon("cpu")}</div>
        <div class="stat-value">${Math.round(info.cpu_percent || 0)}<small>%</small></div>
        <div class="stat-foot">${info.cpu_count} logical cores · ${info.cpu_physical || "?"} physical</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><span>Memory</span>${icon("layers")}</div>
        <div class="stat-value">${info.memory_available_gb}<small>GB free</small></div>
        <div class="stat-foot">${info.memory_total_gb} GB total</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><span>Models</span>${icon("models")}</div>
        <div class="stat-value">${models.length}</div>
        <div class="stat-foot">${models.filter((m) => m.stable).length} stable · ${models.filter((m) => !m.stable).length} candidates</div>
      </div>
    </div>

    <div class="two-col">
      <section class="panel active-run">
        <div class="panel-header">
          <div><h2>${active ? esc(active.name) : "No training run active"}</h2>
          <p>${active ? `${esc(active.mode)} · ${esc(active.id)}` : "Training never starts automatically."}</p></div>
          ${active ? pill(active.status) : `<span class="pill muted-pill">idle</span>`}
        </div>
        <div class="panel-body">
          ${active ? `
            <div class="progress-wrapper" style="padding-inline:0">
              <div class="progress-label"><span>Progress</span><strong>${fmtNumber(step)} / ${fmtNumber(totalSteps)}</strong></div>
              <div class="progress-track"><div class="progress-fill" style="width:${progress}%"></div></div>
            </div>
            <div class="metadata-list" style="padding-inline:0">
              <div class="metadata-row"><span>Workers</span><strong>${esc(active.config?.n_workers ?? "—")}</strong></div>
              <div class="metadata-row"><span>Device</span><strong>${esc(active.config?.device ?? "auto")}</strong></div>
              <div class="metadata-row"><span>Mode</span><strong>${esc(active.mode)}</strong></div>
              <div class="metadata-row"><span>Message</span><strong>${esc(active.message || "—")}</strong></div>
            </div>
            <div class="button-row">
              <a class="button secondary" href="#/monitoring?run=${encodeURIComponent(active.id)}">${icon("monitoring")} Monitor</a>
              <button class="button ghost" data-action="pause" data-run="${esc(active.id)}" ${["paused", "stopping"].includes(active.status) ? "disabled" : ""}>${icon("pause")} Pause</button>
              <button class="button danger" data-action="stop" data-run="${esc(active.id)}" ${active.status === "stopping" ? "disabled" : ""}>${icon("stop")} Stop</button>
            </div>` : `
            <div class="empty-state" style="padding:8px 0 24px">
              <span class="empty-symbol">${icon("training")}</span>
              <h2>Nothing is training</h2>
              <p>Choose a mode and start a run. The first baseline is 1v1 against the scripted chaser.</p>
              <a class="button primary" href="#/training">${icon("play")} Configure a run</a>
            </div>`}
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Readiness</h2><p>Real dependency checks from this machine.</p></div></div>
        <div class="readiness-list">${readiness || emptyState("No diagnostics", "The diagnostics endpoint returned no checks.", "info")}</div>
      </section>
    </div>

    <div class="two-col balanced">
      <section class="panel">
        <div class="panel-header">
          <div><h2>Recent runs</h2><p>Latest training processes, including failures.</p></div>
          <a class="text-button" href="#/monitoring">All runs ${icon("arrow")}</a>
        </div>
        ${runs.length ? `<div class="table-wrap"><table class="compact"><thead><tr><th>Run</th><th>Status</th><th>Steps</th><th>Created</th><th></th></tr></thead><tbody>${runRows}</tbody></table></div>` : emptyState("No runs yet", "Start a training run to populate this table.", "training")}
      </section>

      <section class="panel">
        <div class="panel-header">
          <div><h2>Recent evaluations</h2><p>Deterministic matches against fixed opponents.</p></div>
          <a class="text-button" href="#/evaluation">Details ${icon("arrow")}</a>
        </div>
        ${evaluations.length ? `<div class="table-wrap"><table class="compact"><thead><tr><th>Model</th><th>Record</th><th>Win rate</th><th>Goals</th></tr></thead><tbody>${evaluationsHtml}</tbody></table></div>` : emptyState("No evaluations", "Evaluate a checkpoint to compare models honestly.", "evaluation")}
      </section>
    </div>`;

  $$("[data-action]", $("#view")).forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.run;
      try {
        await api(`/api/runs/${encodeURIComponent(id)}/${button.dataset.action}`, { method: "POST" });
        toast(`Run ${button.dataset.action === "stop" ? "stop requested" : "paused"}`);
        setTimeout(render, 800);
      } catch (error) {
        toast(error.message, "error");
      }
    });
  });
};

// ---------------------------------------------------------------- training
views.training = async () => {
  const [modes, defaults, runs] = await Promise.all([
    api("/api/modes"),
    api("/api/config/defaults"),
    api("/api/runs?limit=20"),
  ]);
  app.cache.modes = modes;
  app.cache.defaults = defaults;
  const config = defaults.training;
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const resumeFrom = params.get("resume");
  const modeOptions = modes
    .map((mode) => `<option value="${mode.id}" ${mode.id === config.mode ? "selected" : ""}>${esc(mode.name)} (${mode.players} cars)</option>`)
    .join("");
  const presets = defaults.presets
    .map(
      (preset, index) => `<button class="preset-card" type="button" data-preset="${index}">
        <span class="preset-top"><strong>${esc(preset.name)}</strong><span class="pill muted-pill">${esc(preset.config.mode)}</span></span>
        <p>${esc(preset.description)}</p></button>`
    )
    .join("");

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Training engine</span><h1>New run</h1>
      <p>PPO with parallel RocketSim environments. Configuration is validated by the server before anything starts.</p></div>
      <div class="heading-actions"><a class="button ghost" href="#/monitoring">${icon("monitoring")} Existing runs</a></div>
    </div>

    ${resumeFrom ? `<div class="callout success">
      <div class="callout-copy"><strong>Continuing from checkpoint ${esc(resumeFrom)}</strong>
      <p>Model weights, optimizer state and step count are restored when the run starts. The original checkpoint is not modified.</p></div>
    </div>` : ""}

    <div class="callout success">
      <div class="callout-copy"><strong>Starting a run is explicit</strong>
      <p>Nothing starts when the app launches. A run consumes CPU until it is stopped or reaches its step budget.</p></div>
    </div>

    <section class="panel" style="margin-bottom:18px">
      <div class="panel-header"><div><h2>Presets</h2><p>Staged recipes: ball control → 1v1 → teams → Heatseeker.</p></div></div>
      <div class="preset-grid" style="padding:0 20px 18px">${presets}</div>
    </section>

    <form id="training-form" class="config-layout">
      <section class="panel">
        <div class="panel-header"><div><h2>Run configuration</h2><p>Values map 1:1 to the server-side schema.</p></div><span class="state-chip" id="form-mode-chip">${esc(config.mode)}</span></div>
        <div class="form-grid">
          <label class="field full"><span>Run name</span><input name="run_name" value="${esc(config.run_name)}" maxlength="80" required /></label>
          <label class="field"><span>Game mode</span><select name="mode">${modeOptions}</select></label>
          <label class="field"><span>Observation layout</span><select name="obs_version">${[[2, "v2 · field frame (198)"], [1, "v1 · legacy (190)"]].map(([value, label]) => `<option value="${value}" ${Number(config.obs_version) === value ? "selected" : ""}>${label}</option>`).join("")}</select><small>v1 only for older checkpoints and comparisons.</small></label>
          <label class="field"><span>Device</span><select name="device">${["auto", "cpu", "cuda", "mps"].map((d) => `<option ${d === config.device ? "selected" : ""}>${d}</option>`).join("")}</select></label>
          <label class="field"><span>Environment workers</span><input name="n_workers" type="number" min="1" max="32" value="${config.n_workers}" /><small>One RocketSim arena per worker process.</small></label>
          <label class="field"><span>Total environment steps</span><input name="total_steps" type="number" min="2000" step="2000" value="${config.total_steps}" /><small>Budget across all workers.</small></label>
          <label class="field"><span>Rollout steps per worker</span><input name="rollout_steps" type="number" min="32" max="8192" value="${config.rollout_steps}" /></label>
          <label class="field"><span>Episode limit (seconds)</span><input name="episode_seconds" type="number" min="5" max="600" value="${config.episode_seconds}" /><small>Episodes also end on a goal.</small></label>
          <label class="field"><span>Learning rate</span><input name="learning_rate" type="number" step="0.00001" min="0.000001" value="${config.learning_rate}" /></label>
          <label class="field"><span>Entropy coefficient</span><input name="entropy_coef" type="number" step="0.001" min="0" value="${config.entropy_coef}" /></label>
          <label class="field"><span>Gamma</span><input name="gamma" type="number" step="0.001" min="0.8" max="0.9999" value="${config.gamma}" /></label>
          <label class="field"><span>Opponent</span><select name="opponent">${["chase", "random", "idle"].map((o) => `<option ${o === config.opponent ? "selected" : ""}>${o}</option>`).join("")}</select><small>Ignored when self-play or the curriculum selects the opponent.</small></label>
          <label class="field"><span>Checkpoint every N iterations</span><input name="checkpoint_interval" type="number" min="1" max="10000" value="${config.checkpoint_interval}" /></label>
          <label class="field"><span>Seed</span><input name="seed" type="number" value="${config.seed}" /><small>Controls kickoffs and action sampling.</small></label>
          <label class="field"><span>Evaluation every N iterations</span><input name="eval_interval" type="number" min="0" max="100000" value="${config.eval_interval}" /><small>0 disables periodic evaluation.</small></label>
          <label class="field"><span>Evaluation games</span><input name="eval_games" type="number" min="1" max="100" value="${config.eval_games}" /><small>Fewer games mean a wider confidence interval.</small></label>
          <label class="field"><span>Evaluation match seconds</span><input name="eval_match_seconds" type="number" min="5" max="600" value="${config.eval_match_seconds}" /></label>
          <label class="field"><span>Evaluation opponent</span><select name="eval_opponent">${["chase", "random", "idle", "self"].map((o) => `<option ${o === config.eval_opponent ? "selected" : ""}>${o}</option>`).join("")}</select></label>
          <label class="field full approval-check"><input type="checkbox" name="self_play" ${config.self_play ? "checked" : ""} /><span>Self-play with an opponent pool of frozen snapshots (PFSP-weighted Elo pool).</span></label>
          <label class="field full approval-check"><input type="checkbox" name="curriculum" ${config.curriculum ? "checked" : ""} /><span>Curriculum: promote through measured stage targets (idle → chaser → mixed → self-play).</span></label>
        </div>
        <details class="advanced-config">
          <summary>PPO details</summary>
          <div class="form-grid" style="padding:12px 0 0">
            <label class="field"><span>Epochs per rollout</span><input name="epochs" type="number" min="1" max="30" value="${config.epochs}" /></label>
            <label class="field"><span>Minibatch size</span><input name="minibatch_size" type="number" min="64" max="65536" value="${config.minibatch_size}" /></label>
            <label class="field"><span>Clip coefficient</span><input name="clip_coef" type="number" step="0.01" min="0.01" max="1" value="${config.clip_coef}" /></label>
            <label class="field"><span>Value clip</span><input name="clip_value" type="number" step="0.01" min="0" max="10" value="${config.clip_value}" /><small>0 disables value clipping.</small></label>
            <label class="field"><span>Target KL</span><input name="target_kl" type="number" step="0.005" min="0" max="10" value="${config.target_kl}" /><small>Early-stop threshold; 0 disables it.</small></label>
            <label class="field"><span>GAE lambda</span><input name="gae_lambda" type="number" step="0.01" min="0.8" max="0.999" value="${config.gae_lambda}" /></label>
            <label class="field"><span>Value coefficient</span><input name="value_coef" type="number" step="0.05" min="0" max="10" value="${config.value_coef}" /></label>
            <label class="field"><span>Max gradient norm</span><input name="max_grad_norm" type="number" step="0.1" min="0.1" max="10" value="${config.max_grad_norm}" /></label>
            <label class="field"><span>Tick skip</span><input name="tick_skip" type="number" min="1" max="16" value="${config.tick_skip}" /><small>Simulator ticks per decision.</small></label>
            <label class="field"><span>Pool snapshot every N iterations</span><input name="snapshot_interval" type="number" min="1" max="1000" value="${config.snapshot_interval}" /><small>Frozen copies for self-play.</small></label>
            <label class="field"><span>Keep checkpoints</span><input name="keep_checkpoints" type="number" min="1" max="100" value="${config.keep_checkpoints}" /></label>
          </div>
        </details>
        <details class="advanced-config">
          <summary>Reward weights</summary>
          <div class="form-grid" style="padding:12px 0 0">
            ${["goal", "concede", "touch", "progress", "alignment", "approach", "boost", "defense", "bump"]
              .map((key) => `<label class="field"><span>${key}</span><input name="reward_${key}" type="number" step="0.01" min="0" value="${config.reward[key] ?? 0}" /></label>`)
              .join("")}
          </div>
          <p class="subtle">Goals dominate; progress and alignment are potential-based shaping. Reward components are logged with the run.</p>
        </details>
        <div class="config-actions">
          <button class="button ghost" type="button" id="validate-button">${icon("shield")} Validate only</button>
          <button class="button primary" type="submit">${icon("play")} Start training</button>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Runs</h2><p>Pause or stop without losing the last checkpoint.</p></div></div>
        ${runs.length ? `<div class="run-list">
          ${runs.map((run) => `<div class="run-row">
            <div class="run-main">
              <strong>${esc(run.name)}</strong>
              <small>${esc(run.mode)} · step ${fmtNumber(run.step)}</small>
            </div>
            ${pill(run.status)}
            <div class="run-actions">
              <a class="button ghost" href="#/monitoring?run=${encodeURIComponent(run.id)}" title="Open telemetry">${icon("chart")}</a>
              <button class="button ghost" data-run-action="pause" data-run="${esc(run.id)}" ${["running", "starting"].includes(run.status) ? "" : "disabled"} title="Pause">${icon("pause")}</button>
              <button class="button danger" data-run-action="stop" data-run="${esc(run.id)}" ${["running", "starting", "paused"].includes(run.status) ? "" : "disabled"} title="Stop">${icon("stop")}</button>
            </div>
          </div>`).join("")}
        </div>` : emptyState("No runs yet", "Your first configuration is on the left.")}
      </section>
    </form>`;

  const form = $("#training-form");
  form.elements.mode.addEventListener("change", () => {
    $("#form-mode-chip").textContent = form.elements.mode.value;
  });
  $$("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => fillForm(form, defaults.presets[Number(button.dataset.preset)].config));
  });
  $$("[data-run-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/api/runs/${encodeURIComponent(button.dataset.run)}/${button.dataset.runAction}`, { method: "POST" });
        toast(`Run ${button.dataset.runAction === "stop" ? "stopping" : "pause requested"}`);
        setTimeout(render, 700);
      } catch (error) {
        toast(error.message, "error");
      }
    });
  });

  const configFromForm = () => {
    const data = new FormData(form);
    const reward = {};
    Object.keys(config.reward).forEach((key) => {
      const value = data.get(`reward_${key}`);
      if (value !== null && value !== "") reward[key] = Number(value);
    });
    return {
      run_name: String(data.get("run_name") || "").trim() || "training run",
      mode: data.get("mode"),
      device: data.get("device"),
      n_workers: Number(data.get("n_workers")),
      total_steps: Number(data.get("total_steps")),
      rollout_steps: Number(data.get("rollout_steps")),
      episode_seconds: Number(data.get("episode_seconds")),
      learning_rate: Number(data.get("learning_rate")),
      entropy_coef: Number(data.get("entropy_coef")),
      gamma: Number(data.get("gamma")),
      opponent: data.get("opponent"),
      checkpoint_interval: Number(data.get("checkpoint_interval")),
      seed: Number(data.get("seed")),
      epochs: Number(data.get("epochs")),
      minibatch_size: Number(data.get("minibatch_size")),
      clip_coef: Number(data.get("clip_coef")),
      gae_lambda: Number(data.get("gae_lambda")),
      tick_skip: Number(data.get("tick_skip")),
      keep_checkpoints: Number(data.get("keep_checkpoints")),
      obs_version: Number(data.get("obs_version")),
      clip_value: Number(data.get("clip_value")),
      target_kl: Number(data.get("target_kl")),
      value_coef: Number(data.get("value_coef")),
      max_grad_norm: Number(data.get("max_grad_norm")),
      snapshot_interval: Number(data.get("snapshot_interval")),
      eval_interval: Number(data.get("eval_interval")),
      eval_games: Number(data.get("eval_games")),
      eval_match_seconds: Number(data.get("eval_match_seconds")),
      eval_opponent: data.get("eval_opponent"),
      self_play: data.get("self_play") === "on",
      curriculum: data.get("curriculum") === "on",
      reward,
      ...(resumeFrom ? { resume_from: resumeFrom } : {}),
    };
  };

  $("#validate-button").addEventListener("click", async () => {
    try {
      const result = await api("/api/config/validate", { method: "POST", body: JSON.stringify(configFromForm()) });
      const warning = result.warnings && result.warnings.length ? ` — ${result.warnings[0]}` : "";
      toast(`Configuration valid · ${result.config.mode} · ${fmtNumber(result.config.total_steps)} steps${warning}`);
    } catch (error) {
      toast(error.message, "error");
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = form.querySelector('button[type="submit"]');
    submit.disabled = true;
    try {
      const run = await api("/api/runs", { method: "POST", body: JSON.stringify(configFromForm()) });
      toast(`Run ${run.id} started`);
      location.hash = `#/monitoring?run=${encodeURIComponent(run.id)}`;
    } catch (error) {
      toast(error.message, "error");
      submit.disabled = false;
    }
  });
};

function fillForm(form, config) {
  const assign = (name, value) => {
    const field = form.elements[name];
    if (!field) return;
    if (field.type === "checkbox") field.checked = Boolean(value);
    else field.value = value ?? "";
  };
  Object.entries(config).forEach(([key, value]) => {
    if (key === "reward") {
      Object.entries(value).forEach(([rewardKey, rewardValue]) => assign(`reward_${rewardKey}`, rewardValue));
    } else if (key !== "policy") {
      assign(key, value);
    }
  });
  $("#form-mode-chip").textContent = config.mode;
  toast("Preset loaded — review the values, then start");
}

// ---------------------------------------------------------------- monitoring
views.monitoring = async () => {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const runs = await api("/api/runs?limit=50");
  if (!runs.length) {
    $("#view").innerHTML = `
      <div class="page-heading"><div><span class="eyebrow">Telemetry</span><h1>Monitoring</h1><p>No runs exist yet.</p></div></div>
      ${emptyState("Nothing to monitor", "Start a training run first; metrics and logs are written by the training process.", "monitoring")}`;
    return;
  }
  const active = runs.find((r) => r.id === params.get("run")) || runs[0];
  const runId = active.id;
  const [detail, metrics, resources] = await Promise.all([
    api(`/api/runs/${encodeURIComponent(runId)}`),
    api(`/api/runs/${encodeURIComponent(runId)}/metrics`),
    api("/api/system/resources"),
  ]);
  const initialLogs = await api(`/api/runs/${encodeURIComponent(runId)}/logs?limit=200`);
  const series = (name) => downsample(metrics.filter((m) => m.name === name));
  const summary = detail.episode_summary || {};
  const cfg = detail.config || {};
  const total = Number(detail.total_steps || cfg.total_steps || 0);
  const progress = total ? Math.min(100, (Number(detail.step || 0) / total) * 100) : 0;

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Telemetry</span><h1>Monitoring</h1><p>Live metrics and logs from the training subprocess.</p></div>
      <div class="heading-actions">
        <select id="run-select" class="log-search" style="width:auto">
          ${runs.map((run) => `<option value="${esc(run.id)}" ${run.id === runId ? "selected" : ""}>${esc(run.name)} · ${esc(run.id)}</option>`).join("")}
        </select>
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-card accent">
        <div class="stat-label"><span>Run status</span>${icon("activity")}</div>
        <div class="stat-value" style="font-size:22px">${esc(detail.status)}</div>
        <div class="stat-foot">${esc(detail.mode)} · ${esc(runId)}</div>
      </div>
      <div class="stat-card"><div class="stat-label"><span>Steps</span>${icon("layers")}</div>
        <div class="stat-value">${fmtNumber(detail.step)}</div>
        <div class="stat-foot">of ${fmtNumber(total)} · ${progress.toFixed(1)} %</div></div>
      <div class="stat-card"><div class="stat-label"><span>Episodes (last ${summary.count || 0})</span>${icon("trophy")}</div>
        <div class="stat-value">${summary.wins ?? 0}<small>W</small></div>
        <div class="stat-foot">${summary.losses ?? 0}L · ${summary.draws ?? 0}D</div></div>
      <div class="stat-card"><div class="stat-label"><span>Throughput</span>${icon("bolt")}</div>
        <div class="stat-value" id="live-sps">—</div>
        <div class="stat-foot">environment steps / s</div></div>
    </div>

    <div class="two-col">
      <section class="panel chart-panel">
        <div class="panel-header"><div><h2>Episodic return</h2><p>Mean reward per finished episode.</p></div><span class="pill muted-pill">${series("charts/episodic_return").length} points</span></div>
        <div class="chart" id="chart-return">${lineChart(series("charts/episodic_return"))}</div>
      </section>
      <section class="panel chart-panel">
        <div class="panel-header"><div><h2>Win rate</h2><p>Share of episodes won against the configured opponent.</p></div><span class="pill muted-pill">${series("charts/win_rate").length} points</span></div>
        <div class="chart" id="chart-winrate">${lineChart(series("charts/win_rate"), { color: "var(--info)" })}</div>
      </section>
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Resources</h2><p>Measured on this machine, updated every 2 s.</p></div></div>
        <div class="panel-body">
          ${resourceRow("CPU", "cpu", resources.cpu_percent, `${resources.cpu_count} cores`)}
          ${resourceRow("Memory", "memory", resources.memory_percent, `${resources.memory_used_gb} / ${resources.memory_total_gb} GB`)}
          ${resources.gpu ? resourceRow("GPU", "gpu", 0, `${resources.gpu.name} · ${fmtNumber(resources.gpu.memory_total_mb)} MB total`) : `<p class="resource-note">No GPU detected. Training runs on CPU; that is supported, just slower.</p>`}
        </div>
      </section>
      <section class="panel">
        <div class="panel-header"><div><h2>Episode summary</h2><p>Rolling window of the latest episodes.</p></div></div>
        <div class="metadata-list">
          <div class="metadata-row"><span>Mean reward</span><strong>${Number(summary.mean_reward || 0).toFixed(3)}</strong></div>
          <div class="metadata-row"><span>Mean length</span><strong>${Math.round(summary.mean_length || 0)} ticks</strong></div>
          <div class="metadata-row"><span>Goals for / against</span><strong>${summary.goals_for ?? 0} / ${summary.goals_against ?? 0}</strong></div>
          <div class="metadata-row"><span>Workers</span><strong>${esc(cfg.n_workers ?? "—")}</strong></div>
          <div class="metadata-row"><span>Opponent</span><strong>${esc(cfg.self_play ? "self-play pool" : cfg.opponent ?? "—")}</strong></div>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Logs</h2><p>Structured output written by the trainer.</p></div><span class="live-indicator"><i></i> streaming</span></div>
      <div class="log-toolbar">
        <div class="log-filters">
          ${["all", "info", "warning", "error"].map((level) => `<button data-log-level="${level}" class="${level === "all" ? "selected" : ""}">${level}</button>`).join("")}
        </div>
        <input class="log-search" id="log-search" placeholder="Filter log lines" />
      </div>
      <div class="log-stream" id="log-stream">${initialLogs.map(logLine).join("") || `<div class="log-line"><span class="log-message">No log lines yet.</span></div>`}</div>
      <div class="log-footer"><span id="log-count">${initialLogs.length} lines</span><span>${esc(detail.message || "")}</span></div>
    </section>`;

  $("#run-select").addEventListener("change", (event) => {
    location.hash = `#/monitoring?run=${encodeURIComponent(event.target.value)}`;
  });

  let allLogs = initialLogs;
  let logLevel = "all";
  let logSearch = "";
  const paintLogs = () => {
    const filtered = allLogs.filter((item) => {
      const levelOk = logLevel === "all" || item.level.toLowerCase() === logLevel;
      const searchOk = !logSearch || item.message.toLowerCase().includes(logSearch);
      return levelOk && searchOk;
    });
    $("#log-stream").innerHTML = filtered.slice(-400).map(logLine).join("") || `<div class="log-line"><span class="log-message">No matching lines.</span></div>`;
    $("#log-count").textContent = `${filtered.length} lines`;
    const stream = $("#log-stream");
    stream.scrollTop = stream.scrollHeight;
  };
  $$("[data-log-level]").forEach((button) => {
    button.addEventListener("click", () => {
      logLevel = button.dataset.logLevel;
      $$("[data-log-level]").forEach((b) => b.classList.toggle("selected", b === button));
      paintLogs();
    });
  });
  $("#log-search").addEventListener("input", (event) => {
    logSearch = event.target.value.toLowerCase();
    paintLogs();
  });

  let returnPoints = series("charts/episodic_return").map((p) => ({ step: p.step, value: p.value }));
  let winratePoints = series("charts/win_rate").map((p) => ({ step: p.step, value: p.value }));

  app.eventSource = new EventSource(`/api/events?run_id=${encodeURIComponent(runId)}`);
  app.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "log") {
      allLogs.push(payload);
      paintLogs();
    } else if (payload.type === "metrics") {
      if (payload.run) {
        $("#live-sps").textContent = fmtNumber(payload.metrics["time/sps"] || 0);
      }
      if (payload.metrics["charts/episodic_return"] !== undefined) {
        returnPoints.push({ step: payload.run?.step || 0, value: payload.metrics["charts/episodic_return"] });
        returnPoints = downsample(returnPoints, 240);
        $("#chart-return").innerHTML = lineChart(returnPoints);
      }
      if (payload.metrics["charts/win_rate"] !== undefined) {
        winratePoints.push({ step: payload.run?.step || 0, value: payload.metrics["charts/win_rate"] });
        winratePoints = downsample(winratePoints, 240);
        $("#chart-winrate").innerHTML = lineChart(winratePoints, { color: "var(--info)" });
      }
    }
  };
  app.eventSource.onerror = () => {
    /* browser retries automatically */
  };

  const resourceTimer = setInterval(async () => {
    try {
      const latest = await api("/api/system/resources");
      const cpu = $("#resource-cpu");
      const mem = $("#resource-memory");
      if (cpu) {
        cpu.style.width = `${Math.min(latest.cpu_percent, 100)}%`;
        cpu.className = `meter-fill ${latest.cpu_percent > 90 ? "danger" : latest.cpu_percent > 70 ? "warning" : "green"}`;
        const label = $("#resource-cpu-label");
        if (label) label.textContent = `${Math.round(latest.cpu_percent)} %`;
      }
      if (mem) {
        mem.style.width = `${Math.min(latest.memory_percent, 100)}%`;
        const label = $("#resource-memory-label");
        if (label) label.textContent = `${latest.memory_used_gb} / ${latest.memory_total_gb} GB`;
      }
    } catch {
      /* API restarting */
    }
  }, 2000);
  app.timers.push(resourceTimer);
};

function resourceRow(label, key, value, detail) {
  const width = Math.max(0, Math.min(Number(value) || 0, 100));
  return `<div class="resource-item">
    <div class="resource-label"><span>${esc(label)}</span><strong id="resource-${key}-label">${Math.round(width)} %</strong></div>
    <div class="meter-track"><div class="meter-fill green" id="resource-${key}" style="width:${width}%"></div></div>
    <p class="resource-note">${esc(detail)}</p>
  </div>`;
}

function logLine(item) {
  const date = new Date((item.ts || 0) * 1000);
  const time = Number.isFinite(date.getTime()) ? date.toLocaleTimeString() : "—";
  const level = (item.level || "INFO").toLowerCase();
  return `<div class="log-line"><span class="log-time">${time}</span><span class="log-level ${level}">${esc(item.level || "INFO")}</span><span class="log-message">${esc(item.message)}</span></div>`;
}

// ---------------------------------------------------------------- models
views.models = async () => {
  const [models, evaluations] = await Promise.all([api("/api/models"), api("/api/evaluations?limit=200")]);
  $("#model-count").textContent = models.length;
  const evalByModel = {};
  evaluations.forEach((item) => {
    const current = evalByModel[item.model_id];
    const winRate = item.wins / Math.max(item.games, 1);
    if (!current || winRate > current.winRate) evalByModel[item.model_id] = { winRate, item };
  });

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Model registry</span><h1>Models</h1>
      <p>Checkpoints are immutable; promotion only moves the stable pointer. Nothing is promoted automatically.</p></div>
      <div class="heading-actions"><a class="button secondary" href="#/evaluation">${icon("evaluation")} Evaluate</a></div>
    </div>

    ${models.length ? `<section class="panel">
      <div class="panel-header"><div><h2>Checkpoints</h2><p>Sorted newest first. ${models.filter((m) => m.stable).length} stable model(s).</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th>Model</th><th>State</th><th>Mode</th><th>Step</th><th>Best win rate</th><th>Created</th><th></th></tr></thead>
        <tbody>${models.map((model) => {
          const evaluation = evalByModel[model.id];
          return `<tr>
            <td><strong>${esc(model.id)}</strong><small>${fmtNumber(model.size_bytes)} bytes${model.stable ? " · stable pointer" : ""}${model.best ? " · best measured" : ""}</small></td>
            <td>${modelState(model.state, model.stable)}</td>
            <td>${esc(model.mode || "—")}</td>
            <td class="table-strong">${fmtNumber(model.step)}</td>
            <td>${evaluation ? `${(evaluation.winRate * 100).toFixed(0)} % <small>vs ${esc(evaluation.item.opponent)}</small>` : "—"}</td>
            <td>${esc((model.created_at || "").replace("T", " ").slice(0, 19))}</td>
            <td class="actions">
              <a class="button ghost" href="#/training?resume=${encodeURIComponent(model.id)}" title="Continue training from this checkpoint">${icon("training")}</a>
              <button class="button ghost" data-evaluate="${esc(model.id)}">${icon("play")} Evaluate</button>
              <button class="button ghost" data-promote="${esc(model.id)}" ${model.stable ? "disabled" : ""}>${icon("check")} Promote</button>
              <button class="button danger" data-delete="${esc(model.id)}" ${model.stable ? "disabled" : ""} title="Delete checkpoint">${icon("delete")}</button>
            </td></tr>`;
        }).join("")}</tbody>
      </table></div>
    </section>` : emptyState("No checkpoints", "Checkpoints are written during training. Start a run first.", "models")}

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Evaluation</h2><p>Run a deterministic match series against a fixed opponent.</p></div></div>
      <div class="form-grid">
        <label class="field"><span>Model</span><select id="eval-model">${models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("")}</select></label>
        <label class="field"><span>Opponent</span><select id="eval-opponent"><option>chase</option><option>idle</option><option>random</option><option>self</option></select></label>
        <label class="field"><span>Games</span><input id="eval-games" type="number" min="1" max="200" value="20" /></label>
        <label class="field"><span>Match seconds</span><input id="eval-seconds" type="number" min="10" max="600" value="120" /></label>
        <label class="field"><span>Seed</span><input id="eval-seed" type="number" value="1000" /></label>
        <label class="field"><span>Device</span><select id="eval-device"><option>cpu</option><option>cuda</option></select></label>
      </div>
      <div class="config-actions"><button class="button primary" id="eval-start" ${models.length ? "" : "disabled"}>${icon("play")} Start evaluation</button></div>
    </section>`;

  $$("[data-evaluate]").forEach((button) => button.addEventListener("click", () => {
    $("#eval-model").value = button.dataset.evaluate;
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }));
  $$("[data-promote]").forEach((button) => button.addEventListener("click", () => confirmAction(
    "Promote checkpoint",
    `Promote ${button.dataset.promote} to the stable pointer? The previous stable model is archived, never deleted.`,
    async () => {
      await api(`/api/models/${encodeURIComponent(button.dataset.promote)}/promote`, { method: "POST" });
      toast("Model promoted to stable");
      render();
    }
  )));
  $$("[data-delete]").forEach((button) => button.addEventListener("click", () => confirmAction(
    "Delete checkpoint",
    `Permanently delete ${button.dataset.delete}? Stable models and the last remaining checkpoint are protected.`,
    async () => {
      await api(`/api/models/${encodeURIComponent(button.dataset.delete)}`, { method: "DELETE" });
      toast("Checkpoint deleted");
      render();
    }
  )));
  $("#eval-start")?.addEventListener("click", async () => {
    const modelId = $("#eval-model").value;
    try {
      await api(`/api/models/${encodeURIComponent(modelId)}/evaluate`, {
        method: "POST",
        body: JSON.stringify({
          model_id: modelId,
          opponent: $("#eval-opponent").value,
          games: Number($("#eval-games").value),
          match_seconds: Number($("#eval-seconds").value),
          seed: Number($("#eval-seed").value),
          device: $("#eval-device").value,
        }),
      });
      toast("Evaluation started. Results appear when the match series finishes.");
      setTimeout(() => location.hash === "#/evaluation" && render(), 2500);
    } catch (error) {
      toast(error.message, "error");
    }
  });
};

function confirmAction(title, body, action) {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-body").textContent = body;
  dialog.showModal();
  const handler = async (event) => {
    if (!event.target.closest("#confirm-ok")) return;
    event.preventDefault();
    dialog.close();
    try {
      await action();
    } catch (error) {
      toast(error.message, "error");
    }
  };
  dialog.addEventListener("click", handler, { once: false });
  dialog.addEventListener("close", () => dialog.removeEventListener("click", handler), { once: true });
}

// ---------------------------------------------------------------- evaluation
views.evaluation = async () => {
  const [evaluations, models] = await Promise.all([api("/api/evaluations?limit=200"), api("/api/models")]);
  const best = {};
  evaluations.forEach((item) => {
    const key = item.model_id || "unknown";
    const rate = item.wins / Math.max(item.games, 1);
    if (!best[key] || rate > best[key].rate) best[key] = { ...item, rate };
  });

  $("#view").innerHTML = `
    <div class="page-heading"><div><span class="eyebrow">Benchmarks</span><h1>Evaluation</h1>
    <p>Every result below is a real match series with fixed seeds. Draws are possible.</p></div></div>

    ${evaluations.length ? `
    <div class="stat-grid">
      <div class="stat-card accent"><div class="stat-label"><span>Evaluations</span>${icon("evaluation")}</div><div class="stat-value">${evaluations.length}</div><div class="stat-foot">all time</div></div>
      <div class="stat-card"><div class="stat-label"><span>Models evaluated</span>${icon("models")}</div><div class="stat-value">${Object.keys(best).length}</div><div class="stat-foot">of ${models.length} checkpoints</div></div>
      <div class="stat-card"><div class="stat-label"><span>Games played</span>${icon("activity")}</div><div class="stat-value">${fmtNumber(evaluations.reduce((sum, item) => sum + item.games, 0))}</div><div class="stat-foot">deterministic simulations</div></div>
      <div class="stat-card"><div class="stat-label"><span>Goals</span>${icon("trophy")}</div><div class="stat-value">${fmtNumber(evaluations.reduce((sum, item) => sum + item.goals_for, 0))}</div><div class="stat-foot">${evaluations.reduce((sum, item) => sum + item.goals_against, 0)} conceded</div></div>
    </div>

    <section class="panel" style="margin-bottom:18px">
      <div class="panel-header"><div><h2>History</h2><p>Latest first. Compare win rate and goal difference.</p></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Mode</th><th>Opponent</th><th>Record</th><th>Win rate (95 % CI)</th><th>Goals</th><th>When</th></tr></thead>
        <tbody>${evaluations.map((item) => {
          const ci = item.details?.win_rate_ci95;
          return `<tr>
          <td><strong>${esc(item.model_id || "—")}</strong><small>${fmtNumber(item.games)} games · seed ${item.seed ?? "—"}</small></td>
          <td>${esc(item.mode)}</td><td>${esc(item.opponent)}</td>
          <td class="table-strong">${item.wins}W · ${item.losses}L · ${item.draws}D</td>
          <td>${pct(item.wins / Math.max(item.games, 1))}<small>${ci ? `${pct(ci[0])} – ${pct(ci[1])}` : "no interval recorded"}</small></td>
          <td>${item.goals_for} : ${item.goals_against}</td>
          <td>${esc((item.created_at || "").replace("T", " ").slice(0, 19))}</td>
        </tr>`;
        }).join("")}</tbody>
      </table></div>
    </section>

    <section class="panel">
      <div class="panel-header"><div><h2>Per-model comparison</h2><p>Best recorded series per checkpoint.</p></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Mode</th><th>Opponent</th><th>Win rate</th><th>Goal diff</th><th>Games</th></tr></thead>
        <tbody>${Object.entries(best).map(([model, item]) => `<tr>
          <td><strong>${esc(model)}</strong></td><td>${esc(item.mode)}</td><td>${esc(item.opponent)}</td>
          <td class="table-strong">${pct(item.rate)}</td>
          <td>${item.goals_for - item.goals_against >= 0 ? "+" : ""}${item.goals_for - item.goals_against}</td>
          <td>${item.games}</td></tr>`).join("")}</tbody>
      </table></div>
    </section>` : emptyState("No evaluations yet", "Open Models and evaluate a checkpoint against the chaser or idle baseline.", "evaluation")}`;
};

// ---------------------------------------------------------------- live
views.live = async () => {
  const [status, models] = await Promise.all([api("/api/live/status"), api("/api/models")]);
  const modes = app.cache.modes || (await api("/api/modes"));
  $("#view").innerHTML = `
    <div class="page-heading"><div><span class="eyebrow">Deployment</span><h1>Live match</h1>
    <p>RLBot v5 bridge. Training and evaluation never require Rocket League; only this page does.</p></div></div>

    <div class="callout ${status.launch_supported ? "success" : "warning"}">
      <div class="callout-copy">
        <strong>${status.launch_supported ? "Live launch available" : "Live launch unavailable in this environment"}</strong>
        <p>${esc(status.note)}</p>
      </div>
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Deployment status</h2><p>Checked on the server, not simulated.</p></div></div>
        <div class="metadata-list">
          <div class="metadata-row"><span>Platform</span><strong>${esc(status.platform)}</strong></div>
          <div class="metadata-row"><span>RLBot installed</span><strong>${status.rlbot_installed ? "yes" : "no"}</strong></div>
          <div class="metadata-row"><span>Stable model</span><strong>${esc(status.stable_model?.id || "none")}</strong></div>
          <div class="metadata-row"><span>Launch supported</span><strong>${status.launch_supported ? "yes" : "no"}</strong></div>
        </div>
        <div class="config-actions"><button class="button" id="live-checks-run">${icon("shield")} Run setup checks</button></div>
        <div class="metadata-list" id="live-checks"><div class="metadata-row"><span>Checks</span><strong>not run yet</strong></div></div>
        ${models.length ? `
        <div class="form-grid">
          <label class="field"><span>Model</span><select id="live-model">${models.map((m) => `<option value="${esc(m.id)}" ${m.stable ? "selected" : ""}>${esc(m.id)}${m.stable ? " (stable)" : ""}${m.best ? " · best" : ""}</option>`).join("")}</select></label>
          <label class="field"><span>Mode</span><select id="live-mode">${modes.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("")}</select></label>
        </div>
        <div class="config-actions">
          <button class="button primary" id="live-launch" ${status.launch_supported ? "" : "disabled"}>${icon("play")} Launch match</button>
          <button class="button" id="live-prepare">${icon("file")} Prepare files only</button>
        </div>
        <p class="subtle">${status.launch_supported ? "Launch starts RLBot with the written match files." : "Launch is disabled in this environment: Windows and RLBot v5 are required. Preparing files still works and shows exactly what a launch would write."}</p>` : `<p class="subtle">No checkpoints exist yet. Train or copy a model into data/models first.</p>`}
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Windows checklist</h2><p>Exact steps for your Epic installation.</p></div></div>
        <div class="panel-body">
          <ol class="guide-list">
            <li>Install Python 3.11 or 3.12 (64-bit) and tick <strong>Add to PATH</strong>.</li>
            <li>In the repository: <strong>powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1</strong></li>
            <li>Install RLBot v5 from <strong>rlbot.org</strong> and start Rocket League through Epic once.</li>
            <li>Start the app: <strong>.venv\\Scripts\\python -m rlai.cli serve --port 8000</strong></li>
            <li>Train or promote a model, then use this page to launch the match.</li>
          </ol>
          <p class="subtle">The bot loads the stable checkpoint unless a different model is selected. The match is created with a human on blue and the policy on orange.</p>
        </div>
      </section>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Adapter verification</h2><p>What has and has not been tested.</p></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Check</th><th>Status</th><th>Evidence</th></tr></thead>
        <tbody>
          <tr><td><strong>Observation parity</strong></td><td><span class="state-chip stable">verified</span></td><td>Simulator state → GamePacket → observation matches the training observation (max diff &lt; 3e-7 over 1v1, 3v3 and Heatseeker).</td></tr>
          <tr><td><strong>Action mapping</strong></td><td><span class="state-chip stable">verified</span></td><td>All 90 actions map to valid controller values; every index is asserted in the test suite.</td></tr>
          <tr><td><strong>Model loading</strong></td><td><span class="state-chip stable">verified</span></td><td>Checkpoint payload loads and runs inference.</td></tr>
          <tr><td><strong>Bot process in RLBot</strong></td><td><span class="state-chip untested">not run</span></td><td>Requires Windows, Rocket League and RLBot v5. Not available in the build sandbox.</td></tr>
        </tbody>
      </table></div>
    </section>`;

  const renderChecks = (report) => {
    const node = $("#live-checks");
    if (!node) return;
    node.innerHTML = report.checks.map((check) => `
      <div class="metadata-row"><span>${esc(check.name)}${check.required ? "" : " (optional)"}</span>
      <strong>${check.ok ? "ok" : "missing"} · ${esc(check.detail)}</strong></div>`).join("");
    toast(report.ok ? "Required checks passed" : "Required checks failed — see details", report.ok ? "info" : "error");
  };
  $("#live-checks-run")?.addEventListener("click", async () => {
    try {
      renderChecks(await api("/api/live/preflight"));
    } catch (error) {
      toast(error.message, "error");
    }
  });

  $("#live-launch")?.addEventListener("click", async () => {
    try {
      const result = await api("/api/live/launch", {
        method: "POST",
        body: JSON.stringify({ model_id: $("#live-model").value, mode: $("#live-mode").value }),
      });
      toast(`Match launched · pid ${result.pid} · log ${result.log}`);
    } catch (error) {
      toast(error.message, "error");
    }
  });

  $("#live-prepare")?.addEventListener("click", async () => {
    try {
      const result = await api("/api/live/launch", {
        method: "POST",
        body: JSON.stringify({ model_id: $("#live-model").value, mode: $("#live-mode").value, dry_run: true }),
      });
      toast(`Files prepared · ${result.directory}`);
    } catch (error) {
      toast(error.message, "error");
    }
  });
};

// ---------------------------------------------------------------- settings
views.settings = async () => {
  const [diagnostics, system] = await Promise.all([api("/api/diagnostics"), api("/api/system")]);
  const info = diagnostics.info;
  $("#view").innerHTML = `
    <div class="page-heading"><div><span class="eyebrow">Runtime</span><h1>Settings &amp; diagnostics</h1>
    <p>Detected hardware, installed versions and measured simulator throughput.</p></div></div>

    <div class="callout ${diagnostics.ok ? "success" : "warning"}">
      <div class="callout-copy"><strong>${diagnostics.ok ? "Required dependencies are present" : "Required dependencies are missing"}</strong>
      <p>This panel is generated from the running Python environment. Python ${esc(info.python)} on ${esc(info.platform)}.</p></div>
    </div>

    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label"><span>CPU</span>${icon("cpu")}</div><div class="stat-value">${info.cpu_count}</div><div class="stat-foot">${info.cpu_physical || "?"} physical cores</div></div>
      <div class="stat-card"><div class="stat-label"><span>Memory</span>${icon("layers")}</div><div class="stat-value">${info.memory_total_gb}<small>GB</small></div><div class="stat-foot">${info.memory_available_gb} GB available</div></div>
      <div class="stat-card"><div class="stat-label"><span>Device</span>${icon("bolt")}</div><div class="stat-value" style="font-size:22px">${info.torch?.cuda_available ? "CUDA" : "CPU"}</div><div class="stat-foot">${info.torch?.cuda_available ? esc(info.torch.devices?.[0] || "") : "GPU is optional"}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Disk free</span>${icon("database")}</div><div class="stat-value">${info.disk_free_gb}<small>GB</small></div><div class="stat-foot">${esc(info.data_dir)}</div></div>
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Dependency checks</h2><p>Required items must pass before training.</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead>
        <tbody>${diagnostics.checks.map((check) => `<tr>
          <td><strong>${esc(check.name)}</strong></td>
          <td>${check.ok ? `<span class="state-chip stable">ok</span>` : check.required ? `<span class="state-chip rejected">missing</span>` : `<span class="state-chip">optional</span>`}</td>
          <td>${esc(check.detail || "")}</td></tr>`).join("")}</tbody></table></div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Throughput benchmark</h2><p>Measured now, not estimated.</p></div></div>
        <div class="form-grid">
          <label class="field"><span>Mode</span><select id="bench-mode">${(app.cache.modes || await api("/api/modes")).map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("")}</select></label>
          <label class="field"><span>Workers</span><input id="bench-workers" type="number" min="1" max="8" value="${Math.max(1, Math.min(4, (info.cpu_count || 2) - 1))}" /></label>
          <label class="field"><span>Seconds</span><input id="bench-seconds" type="number" min="0.5" max="10" step="0.5" value="3" /></label>
        </div>
        <div class="config-actions"><button class="button secondary" id="bench-run">${icon("gauge")} Measure</button></div>
        <div class="panel-body" id="bench-result"><p class="subtle">No benchmark has been run in this session.</p></div>
      </section>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Environment</h2><p>Paths and package versions reported by the backend.</p></div></div>
      <div class="two-col" style="padding:0 20px 18px;margin:0">
        <div class="metadata-list" style="padding:0">
          <div class="metadata-row"><span>Project</span><strong>${esc(info.project_root)}</strong></div>
          <div class="metadata-row"><span>Data directory</span><strong>${esc(info.data_dir)}</strong></div>
          <div class="metadata-row"><span>Python</span><strong>${esc(info.python)}</strong></div>
          <div class="metadata-row"><span>Executable</span><strong>${esc(info.executable)}</strong></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>Package</th><th>Version</th></tr></thead><tbody>
          ${Object.entries(info.packages).map(([name, version]) => `<tr><td>${esc(name)}</td><td>${version ? esc(version) : `<span class="muted">not installed</span>`}</td></tr>`).join("")}
        </tbody></table></div>
      </div>
    </section>`;

  $("#bench-run").addEventListener("click", async () => {
    const button = $("#bench-run");
    button.disabled = true;
    $("#bench-result").innerHTML = `<div class="loading-state" style="min-height:80px"><span class="spinner"></span> Measuring…</div>`;
    try {
      const result = await api("/api/benchmark", {
        method: "POST",
        body: JSON.stringify({
          mode: $("#bench-mode").value,
          workers: Number($("#bench-workers").value),
          seconds: Number($("#bench-seconds").value),
        }),
      });
      $("#bench-result").innerHTML = `<div class="metadata-list" style="padding:0">
        <div class="metadata-row"><span>Environment steps / s</span><strong>${fmtNumber(result.env_steps_per_s)}</strong></div>
        <div class="metadata-row"><span>Simulator ticks / s</span><strong>${fmtNumber(result.sim_ticks_per_s)}</strong></div>
        <div class="metadata-row"><span>Workers</span><strong>${result.workers}</strong></div>
        <div class="metadata-row"><span>Measured seconds</span><strong>${result.seconds.toFixed(2)}</strong></div>
      </div>`;
    } catch (error) {
      $("#bench-result").innerHTML = `<div class="callout error"><div class="callout-copy"><strong>Benchmark failed</strong><p>${esc(error.message)}</p></div></div>`;
    } finally {
      button.disabled = false;
    }
  });
};

// ---------------------------------------------------------------- bootstrap
async function refreshConnection() {
  try {
    const [health, models, system] = await Promise.all([
      api("/api/health"),
      api("/api/models"),
      api("/api/system"),
    ]);
    const connection = $("#connection");
    connection.className = "connection online";
    connection.innerHTML = `<i></i><span>API ${esc(health.version)}</span>`;
    $("#connection-dot").style.background = "var(--accent)";
    $("#model-count").textContent = models.length;
    const active = system.active_run;
    $("#workspace-mode").textContent = active ? `${active.mode} · ${active.status}` : "no active run";
    $("#footer-status").textContent = `API · ${health.version} · ${new Date().toLocaleTimeString()}`;
  } catch {
    $("#connection").className = "connection offline";
    $("#connection").innerHTML = `<i></i><span>API offline</span>`;
    $("#connection-dot").style.background = "var(--danger)";
    $("#footer-status").textContent = "API · offline";
  }
}

$("#refresh-button").addEventListener("click", () => {
  refreshConnection();
  render();
});

window.addEventListener("hashchange", render);
window.addEventListener("DOMContentLoaded", async () => {
  await refreshConnection();
  setInterval(refreshConnection, 10000);
  render();
});
