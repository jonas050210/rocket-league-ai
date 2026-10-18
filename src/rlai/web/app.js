/* Rocket League AI — local control room. Every value comes from the API. */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const icon = (name, cls = "icon") =>
  `<svg class="${cls}"><use href="/theme/icons.svg#icon-${name}"></use></svg>`;

/** localStorage can be unavailable (file:// origins, private mode); never crash on it. */
const localStore = {
  get(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* ignored: the token then lives only for this page load */
    }
  },
  remove(key) {
    try {
      window.localStorage.removeItem(key);
    } catch {
      /* ignored */
    }
  },
};

const app = {
  route: "dashboard",
  cleanup: null,
  eventSource: null,
  timers: [],
  cache: {},
  token: localStore.get("rlai.token") || "",
};

// ---------------------------------------------------------------- utilities
function setToken(value) {
  app.token = String(value ?? "").trim();
  if (app.token) localStore.set("rlai.token", app.token);
  else localStore.remove("rlai.token");
}

async function api(path, options = {}) {
  const { headers: extra, ...rest } = options;
  const headers = { "Content-Type": "application/json", ...(extra || {}) };
  if (app.token) headers["x-rlai-token"] = app.token;
  const response = await fetch(path, { ...rest, headers });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (response.status === 401) {
    toast("The API needs an access token — open Settings and paste it there.", "error");
    throw new Error("unauthorized (401): missing or wrong API token");
  }
  if (!response.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : response.statusText;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

/** EventSource cannot send headers, so the token travels as a query parameter. */
function eventUrl(path, params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  });
  if (app.token) search.set("token", app.token);
  const query = search.toString();
  return query ? `${path}?${query}` : path;
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

/** Small labelled chip used for compatibility / protocol information. */
function chip(text, kind = "") {
  return `<span class="state-chip ${kind}">${esc(text)}</span>`;
}

/**
 * Render a value that may be missing. Reports use {available:false, reason}
 * for anything that was never measured; those must be labelled, never guessed.
 */
function measured(value, render = (v) => esc(v), fallback = "not measured") {
  if (value === null || value === undefined || value === "") {
    return `<span class="muted">${esc(fallback)}</span>`;
  }
  if (typeof value === "object" && value.available === false) {
    return `<span class="muted" title="${esc(value.reason || "")}">unavailable${value.reason ? ` · ${esc(value.reason)}` : ""}</span>`;
  }
  return render(value);
}

function reasonList(title, items) {
  if (!items || !items.length) return "";
  return `<div class="reason-list"><strong>${esc(title)}</strong><ul>${items
    .map((item) => `<li>${esc(typeof item === "string" ? item : JSON.stringify(item))}</li>`)
    .join("")}</ul></div>`;
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
  const label = { dashboard: "Dashboard", training: "Training", monitoring: "Monitoring", reports: "Reports", report: "Run report", models: "Models", evaluation: "Evaluation", live: "Live match", settings: "Settings" }[route] || "Dashboard";
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
  const [modes, defaults, runs, models] = await Promise.all([
    api("/api/modes"),
    api("/api/config/defaults"),
    api("/api/runs?limit=20"),
    api("/api/models"),
  ]);
  app.cache.modes = modes;
  app.cache.defaults = defaults;
  app.cache.models = models;
  const config = defaults.training;
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const resumeFrom = params.get("resume") || "";
  const resumeMode = params.get("mode") || (resumeFrom ? "continue" : "fresh");
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

    <div class="callout success">
      <div class="callout-copy"><strong>Starting a run is explicit</strong>
      <p>Nothing starts, resumes or is scheduled by itself — not on launch, not after a crash, not when a checkpoint appears. Every run below starts only when you press the button and confirm.</p></div>
    </div>

    <section class="panel" style="margin-bottom:18px">
      <div class="panel-header"><div><h2>Presets</h2><p>Staged recipes: ball control → 1v1 → teams → Heatseeker.</p></div></div>
      <div class="preset-grid" style="padding:0 20px 18px">${presets}</div>
    </section>

    <form id="training-form" class="config-layout">
      <section class="panel">
        <div class="panel-header"><div><h2>Run configuration</h2><p>Values map 1:1 to the server-side schema.</p></div><span class="state-chip" id="form-mode-chip">${esc(config.mode)}</span></div>

        <div class="resume-block">
          <div class="resume-head">
            <div><h3>Continuation</h3><p>Pick explicitly how this run starts. The plan is validated against the checkpoint before anything is restored.</p></div>
            <span class="state-chip" id="resume-chip">${esc(resumeMode)}</span>
          </div>
          <div class="form-grid">
            <label class="field"><span>Start from</span>
              <select name="resume_mode">
                <option value="fresh" ${resumeMode === "fresh" ? "selected" : ""}>Fresh — new random policy, nothing restored</option>
                <option value="continue" ${resumeMode === "continue" ? "selected" : ""}>Continue — weights, optimiser, step, RNG, curriculum</option>
                <option value="transfer" ${resumeMode === "transfer" ? "selected" : ""}>Transfer — weights only, step counter restarts at 0</option>
              </select>
              <small>Continue keeps the step budget; transfer fine-tunes an existing policy on a new task.</small>
            </label>
            <label class="field"><span>Checkpoint</span>
              <select name="resume_from" ${models.length ? "" : "disabled"}>
                <option value="">Automatic — newest valid, compatible checkpoint of this lineage</option>
                ${models.map((model) => `<option value="${esc(model.id)}" ${model.id === resumeFrom ? "selected" : ""}>${esc(model.id)} · step ${fmtNumber(model.step)} · ${esc(model.mode || "?")}${model.stable ? " · stable" : ""}${model.best ? " · best" : ""}</option>`).join("")}
              </select>
              <small id="resume-from-hint">${models.length ? "Automatic selection refuses an incompatible checkpoint instead of starting from scratch." : "No checkpoints exist yet, so only fresh training is possible."}</small>
            </label>
          </div>
          <div class="resume-actions">
            <button class="button ghost" type="button" id="plan-button">${icon("shield")} Preview what would be restored</button>
            <label class="field approval-check inline"><input type="checkbox" name="allow_concurrent" /><span>Allow a second run next to an active one (they share the CPU)</span></label>
          </div>
          <div id="resume-plan">${models.length || resumeMode !== "fresh" ? "" : `<p class="subtle">Fresh training ignores every checkpoint.</p>`}</div>
        </div>

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
          <label class="field"><span>Evaluation opponent</span><select name="eval_opponent">${["chase", "random", "idle", "self"].map((o) => `<option ${o === config.eval_opponent ? "selected" : ""}>${o}</option>`).join("")}</select><small>idle/random are diagnostic baselines, not strength measures.</small></label>
          <label class="field"><span>Evaluation suite (optional)</span><input name="eval_opponents" value="${esc((config.eval_opponents || []).join(","))}" placeholder="chase,mixed,self" /><small>Comma-separated. Strength opponents are reported separately from diagnostics.</small></label>
          <label class="field"><span>Evaluation sides</span><select name="eval_sides">${[["blue", "blue only"], ["orange", "orange only"], ["blue,orange", "both (side-balanced)"]].map(([value, label]) => `<option value="${value}" ${(config.eval_sides || ["blue"]).join(",") === value ? "selected" : ""}>${label}</option>`).join("")}</select><small>Both sides alternate per game and mirror the observation.</small></label>
          <label class="field full approval-check"><input type="checkbox" name="eval_mirror" ${config.eval_mirror === false ? "" : "checked"} /><span>Mirror the observation when the policy plays orange (car-frame features, verified equivalent).</span></label>
          <label class="field full approval-check"><input type="checkbox" name="eval_before" ${config.eval_before === false ? "" : "checked"} /><span>Measure a baseline evaluation before training so before/after is comparable.</span></label>
          <label class="field full approval-check"><input type="checkbox" name="side_randomization" ${config.side_randomization ? "checked" : ""} /><span>Randomise which side the learner plays during training rollouts.</span></label>
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
            <label class="field"><span>Store pool weights in checkpoints</span><select name="league_states">${[[true, "yes — a resume keeps the real opponents"], [false, "no — ratings only (smaller files)"]].map(([value, label]) => `<option value="${value}" ${Boolean(config.league_states) === value ? "selected" : ""}>${label}</option>`).join("")}</select><small>About one policy size per snapshot (max 8).</small></label>
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
      league_states: data.get("league_states") === "true",
      eval_interval: Number(data.get("eval_interval")),
      eval_games: Number(data.get("eval_games")),
      eval_match_seconds: Number(data.get("eval_match_seconds")),
      eval_opponent: data.get("eval_opponent"),
      eval_opponents: splitList(data.get("eval_opponents")),
      eval_sides: splitList(data.get("eval_sides")).length ? splitList(data.get("eval_sides")) : ["blue"],
      eval_mirror: data.get("eval_mirror") === "on",
      eval_before: data.get("eval_before") === "on",
      side_randomization: data.get("side_randomization") === "on",
      self_play: data.get("self_play") === "on",
      curriculum: data.get("curriculum") === "on",
      reward,
    };
  };

  const resumeFromForm = () => {
    const data = new FormData(form);
    const mode = data.get("resume_mode") || "fresh";
    const reference = mode === "fresh" ? "" : String(data.get("resume_from") || "");
    return { mode, reference, allowConcurrent: data.get("allow_concurrent") === "on" };
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

  const planPreview = async () => {
    const node = $("#resume-plan");
    const { mode, reference } = resumeFromForm();
    if (mode === "fresh") {
      node.innerHTML = `<p class="subtle">Fresh training: no checkpoint is read, no state is restored. Existing checkpoints stay untouched.</p>`;
      return;
    }
    node.innerHTML = `<div class="loading-state" style="min-height:60px"><span class="spinner"></span> Validating checkpoints…</div>`;
    try {
      const plan = await api("/api/resume/plan", {
        method: "POST",
        body: JSON.stringify({ config: configFromForm(), mode, checkpoint: reference || null }),
      });
      node.innerHTML = renderResumePlan(plan);
    } catch (error) {
      node.innerHTML = `<div class="callout error"><div class="callout-copy"><strong>Plan failed</strong><p>${esc(error.message)}</p></div></div>`;
    }
  };

  $("#plan-button").addEventListener("click", planPreview);
  form.elements.resume_mode.addEventListener("change", () => {
    const { mode } = resumeFromForm();
    $("#resume-chip").textContent = mode;
    form.elements.resume_from.disabled = mode === "fresh" || !models.length;
    planPreview();
  });
  form.elements.resume_from.addEventListener("change", planPreview);
  if (resumeMode !== "fresh") planPreview();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submit = form.querySelector('button[type="submit"]');
    const payload = configFromForm();
    const { mode, reference, allowConcurrent } = resumeFromForm();
    if (mode !== "fresh") {
      payload.resume_mode = mode;
      if (reference) payload.resume_from = reference;
    } else {
      payload.resume_mode = "fresh";
    }
    payload.allow_concurrent = allowConcurrent;
    const summary = [
      `Mode ${payload.mode} · ${fmtNumber(payload.total_steps)} steps · ${payload.n_workers} worker(s) · device ${payload.device}.`,
      mode === "fresh"
        ? "Fresh training: no checkpoint is restored."
        : `${mode === "continue" ? "Continue" : "Transfer"} from ${reference || "the newest compatible checkpoint"}.`,
      allowConcurrent ? "Concurrent runs are allowed; they will compete for CPU." : "",
      "Training starts immediately and keeps running until it is stopped or the budget is reached.",
    ].filter(Boolean).join(" ");
    confirmAction(`Start ${mode} training`, summary, async () => {
      submit.disabled = true;
      try {
        const run = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
        toast(`Run ${run.id} started (${mode})`);
        location.hash = `#/monitoring?run=${encodeURIComponent(run.id)}`;
      } catch (error) {
        toast(error.message, "error");
        submit.disabled = false;
      }
    });
  });
};

function splitList(value) {
  return String(value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

/** Render a resume plan exactly as the server computed it — no invented state. */
function renderResumePlan(plan) {
  if (!plan) return "";
  if (plan.mode === "fresh" || !plan.will_resume) {
    return `<div class="callout ${plan.explicit_unavailable ? "error" : "warning"}">
      <div class="callout-copy">
        <strong>${plan.will_resume ? "Resume planned" : plan.explicit_unavailable ? "This checkpoint cannot be used" : "Nothing to resume from"}</strong>
        <p>${esc(plan.explicit_unavailable ? "The requested checkpoint was rejected. No other checkpoint is substituted for an explicit request." : "Training would start from scratch with a fresh policy.")}</p>
        ${reasonList("Warnings", plan.warnings)}
        ${(plan.rejected || []).length ? `<div class="table-wrap"><table class="compact">
          <thead><tr><th>Rejected checkpoint</th><th>Step</th><th>Reason</th></tr></thead>
          <tbody>${plan.rejected.map((item) => `<tr>
            <td><strong>${esc(item.id || item.absolute_path || "?")}</strong></td>
            <td>${fmtNumber(item.step)}</td>
            <td>${(item.reasons || []).map((reason) => esc(reason)).join("<br>")}</td>
          </tr>`).join("")}</tbody></table></div>` : ""}
      </div></div>`;
  }
  return `<div class="callout success">
    <div class="callout-copy">
      <strong>${plan.mode === "transfer" ? "Transfer from" : "Continue from"} ${esc(plan.model_id || plan.checkpoint_path)}</strong>
      <p>Step counter ${plan.mode === "transfer" ? `restarts at 0 (parent was at ${fmtNumber(plan.parent_step)})` : `restored at ${fmtNumber(plan.restored_step)}`} · parent run ${esc(plan.parent_run || "unknown")}${plan.fallback_used ? " · fallback after a rejected checkpoint" : ""}.</p>
      <div class="chip-row">${(plan.restores || []).map((item) => chip(item, "stable")).join("")}</div>
      ${(plan.missing || []).length ? `<p class="subtle">Not stored in this checkpoint: ${plan.missing.map((item) => esc(item)).join(", ")}.</p>` : ""}
      ${reasonList("Warnings", plan.warnings)}
      ${(plan.overrides || []).length ? `<div class="table-wrap"><table class="compact">
        <thead><tr><th>Config field</th><th>Checkpoint</th><th>This form</th><th>Used</th></tr></thead>
        <tbody>${plan.overrides.map((item) => `<tr>
          <td><strong>${esc(item.field)}</strong></td>
          <td>${esc(JSON.stringify(item.checkpoint))}</td>
          <td>${esc(JSON.stringify(item.requested))}</td>
          <td>${item.error ? `<span class="state-chip rejected">${esc(item.error)}</span>` : esc(JSON.stringify(item.used))}</td>
        </tr>`).join("")}</tbody></table></div>
        <p class="subtle">Inherited from the checkpoint: ${(plan.inherited || []).map((name) => esc(name)).join(", ") || "nothing"}.</p>` : ""}
    </div></div>`;
}

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
    } else if (key === "eval_sides" || key === "eval_opponents") {
      assign(key, Array.isArray(value) ? value.join(",") : value);
    } else if (key === "league_states") {
      assign(key, String(Boolean(value)));
    } else if (key === "resume_mode" || key === "resume_from" || key === "allow_concurrent") {
      /* continuation is chosen by the user, never by a preset */
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
  const latestOf = (name) => {
    const rows = metrics.filter((m) => m.name === name);
    return rows.length ? rows[rows.length - 1].value : null;
  };

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Telemetry</span><h1>Monitoring</h1><p>Live metrics and logs from the training subprocess.</p></div>
      <div class="heading-actions">
        <select id="run-select" class="log-search" style="width:auto">
          ${runs.map((run) => `<option value="${esc(run.id)}" ${run.id === runId ? "selected" : ""}>${esc(run.name)} · ${esc(run.id)}</option>`).join("")}
        </select>
        <a class="button ghost" href="#/report?run=${encodeURIComponent(runId)}">${icon("file")} Report</a>
        <a class="button ghost" href="#/reports">${icon("chart")} All reports</a>
      </div>
    </div>

    ${detail.resumed_step || detail.parent_run ? `<div class="callout success">
      <div class="callout-copy"><strong>Continuation of ${esc(detail.parent_run || "an earlier run")}</strong>
      <p>Step counter was restored at ${fmtNumber(detail.resumed_step)}; the numbers below and in the report count only what this process did (${fmtNumber(Math.max(0, Number(detail.step || 0) - Number(detail.resumed_step || 0)))} steps so far).</p></div>
    </div>` : ""}

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
      <div class="panel-header"><div><h2>Iteration cost</h2>
      <p>Measured separately per phase: simulation and inference happen in the worker processes, the PPO update in the learner.</p></div>
      <span class="pill muted-pill" id="timing-step">step ${fmtNumber(detail.step)}</span></div>
      <div class="metadata-list">
        ${TIMING_ROWS.map(([name, label]) => `<div class="metadata-row"><span>${esc(label)}</span>
          <strong id="timing-${timingSlug(name)}">${latestOf(name) === null ? "not measured yet" : `${Number(latestOf(name)).toFixed(4)} s`}</strong></div>`).join("")}
        <div class="metadata-row"><span>Curriculum stage</span><strong>${esc(latestOf("curriculum/stage") === null ? (cfg.curriculum ? "not reported yet" : "disabled") : `stage ${Math.round(latestOf("curriculum/stage"))}`)}</strong></div>
        <div class="metadata-row"><span>Opponent pool</span><strong>${esc(latestOf("league/pool_size") === null ? (cfg.self_play ? "empty" : "not used") : `${Math.round(latestOf("league/pool_size"))} snapshot(s)`)}</strong></div>
      </div>
      <p class="report-note">The report after the run repeats these numbers as mean/min/max over all iterations, together with wall-clock steps/s.</p>
    </section>

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

  app.eventSource = new EventSource(eventUrl("/api/events", { run_id: runId }));
  app.eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "log") {
      allLogs.push(payload);
      paintLogs();
    } else if (payload.type === "metrics") {
      if (payload.run) {
        $("#live-sps").textContent = fmtNumber(payload.metrics["time/sps"] || 0);
        const stepNode = $("#timing-step");
        if (stepNode) stepNode.textContent = `step ${fmtNumber(payload.run.step || 0)}`;
        TIMING_ROWS.forEach(([name]) => {
          const node = $(`#timing-${timingSlug(name)}`);
          const value = payload.metrics[name];
          if (node && value !== undefined && value !== null) node.textContent = `${Number(value).toFixed(4)} s`;
        });
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

const TIMING_ROWS = [
  ["time/collect_seconds", "Collect (rollout)"],
  ["time/sim_seconds", "— of which simulation"],
  ["time/infer_seconds", "— of which policy inference"],
  ["time/update_seconds", "PPO update"],
  ["time/eval_seconds", "Evaluation"],
  ["time/checkpoint_seconds", "Checkpointing"],
  ["time/iteration_seconds", "Whole iteration"],
];

const timingSlug = (name) => String(name).replace(/[^a-z0-9]+/gi, "-");

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

// ---------------------------------------------------------------- reports
const REPORT_CURVES = [
  ["charts/episodic_return", "Episodic return", "var(--accent)"],
  ["charts/episode_length", "Episode length", "var(--info)"],
  ["losses/policy", "Policy loss", "var(--accent)"],
  ["losses/value", "Value loss", "var(--info)"],
  ["losses/approx_kl", "Approx KL", "var(--warning, #d9a13b)"],
  ["losses/entropy", "Entropy", "var(--success, #4caf7d)"],
  ["losses/explained_variance", "Explained variance", "var(--info)"],
  ["eval/win_rate", "Evaluation win rate", "var(--accent)"],
  ["time/sps", "Steps / s per iteration", "var(--info)"],
];

function statRow(label, value) {
  return `<div class="metadata-row"><span>${esc(label)}</span><strong>${value}</strong></div>`;
}

function num(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

/** A metric block from the report: labelled, or explicitly unavailable. */
function metricCells(entry, keys) {
  if (!entry || entry.available === false) {
    return `<td colspan="${keys.length}" class="muted">${esc((entry && entry.reason) || "not measured")}</td>`;
  }
  return keys.map((key) => `<td>${key === "count" ? fmtNumber(entry[key]) : num(entry[key], 6)}</td>`).join("");
}

function protocolChips(item) {
  if (!item) return "";
  const protocol = item.protocol || {};
  const sides = (item.sides || protocol.sides || []).join("+") || "blue";
  return [
    chip(`${item.games ?? "?"} games`, ""),
    chip(`sides ${sides}`, ""),
    chip(item.mirror === false ? "no mirror" : "mirrored", ""),
    chip(`seed ${protocol.seed ?? item.seed ?? "?"}`, ""),
    chip(`${protocol.match_seconds ?? "?"} s matches`, ""),
    protocol.stochastic === undefined ? "" : chip(protocol.stochastic ? "stochastic" : "deterministic", ""),
    protocol.obs_version ? chip(`obs v${protocol.obs_version}`, "") : "",
  ].filter(Boolean).join("");
}

function evaluationCard(title, item) {
  if (!item) return `<div class="report-block"><strong>${esc(title)}</strong><p class="muted">not measured</p></div>`;
  if (item.available === false) {
    return `<div class="report-block"><strong>${esc(title)}</strong><p class="muted">${esc(item.reason || "unavailable")}</p></div>`;
  }
  const ci = item.win_rate_ci95;
  return `<div class="report-block">
    <strong>${esc(title)}</strong>
    <div class="metadata-list">
      ${statRow("Opponent", esc(item.opponent || "—"))}
      ${statRow("Record", `${item.wins ?? 0} W · ${item.losses ?? 0} L · ${item.draws ?? 0} D`)}
      ${statRow("Win rate", `${pct(item.win_rate)}${ci ? ` <small>95 % CI ${pct(ci[0])} – ${pct(ci[1])}</small>` : " <small>no interval</small>"}`)}
      ${statRow("Decisive win rate", pct(item.win_rate_decisive))}
      ${statRow("Goals", `${item.goals_for ?? 0} : ${item.goals_against ?? 0}`)}
      ${statRow("Model", esc(item.model_id || "—"))}
      ${statRow("Duration", item.duration_s ? fmtDuration(item.duration_s) : "—")}
    </div>
    <div class="chip-row">${protocolChips(item)}</div>
    ${item.complete === false ? `<p class="muted" style="margin-top:6px">Incomplete: no game finished.</p>` : ""}
  </div>`;
}

function renderEvaluation(section) {
  if (!section) return "";
  const delta = section.delta || {};
  const periodic = section.periodic || [];
  return `
    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Evaluation (measured playing strength)</h2>
      <p>Real match series against fixed opponents under a recorded protocol. Training reward is not evidence of strength.</p></div></div>
      <div class="report-grid">
        ${evaluationCard("Before training", section.before)}
        ${evaluationCard("After training", section.after)}
        <div class="report-block"><strong>Change</strong>
          ${delta.available === false
            ? `<p class="muted">${esc(delta.reason || "unavailable")}</p>`
            : `<div class="metadata-list">
                ${statRow("Win rate before", pct(delta.win_rate_before))}
                ${statRow("Win rate after", pct(delta.win_rate_after))}
                ${statRow("Change", `${delta.win_rate_change === null || delta.win_rate_change === undefined ? "—" : `${delta.win_rate_change > 0 ? "+" : ""}${(delta.win_rate_change * 100).toFixed(1)} pp`}`)}
                ${statRow("Goal difference", `${delta.goal_diff_before ?? "—"} → ${delta.goal_diff_after ?? "—"}`)}
                ${statRow("Same protocol", delta.comparable_protocol ? "yes" : "no")}
                ${statRow("Same kickoff seed", delta.same_seed === undefined ? "—" : delta.same_seed ? "yes" : "no")}
              </div>
              ${delta.warning ? `<p class="muted" style="margin-top:6px">${esc(delta.warning)}</p>` : ""}
              ${delta.note ? `<p class="muted" style="margin-top:6px">${esc(delta.note)}</p>` : ""}`}
        </div>
        <div class="report-block"><strong>Best model pointer</strong>
          ${section.best && section.best.available !== false
            ? `<div class="metadata-list">
                ${statRow("Model", esc(section.best.model_id || section.best.id || "—"))}
                ${statRow("Win rate", pct(section.best.win_rate))}
                ${statRow("Opponent", esc(section.best.opponent || "—"))}
                ${statRow("Protocol", esc(section.best.protocol || "—"))}
              </div>`
            : `<p class="muted">${esc((section.best && section.best.reason) || "no best-model pointer has been set")}</p>`}
        </div>
      </div>
      ${(section.pairs || []).length ? `<div class="table-wrap" style="padding:0 20px 6px"><table class="compact">
        <thead><tr><th>Opponent</th><th>Class</th><th>Before</th><th>After</th><th>Win-rate change</th><th>Goal diff</th><th>Seeds</th></tr></thead>
        <tbody>${section.pairs.map((pair) => {
          const d = pair.delta;
          const change = d.win_rate_change === null || d.win_rate_change === undefined
            ? "—" : `${d.win_rate_change > 0 ? "+" : ""}${(d.win_rate_change * 100).toFixed(1)} pp`;
          return `<tr>
            <td><strong>${esc(d.opponent)}</strong></td>
            <td>${chip(d.opponent_class, d.opponent_class === "strength" ? "stable" : "")}</td>
            <td>${d.wins_before ?? 0}W · ${d.games_before ?? 0} games · ${pct(d.win_rate_before)}</td>
            <td>${d.wins_after ?? 0}W · ${d.games_after ?? 0} games · ${pct(d.win_rate_after)}</td>
            <td class="table-strong">${change}</td>
            <td>${d.goal_diff_before ?? "—"} → ${d.goal_diff_after ?? "—"}</td>
            <td>${d.same_seed ? "identical" : `${esc(d.seed_before)} → ${esc(d.seed_after)}`}</td>
          </tr>`;
        }).join("")}</tbody></table>
        <p class="report-note" style="padding:0">Each row pairs the baseline and the final series for one opponent under an identical protocol. Diagnostic opponents (idle, random) prove the pipeline works; only strength opponents say something about football.</p></div>` : ""}

      ${periodic.length ? `<div class="table-wrap" style="padding:0 20px 18px"><table class="compact">
        <thead><tr><th>When</th><th>Opponent</th><th>Record</th><th>Win rate (95 % CI)</th><th>Goals</th><th>Protocol</th></tr></thead>
        <tbody>${periodic.slice().reverse().map((item) => `<tr>
          <td>${esc((item.created_at || "").replace("T", " ").slice(0, 19))}</td>
          <td>${esc(item.opponent || "—")}</td>
          <td class="table-strong">${item.wins ?? 0}W · ${item.losses ?? 0}L · ${item.draws ?? 0}D</td>
          <td>${pct(item.win_rate)}${item.win_rate_ci95 ? ` <small>${pct(item.win_rate_ci95[0])} – ${pct(item.win_rate_ci95[1])}</small>` : ""}</td>
          <td>${item.goals_for ?? 0} : ${item.goals_against ?? 0}</td>
          <td><div class="chip-row">${protocolChips(item)}</div></td>
        </tr>`).join("")}</tbody></table></div>` : ""}
      ${(section.incomplete || []).length ? `<div class="callout warning" style="margin:0 20px 18px"><div class="callout-copy">
        <strong>${section.incomplete.length} evaluation(s) incomplete</strong>
        <p>${section.incomplete.map((item) => esc(`${item.opponent || "?"} @ ${(item.created_at || "").slice(0, 19)}`)).join(", ")} — no game finished, so no win rate can be reported.</p>
      </div></div>` : ""}
    </section>`;
}

function renderEpisodes(episodes) {
  const block = (title, stats) => {
    if (!stats || stats.available === false) {
      return `<div class="report-block"><strong>${esc(title)}</strong><p class="muted">${esc((stats && stats.reason) || "no episodes recorded")}</p></div>`;
    }
    return `<div class="report-block"><strong>${esc(title)}</strong>
      <div class="metadata-list">
        ${statRow("Episodes", fmtNumber(stats.count))}
        ${statRow("Mean reward", num(stats.mean_reward, 3))}
        ${statRow("Reward range", `${num(stats.min_reward, 3)} … ${num(stats.max_reward, 3)}`)}
        ${statRow("Mean length", `${num(stats.mean_length, 1)} steps`)}
        ${statRow("Record", `${stats.wins ?? 0} W · ${stats.losses ?? 0} L · ${stats.draws ?? 0} D`)}
        ${statRow("Training win rate", pct(stats.win_rate))}
        ${statRow("Truncations", fmtNumber(stats.truncations))}
        ${statRow("Goals", `${stats.goals_for ?? 0} : ${stats.goals_against ?? 0}`)}
        ${statRow("Opponents", esc(Object.entries(stats.opponents || {}).map(([name, count]) => `${name} ×${count}`).join(", ") || "—"))}
      </div></div>`;
  };
  return `<section class="panel" style="margin-top:18px">
    <div class="panel-header"><div><h2>Episodes</h2><p>Rollout statistics from the training environment. These describe optimisation, not playing strength.</p></div></div>
    <div class="report-grid">
      ${block("This run", episodes.this_run)}
      ${block("All time (including parents)", episodes.all_time)}
      ${block("Last 100", episodes.last_100)}
    </div></section>`;
}

views.reports = async () => {
  const [reports, runs] = await Promise.all([
    api("/api/reports?limit=100"),
    api("/api/runs?limit=100"),
  ]);
  const byRun = {};
  reports.forEach((item) => { byRun[item.run_id] = item; });
  const rows = runs.map((run) => ({ run, report: byRun[run.id] }));

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Statistics</span><h1>Run reports</h1>
      <p>A report is written for every run that finishes, is stopped or fails — including the values that could not be measured, labelled as unavailable.</p></div>
      <div class="heading-actions"><a class="button ghost" href="#/training">${icon("training")} New run</a></div>
    </div>

    ${rows.length ? `<section class="panel">
      <div class="panel-header"><div><h2>Runs</h2><p>${reports.length} report(s) stored for ${runs.length} run(s).</p></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th></th><th>Run</th><th>Status</th><th>Mode</th><th>Steps</th><th>Duration</th><th>Steps / s</th><th>Final eval</th><th>Checkpoint</th><th></th></tr></thead>
        <tbody>${rows.map(({ run, report }) => {
          const summary = (report && report.summary) || {};
          const steps = summary.steps || {};
          const after = (summary.evaluation || {}).after || {};
          const checkpoint = summary.final_checkpoint || {};
          return `<tr>
            <td><input type="checkbox" class="compare-check" value="${esc(run.id)}" /></td>
            <td><strong>${esc(run.name)}</strong><small>${esc(run.id)}${run.parent_run ? ` · continues ${esc(run.parent_run)}` : ""}</small></td>
            <td>${pill(run.status)}</td>
            <td>${esc(run.mode)}</td>
            <td class="table-strong">${fmtNumber(steps.completed ?? run.step)}<small> of ${fmtNumber(steps.target ?? run.total_steps)}</small></td>
            <td>${report ? fmtDuration(summary.duration_s) : `<span class="muted">no report</span>`}</td>
            <td>${report ? num(summary.steps_per_second, 1) : "—"}</td>
            <td>${after && after.available !== false ? `${pct(after.win_rate)} <small>vs ${esc(after.opponent || "?")}</small>` : `<span class="muted">${esc((after && after.reason) || "not measured")}</span>`}</td>
            <td>${checkpoint && checkpoint.available !== false ? (checkpoint.valid ? chip("valid", "stable") : chip("invalid", "rejected")) : `<span class="muted">none</span>`}</td>
            <td class="actions">
              ${report
                ? `<a class="button ghost" href="#/report?run=${encodeURIComponent(run.id)}">${icon("file")} Report</a>`
                : `<button class="button ghost" data-rebuild="${esc(run.id)}">${icon("refresh")} Rebuild</button>`}
              <a class="button ghost" href="#/monitoring?run=${encodeURIComponent(run.id)}" title="Telemetry">${icon("chart")}</a>
            </td></tr>`;
        }).join("")}</tbody>
      </table></div>
      <div class="compare-picker">
        <button class="button secondary" id="compare-button">${icon("chart")} Compare selected</button>
        <span class="subtle">Tick two or more runs. Runs with different modes or observation layouts are flagged as not comparable.</span>
      </div>
      <div id="compare-result" style="padding:0 20px 18px"></div>
    </section>` : emptyState("No runs yet", "Start a training run; its report appears here when it ends, is stopped or fails.", "file")}

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>What a report contains</h2><p>Everything below comes from the run itself.</p></div></div>
      <div class="panel-body"><ul class="guide-list">
        <li>Steps at start and end, steps completed in this run, target and progress.</li>
        <li>Wall-clock duration and two throughput numbers: wall-clock steps/s and per-iteration steps/s (collect + update only).</li>
        <li>Device, worker count, platform and torch threads; the full configuration.</li>
        <li>Resume mode, parent run, parent checkpoint and the restored step.</li>
        <li>PPO losses, KL, entropy, clip fraction, explained variance and gradient norm (first/last/mean/min/max).</li>
        <li>Where the time went: collection, simulation, inference, PPO update, evaluation, checkpointing.</li>
        <li>Reward components, episode rewards and lengths, goals and truncations.</li>
        <li>Before/after evaluation with opponent, games, record, goals, win rate, 95 % CI and the exact protocol.</li>
        <li>Final checkpoint path, validity check and warnings; incomplete evaluations are listed, never hidden.</li>
        <li>Historical curves and comparisons with other runs that used the same mode and observation layout.</li>
      </ul></div>
    </section>`;

  $$("[data-rebuild]").forEach((button) => button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(`/api/runs/${encodeURIComponent(button.dataset.rebuild)}/report?rebuild=true`);
      toast("Report rebuilt from the recorded metrics");
      location.hash = `#/report?run=${encodeURIComponent(button.dataset.rebuild)}`;
    } catch (error) {
      toast(error.message, "error");
      button.disabled = false;
    }
  }));

  $("#compare-button")?.addEventListener("click", async () => {
    const selected = $$(".compare-check").filter((box) => box.checked).map((box) => box.value);
    const node = $("#compare-result");
    if (selected.length < 2) {
      toast("Select at least two runs to compare", "error");
      return;
    }
    node.innerHTML = `<div class="loading-state" style="min-height:60px"><span class="spinner"></span> Comparing…</div>`;
    try {
      node.innerHTML = renderComparison(await api("/api/runs/compare", {
        method: "POST",
        body: JSON.stringify({ run_ids: selected }),
      }));
    } catch (error) {
      node.innerHTML = `<div class="callout error"><div class="callout-copy"><strong>Comparison failed</strong><p>${esc(error.message)}</p></div></div>`;
    }
  });
};

function renderComparison(payload) {
  const runs = payload.runs || [];
  return `<div class="callout ${payload.comparable ? "success" : "warning"}" style="margin:0 0 12px">
      <div class="callout-copy"><strong>${payload.comparable ? "These runs are comparable" : "Not comparable"}</strong>
      <p>${esc(payload.warning || `Same mode (${(payload.modes || []).join(", ") || "unknown"}); differences come from configuration and hardware.`)}</p></div>
    </div>
    <div class="table-wrap"><table class="compact">
      <thead><tr><th>Run</th><th>Mode</th><th>Status</th><th>Steps</th><th>Duration</th><th>Steps / s</th><th>Win rate</th><th>Mean return</th><th>Parent</th></tr></thead>
      <tbody>${runs.map((item) => `<tr>
        <td><a href="#/report?run=${encodeURIComponent(item.run_id)}"><strong>${esc(item.name || item.run_id)}</strong></a><small>${esc(item.run_id)}</small></td>
        <td>${esc(item.mode || "—")}</td>
        <td>${item.status ? pill(item.status) : "—"}</td>
        <td>${fmtNumber(item.steps_completed ?? (item.steps || {}).completed)}<small> ${fmtNumber(item.steps_start ?? 0)} → ${fmtNumber(item.steps_end ?? 0)}</small></td>
        <td>${item.duration_s ? fmtDuration(item.duration_s) : "—"}</td>
        <td>${num(item.steps_per_second, 1)}</td>
        <td>${item.final_win_rate === null || item.final_win_rate === undefined ? `<span class="muted">not measured</span>` : pct(item.final_win_rate)}</td>
        <td>${num(item.mean_return, 3)}</td>
        <td>${esc(item.parent_run || "—")}</td>
      </tr>`).join("")}</tbody></table></div>`;
}

views.report = async () => {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const runs = await api("/api/runs?limit=100");
  if (!runs.length) {
    $("#view").innerHTML = `${emptyState("No runs yet", "A report is written when a training run ends.", "file")}`;
    return;
  }
  const runId = params.get("run") || runs[0].id;
  const rebuild = params.get("rebuild") === "1";
  let report = null;
  let failure = null;
  try {
    report = await api(`/api/runs/${encodeURIComponent(runId)}/report${rebuild ? "?rebuild=true" : ""}`);
  } catch (error) {
    failure = error;
  }
  const checkpoint = (report && report.final_checkpoint) || {};
  const throughput = (report && report.throughput) || {};
  const iteration = throughput.iteration_steps_per_second || {};
  const hardware = (report && report.hardware) || {};
  const resume = (report && report.resume) || {};
  const steps = (report && report.steps) || {};
  const timings = (report && report.timings) || {};
  const optimisation = (report && report.optimisation) || {};
  const rewardComponents = (report && report.reward_components) || {};
  const optimKeys = ["first", "last", "mean", "min", "max", "count"];
  const timingKeys = ["mean", "min", "max", "last", "count"];
  const curvePairs = [];
  for (let i = 0; i < REPORT_CURVES.length; i += 2) curvePairs.push(REPORT_CURVES.slice(i, i + 2));

  $("#view").innerHTML = `
    <div class="page-heading">
      <div><span class="eyebrow">Statistics</span><h1>${esc((report && report.name) || runId)}</h1>
      <p>${esc(runId)}${report ? ` · ${esc(report.mode || "?")} · started ${esc((report.started_at || "").replace("T", " ").slice(0, 19))}` : ""}</p></div>
      <div class="heading-actions">
        <select id="report-run-select" class="log-search" style="width:auto">
          ${runs.map((run) => `<option value="${esc(run.id)}" ${run.id === runId ? "selected" : ""}>${esc(run.name)} · ${esc(run.id)}</option>`).join("")}
        </select>
        <a class="button ghost" href="#/monitoring?run=${encodeURIComponent(runId)}">${icon("chart")} Telemetry</a>
        <a class="button ghost" href="/api/runs/${encodeURIComponent(runId)}/report" target="_blank" rel="noopener">${icon("file")} JSON</a>
      </div>
    </div>

    ${failure ? `<div class="callout error"><div class="callout-copy">
      <strong>No stored report</strong><p>${esc(failure.message)}</p></div>
      <a class="button secondary" href="#/report?run=${encodeURIComponent(runId)}&rebuild=1">${icon("refresh")} Rebuild from recorded metrics</a>
    </div>` : `
    ${report.status ? `<div class="callout ${report.status === "completed" ? "success" : report.status === "failed" ? "error" : "warning"}">
      <div class="callout-copy"><strong>Run ${esc(report.status)}</strong>
      <p>${esc(report.message || "No message recorded.")}</p></div></div>` : ""}

    <div class="stat-grid">
      <div class="stat-card accent"><div class="stat-label"><span>Status</span>${icon("activity")}</div>
        <div class="stat-value" style="font-size:22px">${esc(report.status || "—")}</div>
        <div class="stat-foot">${esc(report.mode || "—")} · ${fmtNumber(report.iterations)} iterations</div></div>
      <div class="stat-card"><div class="stat-label"><span>Steps</span>${icon("layers")}</div>
        <div class="stat-value">${fmtNumber(steps.completed)}</div>
        <div class="stat-foot">${fmtNumber(steps.start)} → ${fmtNumber(steps.end)} of ${fmtNumber(steps.target)} · ${steps.progress === null || steps.progress === undefined ? "—" : pct(steps.progress, 1)}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Duration</span>${icon("clock")}</div>
        <div class="stat-value">${fmtDuration(report.duration_s)}</div>
        <div class="stat-foot">${esc((report.started_at || "?").replace("T", " ").slice(0, 19))} → ${esc((report.ended_at || "?").replace("T", " ").slice(0, 19))}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Steps / s (wall clock)</span>${icon("bolt")}</div>
        <div class="stat-value">${num(throughput.wall_clock_steps_per_second, 1)}</div>
        <div class="stat-foot">per iteration: ${iteration.available === false ? "not measured" : num(iteration.mean, 1)}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Device</span>${icon("cpu")}</div>
        <div class="stat-value" style="font-size:22px">${measured(hardware.device, (v) => esc(String(v).toUpperCase()))}</div>
        <div class="stat-foot">${measured(hardware.workers, (v) => `${esc(v)} worker(s)`)} · ${hardware.gpu_name ? esc(hardware.gpu_name) : "no GPU"}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Final checkpoint</span>${icon("models")}</div>
        <div class="stat-value" style="font-size:22px">${checkpoint.available === false ? "none" : checkpoint.valid ? "valid" : "invalid"}</div>
        <div class="stat-foot">${esc(checkpoint.model_id || (checkpoint.reason ?? "no checkpoint written"))}</div></div>
    </div>

    <div class="two-col" style="margin-top:18px">
      <section class="panel">
        <div class="panel-header"><div><h2>Continuation</h2><p>What this run restored, and from where.</p></div></div>
        <div class="metadata-list" style="padding:0 20px 12px">
          ${statRow("Resume mode", esc(resume.mode || "fresh"))}
          ${statRow("Parent run", esc(resume.parent_run || "—"))}
          ${statRow("Restored step", fmtNumber(resume.restored_step))}
          ${statRow("Checkpoint", esc(resume.checkpoint || resume.model_id || "—"))}
          ${statRow("Will resume", resume.will_resume === undefined ? "—" : resume.will_resume ? "yes" : "no")}
        </div>
        <div style="padding:0 20px 18px">
          ${(resume.restores || []).length ? `<div class="chip-row">${resume.restores.map((item) => chip(item, "stable")).join("")}</div>` : ""}
          ${reasonList("Resume warnings", resume.warnings)}
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Hardware &amp; throughput</h2><p>Measured separately, never estimated.</p></div></div>
        <div class="metadata-list" style="padding:0 20px 12px">
          ${statRow("Platform", esc(hardware.platform || "—"))}
          ${statRow("Logical CPUs", esc(hardware.cpu_logical ?? "—"))}
          ${statRow("Python / torch", `${esc(hardware.python || "?")} · ${esc(hardware.torch || "?")}`)}
          ${statRow("Torch threads", esc(hardware.torch_threads ?? "—"))}
          ${statRow("GPU available", hardware.gpu_available === undefined ? "—" : hardware.gpu_available ? "yes" : "no")}
          ${statRow("Wall-clock steps/s", num(throughput.wall_clock_steps_per_second, 2))}
          ${statRow("Iteration steps/s", iteration.available === false ? "not measured" : `${num(iteration.mean, 2)} (min ${num(iteration.min, 2)} · max ${num(iteration.max, 2)})`)}
        </div>
        <p class="report-note" style="padding:0 20px 18px">${esc(report.steps_per_second_basis || (iteration.basis ?? ""))}</p>
      </section>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Where the time went</h2><p>Seconds per iteration, averaged over ${fmtNumber(report.iterations)} iterations.</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th>Phase</th>${timingKeys.map((key) => `<th>${esc(key)}</th>`).join("")}</tr></thead>
        <tbody>${Object.entries(timings).map(([name, entry]) => `<tr>
          <td><strong>${esc(name)}</strong></td>${metricCells(entry, timingKeys)}</tr>`).join("")}</tbody>
      </table></div>
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Optimisation</h2><p>PPO statistics recorded by the learner after the restored step.</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th>Metric</th>${optimKeys.map((key) => `<th>${esc(key)}</th>`).join("")}</tr></thead>
        <tbody>${Object.entries(optimisation).map(([name, entry]) => `<tr>
          <td><strong>${esc(name)}</strong></td>${metricCells(entry, optimKeys)}</tr>`).join("")}</tbody>
      </table></div>
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Reward components</h2><p>Mean per environment step. Shaping terms are potential-based; only goals change the score.</p></div></div>
      ${rewardComponents.available === false
        ? `<p class="report-note">${esc(rewardComponents.reason || "not measured")}</p>`
        : `<div class="table-wrap"><table class="compact">
            <thead><tr><th>Component</th><th>Mean</th><th>Last</th><th>Samples</th></tr></thead>
            <tbody>${Object.entries(rewardComponents).map(([name, entry]) => `<tr>
              <td><strong>${esc(name.replace("reward/", ""))}</strong></td>
              <td>${num(entry.mean, 4)}</td><td>${num(entry.last, 4)}</td><td>${fmtNumber(entry.count)}</td>
            </tr>`).join("")}</tbody></table></div>`}
    </section>

    ${renderEpisodes((report && report.episodes) || {})}
    ${renderEvaluation(report && report.evaluation)}

    ${curvePairs.map((pair) => `<div class="two-col" style="margin-top:18px">
      ${pair.map(([name, label, color]) => {
        const points = (report.curves || {})[name] || [];
        return `<section class="panel chart-panel">
          <div class="panel-header"><div><h2>${esc(label)}</h2><p>${esc(name)}</p></div>
          <span class="pill muted-pill">${points.length} points</span></div>
          <div class="chart">${lineChart(points, { color })}</div>
        </section>`;
      }).join("")}
    </div>`).join("")}

    ${(report.comparisons || []).length ? `<section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Other runs with the same setup</h2><p>Same mode, observation layout and policy shape.</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th>Run</th><th>Status</th><th>Steps</th><th>Duration</th><th>Steps / s</th><th>Final eval win rate</th><th>Mean return</th></tr></thead>
        <tbody>${report.comparisons.map((item) => `<tr>
          <td><a href="#/report?run=${encodeURIComponent(item.run_id)}"><strong>${esc(item.name || item.run_id)}</strong></a><small>${esc(item.run_id)}</small></td>
          <td>${pill(item.status)}</td>
          <td>${fmtNumber(item.steps_completed)}<small> ${fmtNumber(item.steps_start)} → ${fmtNumber(item.steps_end)}</small></td>
          <td>${item.duration_s ? fmtDuration(item.duration_s) : "—"}</td>
          <td>${num(item.steps_per_second, 1)}</td>
          <td>${item.final_win_rate === null || item.final_win_rate === undefined ? `<span class="muted">not measured</span>` : pct(item.final_win_rate)}</td>
          <td>${num(item.mean_return, 3)}</td>
        </tr>`).join("")}</tbody></table></div>
    </section>` : ""}

    ${checkpoint.available !== false ? `<section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Final checkpoint</h2><p>Re-validated when the report was written.</p></div></div>
      <div class="metadata-list" style="padding:0 20px 12px">
        ${statRow("Model id", esc(checkpoint.model_id || "—"))}
        ${statRow("Step", fmtNumber(checkpoint.step))}
        ${statRow("Label", esc(checkpoint.label || "—"))}
        ${statRow("Size", `${fmtNumber(checkpoint.size_bytes)} bytes`)}
        ${statRow("Valid", checkpoint.valid ? "yes" : "no")}
        ${statRow("Path", esc(checkpoint.path || "—"))}
      </div>
      <div style="padding:0 20px 18px">
        ${(checkpoint.validation && checkpoint.validation.reasons && checkpoint.validation.reasons.length)
          ? reasonList("Validation findings", checkpoint.validation.reasons)
          : `<p class="subtle">Weights are finite and match the recorded architecture.</p>`}
        ${checkpoint.model_id ? `<a class="button ghost" href="#/training?resume=${encodeURIComponent(checkpoint.model_id)}&mode=continue">${icon("training")} Continue from this checkpoint</a>` : ""}
      </div>
    </section>` : ""}

    ${(report.warnings || []).length ? `<section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Warnings</h2><p>Everything the run could not do or measure.</p></div></div>
      <div class="panel-body">${reasonList("", report.warnings)}</div>
    </section>` : ""}

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Configuration</h2><p>The exact configuration this run used.</p></div></div>
      <pre class="proposal-target" style="margin:0 20px 18px">${esc(JSON.stringify(report.configuration || {}, null, 2))}</pre>
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>How to read this report</h2></div></div>
      <div class="panel-body"><ul class="guide-list">
        ${(report.notes || []).map((note) => `<li>${esc(note)}</li>`).join("")}
      </ul></div>
    </section>`}`;

  $("#report-run-select")?.addEventListener("change", (event) => {
    location.hash = `#/report?run=${encodeURIComponent(event.target.value)}`;
  });
};

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
              <a class="button ghost" href="#/training?resume=${encodeURIComponent(model.id)}&mode=continue" title="Continue training from this checkpoint">${icon("training")} Continue</a>
              <button class="button ghost" data-validate="${esc(model.id)}" title="Check compatibility with the default configuration">${icon("shield")} Validate</button>
              <button class="button ghost" data-evaluate="${esc(model.id)}">${icon("play")} Evaluate</button>
              <button class="button ghost" data-promote="${esc(model.id)}" ${model.stable ? "disabled" : ""}>${icon("check")} Promote</button>
              <button class="button danger" data-delete="${esc(model.id)}" ${model.stable ? "disabled" : ""} title="Delete checkpoint">${icon("delete")}</button>
            </td></tr>`;
        }).join("")}</tbody>
      </table></div>
    </section>` : emptyState("No checkpoints", "Checkpoints are written during training. Start a run first.", "models")}

    <section class="panel" style="margin-top:18px" id="model-validation-panel" hidden>
      <div class="panel-header"><div><h2>Checkpoint compatibility</h2><p>Validated against the default configuration; the weights are re-read and checked, never modified.</p></div></div>
      <div class="panel-body" id="model-validation"></div>
    </section>

    ${evaluationFormHtml(models, "eval", "Evaluate a checkpoint", "One opponent runs a single series; two or more run a suite that separates strength opponents from diagnostic baselines.")}`;

  $$("[data-evaluate]").forEach((button) => button.addEventListener("click", () => {
    $("#eval-model").value = button.dataset.evaluate;
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }));
  $$("[data-validate]").forEach((button) => button.addEventListener("click", async () => {
    const node = $("#model-validation");
    $("#model-validation-panel").hidden = false;
    node.innerHTML = `<div class="loading-state" style="min-height:60px"><span class="spinner"></span> Reading and validating ${esc(button.dataset.validate)}…</div>`;
    node.scrollIntoView({ behavior: "smooth", block: "center" });
    try {
      const defaults = app.cache.defaults || (await api("/api/config/defaults"));
      const model = models.find((item) => item.id === button.dataset.validate) || {};
      const config = { ...defaults.training, mode: model.mode || defaults.training.mode };
      const result = await api(`/api/models/${encodeURIComponent(button.dataset.validate)}/validate`, {
        method: "POST",
        body: JSON.stringify({ config, require_optimizer: false }),
      });
      node.innerHTML = `
        <div class="callout ${result.ok ? "success" : "error"}">
          <div class="callout-copy">
            <strong>${esc(result.model_id)} — ${result.ok ? "usable" : "not usable"} for ${esc(config.mode)}</strong>
            <p>Step ${fmtNumber(result.step)} · label ${esc(result.label || "—")} · created ${esc((result.created_at || "").replace("T", " ").slice(0, 19))} · ${fmtNumber(result.size_bytes)} bytes.</p>
            <div class="chip-row">
              ${chip(result.ok ? "weights ok" : "rejected", result.ok ? "stable" : "rejected")}
              ${chip(`obs v${result.obs_version ?? "?"}`, "")}
              ${chip(result.has_optimizer ? "optimiser stored" : "no optimiser", result.has_optimizer ? "stable" : "")}
              ${chip(result.has_rng ? "RNG stored" : "no RNG", result.has_rng ? "stable" : "")}
              ${chip(result.has_curriculum ? "curriculum stored" : "no curriculum", result.has_curriculum ? "stable" : "")}
              ${chip(result.has_league ? "pool ratings stored" : "no pool ratings", result.has_league ? "stable" : "")}
            </div>
            ${reasonList("Findings", result.reasons)}
            <p class="subtle" style="margin-top:8px">Continue (restore everything): ${result.resume.continue ? "possible" : "not possible"}. Transfer (weights only): ${result.resume.transfer ? "possible" : "not possible"}.</p>
          </div>
        </div>`;
    } catch (error) {
      node.innerHTML = `<div class="callout error"><div class="callout-copy"><strong>Validation failed</strong><p>${esc(error.message)}</p></div></div>`;
    }
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
  wireEvaluationForm("eval");
};

/**
 * The evaluation form is shared by the Models and Evaluation views so both
 * send exactly the same protocol fields (sides, mirroring, seeds, phase).
 */
function evaluationFormHtml(models, prefix, title = "Evaluation", description) {
  return `
    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>${esc(title)}</h2><p>${esc(description || "Reproducible match series: fixed seeds, recorded protocol, both sides when you ask for it.")}</p></div></div>
      <div class="form-grid">
        <label class="field"><span>Model</span><select id="${prefix}-model">${models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}${m.stable ? " · stable" : ""}${m.best ? " · best" : ""}</option>`).join("")}</select></label>
        <label class="field"><span>Opponents</span><input id="${prefix}-opponents" value="chase" placeholder="chase or chase,mixed,self" /><small>idle/random are diagnostic baselines (they cannot lose on purpose); chase/mixed/self/model:* measure strength.</small></label>
        <label class="field"><span>Games per opponent</span><input id="${prefix}-games" type="number" min="1" max="200" value="20" /><small>More games narrow the 95 % confidence interval.</small></label>
        <label class="field"><span>Match seconds</span><input id="${prefix}-seconds" type="number" min="5" max="600" value="120" /></label>
        <label class="field"><span>Seed</span><input id="${prefix}-seed" type="number" value="1000" /><small>Game i uses seed + i · 7919, so a series is reproducible.</small></label>
        <label class="field"><span>Sides</span><select id="${prefix}-sides"><option value="blue">blue only</option><option value="orange">orange only</option><option value="blue,orange">both (side-balanced)</option></select><small>Both sides alternate per game; kickoffs are seeded per game.</small></label>
        <label class="field"><span>Device</span><select id="${prefix}-device"><option>cpu</option><option>cuda</option></select></label>
        <label class="field"><span>Phase label</span><select id="${prefix}-phase"><option>manual</option><option>before</option><option>periodic</option><option>after</option></select><small>Stored with the result; before/after are used by run reports.</small></label>
        <label class="field full approval-check"><input type="checkbox" id="${prefix}-mirror" checked /><span>Mirror the observation when the policy plays orange (180° about z; car-frame features, verified equivalent in the test suite).</span></label>
        <label class="field full approval-check"><input type="checkbox" id="${prefix}-stochastic" /><span>Sample actions instead of argmax (matches the training policy; runs then differ).</span></label>
      </div>
      <div class="config-actions">
        <button class="button primary" id="${prefix}-start" ${models.length ? "" : "disabled"}>${icon("play")} Start evaluation</button>
        <span class="subtle" id="${prefix}-budget"></span>
      </div>
    </section>`;
}

function wireEvaluationForm(prefix) {
  const field = (name) => $(`#${prefix}-${name}`);
  if (!field("model")) return;
  const payload = () => {
    const opponents = splitList(field("opponents").value);
    return {
      model_id: field("model").value,
      opponent: opponents[0] || "chase",
      opponents,
      games: Number(field("games").value),
      match_seconds: Number(field("seconds").value),
      seed: Number(field("seed").value),
      device: field("device").value,
      sides: splitList(field("sides").value),
      mirror: field("mirror").checked,
      stochastic: field("stochastic").checked,
      phase: field("phase").value,
    };
  };
  const updateBudget = () => {
    const node = field("budget");
    if (!node) return;
    const request = payload();
    const games = request.games * Math.max(1, request.sides.length);
    const minutes = (games * Math.max(1, request.opponents.length) * request.match_seconds) / 60;
    node.textContent = `${request.opponents.length} opponent(s) · ${games} game(s) each · about ${minutes.toFixed(1)} min of simulated matches`;
  };
  ["games", "seconds", "seed"].forEach((name) => field(name)?.addEventListener("input", updateBudget));
  ["opponents", "sides"].forEach((name) => field(name)?.addEventListener("change", updateBudget));
  updateBudget();

  field("start")?.addEventListener("click", () => {
    const request = payload();
    const suite = request.opponents.length > 1;
    confirmAction(
      suite ? "Start evaluation suite" : "Start evaluation",
      `${request.games} game(s) per side against ${request.opponents.join(", ")} on sides ${request.sides.join("+")} ` +
        `(${request.match_seconds} s matches, seed ${request.seed}, ${request.mirror ? "mirrored" : "not mirrored"}, ` +
        `${request.stochastic ? "stochastic" : "deterministic"} policy). The series occupies the CPU until it finishes; nothing else is started.`,
      async () => {
        const url = suite
          ? "/api/evaluations/suite"
          : `/api/models/${encodeURIComponent(request.model_id)}/evaluate`;
        const result = await api(url, { method: "POST", body: JSON.stringify(request) });
        toast(`Evaluation started (${result.kind}) · job ${result.job_id}. Results appear when the series finishes.`);
        setTimeout(() => location.hash === "#/evaluation" && render(), 2500);
      }
    );
  });
}

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
  const classify = (item) =>
    (item.details && item.details.opponent_class) ||
    (["idle", "random"].includes(String(item.opponent || "").trim()) ? "diagnostic" : "strength");
  const strength = evaluations.filter((item) => classify(item) === "strength");
  const diagnostic = evaluations.filter((item) => classify(item) === "diagnostic");
  const bestPerModel = (rows) => {
    const best = {};
    rows.forEach((item) => {
      const key = `${item.model_id || "unknown"}|${item.opponent}`;
      const rate = item.wins / Math.max(item.games, 1);
      if (!best[key] || rate > best[key].rate) best[key] = { ...item, rate };
    });
    return Object.values(best);
  };
  const bestStrength = bestPerModel(strength);
  const modelsWithStrength = new Set(bestStrength.map((item) => item.model_id));

  $("#view").innerHTML = `
    <div class="page-heading"><div><span class="eyebrow">Benchmarks</span><h1>Evaluation</h1>
    <p>Every result below is a real match series with fixed seeds and a recorded protocol. Draws are possible, and a high training reward is not evidence of strength.</p></div>
    <div class="heading-actions"><a class="button ghost" href="#/reports">${icon("file")} Run reports</a></div></div>

    ${models.length ? evaluationFormHtml(models, "series", "New match series", "Side-balanced, seeded and reproducible. Diagnostic baselines are reported separately from strength opponents.") : ""}

    ${evaluations.length ? `
    <div class="stat-grid" style="margin-top:18px">
      <div class="stat-card accent"><div class="stat-label"><span>Series</span>${icon("evaluation")}</div><div class="stat-value">${evaluations.length}</div><div class="stat-foot">${strength.length} strength · ${diagnostic.length} diagnostic</div></div>
      <div class="stat-card"><div class="stat-label"><span>Models with a strength result</span>${icon("models")}</div><div class="stat-value">${modelsWithStrength.size}</div><div class="stat-foot">of ${models.length} checkpoints</div></div>
      <div class="stat-card"><div class="stat-label"><span>Games played</span>${icon("activity")}</div><div class="stat-value">${fmtNumber(evaluations.reduce((sum, item) => sum + item.games, 0))}</div><div class="stat-foot">simulated matches</div></div>
      <div class="stat-card"><div class="stat-label"><span>Goals</span>${icon("trophy")}</div><div class="stat-value">${fmtNumber(evaluations.reduce((sum, item) => sum + item.goals_for, 0))}</div><div class="stat-foot">${evaluations.reduce((sum, item) => sum + item.goals_against, 0)} conceded</div></div>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Playing strength</h2><p>Best recorded series per checkpoint and strength opponent. Diagnostic baselines (idle, random) are excluded: they measure that the pipeline works, not how good the policy is.</p></div></div>
      ${bestStrength.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Mode</th><th>Opponent</th><th>Win rate (95 % CI)</th><th>Record</th><th>Goal diff</th><th>Games</th><th>Protocol</th></tr></thead>
        <tbody>${bestStrength.map((item) => `<tr>
          <td><strong>${esc(item.model_id || "—")}</strong></td><td>${esc(item.mode)}</td><td>${esc(item.opponent)}</td>
          <td class="table-strong">${pct(item.rate)}<small>${item.details?.win_rate_ci95 ? `${pct(item.details.win_rate_ci95[0])} – ${pct(item.details.win_rate_ci95[1])}` : "no interval"}</small></td>
          <td>${item.wins}W · ${item.losses}L · ${item.draws}D</td>
          <td>${item.goals_for - item.goals_against >= 0 ? "+" : ""}${item.goals_for - item.goals_against}</td>
          <td>${item.games}</td>
          <td><div class="chip-row">${protocolChips(item.details || item)}</div></td>
        </tr>`).join("")}</tbody></table></div>`
        : `<p class="report-note">No strength evaluation has been recorded yet. Run a series against chase, mixed, self or a model checkpoint.</p>`}
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>History</h2><p>Latest first, with the phase label and the protocol each series used. Series with different protocols are not comparable.</p></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Model</th><th>Mode</th><th>Opponent</th><th>Class</th><th>Phase</th><th>Record</th><th>Win rate (95 % CI)</th><th>Goals</th><th>Protocol</th><th>When</th></tr></thead>
        <tbody>${evaluations.map((item) => {
          const details = item.details || {};
          const ci = details.win_rate_ci95;
          const kind = classify(item);
          return `<tr>
          <td><strong>${esc(item.model_id || "—")}</strong><small>${fmtNumber(item.games)} games · seed ${item.seed ?? details.protocol?.seed ?? "—"}</small></td>
          <td>${esc(item.mode)}</td><td>${esc(item.opponent)}</td>
          <td>${chip(kind, kind === "strength" ? "stable" : "")}</td>
          <td>${esc(item.phase || "—")}</td>
          <td class="table-strong">${item.wins}W · ${item.losses}L · ${item.draws}D</td>
          <td>${pct(item.wins / Math.max(item.games, 1))}<small>${ci ? `${pct(ci[0])} – ${pct(ci[1])}` : "no interval recorded"}</small></td>
          <td>${item.goals_for} : ${item.goals_against}</td>
          <td><div class="chip-row">${protocolChips(details)}</div></td>
          <td>${esc((item.created_at || "").replace("T", " ").slice(0, 19))}</td>
        </tr>`;
        }).join("")}</tbody>
      </table></div>
    </section>

    ${diagnostic.length ? `<section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Diagnostic baselines</h2><p>Against idle and random opponents. A win rate here says the observation, action and reward pipeline works — nothing about football skill.</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th>Model</th><th>Opponent</th><th>Win rate</th><th>Record</th><th>Touches</th><th>Games</th><th>When</th></tr></thead>
        <tbody>${diagnostic.map((item) => `<tr>
          <td><strong>${esc(item.model_id || "—")}</strong></td><td>${esc(item.opponent)}</td>
          <td class="table-strong">${pct(item.wins / Math.max(item.games, 1))}</td>
          <td>${item.wins}W · ${item.losses}L · ${item.draws}D</td>
          <td>${item.details?.touches ?? "—"}</td>
          <td>${item.games}</td>
          <td>${esc((item.created_at || "").replace("T", " ").slice(0, 19))}</td>
        </tr>`).join("")}</tbody></table></div>
    </section>` : ""}` : emptyState("No evaluations yet", "Use the form above to run a seeded, side-balanced series against a diagnostic baseline and a strength opponent.", "evaluation")}

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>How these numbers are produced</h2></div></div>
      <div class="panel-body"><ul class="guide-list">
        <li>Each game is a real RocketSim match; the policy is loaded from the checkpoint and never retrained during evaluation.</li>
        <li>Sides alternate per game when both are selected, and the orange-side observation is rotated 180° about z (car-frame features, verified equivalent). Disabling mirroring measures the raw, unrotated behaviour.</li>
        <li>Game i always uses seed + i · 7919, so repeating a series with the same settings reproduces the same kickoffs.</li>
        <li>The 95 % interval is a Wilson score interval over the played games; with few games it is wide, and that width is shown rather than hidden.</li>
        <li>Results are only comparable when the protocol matches: same mode, opponent, game count, match length, sides, mirroring, sampling and observation version.</li>
        <li>The best-model pointer is only moved by a series that produced at least one win under a recorded protocol.</li>
      </ul></div>
    </section>`;

  wireEvaluationForm("series");
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
      const selected = $("#live-model")?.value;
      const url = selected
        ? `/api/live/preflight?model_id=${encodeURIComponent(selected)}`
        : "/api/live/preflight";
      renderChecks(await api(url));
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
          <label class="field"><span>Seconds per worker</span><input id="bench-seconds" type="number" min="0.5" max="10" step="0.5" value="3" /></label>
          <label class="field full approval-check"><input type="checkbox" id="bench-parallel" checked /><span>Run the workers in parallel processes and measure the wall clock between a start signal and the results (the honest number for training).</span></label>
          <label class="field full approval-check"><input type="checkbox" id="bench-inference" checked /><span>Also measure policy inference separately (decisions per second for one agent step).</span></label>
        </div>
        <div class="config-actions"><button class="button secondary" id="bench-run">${icon("gauge")} Measure</button></div>
        <div class="panel-body" id="bench-result"><p class="subtle">No benchmark has been run in this session. Environment construction is excluded from both timings.</p></div>
      </section>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>API access token</h2><p>Only needed when the server was started with RLAI_AUTH_TOKEN.</p></div></div>
      <div class="panel-body">
        <div class="form-grid">
          <label class="field full"><span>Token</span><input id="token-input" type="password" value="${esc(app.token)}" placeholder="paste the value of RLAI_AUTH_TOKEN" autocomplete="off" />
          <small>Stored in this browser only (localStorage) and sent as the x-rlai-token header. Read-only endpoints stay open; starting training, evaluating, promoting or launching a match requires it.</small></label>
        </div>
        <div class="config-actions">
          <button class="button secondary" id="token-save">${icon("lock")} Save token</button>
          <button class="button ghost" id="token-clear">${icon("delete")} Clear</button>
        </div>
      </div>
    </section>

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

  $("#token-save").addEventListener("click", () => {
    setToken($("#token-input").value);
    toast(app.token ? "Token saved in this browser" : "Token cleared");
    refreshConnection();
  });
  $("#token-clear").addEventListener("click", () => {
    setToken("");
    $("#token-input").value = "";
    toast("Token cleared");
    refreshConnection();
  });

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
          parallel: $("#bench-parallel").checked,
          include_inference: $("#bench-inference").checked,
        }),
      });
      $("#bench-result").innerHTML = `<div class="metadata-list" style="padding:0">
        <div class="metadata-row"><span>Environment steps / s</span><strong>${fmtNumber(result.env_steps_per_s)}</strong></div>
        <div class="metadata-row"><span>Simulator ticks / s</span><strong>${fmtNumber(result.sim_ticks_per_s)}</strong></div>
        <div class="metadata-row"><span>Workers</span><strong>${result.workers} · ${result.parallel ? "parallel processes" : "sequential in one process"}</strong></div>
        <div class="metadata-row"><span>Measured wall clock</span><strong>${Number(result.wall_seconds ?? result.seconds ?? 0).toFixed(3)} s of ${Number(result.requested_seconds ?? 0).toFixed(1)} s requested</strong></div>
        ${result.sequential_seconds ? `<div class="metadata-row"><span>Sequential equivalent</span><strong>${Number(result.sequential_seconds).toFixed(3)} s</strong></div>` : ""}
        ${result.single_worker_steps_per_s ? `<div class="metadata-row"><span>Single worker</span><strong>${fmtNumber(result.single_worker_steps_per_s)} steps / s</strong></div>` : ""}
        ${result.speedup ? `<div class="metadata-row"><span>Parallel speed-up</span><strong>${Number(result.speedup).toFixed(2)} ×</strong></div>` : ""}
        ${result.inference ? `<div class="metadata-row"><span>Inference</span><strong>${fmtNumber(result.inference.decisions_per_s)} decisions / s · ${Number(result.inference.milliseconds_per_decision ?? 0).toFixed(2)} ms each</strong></div>` : ""}
        <div class="metadata-row"><span>Total</span><strong>${fmtNumber(result.total_steps)} steps · ${fmtNumber(result.total_ticks)} ticks</strong></div>
      </div>
      <p class="resource-note" style="margin-top:8px">${esc(result.measurement || "")}</p>
      ${result.parallel_error ? `<p class="resource-note">Parallel measurement failed and fell back: ${esc(result.parallel_error)}</p>` : ""}
      ${(result.per_worker || []).length ? `<div class="table-wrap" style="margin-top:8px"><table class="compact">
        <thead><tr><th>Worker</th><th>Steps</th><th>Ticks</th><th>Seconds</th><th>Steps / s</th></tr></thead>
        <tbody>${result.per_worker.map((worker) => `<tr>
          <td>${esc(worker.worker ?? worker.index ?? "?")}</td>
          <td>${fmtNumber(worker.steps)}</td><td>${fmtNumber(worker.ticks)}</td>
          <td>${Number(worker.seconds ?? 0).toFixed(3)}</td>
          <td>${fmtNumber(worker.steps_per_s ?? (worker.seconds ? worker.steps / worker.seconds : null))}</td>
        </tr>`).join("")}</tbody></table></div>` : ""}`;
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
