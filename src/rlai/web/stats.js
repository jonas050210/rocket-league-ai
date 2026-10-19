/* ---------------------------------------------------------------- statistics
 *
 * The Stats tab is the "what actually happened" view: training time,
 * progress, learning metrics, throughput, checkpoints, evaluations and the
 * session history.  Every number comes from the API; nothing is derived
 * in the browser, and a value that was never measured is shown as
 * "not measured" instead of a zero.
 */

const {
  views,
  api,
  $,
  $$,
  esc,
  icon,
  chip,
  pill,
  emptyState,
  lineChart,
  fmtNumber,
  fmtDuration,
  pct,
  renderComparison,
} = window.RLAI;

/** ISO timestamp (UTC) -> local time string. */
function fmtLocalTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** ISO timestamp -> short "HH:MM:SS" in the local time zone. */
function fmtLocalClock(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Duration in seconds -> "1h 02m 03s" (or "—" when unknown). */
function fmtDurationLong(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) {
    return "—";
  }
  const total = Math.max(0, Math.floor(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(secs).padStart(2, "0")}s`;
  if (minutes) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  return `${secs} s`;
}

function sessionStatusChip(status) {
  const map = {
    completed: "stable",
    stopped: "",
    interrupted: "rejected",
    failed: "rejected",
    inherited: "candidate",
    running: "candidate",
  };
  return chip(status || "unknown", map[status] || "");
}

function statCard(label, value, foot, iconName = "activity", cls = "") {
  return `<div class="stat-card ${cls}">
    <div class="stat-label"><span>${esc(label)}</span>${icon(iconName)}</div>
    <div class="stat-value">${value}</div>
    <div class="stat-foot">${foot || ""}</div>
  </div>`;
}

/** One chart with a title, a unit hint and an "unavailable" fallback. */
function metricChart(title, points, { unit = "", hint = "" } = {}) {
  const body = points && points.length
    ? lineChart(points)
    : `<div class="chart-empty">${icon("chart")}<span>Not measured yet</span></div>`;
  return `<section class="panel">
    <div class="panel-header"><div><h2>${esc(title)}</h2><p>${esc(hint || unit)}</p></div></div>
    <div class="panel-body">${body}</div>
  </section>`;
}

views.stats = async () => {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const runs = await api("/api/runs?limit=50");
  if (!runs.length) {
    $("#view").innerHTML = `
      <div class="page-heading"><div><span class="eyebrow">Statistics</span><h1>Stats</h1><p>No runs exist yet.</p></div></div>
      ${emptyState("Nothing to measure", "Start a training run; every measurement in this tab comes from that run.", "chart")}`;
    return;
  }
  const selected = params.get("run") || runs[0].id;
  const [stats, resources, overview, reports] = await Promise.all([
    api(`/api/stats?run_id=${encodeURIComponent(selected)}`),
    api("/api/system/resources"),
    api("/api/stats/overview?limit=12"),
    api("/api/reports?limit=100"),
  ]);
  if (!stats || stats.available === false) {
    $("#view").innerHTML = `
      <div class="page-heading"><div><span class="eyebrow">Statistics</span><h1>Stats</h1></div></div>
      ${emptyState("No statistics", stats?.reason || "The backend returned no statistics.", "chart")}`;
    return;
  }

  const timing = stats.timing || {};
  const progress = stats.progress || {};
  const learning = stats.learning || {};
  const performance = stats.performance || {};
  const evaluation = stats.evaluation || {};
  const checkpoints = stats.checkpoints || [];
  // the session history lives in the timing block; ``history`` is the
  // alias the API keeps for convenience
  const history = timing.sessions || stats.history || [];
  const curves = stats.curves || {};
  const resourcesBlock = performance.resources || resources || {};
  const gpu = resourcesBlock.gpu || null;

  const learningValue = (name) => {
    const entry = learning[name];
    if (!entry) return "—";
    return fmtNumber(entry.last, 4);
  };
  const learningMean = (name) => {
    const entry = learning[name];
    if (!entry) return "—";
    return fmtNumber(entry.mean, 4);
  };

  const sessionRows = history.length
    ? history.map((session) => `<tr>
        <td>${esc(fmtLocalTime(session.started_at))}</td>
        <td>${esc(fmtLocalClock(session.ended_at))}</td>
        <td class="table-strong">${session.duration_available === false || session.duration_s === null ? `<span class="muted">unknown</span>` : esc(fmtDurationLong(session.duration_s))}</td>
        <td>${sessionStatusChip(session.status)}</td>
        <td>${fmtNumber(session.start_step)} → ${fmtNumber(session.end_step)}</td>
        <td class="muted">${esc(session.note || "")}</td>
      </tr>`).join("")
    : "";

  const checkpointRows = checkpoints.length
    ? checkpoints.map((row) => `<tr>
        <td><strong>${esc(row.id)}</strong><small>${esc(row.label || "candidate")}${row.stable ? " · stable" : ""}${row.best ? " · best" : ""}</small></td>
        <td>${esc(fmtLocalTime(row.created_at))}</td>
        <td class="table-strong">${row.duration_available ? esc(fmtDurationLong(row.duration_s)) : `<span class="muted">unknown</span>`}</td>
        <td>${fmtNumber(row.step)}</td>
        <td>${row.eval_score === null || row.eval_score === undefined ? `<span class="muted">not measured</span>` : pct(row.eval_score)}</td>
      </tr>`).join("")
    : "";

  const opponentRows = (evaluation.per_opponent || []).length
    ? evaluation.per_opponent.map((row) => `<tr>
        <td><strong>${esc(row.opponent)}</strong><small>${fmtNumber(row.evaluations)} evaluation(s)</small></td>
        <td class="table-strong">${fmtNumber(row.wins)} / ${fmtNumber(row.losses)} / ${fmtNumber(row.draws)}</td>
        <td>${row.win_rate === null || row.win_rate === undefined ? `<span class="muted">not measured</span>` : pct(row.win_rate)}</td>
        <td>${fmtNumber(row.games)}</td>
      </tr>`).join("")
    : "";

  const ranking = (stats.comparison || {}).ranking || [];
  const rankingRows = ranking.length
    ? ranking.map((row, index) => `<tr>
        <td>${index + 1}</td>
        <td><strong>${esc(row.id)}</strong></td>
        <td>${fmtNumber(row.step)}</td>
        <td class="table-strong">${pct(row.eval_score)}</td>
        <td>${row.duration_s === null || row.duration_s === undefined ? `<span class="muted">unknown</span>` : esc(fmtDurationLong(row.duration_s))}</td>
      </tr>`).join("")
    : "";

  const historyRows = (overview.runs || []).map((row) => `<tr>
      <td><strong>${esc(row.name || row.id)}</strong><small>${esc(row.mode)} · ${esc(row.id)}</small></td>
      <td>${pill(row.status)}</td>
      <td class="table-strong">${fmtNumber(row.step)}${row.total_steps ? ` / ${fmtNumber(row.total_steps)}` : ""}</td>
      <td>${row.accumulated_s === null || row.accumulated_s === undefined ? `<span class="muted">unknown</span>` : esc(fmtDurationLong(row.accumulated_s))}</td>
      <td>${fmtNumber(row.session_count)}</td>
      <td>${fmtNumber(row.interrupted_count)}</td>
      <td>${esc(fmtLocalTime(row.started_at))}</td>
    </tr>`).join("");

  // one full report per finished/stopped/failed run; stored summaries keep
  // the history table cheap without re-reading the run reports
  const reportByRun = {};
  (reports || []).forEach((item) => { reportByRun[item.run_id] = item; });
  const reportRows = runs.map((run) => {
    const report = reportByRun[run.id];
    const summary = (report && report.summary) || {};
    const steps = summary.steps || {};
    const after = (summary.evaluation || {}).after || {};
    const finalCheckpoint = summary.final_checkpoint || {};
    return `<tr>
      <td><input type="checkbox" class="compare-check" value="${esc(run.id)}" aria-label="Select ${esc(run.name)} for comparison" /></td>
      <td><strong>${esc(run.name)}</strong><small>${esc(run.id)}${run.parent_run ? ` · continues ${esc(run.parent_run)}` : ""}</small></td>
      <td>${pill(run.status)}</td>
      <td class="table-strong">${fmtNumber(steps.completed ?? run.step)}<small> of ${fmtNumber(steps.target ?? run.total_steps)}</small></td>
      <td>${report ? fmtDuration(summary.duration_s) : `<span class="muted">no report</span>`}</td>
      <td>${report ? (summary.steps_per_second === null || summary.steps_per_second === undefined ? "—" : fmtNumber(summary.steps_per_second, 1)) : "—"}</td>
      <td>${after && after.available !== false ? `${pct(after.win_rate)} <small>vs ${esc(after.opponent || "?")}</small>` : `<span class="muted">not measured</span>`}</td>
      <td>${finalCheckpoint && finalCheckpoint.available !== false ? (finalCheckpoint.valid ? chip("valid", "stable") : chip("invalid", "rejected")) : `<span class="muted">none</span>`}</td>
      <td class="actions">
        ${report ? `<a class="button ghost" href="#/report?run=${encodeURIComponent(run.id)}">${icon("file")} Report</a>` : `<button class="button ghost" data-rebuild="${esc(run.id)}">${icon("refresh")} Rebuild</button>`}
      </td>
    </tr>`;
  }).join("");

  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Statistics</span>
        <h1>Stats</h1>
        <p>Training time, progress, learning metrics, throughput, checkpoints and evaluations.</p>
      </div>
      <div class="heading-actions">
        <select id="stats-run" class="log-search" style="width:auto">
          ${runs.map((run) => `<option value="${esc(run.id)}" ${run.id === selected ? "selected" : ""}>${esc(run.name)} · ${esc(run.id)}</option>`).join("")}
        </select>
        <a class="button ghost" href="#/report?run=${encodeURIComponent(selected)}">${icon("file")} Report</a>
      </div>
    </div>

    <div class="stat-grid">
      ${statCard("Current session", esc(fmtDurationLong(timing.current_session_s)), timing.current_session_available === false ? `<span class="muted">no session running</span>` : "measured with a monotonic clock", "activity", "accent")}
      ${statCard("Accumulated training time", esc(fmtDurationLong(timing.accumulated_s)), `${fmtNumber(timing.session_count)} session(s) · ${fmtNumber(timing.interrupted_count)} interrupted`, "layers")}
      ${statCard("Environment steps", fmtNumber(progress.step), progress.total_steps ? `of ${fmtNumber(progress.total_steps)} · ${pct(progress.progress, 1)}` : "no target recorded", "chart")}
      ${statCard("Throughput", `${fmtNumber(performance.steps_per_second)}<small> steps/s</small>`, "wall clock, whole run", "monitoring")}
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Training time</h2><p>Measured, never estimated from the wall clock.</p></div></div>
        <div class="panel-body">
          <div class="metadata-list" style="padding-inline:0">
            <div class="metadata-row"><span>Current session</span><strong>${esc(fmtDurationLong(timing.current_session_s))}</strong></div>
            <div class="metadata-row"><span>Accumulated (all sessions)</span><strong>${esc(fmtDurationLong(timing.accumulated_s))}</strong></div>
            <div class="metadata-row"><span>Started</span><strong>${esc(fmtLocalTime(timing.started_at))}</strong></div>
            <div class="metadata-row"><span>Last session ended</span><strong>${esc(fmtLocalTime(timing.ended_at))}</strong></div>
            <div class="metadata-row"><span>Latest checkpoint</span><strong>${esc(fmtLocalTime(timing.last_checkpoint_at))}</strong></div>
            <div class="metadata-row"><span>Sessions</span><strong>${fmtNumber(timing.session_count)} · ${fmtNumber(timing.interrupted_count)} interrupted</strong></div>
          </div>
          ${timing.available === false ? `<p class="resource-note">This run predates session tracking - the training time is unknown, not zero.</p>` : ""}
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Progress</h2><p>Where the training stands right now.</p></div></div>
        <div class="panel-body">
          <div class="metadata-list" style="padding-inline:0">
            <div class="metadata-row"><span>Status</span><strong>${esc(progress.status || "—")}</strong></div>
            <div class="metadata-row"><span>Environment steps</span><strong>${fmtNumber(progress.step)}</strong></div>
            <div class="metadata-row"><span>Iterations</span><strong>${fmtNumber(progress.iterations)}</strong></div>
            <div class="metadata-row"><span>Finished episodes</span><strong>${fmtNumber(progress.episode_count)}</strong></div>
            <div class="metadata-row"><span>Episode win rate</span><strong>${progress.episodes && progress.episodes.win_rate !== null && progress.episodes.win_rate !== undefined ? pct(progress.episodes.win_rate) : `<span class="muted">not measured</span>`}</strong></div>
            <div class="metadata-row"><span>Mean episode reward</span><strong>${progress.episodes && progress.episodes.mean_reward !== null && progress.episodes.mean_reward !== undefined ? fmtNumber(progress.episodes.mean_reward, 3) : `<span class="muted">not measured</span>`}</strong></div>
          </div>
        </div>
      </section>
    </div>

    <div class="two-col">
      ${metricChart("Episodic return", curves["charts/episodic_return"], { hint: "optimisation signal, not playing strength" })}
      ${metricChart("Episode win rate", curves["charts/win_rate"], { hint: "share of finished episodes that were won" })}
      ${metricChart("Throughput (steps / s)", curves["time/sps"], { hint: "environment steps per second, measured per iteration" })}
      ${metricChart("Policy loss", curves["losses/policy"], { hint: "PPO surrogate loss" })}
    </div>

    <div class="two-col">
      ${metricChart("Explained variance", curves["losses/explained_variance"], { hint: "how well the critic predicts the returns" })}
      ${metricChart("Evaluation win rate", curves["eval/win_rate"], { hint: "measured playing strength at periodic evaluations" })}
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Learning metrics</h2><p>Last value and mean of the optimisation metrics.</p></div></div>
        <div class="panel-body">
          <div class="table-wrap"><table class="compact">
            <thead><tr><th>Metric</th><th>Last</th><th>Mean</th></tr></thead>
            <tbody>
              ${[
                ["charts/episodic_return", "Episodic return"],
                ["charts/win_rate", "Episode win rate"],
                ["losses/policy", "Policy loss"],
                ["losses/value", "Value loss"],
                ["losses/entropy", "Entropy"],
                ["losses/approx_kl", "Approx. KL"],
                ["losses/clipfrac", "Clip fraction"],
                ["losses/explained_variance", "Explained variance"],
                ["losses/learning_rate", "Learning rate"],
                ["time/sps", "Steps / s (iteration)"],
                ["time/collect_seconds", "Collect time (s)"],
                ["time/update_seconds", "Update time (s)"],
                ["time/sim_seconds", "Simulator time (s)"],
                ["time/infer_seconds", "Inference time (s)"],
                ["eval/win_rate", "Evaluation win rate"],
              ].map(([key, label]) => `<tr>
                  <td>${esc(label)}</td>
                  <td class="table-strong">${esc(learningValue(key))}</td>
                  <td>${esc(learningMean(key))}</td>
                </tr>`).join("")}
            </tbody>
          </table></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Performance</h2><p>Throughput and the machine's load.</p></div></div>
        <div class="panel-body">
          <div class="metadata-list" style="padding-inline:0">
            <div class="metadata-row"><span>Steps / s (wall clock)</span><strong>${fmtNumber(performance.steps_per_second)}</strong></div>
            <div class="metadata-row"><span>CPU</span><strong>${Math.round(resourcesBlock.cpu_percent || 0)} %</strong></div>
            <div class="metadata-row"><span>Memory</span><strong>${Math.round(resourcesBlock.memory_percent || 0)} % · ${resourcesBlock.memory_used_gb} / ${resourcesBlock.memory_total_gb} GB</strong></div>
            <div class="metadata-row"><span>GPU</span><strong>${gpu ? esc(gpu.name || "CUDA device") : `<span class="muted">not available</span>`}</strong></div>
            ${resourcesBlock.process ? `<div class="metadata-row"><span>API process</span><strong>${fmtNumber(resourcesBlock.process.rss_mb)} MB · ${fmtNumber(resourcesBlock.process.threads)} threads</strong></div>` : ""}
            <div class="metadata-row"><span>Metric rows kept</span><strong>${fmtNumber((stats.retention || {}).rows)}</strong></div>
          </div>
          <p class="resource-note">The trainer runs in its own process; the numbers above describe this machine, not a single process.</p>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Checkpoints</h2><p>Every checkpoint with the training time it represents.</p></div>
        <a class="text-button" href="#/models">Models ${icon("arrow")}</a></div>
      <div class="panel-body">
        ${checkpointRows ? `<div class="table-wrap"><table class="compact">
          <thead><tr><th>Checkpoint</th><th>Created</th><th>Training time</th><th>Step</th><th>Win rate</th></tr></thead>
          <tbody>${checkpointRows}</tbody></table></div>`
          : emptyState("No checkpoints", "This run has not written a checkpoint yet.", "models")}
      </div>
    </section>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>Evaluation</h2><p>Measured results against fixed opponents.</p></div></div>
        <div class="panel-body">
          ${opponentRows ? `<div class="table-wrap"><table class="compact">
            <thead><tr><th>Opponent</th><th>W / L / D</th><th>Win rate</th><th>Games</th></tr></thead>
            <tbody>${opponentRows}</tbody></table></div>`
            : emptyState("No evaluation", "Nothing was evaluated for this run yet.", "evaluation")}
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Checkpoint comparison</h2><p>Ranked by the measured win rate.</p></div></div>
        <div class="panel-body">
          ${rankingRows ? `<div class="table-wrap"><table class="compact">
            <thead><tr><th>#</th><th>Checkpoint</th><th>Step</th><th>Win rate</th><th>Training time</th></tr></thead>
            <tbody>${rankingRows}</tbody></table></div>`
            : emptyState("Nothing to compare", "No checkpoint of this run has a measured score yet.", "models")}
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Session history</h2><p>Every session of this run with its measured duration.</p></div></div>
      <div class="panel-body">
        ${sessionRows ? `<div class="table-wrap"><table class="compact">
          <thead><tr><th>Started</th><th>Ended</th><th>Duration</th><th>Status</th><th>Steps</th><th>Note</th></tr></thead>
          <tbody>${sessionRows}</tbody></table></div>`
          : emptyState("No sessions", "This run has no recorded session.", "activity")}
      </div>
    </section>

    <section class="panel">
      <div class="panel-header"><div><h2>Training history</h2><p>Every run with its accumulated training time.</p></div>
        <a class="text-button" href="#/monitoring">All runs ${icon("arrow")}</a></div>
      <div class="panel-body">
        ${historyRows ? `<div class="table-wrap"><table class="compact">
          <thead><tr><th>Run</th><th>Status</th><th>Steps</th><th>Training time</th><th>Sessions</th><th>Interrupted</th><th>Started</th></tr></thead>
          <tbody>${historyRows}</tbody></table></div>`
          : emptyState("No runs", "Nothing has been trained yet.", "training")}
      </div>
    </section>

    <section class="panel">
      <div class="panel-header"><div><h2>Run reports</h2><p>A report is written for every run that finishes, is stopped or fails — including the values that could not be measured, labelled as unavailable.</p></div></div>
      <div class="table-wrap"><table class="compact">
        <thead><tr><th></th><th>Run</th><th>Status</th><th>Steps</th><th>Duration</th><th>Steps / s</th><th>Final eval</th><th>Checkpoint</th><th></th></tr></thead>
        <tbody>${reportRows}</tbody>
      </table></div>
      <div class="compare-picker">
        <button class="button secondary" id="compare-button">${icon("chart")} Compare selected</button>
        <span class="subtle">Tick two or more runs. Runs with different modes or observation layouts are flagged as not comparable.</span>
      </div>
      <div id="compare-result" style="padding:0 20px 18px"></div>
    </section>`;

  const select = $("#stats-run");
  if (select) {
    select.addEventListener("change", () => {
      location.hash = `#/stats?run=${encodeURIComponent(select.value)}`;
    });
  }

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
