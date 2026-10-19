/* ---------------------------------------------------------------- bot lab
 *
 * Everything that is about the *opponents* of the 1v1 policy, in one place:
 * repeatable evaluation experiments, match series, opponent difficulty
 * levels, saved bot teams, match analysis and the player memory.
 *
 * Subtabs live in the route query (``#/lab?tab=series``) so a view stays
 * linked.  Every value comes from the API: a difficulty level shows its
 * *target* and, once an evaluation exists, the measured value - the two are
 * never merged.
 */

const {
  views,
  api,
  $,
  $$,
  esc,
  icon,
  chip,
  emptyState,
  fmtNumber,
  pct,
  confirmAction,
  toast,
  evaluationFormHtml,
  wireEvaluationForm,
  protocolChips,
} = window.RLAI;

const LAB_TABS = ["experiments", "series", "difficulty", "teams", "memory"];
const LAB_TAB_LABELS = {
  experiments: "Experiments",
  series: "Match series",
  difficulty: "Difficulty",
  teams: "Teams",
  memory: "Player memory",
};

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
  });
}

function labSubtabs(active) {
  return `<div class="subtabs" role="tablist" aria-label="Bot Lab sections">
    ${LAB_TABS.map((tab) => `<a
      class="subtab ${tab === active ? "active" : ""}"
      href="#/lab?tab=${tab}"
      role="tab"
      aria-selected="${tab === active}">${esc(LAB_TAB_LABELS[tab])}</a>`).join("")}
  </div>`;
}

views.lab = async () => {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const tab = LAB_TABS.includes(params.get("tab")) ? params.get("tab") : "experiments";
  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Opponents &amp; measurement</span>
        <h1>Bot Lab</h1>
        <p>Measure the 1v1 policy: repeatable experiments, match series, opponent difficulty and what the agent remembers.</p>
      </div>
    </div>
    ${labSubtabs(tab)}
    <div id="lab-body" style="margin-top:18px">
      <div class="loading-state"><span class="spinner"></span> Loading…</div>
    </div>`;
  const body = $("#lab-body");
  const renderTab = { experiments: renderLabExperiments, series: renderLabSeries, difficulty: renderLabDifficulty, teams: renderLabTeams, memory: renderLabMemory };
  try {
    await renderTab[tab](body);
  } catch (error) {
    body.innerHTML = `${emptyState("Could not load this section", error.message, "warning")}`;
    toast(error.message, "error");
  }
};

// ---------------------------------------------------------------- experiments
async function renderLabExperiments(body) {
  const [lab, analyses] = await Promise.all([api("/api/lab"), api("/api/analysis")]);
  const models = lab.models || [];

  const modelOptions = models
    .map(
      (model) =>
        `<option value="${esc(model.id)}">${esc(model.id)} · step ${fmtNumber(model.step)}${model.eval_score === null || model.eval_score === undefined ? "" : ` · ${pct(model.eval_score)}`}</option>`,
    )
    .join("");

  const evaluationRows = (lab.recent_evaluations || []).length
    ? lab.recent_evaluations
        .map(
          (row) => `<tr>
            <td><strong>${esc(row.model_id || "—")}</strong><small>${esc(row.opponent)} · ${esc(row.phase || "manual")}</small></td>
            <td class="table-strong">${fmtNumber(row.wins)} / ${fmtNumber(row.losses)} / ${fmtNumber(row.draws)}</td>
            <td>${(row.details || {}).win_rate === null || (row.details || {}).win_rate === undefined ? `<span class="muted">not measured</span>` : pct(row.details.win_rate)}</td>
            <td>${fmtNumber(row.games)}</td>
            <td>${esc(fmtLocalTime(row.created_at))}</td>
          </tr>`,
        )
        .join("")
    : "";

  const analysis = analyses.evaluations || {};
  const analysisRows = (analysis.by_opponent || []).length
    ? analysis.by_opponent
        .map(
          (row) => `<tr>
            <td><strong>${esc(row.opponent)}</strong></td>
            <td class="table-strong">${fmtNumber(row.wins)} / ${fmtNumber(row.losses)} / ${fmtNumber(row.draws)}</td>
            <td>${row.win_rate === null || row.win_rate === undefined ? `<span class="muted">not measured</span>` : pct(row.win_rate)}</td>
            <td>${fmtNumber(row.goals_for)} : ${fmtNumber(row.goals_against)}</td>
            <td>${fmtNumber(row.mean_touches, 1)}</td>
          </tr>`,
        )
        .join("")
    : "";

  body.innerHTML = `
    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>New experiment</h2><p>Planning never starts a simulation.</p></div></div>
        <div class="panel-body">
          <div class="form-grid">
            <label class="field"><span>Checkpoints</span>
              <select id="lab-models" multiple size="4" aria-label="Checkpoints">${modelOptions || `<option value="">no checkpoint available</option>`}</select>
            </label>
            <label class="field"><span>Opponents</span>
              <input id="lab-opponents" type="text" value="chase" placeholder="chase, idle, random" />
              <small>chase/mixed/self/model:* measure strength; idle/random are diagnostic baselines.</small>
            </label>
            <label class="field"><span>Games per opponent</span><input id="lab-games" type="number" min="1" max="200" value="6" /></label>
            <label class="field"><span>Match seconds</span><input id="lab-seconds" type="number" min="5" max="600" value="30" /></label>
            <label class="field"><span>Seed</span><input id="lab-seed" type="number" value="1000" /></label>
            <label class="field"><span>Mode</span>
              <select id="lab-mode">
                <option value="soccer-1v1" selected>1v1</option>
                <option value="soccer-2v2">2v2</option>
                <option value="soccer-3v3">3v3</option>
                <option value="heatseeker-2v2">Heatseeker 2v2</option>
              </select>
            </label>
          </div>
          <div class="button-row" style="margin-top:12px">
            <button class="button ghost" id="lab-plan">${icon("chart")} Plan</button>
            <button class="button primary" id="lab-run">${icon("play")} Plan &amp; run</button>
          </div>
          <div id="lab-plan-result" style="margin-top:12px"></div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Match analysis</h2><p>What the recorded games actually show.</p></div></div>
        <div class="panel-body">
          ${
            analysisRows
              ? `<div class="table-wrap"><table class="compact">
                  <thead><tr><th>Opponent</th><th>W / L / D</th><th>Win rate</th><th>Goals</th><th>Mean touches</th></tr></thead>
                  <tbody>${analysisRows}</tbody></table></div>`
              : emptyState("No analysed games", "Run an evaluation to produce per-game data.", "chart")
          }
          <p class="resource-note">${esc((analyses.replays || {}).reason || "")}</p>
        </div>
      </section>
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Recent evaluations</h2><p>Every measured series, newest first.</p></div>
        <a class="text-button" href="#/lab?tab=series">Full history ${icon("arrow")}</a></div>
      <div class="panel-body">
        ${evaluationRows ? `<div class="table-wrap"><table class="compact">
          <thead><tr><th>Model</th><th>Record</th><th>Win rate</th><th>Games</th><th>When</th></tr></thead>
          <tbody>${evaluationRows}</tbody></table></div>`
          : emptyState("No evaluations", "Nothing has been evaluated yet.", "evaluation")}
      </div>
    </section>`;

  const collect = () => ({
    model_ids: $$("#lab-models option:checked").map((option) => option.value),
    opponents: $("#lab-opponents").value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    games: Number($("#lab-games").value || 6),
    match_seconds: Number($("#lab-seconds").value || 30),
    seed: Number($("#lab-seed").value || 1000),
    mode: $("#lab-mode").value,
    sides: ["blue"],
    mirror: true,
  });

  const showPlan = async () => {
    const payload = collect();
    const plan = await api("/api/lab/plan", { method: "POST", body: JSON.stringify(payload) });
    $("#lab-plan-result").innerHTML = `<div class="callout success"><div class="callout-copy">
      <strong>Planned experiment</strong>
      <p>${fmtNumber(plan.total_games)} game(s) · about ${fmtNumber(plan.projected_sim_minutes, 1)} min of simulated match time.
      Opponents: ${esc(plan.opponents.join(", "))}. Nothing has been started yet.</p>
    </div></div>`;
    return plan;
  };

  $("#lab-plan").addEventListener("click", async () => {
    try {
      await showPlan();
    } catch (error) {
      toast(error.message, "error");
    }
  });

  $("#lab-run").addEventListener("click", () => {
    confirmAction(
      "Run the evaluation",
      "This starts a real simulation. It can take minutes. Continue?",
      async () => {
        try {
          const payload = collect();
          const plan = await api("/api/lab/plan", { method: "POST", body: JSON.stringify(payload) });
          const result = await api("/api/lab/run", {
            method: "POST",
            body: JSON.stringify({ ...payload, execute: true }),
          });
          toast(`${result.count} series evaluated.`);
          render();
        } catch (error) {
          toast(error.message, "error");
        }
      },
    );
  });
}

// ---------------------------------------------------------------- match series
async function renderLabSeries(body) {
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

  body.innerHTML = `
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
        }).join("")}</tbody></table></div>
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
}

// ---------------------------------------------------------------- difficulty
async function renderLabDifficulty(body) {
  const [difficulties, behaviours] = await Promise.all([api("/api/difficulties"), api("/api/bot-behaviour")]);
  const profiles = difficulties.profiles || [];
  const behaviourProfiles = behaviours.profiles || [];

  const profileRows = profiles
    .map(
      (profile) => `<tr>
        <td><strong>${esc(profile.label)}</strong><small>${esc(profile.name)}</small></td>
        <td class="table-strong">${Math.round(profile.target_win_rate * 100)} % <span class="state-chip">target</span></td>
        <td>${fmtNumber(profile.mistake_rate, 2)}</td>
        <td>${fmtNumber(profile.reaction_delay, 2)} s</td>
        <td>${fmtNumber(profile.boost_efficiency, 2)}</td>
        <td><span class="state-chip">${esc(profile.behaviour)}</span></td>
        <td class="muted">${esc(profile.description)}</td>
      </tr>`,
    )
    .join("");

  const behaviourRows = behaviourProfiles
    .map(
      (profile) => `<tr>
        <td><strong>${esc(profile.name)}</strong></td>
        <td>${fmtNumber(profile.aggression, 2)}</td>
        <td>${fmtNumber(profile.boost_eagerness, 2)}</td>
        <td>${fmtNumber(profile.rotation, 2)}</td>
        <td>${fmtNumber(profile.shot_preference, 2)}</td>
        <td>${fmtNumber(profile.safety, 2)}</td>
        <td class="muted">${esc(profile.description)}</td>
      </tr>`,
    )
    .join("");

  body.innerHTML = `
    <div class="callout">
      <div class="callout-copy">
        <strong>The difficulty percentages are design targets</strong>
        <p>${esc(difficulties.note || "")} A measured win rate only appears after an evaluation against the level (see Match series).</p>
      </div>
    </div>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Difficulty levels</h2><p>What each level actually does to the bot.</p></div></div>
      <div class="panel-body">
        <div class="table-wrap"><table class="compact">
          <thead><tr><th>Level</th><th>Target</th><th>Mistake rate</th><th>Reaction</th><th>Boost use</th><th>Behaviour</th><th>Description</th></tr></thead>
          <tbody>${profileRows}</tbody>
        </table></div>
      </div>
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-header"><div><h2>Bot behaviour</h2><p>The behaviour profiles of the Bot Brain.</p></div></div>
      <div class="panel-body">
        <div class="table-wrap"><table class="compact">
          <thead><tr><th>Profile</th><th>Aggression</th><th>Boost</th><th>Rotation</th><th>Shot</th><th>Safety</th><th>Description</th></tr></thead>
          <tbody>${behaviourRows}</tbody>
        </table></div>
      </div>
    </section>`;
}

// ---------------------------------------------------------------- teams
async function renderLabTeams(body) {
  const [teams, difficulties, behaviours] = await Promise.all([
    api("/api/teams"),
    api("/api/difficulties"),
    api("/api/bot-behaviour"),
  ]);
  const profiles = difficulties.profiles || [];
  const behaviourProfiles = behaviours.profiles || [];

  const difficultyOptions = (selected) =>
    profiles
      .map(
        (profile) =>
          `<option value="${esc(profile.name)}" ${profile.name === selected ? "selected" : ""}>${esc(profile.label)} (target ${Math.round(profile.target_win_rate * 100)} %)</option>`,
      )
      .join("");

  const behaviourOptions = (selected) =>
    behaviourProfiles
      .map(
        (profile) =>
          `<option value="${esc(profile.name)}" ${profile.name === selected ? "selected" : ""}>${esc(profile.name)}</option>`,
      )
      .join("");

  const teamRows = (teams.teams || []).length
    ? teams.teams
        .map(
          (team) => `<tr>
            <td><strong>${esc(team.name)}</strong><small>${esc(team.mode)} · ${esc(team.description || "")}</small></td>
            <td>${team.bots
              .map(
                (bot) =>
                  `${esc(bot.name)} <span class="state-chip">${esc(bot.difficulty)}</span> <span class="state-chip">${esc(bot.behaviour)}</span>`,
              )
              .join("<br>")}</td>
            <td class="table-strong">${team.average_difficulty_target === null || team.average_difficulty_target === undefined ? "—" : `${Math.round(team.average_difficulty_target * 100)} %`}</td>
            <td>${esc(fmtLocalTime(team.updated_at))}</td>
            <td><button class="text-button" data-delete-team="${esc(team.name)}">Delete</button></td>
          </tr>`,
        )
        .join("")
    : "";

  body.innerHTML = `
    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>New team</h2><p>One to three bots with their own settings. 1v1 uses a single bot.</p></div></div>
        <div class="panel-body">
          <div class="form-grid">
            <label class="field"><span>Team name</span><input id="team-name" type="text" value="my team" /></label>
            <label class="field"><span>Mode</span>
              <select id="team-mode">
                <option value="soccer-1v1" selected>1v1</option>
                <option value="soccer-2v2">2v2</option>
                <option value="soccer-3v3">3v3</option>
                <option value="heatseeker-2v2">Heatseeker 2v2</option>
              </select>
            </label>
            <label class="field"><span>Description</span><input id="team-description" type="text" placeholder="optional" /></label>
          </div>
          <div id="team-bots"></div>
          <div class="button-row" style="margin-top:12px">
            <button class="button ghost" id="add-bot">${icon("plus")} Add bot</button>
            <button class="button primary" id="save-team">${icon("check")} Save team</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Saved teams</h2><p>Stored in the local database.</p></div></div>
        <div class="panel-body">
          ${
            teamRows
              ? `<div class="table-wrap"><table class="compact">
                  <thead><tr><th>Team</th><th>Bots</th><th>Avg. target</th><th>Updated</th><th></th></tr></thead>
                  <tbody>${teamRows}</tbody></table></div>`
              : emptyState("No teams yet", "Build one on the left and save it.", "training")
          }
        </div>
      </section>
    </div>`;

  const botContainer = $("#team-bots");
  let counter = 0;
  const addBot = () => {
    counter += 1;
    const row = document.createElement("div");
    row.className = "team-bot-row";
    row.dataset.index = String(counter);
    row.innerHTML = `
      <div class="form-grid">
        <label class="field"><span>Name</span><input class="bot-name" type="text" value="bot-${counter}" /></label>
        <label class="field"><span>Difficulty</span><select class="bot-difficulty">${difficultyOptions("medium")}</select></label>
        <label class="field"><span>Behaviour</span><select class="bot-behaviour">${behaviourOptions("balanced")}</select></label>
        <label class="field"><span>Role</span>
          <select class="bot-role">
            <option value="flex">flex</option>
            <option value="striker">striker</option>
            <option value="defender">defender</option>
            <option value="support">support</option>
          </select>
        </label>
        <label class="field"><span>Model (optional)</span><input class="bot-model" type="text" placeholder="checkpoint id" /></label>
      </div>`;
    botContainer.appendChild(row);
  };
  addBot();
  $("#add-bot").addEventListener("click", addBot);

  $("#save-team").addEventListener("click", async () => {
    const bots = $$(".team-bot-row", botContainer).map((row) => ({
      name: $(".bot-name", row).value.trim() || "bot",
      difficulty: $(".bot-difficulty", row).value,
      behaviour: $(".bot-behaviour", row).value,
      role: $(".bot-role", row).value,
      model: $(".bot-model", row).value.trim() || null,
    }));
    const payload = {
      name: $("#team-name").value.trim(),
      mode: $("#team-mode").value,
      description: $("#team-description").value.trim(),
      bots,
    };
    if (!payload.name) {
      toast("The team needs a name.", "error");
      return;
    }
    try {
      await api("/api/teams", { method: "POST", body: JSON.stringify(payload) });
      toast(`Team ${payload.name} saved.`);
      render();
    } catch (error) {
      toast(error.message, "error");
    }
  });

  $$("[data-delete-team]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.dataset.deleteTeam;
      confirmAction("Delete team", `Delete the team "${name}"? This cannot be undone.`, async () => {
        try {
          await api(`/api/teams/${encodeURIComponent(name)}`, { method: "DELETE" });
          toast(`Team ${name} deleted.`);
          render();
        } catch (error) {
          toast(error.message, "error");
        }
      });
    });
  });
}

// ---------------------------------------------------------------- player memory
async function renderLabMemory(body) {
  const [memory, rivals] = await Promise.all([api("/api/player-memory"), api("/api/rivals")]);
  const summary = memory.summary || {};
  const players = memory.players || [];

  const rows = players.length
    ? players
        .map(
          (player) => `<tr>
            <td><strong>${esc(player.name)}</strong>${player.note ? `<small>${esc(player.note)}</small>` : ""}</td>
            <td class="table-strong">${fmtNumber(player.games)}</td>
            <td>${fmtNumber(player.wins)} / ${fmtNumber(player.losses)} / ${fmtNumber(player.draws)}</td>
            <td>${player.win_rate === null || player.win_rate === undefined ? `<span class="muted">not measured</span>` : pct(player.win_rate)}</td>
            <td>${fmtNumber(player.goals_for)} : ${fmtNumber(player.goals_against)}</td>
            <td>${esc(fmtLocalTime(player.last_seen ? new Date(player.last_seen * 1000).toISOString() : null))}</td>
            <td>
              <button class="text-button" data-note="${esc(player.name)}">Note</button>
              <button class="text-button" data-delete="${esc(player.name)}">Delete</button>
            </td>
          </tr>`,
        )
        .join("")
    : "";

  const rivalRows = (rivals.rivals || []).length
    ? rivals.rivals
        .map(
          (rival) => `<tr>
            <td><strong>${esc(rival.name)}</strong><small>${esc(rival.kind)} · ${esc(rival.reference || "")}</small></td>
            <td class="table-strong">${fmtNumber(rival.games)}</td>
            <td>${fmtNumber(rival.wins)} / ${fmtNumber(rival.losses)} / ${fmtNumber(rival.draws)}</td>
            <td>${rival.win_rate === null || rival.win_rate === undefined ? `<span class="muted">not measured</span>` : pct(rival.win_rate)}</td>
            <td>${fmtNumber(rival.goal_diff)}</td>
            <td>${fmtNumber(rival.evaluations)}</td>
          </tr>`,
        )
        .join("")
    : "";

  const bottleneck = rivals.bottleneck || {};

  body.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card accent"><div class="stat-label"><span>Players</span>${icon("models")}</div>
        <div class="stat-value">${fmtNumber(summary.players)}</div>
        <div class="stat-foot">limit ${fmtNumber(summary.max_entries)}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Meetings</span>${icon("activity")}</div>
        <div class="stat-value">${fmtNumber(summary.games)}</div>
        <div class="stat-foot">${fmtNumber(summary.wins)} won · ${fmtNumber(summary.losses)} lost</div></div>
      <div class="stat-card"><div class="stat-label"><span>Win rate</span>${icon("chart")}</div>
        <div class="stat-value">${summary.win_rate === null || summary.win_rate === undefined ? "—" : pct(summary.win_rate)}</div>
        <div class="stat-foot">goals ${fmtNumber(summary.goals_for)} : ${fmtNumber(summary.goals_against)}</div></div>
      <div class="stat-card"><div class="stat-label"><span>Hardest rival</span>${icon("warning")}</div>
        <div class="stat-value" style="font-size:20px">${bottleneck.available ? esc(bottleneck.name) : "—"}</div>
        <div class="stat-foot">${bottleneck.available ? `${pct(bottleneck.win_rate)} over ${fmtNumber(bottleneck.games)} games` : esc(bottleneck.reason || "not measured")}</div></div>
    </div>

    <div class="callout" style="margin-top:18px">
      <div class="callout-copy">
        <strong>The memory is data, never model weights</strong>
        <p>${esc(memory.note || "")}</p>
      </div>
    </div>

    <div class="two-col" style="margin-top:18px">
      <section class="panel">
        <div class="panel-header"><div><h2>Known players</h2><p>Limited to ${fmtNumber(summary.max_entries)} entries and ${fmtNumber(summary.max_age_days)} days.</p></div>
          <div class="panel-actions">
            <a class="text-button" href="/api/player-memory/export" download="player-memory.json">${icon("file")} Export</a>
            <button class="button ghost" id="reset-memory">${icon("warning")} Reset</button>
          </div>
        </div>
        <div class="panel-body">
          ${rows ? `<div class="table-wrap"><table class="compact">
            <thead><tr><th>Player</th><th>Games</th><th>W / L / D</th><th>Win rate</th><th>Goals</th><th>Last seen</th><th></th></tr></thead>
            <tbody>${rows}</tbody></table></div>`
            : emptyState("Nothing remembered", "The memory is empty - it fills as the agent meets opponents.", "models")}
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><div><h2>Rivals</h2><p>Named opponents with their match history.</p></div></div>
        <div class="panel-body">
          ${rivalRows ? `<div class="table-wrap"><table class="compact">
            <thead><tr><th>Rival</th><th>Games</th><th>W / L / D</th><th>Win rate</th><th>Goal diff</th><th>Evaluations</th></tr></thead>
            <tbody>${rivalRows}</tbody></table></div>`
            : emptyState("No rivals", "Evaluate a checkpoint against an opponent to create the first rival.", "evaluation")}
        </div>
      </section>
    </div>`;

  $("#reset-memory").addEventListener("click", () => {
    confirmAction(
      "Reset the player memory",
      "Delete every remembered player? The memory is separate from the model weights and the training history, so nothing else is affected.",
      async () => {
        try {
          const result = await api("/api/player-memory/reset", { method: "POST" });
          toast(`${result.deleted} player(s) removed.`);
          render();
        } catch (error) {
          toast(error.message, "error");
        }
      },
    );
  });

  $$("[data-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/api/player-memory/${encodeURIComponent(button.dataset.delete)}`, {
          method: "DELETE",
        });
        toast("Player removed.");
        render();
      } catch (error) {
        toast(error.message, "error");
      }
    });
  });

  $$("[data-note]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.dataset.note;
      const current = (players.find((player) => player.name === name) || {}).note || "";
      const note = window.prompt(`Note for ${name}`, current);
      if (note === null) return;
      api("/api/player-memory/note", {
        method: "POST",
        body: JSON.stringify({ name, note }),
      })
        .then(() => {
          toast("Note saved.");
          render();
        })
        .catch((error) => toast(error.message, "error"));
    });
  });
}
