/* ---------------------------------------------------------------- features
 *
 * The views for the planned product features: the team builder with the
 * difficulty levels and the per-bot behaviour, the persistent player
 * memory, the rival system and the bot evaluation lab.
 *
 * Every view reads its data from the API.  Nothing is invented in the
 * browser: a difficulty level shows its *target* and, once an evaluation
 * exists, the measured value - the two are never merged.
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
  measured,
  fmtNumber,
  fmtDuration,
  pct,
  confirmAction,
  toast,
} = window.RLAI;

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

// ---------------------------------------------------------------- team builder
views.teams = async () => {
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

  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Bot management</span>
        <h1>Teams &amp; bots</h1>
        <p>Build a team, give every bot a difficulty level and a behaviour.</p>
      </div>
    </div>

    <div class="callout">
      <div class="callout-copy">
        <strong>The difficulty percentages are design targets</strong>
        <p>${esc(difficulties.note || "")} A measured win rate only appears after an evaluation against the level.</p>
      </div>
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>New team</h2><p>One to three bots with their own settings.</p></div></div>
        <div class="panel-body">
          <div class="form-grid">
            <label class="field"><span>Team name</span><input id="team-name" type="text" value="my team" /></label>
            <label class="field"><span>Mode</span>
              <select id="team-mode">
                <option value="soccer-1v1">1v1</option>
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
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Difficulty levels</h2><p>What each level actually does to the bot.</p></div></div>
      <div class="panel-body">
        <div class="table-wrap"><table class="compact">
          <thead><tr><th>Level</th><th>Target</th><th>Mistake rate</th><th>Reaction</th><th>Boost use</th><th>Behaviour</th><th>Description</th></tr></thead>
          <tbody>${profileRows}</tbody>
        </table></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-header"><div><h2>Bot behaviour</h2><p>The four behaviour profiles of the Bot Brain.</p></div></div>
      <div class="panel-body">
        <div class="table-wrap"><table class="compact">
          <thead><tr><th>Profile</th><th>Aggression</th><th>Boost</th><th>Rotation</th><th>Shot</th><th>Safety</th><th>Description</th></tr></thead>
          <tbody>${behaviourRows}</tbody>
        </table></div>
      </div>
    </section>`;

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
    button.addEventListener("click", async () => {
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
};

// ---------------------------------------------------------------- player memory
views.memory = async () => {
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

  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Memory &amp; rivals</span>
        <h1>Player memory</h1>
        <p>What the agent remembers about the players it met, and how it fares against its rivals.</p>
      </div>
      <div class="heading-actions">
        <a class="button ghost" href="/api/player-memory/export" download="player-memory.json">${icon("file")} Export</a>
        <button class="button danger" id="reset-memory">${icon("warning")} Reset</button>
      </div>
    </div>

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

    <div class="callout">
      <div class="callout-copy">
        <strong>The memory is data, never model weights</strong>
        <p>${esc(memory.note || "")}</p>
      </div>
    </div>

    <section class="panel">
      <div class="panel-header"><div><h2>Known players</h2><p>Limited to ${fmtNumber(summary.max_entries)} entries and ${fmtNumber(summary.max_age_days)} days.</p></div></div>
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
    </section>`;

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
};

// ---------------------------------------------------------------- evaluation lab
views.lab = async () => {
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

  $("#view").innerHTML = `
    <div class="page-heading">
      <div>
        <span class="eyebrow">Measurement</span>
        <h1>Bot evaluation lab</h1>
        <p>Repeatable experiments, checkpoint comparisons and match analysis.</p>
      </div>
    </div>

    <div class="two-col">
      <section class="panel">
        <div class="panel-header"><div><h2>New experiment</h2><p>Planning never starts a simulation.</p></div></div>
        <div class="panel-body">
          <div class="form-grid">
            <label class="field"><span>Checkpoints</span>
              <select id="lab-models" multiple size="4">${modelOptions || `<option value="">no checkpoint available</option>`}</select>
            </label>
            <label class="field"><span>Opponents</span>
              <input id="lab-opponents" type="text" value="chase" placeholder="chase, idle, random" />
            </label>
            <label class="field"><span>Games</span><input id="lab-games" type="number" min="1" max="200" value="6" /></label>
            <label class="field"><span>Match seconds</span><input id="lab-seconds" type="number" min="5" max="600" value="30" /></label>
            <label class="field"><span>Seed</span><input id="lab-seed" type="number" value="1000" /></label>
            <label class="field"><span>Mode</span>
              <select id="lab-mode">
                <option value="soccer-1v1">1v1</option>
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
      <div class="panel-header"><div><h2>Recent evaluations</h2><p>Every measured series, newest first.</p></div></div>
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
};
