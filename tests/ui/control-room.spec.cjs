const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const path = require("node:path");

const preview = "http://127.0.0.1:8872";
const key = () =>
  fs
    .readFileSync(
      path.join(process.env.RLAI_UI_DATA_DIR, "control/access-token"),
      "utf8",
    )
    .trim();
const snapshot = async (page) =>
  (
    await page.request.get("/api/snapshot", {
      headers: { Authorization: `Bearer ${key()}` },
    })
  ).json();
const connect = async (page) => {
  await page.goto(`/#token=${key()}`);
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  await expect(page).toHaveURL(/#dashboard$/);
  await expect(page.locator("#connect-dialog")).not.toBeVisible();
};
const nav = async (page, route) => {
  await page.locator(`[data-nav="${route}"]`).click();
  await expect(page.locator("#content")).toHaveAttribute("data-page", route);
};
const approveDialog = async (page) => {
  await expect(page.locator("#confirmation-dialog")).toBeVisible();
  await expect(page.locator("#confirm-action")).toBeDisabled();
  await expect(page.locator("#proposal-rationale")).not.toBeEmpty();
  await expect(page.locator("#proposal-impact")).not.toBeEmpty();
  await page.locator("#approval-checkbox").check();
  await expect(page.locator("#confirm-action")).toBeEnabled();
  await page.locator("#confirm-action").click();
  await expect(page.locator("#confirmation-dialog")).not.toBeVisible();
};
const assertReplayGeometry = async (page) => {
  await nav(page, "visualization");
  const geometry = await page.evaluate(() => window.__RLAI_GEOMETRY__.selfTest());
  // selfTest maps a fixed 920x620 canvas, so the pitch spans are deterministic.
  expect(geometry.fieldWidth).toBeCloseTo(715, 6);
  expect(geometry.fieldHeight).toBeCloseTo(572, 6);
  expect(geometry.goalWidth).toBeCloseTo(124.6718, 3);
  expect(geometry.orangeRight).toBe(true);
  expect(geometry.blueLeft).toBe(true);
  expect(geometry.positiveXUp).toBe(true);
  const linePixel = await page.evaluate(() => {
    const canvas = document.getElementById("replay-canvas");
    const m = window.__RLAI_GEOMETRY__.mapping(canvas.width, canvas.height);
    const [x, y] = m.px(3000, 0); // halfway line, clear of the centred empty state
    return [...canvas.getContext("2d").getImageData(Math.round(x), Math.round(y), 1, 1).data];
  });
  expect(linePixel[0]).toBeGreaterThan(120);
};
const prepareDraft = async (page, overrides = {}) => {
  const config = {
    ...(await snapshot(page)).plan,
    name: "Browser CPU lifecycle smoke",
    device: "cpu",
    hidden_sizes: [32, 32],
    total_timesteps: 65536,
    timesteps_per_iteration: 128,
    batch_size: 128,
    buffer_size: 384,
    n_minibatches: 2,
    n_proc: 1,
    checkpoint_every_iterations: 4,
    episode_timeout_seconds: 1,
    no_touch_timeout_seconds: 1,
    ...overrides,
  };
  await nav(page, "training");
  await page.locator(".advanced-config > summary").click();
  await page.locator("#config-json").fill(JSON.stringify(config, null, 2));
  return config;
};

// These tests drive the actual application and real PPO workers, never a mocked API.
test("read-only preview: all six views, honest unavailable features, offline assets, responsive layout", async ({
  page,
}, info) => {
  const errors = [],
    external = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (!request.url().startsWith(preview)) external.push(request.url());
  });
  await page.goto(preview);
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  await expect(page.locator("#navigation [data-nav]")).toHaveCount(6);
  await expect(
    page.getByRole("button", { name: "New training run" }),
  ).toBeDisabled();
  const headings = {
    dashboard: "Training, under control.",
    training: "Training",
    monitoring: "Monitoring",
    models: "Models",
    visualization: "Visualization",
    benchmark: "Benchmark",
  };
  for (const [route, title] of Object.entries(headings)) {
    await nav(page, route);
    await expect(page.locator("h1")).toHaveText(title);
    await page.screenshot({
      path: info.outputPath(`${route}.png`),
      fullPage: true,
    });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
  await nav(page, "visualization");
  await expect(
    page.getByText("Run an evaluation from Models to record a replay."),
  ).toBeVisible();
  await assertReplayGeometry(page);
  await nav(page, "models");
  await expect(page.getByText("Checkpoint evaluation")).toBeVisible();
  await expect(page.getByText("Checkpoint league")).toBeVisible();
  await nav(page, "benchmark");
  await expect(
    page.getByRole("button", { name: "Start benchmark" }),
  ).toBeDisabled();
  await expect(page.getByText("No benchmark has been run yet.")).toBeVisible();
  await nav(page, "training");
  await expect(
    page.getByRole("button", { name: "Validate draft" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Review plan changes" }),
  ).toBeDisabled();
  // Legacy deep links still land on the plan panel in Training.
  await page.goto(`${preview}/#configuration`);
  await expect(page.locator("#content")).toHaveAttribute("data-page", "training");
  await expect(page.locator("h1")).toHaveText("Training");
  await page.setViewportSize({ width: 1000, height: 700 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await nav(page, "dashboard");
  await page.setViewportSize({ width: 420, height: 800 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});

test("capability prompt, wrong-key rejection, URL fragment removal, and no auto-start", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#connect-dialog")).toBeVisible();
  await page.locator("#access-key").fill("deliberately-invalid-local-test-key");
  await page.getByRole("button", { name: "Connect workspace" }).click();
  await expect(page.locator("#connect-dialog")).toBeVisible();
  await page.locator("#access-key").fill(key());
  await page.getByRole("button", { name: "Connect workspace" }).click();
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  expect((await snapshot(page)).runs).toHaveLength(0);
  await connect(page);
  expect(page.url()).not.toContain(key());
  expect((await snapshot(page)).active_run).toBeNull();
});

test("real GUI vertical slice: validate, review, reject, approve, pause, reconnect, resume, stop, models, export, recovery", async ({
  page,
  context,
}, info) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await connect(page);
  const original = (await snapshot(page)).plan;
  const config = await prepareDraft(page);
  await page.locator("#config-json").fill("{ invalid JSON");
  await page.getByRole("button", { name: "Validate draft" }).click();
  await expect(page.locator("#validation-message")).toContainText(
    "Configuration JSON is invalid",
  );
  expect((await snapshot(page)).plan).toEqual(original);
  await page.locator("#config-json").fill(JSON.stringify(config, null, 2));
  await page.getByRole("button", { name: "Validate draft" }).click();
  await expect(page.locator("#validation-message")).toContainText(
    "Valid draft. Nothing applied",
  );
  await page.getByRole("button", { name: "Review plan changes" }).click();
  await expect(page.locator("#proposal-diff")).toContainText("device");
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  expect((await snapshot(page)).plan).toEqual(original);
  await page.getByRole("button", { name: "Review plan changes" }).click();
  await approveDialog(page);
  await expect.poll(async () => (await snapshot(page)).plan).toEqual(config);
  expect((await snapshot(page)).runs).toHaveLength(0);
  await nav(page, "training");
  await page.getByRole("button", { name: "New training run" }).click();
  // The approved plan is unchanged, so the explicit click starts it directly.
  await expect(page.locator("#confirmation-dialog")).not.toBeVisible();
  await expect(
    page.getByRole("button", { name: "Pause", exact: true }),
  ).toBeEnabled({ timeout: 30_000 });
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.locator(".panel .status-pill").first()).toHaveText(
    "PAUSED",
  );
  const run = (await snapshot(page)).active_run;
  expect(run.metrics.steps).toBeGreaterThan(0);
  expect(run.metrics.updates).toBeGreaterThan(0);
  expect(run.metrics.win_rate).toBeNull();
  await page.screenshot({
    path: info.outputPath("real-training-paused.png"),
    fullPage: true,
  });

  // Disconnection must visually disable cached controls, not merely reject their handlers.
  await page.route("**/api/**", (route) => route.abort("failed"));
  await expect(page.locator("#connection-banner")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Resume", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Stop run", exact: true }),
  ).toBeDisabled();
  await assertReplayGeometry(page);
  await nav(page, "models");
  await expect(page.getByText("Checkpoint evaluation")).toBeVisible();
  await expect(page.getByText("Checkpoint league")).toBeVisible();
  await nav(page, "benchmark");
  await expect(
    page.getByRole("button", { name: "Start benchmark" }),
  ).toBeDisabled();
  await expect(page.getByText("No benchmark has been run yet.")).toBeVisible();
  await nav(page, "training");
  await expect(
    page.getByRole("button", { name: "Validate draft" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Review plan changes" }),
  ).toBeDisabled();
  await page.unroute("**/api/**");
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  await expect(
    page.getByRole("button", { name: "Validate draft" }),
  ).toBeEnabled();
  await nav(page, "monitoring");
  await expect(page.locator("#log-stream")).toContainText(
    "Paused at a completed PPO update",
  );
  await page.getByRole("textbox", { name: "Filter logs" }).fill("Checkpoint");
  await expect(page.locator("#log-stream")).toContainText("checkpoint");
  await nav(page, "training");
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await expect
    .poll(async () => (await snapshot(page)).active_run.metrics.steps)
    .toBeGreaterThan(run.metrics.steps);
  await page.close();
  const reconnected = await context.newPage();
  await connect(reconnected);
  await nav(reconnected, "training");
  await expect(
    reconnected.getByRole("button", { name: "Pause", exact: true }),
  ).toBeEnabled();
  expect((await snapshot(reconnected)).active_run.id).toBe(run.id);
  await reconnected
    .getByRole("button", { name: "Stop run", exact: true })
    .click();
  await expect(reconnected.locator(".panel .status-pill").first()).toHaveText(
    "STOPPED",
  );
  await expect
    .poll(async () => (await snapshot(reconnected)).active_run)
    .toBeNull();
  await nav(reconnected, "models");
  expect((await snapshot(reconnected)).checkpoints.length).toBeGreaterThan(1);
  const first = reconnected.locator("tbody tr").first();
  await first.locator('[data-action="inspect-model"]').click();
  await expect(
    reconnected.getByRole("dialog", { name: "Checkpoint metadata" }),
  ).toContainText("No evaluation result is available");
  await reconnected.getByRole("button", { name: "Close metadata" }).click();
  await first.getByRole("button", { name: "Designate", exact: true }).click();
  expect((await snapshot(reconnected)).designated_best).toBeNull();
  await approveDialog(reconnected);
  await expect
    .poll(async () => (await snapshot(reconnected)).designated_best)
    .not.toBeNull();
  await expect(
    first.getByRole("button", { name: "Delete", exact: true }),
  ).toBeDisabled();
  const modelsBefore = (await snapshot(reconnected)).checkpoints;
  const deleting = modelsBefore.at(-1);
  await reconnected
    .locator("tbody tr")
    .last()
    .getByRole("button", { name: "Delete", exact: true })
    .click();
  await expect(reconnected.locator("#proposal-target")).toContainText(
    deleting.id,
  );
  await reconnected
    .getByRole("button", { name: "Cancel", exact: true })
    .click();
  expect((await snapshot(reconnected)).checkpoints).toHaveLength(
    modelsBefore.length,
  );
  await reconnected
    .locator("tbody tr")
    .last()
    .getByRole("button", { name: "Delete", exact: true })
    .click();
  await approveDialog(reconnected);
  await expect
    .poll(async () => (await snapshot(reconnected)).checkpoints.length)
    .toBe(modelsBefore.length - 1);
  expect(
    (await snapshot(reconnected)).checkpoints.find(
      (item) => item.id === deleting.id,
    ),
  ).toBeUndefined();

  await reconnected.screenshot({
    path: info.outputPath("real-checkpoint-inventory.png"),
    fullPage: true,
  });
  const downloadPromise = reconnected.waitForEvent("download");
  await reconnected.getByRole("button", { name: "Export inventory" }).click();
  const download = await downloadPromise;
  const filename = info.outputPath("measured-report.json");
  await download.saveAs(filename);
  const report = JSON.parse(fs.readFileSync(filename, "utf8"));
  expect(report.snapshot.runs[0].metrics.steps).toBeGreaterThan(0);
  expect(report.snapshot.evaluation_available).toBe(true);
  expect(report.snapshot.evaluations).toEqual([]);
  expect(JSON.stringify(report)).not.toContain(key());

  await first.getByRole("button", { name: "Recover", exact: true }).click();
  await expect(reconnected.locator("#proposal-impact")).toContainText(
    "Simulator episodes reset",
  );
  expect((await snapshot(reconnected)).runs).toHaveLength(1);
  // The helper also requires a literal checkbox approval for recovery.
  await approveDialog(reconnected);
  await expect(reconnected.locator("#content")).toHaveAttribute(
    "data-page",
    "training",
  );
  await expect(
    reconnected.getByRole("button", { name: "Stop run", exact: true }),
  ).toBeEnabled();
  await reconnected
    .getByRole("button", { name: "Stop run", exact: true })
    .click();
  await expect(reconnected.locator(".panel .status-pill").first()).toHaveText(
    "STOPPED",
    { timeout: 30_000 },
  );
  await expect
    .poll(async () => (await snapshot(reconnected)).active_run)
    .toBeNull();
  const recovered = (await snapshot(reconnected)).runs[0];
  expect(recovered.id).not.toBe(run.id);
  expect(recovered.parent.run_id).toBe(run.id);
  expect(errors).toEqual([]);
});

test("cached configuration controls disable on disconnect and recover without losing draft", async ({
  page,
}) => {
  await connect(page);
  await nav(page, "training");
  await page.locator("#cfg-name").fill("A local unapproved draft");
  await page.route("**/api/**", (route) => route.abort("failed"));
  await expect(page.locator("#connection-banner")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Validate draft" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Review plan changes" }),
  ).toBeDisabled();
  await expect(page.locator("#cfg-name")).toHaveValue(
    "A local unapproved draft",
  );
  await page.unroute("**/api/**");
  await expect(
    page.getByRole("button", { name: "Validate draft" }),
  ).toBeEnabled();
  await expect(page.locator("#cfg-name")).toHaveValue(
    "A local unapproved draft",
  );
  expect((await snapshot(page)).plan.name).not.toBe("A local unapproved draft");
});

test("actual worker CUDA failure is visible and never silently replaced by CPU", async ({
  page,
}) => {
  test.skip(
    process.env.RLAI_TEST_CUDA_UNAVAILABLE !== "1",
    "Opt in only on a verified CUDA-unavailable host.",
  );
  await connect(page);
  await prepareDraft(page, { device: "cuda:0" });
  await page.getByRole("button", { name: "Review plan changes" }).click();
  await approveDialog(page);
  await nav(page, "training");
  await page.getByRole("button", { name: "New training run" }).click();
  await expect(page.locator("#confirmation-dialog")).not.toBeVisible();
  await expect(page.locator(".callout.error")).toContainText(
    "No CPU fallback occurred",
    { timeout: 30_000 },
  );
  await expect(page.locator(".panel .status-pill").first()).toHaveText(
    "FAILED",
  );
  await expect.poll(async () => (await snapshot(page)).active_run).toBeNull();
});

// A deliberately slow pass over every view: twenty seconds of continuous
// rendering, polling and canvas drawing, which catches refresh loops that
// rebuild the DOM mid-interaction or leak page errors between views.
test("twenty-second smoke: every view keeps rendering under polling", async ({
  page,
}) => {
  test.setTimeout(90_000);
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${preview}/`);
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  const routes = [
    "dashboard",
    "training",
    "monitoring",
    "models",
    "visualization",
    "benchmark",
  ];
  const started = Date.now();
  for (const route of routes) {
    await nav(page, route);
    await expect(page.locator("#content h1")).not.toBeEmpty();
    // Stay on the view across several refresh intervals (the client polls
    // once per second) so re-render bugs surface while it is on screen.
    await page.waitForTimeout(3300);
    await expect(page.locator("#content")).toHaveAttribute("data-page", route);
    await expect(page.locator(".loading-state")).toHaveCount(0);
  }
  const elapsed = Date.now() - started;
  expect(elapsed).toBeGreaterThan(18_000);
  expect(errors).toEqual([]);
});
