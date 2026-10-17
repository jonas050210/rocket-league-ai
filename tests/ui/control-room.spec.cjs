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
  expect(geometry.fieldWidth).toBeCloseTo(715, 6);
  expect(geometry.fieldHeight).toBeCloseTo(572, 6);
  expect(geometry.goalWidth).toBeCloseTo(124.6718, 3);
  expect(geometry.orangeRight).toBe(true);
  expect(geometry.blueLeft).toBe(true);
  expect(geometry.positiveXUp).toBe(true);
  const linePixel = await page.evaluate(() => {
    const canvas = document.getElementById("replay-canvas");
    const m = window.__RLAI_GEOMETRY__.mapping(canvas.width, canvas.height);
    const [x, y] = m.px(3000, 0);
    return [...canvas.getContext("2d").getImageData(Math.round(x), Math.round(y), 1, 1).data];
  });
  expect(linePixel[0]).toBeGreaterThan(120);
};

test("read-only preview: all seven views including Roadmap, simplified Benchmark & Training, offline assets", async ({
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
  await expect(page.locator("#navigation [data-nav]")).toHaveCount(7);
  await expect(
    page.getByRole("button", { name: "New training run" }),
  ).toBeDisabled();
  const headings = {
    dashboard: "Training, under control.",
    roadmap: "Roadmap",
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
  await nav(page, "roadmap");
  await expect(page.getByText("Stufe 1 · Ball Chaser & Kickoff")).toBeVisible();
  await expect(page.getByText("Stufe 5 · Champion Master Bot")).toBeVisible();
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
    page.getByRole("button", { name: "Start benchmark" }).first(),
  ).toBeDisabled();
  await expect(page.getByText("No benchmark has been run yet.")).toBeVisible();
  await nav(page, "training");
  await expect(page.locator("#train-max-steps")).toBeDisabled();
  await expect(page.locator("#train-duration-minutes")).toBeDisabled();
  // Legacy deep links still land on Training.
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

test("cached configuration controls disable on disconnect and recover without losing input", async ({
  page,
}) => {
  await connect(page);
  await nav(page, "training");
  await page.locator("#train-max-steps").fill("500000");
  await page.route("**/api/**", (route) => route.abort("failed"));
  await expect(page.locator("#connection-banner")).toBeVisible();
  await expect(page.locator("#train-max-steps")).toHaveValue("500000");
  await page.unroute("**/api/**");
  await expect(page.locator("#connection")).toHaveText("Backend connected");
  await expect(page.locator("#train-max-steps")).toHaveValue("500000");
});

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
    "roadmap",
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
    await page.waitForTimeout(2800);
    await expect(page.locator("#content")).toHaveAttribute("data-page", route);
    await expect(page.locator(".loading-state")).toHaveCount(0);
  }
  const elapsed = Date.now() - started;
  expect(elapsed).toBeGreaterThan(18_000);
  expect(errors).toEqual([]);
});
