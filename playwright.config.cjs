const { defineConfig } = require("@playwright/test");
const portable = require("@sparticuz/chromium").default;
const path = require("node:path");

const python = process.env.RLAI_TEST_PYTHON || path.resolve(".venv/bin/python");
const data = process.env.RLAI_UI_DATA_DIR;
if (!data)
  throw new Error(
    "Run browser tests with npm test (the runner creates isolated workspaces).",
  );
const args = portable.args.filter(
  (flag) =>
    ![
      "--disable-web-security",
      "--disable-site-isolation-trials",
      "--allow-running-insecure-content",
      "--single-process",
    ].includes(flag),
);
module.exports = defineConfig({
  testDir: "./tests/ui",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  reporter: [["list"], ["junit", { outputFile: "test-results/ui-junit.xml" }]],
  use: {
    baseURL: "http://127.0.0.1:8871",
    viewport: { width: 1440, height: 960 },
    colorScheme: "dark",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    launchOptions: {
      executablePath: process.env.RLAI_CHROMIUM_EXECUTABLE,
      args,
    },
  },
  webServer: [
    {
      command: `"${python}" -m rocket_league_ai backend --host 0.0.0.0 --port 8871 --data-dir "${data}/control"`,
      url: "http://127.0.0.1:8871/api/health",
      timeout: 20_000,
      reuseExistingServer: false,
      gracefulShutdown: { signal: "SIGTERM", timeout: 25_000 },
    },
    {
      command: `"${python}" -m rocket_league_ai backend --host 0.0.0.0 --port 8872 --preview-readonly --data-dir "${data}/preview"`,
      url: "http://127.0.0.1:8872/api/health",
      timeout: 20_000,
      reuseExistingServer: false,
      gracefulShutdown: { signal: "SIGTERM", timeout: 25_000 },
    },
  ],
});
