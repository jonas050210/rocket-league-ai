/* Browser-test runner only. Runtime UI has no Node or browser download requirement. */
const { spawnSync } = require("node:child_process");
const { randomUUID } = require("node:crypto");
const path = require("node:path");
const os = require("node:os");

(async () => {
  const portable = require("@sparticuz/chromium");
  if (!process.env.RLAI_CHROMIUM_EXECUTABLE) {
    process.env.RLAI_CHROMIUM_EXECUTABLE =
      await portable.default.executablePath();
    // This sandbox, like a minimal Linux container, lacks NSS. Extract the package's
    // own libraries explicitly; do not alter the host or disable browser web security.
    await portable.inflate(
      path.resolve("node_modules/@sparticuz/chromium/bin/al2023.tar.br"),
    );
    portable.setupLambdaEnvironment(path.join(os.tmpdir(), "al2023", "lib"));
  }
  process.env.RLAI_UI_DATA_DIR = path.resolve(
    ".rlai",
    "ui-tests",
    randomUUID(),
  );
  const result = spawnSync(
    process.execPath,
    [require.resolve("@playwright/test/cli"), "test", ...process.argv.slice(2)],
    {
      stdio: "inherit",
      env: process.env,
    },
  );
  process.exitCode = result.status ?? 1;
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
