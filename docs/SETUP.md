# Setup, launch and verification

Current contracts/defaults: [PROJECT.md](../PROJECT.md). Recovery: [RECOVERY.md](RECOVERY.md). Exact performed checks: [VERIFICATION.md](VERIFICATION.md).

## 1. Supported host

- **Ubuntu 24.04 on WSL 2** is the recommended target. Ubuntu 22.04 may work with the native prerequisites, but has not been certified here. This is a Linux application; do not install the training profile into native Windows Python.
- Target: i7-12700F, RTX 4060 Ti 8 GB, 32 GB host RAM. The sandbox used for implementation is Debian 12, Python 3.11.2, about 3.8 GiB RAM and **no GPU/display**; it cannot certify WSL/CUDA.
- Python **3.12** is recommended (`.python-version`); Python **3.11–3.12** is the declared range. Select the interpreter explicitly if using 3.11. PySide6 6.10.2 needs a modern Linux native runtime; installing the Python wheel alone is insufficient.
- Put source and `.rlai` in the Linux filesystem, for example `~/projects/rocket-league-ai`, **not `/mnt/c`**. Atomic checkpoint publication, SQLite WAL and `flock` assume ordinary Linux filesystem behavior.

In a Windows terminal, inspect/update WSL with `wsl --status`, `wsl --version`, `wsl --update`, and select an Ubuntu 24.04 WSL 2 distribution. Run the rest of the project commands **inside Ubuntu**. WSLg normally supplies DISPLAY/WAYLAND_DISPLAY automatically; do not overwrite those variables with a guessed Windows IP.

Do not blindly apply the historical research's RAM/core allocation estimates. Inspect `free -h`, `nproc` and the GUI host readings first. Any `.wslconfig` adjustment is an operator change on Windows, requires WSL shutdown/restart, and is not performed by this application.

## 2. System prerequisites

For Ubuntu **24.04**:

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv pipx \
  libgl1 libegl1 libopengl0 libnss3 libnspr4 libdbus-1-3 libxkbcommon0 libxkbcommon-x11-0 \
  libx11-xcb1 libxcb-util1 libxcb-cursor0 libxcb-dri3-0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 \
  libxcb-xinerama0 libxcb-xkb1 libxcomposite1 libxdamage1 libxfixes3 \
  libxrandr2 libxrender1 libxtst6 libxi6 libasound2t64 libgbm1 libxshmfence1 libxkbfile1 \
  fonts-dejavu-core
pipx install uv==0.13.1
pipx ensurepath
```

Open a new Ubuntu shell after `pipx ensurepath`, or add `~/.local/bin` to PATH for the current shell. Ubuntu 22.04 uses `libasound2` instead of `libasound2t64`; do not blindly run the 24.04 command on another distribution. Package installation needs the host's normal distribution repositories; this restricted coding sandbox cannot use them.

For headless **native Qt testing** add `xvfb xauth`. Xvfb is a test display, not a replacement for validating WSLg on the real host.

### NVIDIA / WSL rules

Install/update the **Windows NVIDIA driver** that supports WSL GPU passthrough. Do **not** install a Linux NVIDIA kernel driver, `cuda-drivers` or the `cuda` driver metapackage inside WSL. The locked PyTorch wheel supplies its user-space CUDA runtime; this eager-MLP slice does not require an independently installed CUDA toolkit/compiler.

Check `nvidia-smi` inside WSL (or `/usr/lib/wsl/lib/nvidia-smi`) and `/usr/lib/wsl/lib/libcuda.so.1`, then run the actual application probe below. A GPU name or `nvidia-smi` alone is not proof that the installed PyTorch build can perform CUDA math. This lock resolves Torch 2.9.1's CUDA 12.8 Linux profile; keep the Windows driver compatible/current and verify actual execution.

## 3. Locked Python environment

From the repository root:

```bash
uv --version  # expected tooling profile: 0.13.1
uv sync --frozen --all-extras --python 3.12
uv run --no-sync rocket-ai doctor
uv run --no-sync rocket-ai doctor --probe
```

This installs service, training, desktop and developer checks. For reproducible valuable runs, commit the training code and keep the lock unchanged before starting. A dirty-source hash identifies edits but is not a backup of an uncommitted tree; do not edit source/packages during an active run. Ensure an existing dedicated data directory is private/owned by you; capability files are mode 0600, but existing directory permissions are not silently rewritten.

Narrower installations:

```bash
# Headless service + real training, no desktop:
uv sync --frozen --extra training --python 3.12
# Runtime-only full app, excluding developer tools:
uv sync --frozen --extra training --extra desktop --no-dev --python 3.12
# Service/UI inventory only, no real learner or native shell:
uv sync --frozen --no-dev --python 3.12
```

A service-only installation refuses a training proposal with missing-package instructions; it never substitutes a toy environment. For an existing system Python 3.11, replace `--python 3.12` with `--python 3.11`. If `uv run` subsequently tries to switch back to `.python-version`'s 3.12, pass `--python 3.11` or invoke `.venv/bin/rocket-ai` / `.venv/bin/python` explicitly. Do not change the committed lock merely to run a CPU smoke.

`doctor` without `--probe` reads package metadata and does not initialize a learner. `doctor --probe` performs a small CUDA matmul when available and a real two-car/eight-tick RocketSim probe with 16 meshes; it **does not train**. It exits **1** when CUDA is unavailable (even if simulation passes), making the missing target gate explicit. Do not turn a failed probe into a silent CPU fallback.

### Compatibility notes

- Distribution `rocketsim` imports as **`RocketSim`**.
- Published `rlgym-rocket-league==2.0.1` requires **NumPy <2**. This lock uses 1.26.4. Do not install current `rlviser-py` into the same profile; its NumPy 2 requirement conflicts. Rendering is not implemented.
- The `rlgym-rocket-league` sdist metadata/source and installed wheel are authoritative, not conflicting GitHub HEAD snippets or similarly named `rlgym-sim`/older `gym` integrations.
- RocketSim 2.2.1's internal WHEEL tags are malformed for its ABI3 filename. `uv pip check` reports a CPython-3.9 tag warning even though the installed module and real environment run on 3.11. Standard `python -m pip check` passes. Do not patch/hide the metadata defect; see [UPSTREAM.md](UPSTREAM.md).
- If pip is not installed in uv's venv, a non-mutating check can use `uv run --no-sync --with pip python -m pip check` (the temporary `pip` tool is outside the application lock). This command was used for dependency verification; pip is not a runtime dependency.

## 4. Start the native application

```bash
uv run --no-sync rocket-ai desktop
```

The launcher imports native Qt first, then starts/connects to a detached loopback backend on port 8765. It opens an off-record native window and passes the workspace capability in a fragment that is removed by the client. **No training starts.** Close/reopen the window without shutting down an active run.

Custom workspace/port:

```bash
uv run --no-sync rocket-ai desktop --data-dir .rlai/my-workspace \
  --backend-url http://127.0.0.1:8870
```

Automatic backend startup is local HTTP/loopback only. A running backend belonging to another workspace is rejected, not silently reused. Explicit service mode:

```bash
uv run --no-sync rocket-ai backend --data-dir .rlai/my-workspace --port 8870
uv run --no-sync rocket-ai desktop --data-dir .rlai/my-workspace \
  --backend-url http://127.0.0.1:8870
```

Closing the desktop does not stop this service. Use Ctrl+C in its terminal or SIGTERM to the **backend process** for cooperative worker shutdown. Wait for shutdown acknowledgement; a second forced signal can bypass graceful cleanup. Uvicorn 0.42 re-raises the original termination signal after lifespan cleanup, so a SIGTERM process return of 143 / Python `-15` can accompany a fully successful graceful shutdown. Inspect durable run status and shutdown logs rather than treating exit zero as the only success signal.

### Browser diagnostics / hosted preview

```bash
uv run --no-sync rocket-ai browser-url --data-dir .rlai/my-workspace \
  --backend-url http://127.0.0.1:8870
```

Open the printed capability URL **privately and locally**. Never paste it into issues/chat or share it. Browser code uses relative URLs, not another machine's localhost. When using a hosted proxy, the browser-facing origin must be the proxied origin; do not copy a sandbox loopback address into the user's browser.

For an empty, deliberately public diagnostic workspace only:

```bash
uv run --no-sync rocket-ai backend --host 0.0.0.0 --port 8765 \
  --preview-readonly --data-dir .rlai/preview
```

This opt-in mode permits read-only GETs and embedding; mutation still requires a valid capability. Without one, control buttons are disabled. Do not use it to publish private logs/experiments, and do not expose a normal privileged backend to an untrusted network. This is a trusted-local control plane, not an Internet multi-user product.

## 5. First real smoke from the GUI

The approved default plan explicitly selects `cuda:0`; it must fail rather than silently use CPU if CUDA is unavailable. For a tiny CPU correctness test, choose the following **new reviewed plan**:

```json
{
  "device": "cpu",
  "total_timesteps": 512,
  "timesteps_per_iteration": 128,
  "batch_size": 128,
  "buffer_size": 384,
  "hidden_sizes": [32, 32],
  "n_proc": 1,
  "n_minibatches": 2,
  "checkpoint_every_iterations": 2,
  "episode_timeout_seconds": 1,
  "no_touch_timeout_seconds": 1,
  "pin_memory": false
}
```

These are overrides, **not** a complete copy/paste replacement for the complete JSON editor; preserve the other fields of the current full draft. The form's batch field also sets collection size and 3× buffer capacity. Advanced JSON exposes all validated fields.

1. Configuration → edit → **Validate draft**. Validation does not apply or start anything.
2. **Review & apply** → inspect field differences/rationale/impact → tick review checkbox → confirm. Declining leaves the approved plan unchanged.
3. Training → **New training run** → separately inspect/confirm. Initialization creates the real environment and an initial checkpoint.
4. For pause/resume/stop testing, use a somewhat longer but still bounded budget such as 32,768 steps; a 512-step run may complete before you click pause. Requests acknowledge only at completed updates.
5. Monitoring shows genuine structured logs and counters. Models shows actual checkpoints and explicitly “not evaluated.” Visualization explicitly has no connected renderer.

No skill can be inferred from a few updates, mostly zero kickoff rewards, entropy or a successful checkpoint load. Do not begin competitive training as a side effect of setup.

## 6. Automated checks

Full test environment requires `uv sync --frozen --all-extras`. Tests never use a real user workspace.

```bash
uv run --no-sync ruff check src tests
uv run --no-sync python -m compileall -q src
uv run --no-sync pytest -q -m 'not integration and not desktop'
uv run --no-sync pytest -q -m 'integration and not desktop'
uv run --no-sync pytest -q -m 'not desktop'
```

Integration tests use actual RocketSim/RLGym Learn/PPO/Torch subprocesses, CPU correctness plans, temporary directories, explicit test confirmations and bounded budgets. They include worker/backend failure injection; do not point them at valuable data. Unit storage byte fixtures are explicitly unit-only and are never fed into PPO.

Browser tests require Linux x86_64 and **Node 22.22.3** (the locally verified version) or a compatible newer release:

```bash
npm ci
npm run check
npm test
# Enable the real CUDA-refusal browser test only after verifying this host lacks CUDA:
RLAI_TEST_CUDA_UNAVAILABLE=1 npm test
```

The runner creates unique ignored `.rlai/ui-tests/<uuid>` workspaces and starts test services on ports **8871/8872**; those ports must be free. It extracts an npm-pinned headless Chromium and bundled NSS libraries without downloading a browser at runtime. It **does not disable web security**. `--no-sandbox` is confined to this container browser-test profile, not a production GUI recommendation. `RLAI_CHROMIUM_EXECUTABLE` can select a separately installed compatible browser; `RLAI_TEST_PYTHON` can select the test interpreter.

Artifacts are ignored `test-results/` (native CI PNGs: `native-test-results/`) (screenshots, measured JSON report, failure traces). Test capabilities are disposable; do not publish equivalent traces from a real private workspace. Browser tests explicitly start and recover real small CPU workers; all app assets remain offline.

### Native Qt tests (explicit opt-in)

Under working WSLg/X11:

```bash
RLAI_NATIVE_TEST=1 uv run --no-sync pytest -q -m desktop
```

On an Ubuntu CI/headless host with native dependencies, Xvfb and xauth:

```bash
QT_OPENGL=software QTWEBENGINE_DISABLE_SANDBOX=1 \
QTWEBENGINE_CHROMIUM_FLAGS='--disable-gpu --disable-dev-shm-usage' \
RLAI_NATIVE_TEST=1 xvfb-run -a -s '-screen 0 1440x1000x24' \
  uv run --no-sync pytest -q -m desktop
```

Disabling the WebEngine sandbox is a **trusted ephemeral CI/container workaround only**; do not add it to everyday WSL desktop launch. The application does not set it. These tests actually launch Qt, capture a PNG, reconnect/close during a real paused worker, resume/stop independently, and test native detached-backend bootstrap without autotraining. Without opt-in they skip with a reason. This command has actually passed in Ubuntu 24.04/Python 3.12.15 CI (2 native tests, zero skips). It does not certify interactive WSLg or the save dialog; consult [VERIFICATION.md](VERIFICATION.md).

## 7. Troubleshooting

- **`libGL.so.1`, NSS, xcb, xkbcommon, GBM or Qt platform plugin error:** install matching native packages above. `ldd` the relevant PySide6 Qt library/plugin and inspect missing dependencies. A minimal headless Chromium test passing does not supply the native Qt runtime. Do not fetch random replacement libraries, patch ELF symbols or permanently disable security to claim a successful launch.
- **No display/WSLg window:** verify WSLg is updated and DISPLAY/WAYLAND_DISPLAY are present. Test a simple native Ubuntu GUI independently; use the local diagnostic client to inspect backend state meanwhile. Do not restart training to fix a GUI-only issue.
- **CUDA unavailable:** update/check the Windows driver and WSL passthrough, verify actual Torch probe. An explicit CPU smoke is a separate approved configuration, not an automatic fallback.
- **`Already inited` from outside code:** application-managed initialization is idempotent after its own successful init; do not initialize RocketSim separately against unrelated meshes before importing/using the environment.
- **Workspace key rejected:** match `--data-dir`, backend URL and port. Do not print the capability in shared logs. Stop the service before deliberate key regeneration; there is no unauthenticated reset endpoint.
- **Pausing/stopping:** await a PPO update boundary, inspect heartbeat/progress/logs. Do not assume the click was an instant snapshot. A 120-second health deadline contains a stalled worker; previous committed checkpoints remain available.
- **Orphan lease held:** wait for EOF-driven cleanup; do not delete live lock files or launch a second worker. See [RECOVERY.md](RECOVERY.md).
- **Disk growth:** checkpoints/logs are retained by default; there is no pruning daemon. Review and explicitly approve deletion of unprotected models; preserve a last valid recovery checkpoint and back up valuable runs.
