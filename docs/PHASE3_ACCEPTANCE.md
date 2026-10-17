# Phase 3 — Hardware, CUDA, WSLg & Training Stability Acceptance Report

**Date:** 2026-10-17
**Branch:** `arena/d85c45cf-rocket-league-ai`
**Base commit:** `feacfcb1b6fc9d40fd021bea55a1c70015d5614b` (merge of Phase 2 PR #2)
**Application version:** 0.2.0

## Executive Summary

Phase 3 is **BLOCKED** for hardware-dependent acceptance. This session runs in the Arena cloud sandbox — a Debian 12 container with no NVIDIA GPU, no CUDA drivers, no display server, and no WSL2. The target machine (Windows + WSL2 + RTX 4060 Ti + WSLg) is not accessible.

All software-level checks that do not require hardware pass successfully, confirming the codebase is intact and matches the Phase 2 verified state. The target-host gates documented in the roadmap and project reference remain unexecuted.

## 1. Repository State

| Property | Observed |
|---|---|
| Active branch | `arena/d85c45cf-rocket-league-ai` |
| Base commit | `feacfcb` (Phase 2 merge PR #2) |
| Working tree | Clean, no uncommitted changes |
| Merged PRs | #1 (docs), #2 (Phase 2 foundation) |
| Open PRs | None |

Historical documentation under `docs/history/` is preserved:
- `ARCHITECTURE_PHASE1.md` (613 lines)
- `ROADMAP_PHASE1.md` (446 lines)

## 2. Execution Environment (Actual Observed State)

This is the Arena cloud sandbox, **not** the target training machine.

| Property | Observed | Target (per PROJECT.md) |
|---|---|---|
| OS | Debian 12 (bookworm), kernel 6.1.158+ | Ubuntu 24.04 on WSL 2 |
| Host | e2b.local (Arena container) | User's Windows + WSL 2 machine |
| CPU | Intel Xeon @ 2.60GHz, 2 vCPUs | Intel i7-12700F |
| RAM | 3.8 GiB | 32 GB host RAM |
| GPU | None | RTX 4060 Ti 8 GB |
| CUDA | Not available (Torch built for CUDA 12.8) | Expected available |
| Display | None (no DISPLAY, no WAYLAND_DISPLAY) | WSLg |
| WSL | No (no WSLInterop) | WSL 2 |
| Disk | 21 GB total, 12 GB available | Operator-dependent |
| Python | 3.11.2 | 3.12 recommended, 3.11 supported |
| Node | 22.22.3 | 22.22.3+ |
| uv | 0.13.1 | 0.13.1 |

## 3. Work Package 1: Environment and Dependency Verification

### Status: PARTIALLY COMPLETE (sandbox-only)

#### Commands executed and results:

| Command | Exit code | Result |
|---|---|---|
| `uv sync --frozen --all-extras --python 3.11` | 0 | All locked dependencies installed successfully |
| `uv run --no-sync ruff check src tests scripts` | 0 | All checks passed |
| `uv run --no-sync python -m compileall -q src` | 0 | Passed |
| `uv run --no-sync rocket-ai doctor --probe` | 1 | CUDA unavailable (expected); simulation passed |
| `uv run --no-sync pytest -q -m 'not desktop'` | 0 | **98 passed**, 2 deselected, 1 warning, 43.07s |
| `RLAI_NATIVE_TEST=1 uv run --no-sync pytest -q -m desktop` | 1 | 2 failed (missing libGL.so.1), environment-blocked |
| `npm run check` | 0 | JavaScript syntax passed |
| `RLAI_TEST_CUDA_UNAVAILABLE=1 npm test` | 0 | **5 passed**, 38.7s |

#### Doctor probe output (verbatim):

```json
{
  "python": "3.11.2",
  "platform": "linux",
  "training_packages": {
    "rlgym-api": "2.0.0",
    "rlgym-rocket-league": "2.0.1",
    "rocketsim": "2.2.1",
    "rlgym-learn": "2.0.0",
    "rlgym-learn-algos": "0.4.2",
    "torch": "2.9.1",
    "numpy": "1.26.4"
  },
  "desktop_installed": true,
  "cuda": {
    "probed": true,
    "available": false,
    "torch": "2.9.1+cu128",
    "runtime": "12.8"
  },
  "simulation": {
    "soccar_meshes": 16,
    "ticks": 8,
    "status": "passed"
  }
}
```

#### Findings:

- **All locked dependency versions match** the documented profile exactly.
- **Simulation passes:** 16 soccar meshes, 8 ticks — real RocketSim initialization works.
- **CUDA unavailable:** PyTorch 2.9.1+cu128 is correctly built for CUDA 12.8 but no device is present. Exit code 1 is the expected, intentional behavior.
- **Desktop libraries:** PySide6 is installed; native launch fails with missing `libGL.so.1` — consistent with this headless sandbox (same result as Phase 2).
- **Test parity with Phase 2:** All 98 non-desktop Python tests pass, all 5 browser tests pass. The codebase is identical to the Phase 2 verified state.

#### Blockers for full completion:

- Cannot verify CUDA availability without an NVIDIA GPU.
- Cannot verify actual CUDA math (matmul) on the target device.
- Cannot verify GPU name, VRAM, or compute capability.

## 4. Work Package 2: Actual CUDA Training Acceptance

### Status: BLOCKED (no GPU in this environment)

The CUDA training acceptance cannot be performed because:

1. No NVIDIA GPU device is present (`nvidia-smi` not found, `/dev/nvidia*` absent).
2. No NVIDIA kernel driver or `/usr/lib/wsl/lib/libcuda.so.1`.
3. PyTorch reports `cuda.is_available() == False` (correct — no device to detect).

#### What must be verified on the target machine:

Per the roadmap gate N0 and PROJECT.md §9:

1. `nvidia-smi` inside WSL shows the RTX 4060 Ti.
2. `/usr/lib/wsl/lib/libcuda.so.1` exists.
3. `rocket-ai doctor --probe` exits 0 and reports:
   - CUDA available: true
   - Device name: NVIDIA GeForce RTX 4060 Ti (or similar)
   - CUDA matmul: passed
4. Actual training workload executes on GPU (not just reports availability):
   - Launch a bounded CUDA training plan via the GUI
   - Verify `torch.cuda.max_memory_allocated()` > 0 during training
   - Monitor GPU utilization via nvidia-smi during training
   - Confirm VRAM consumption is within the 8 GB budget
   - Confirm no silent CPU fallback
5. Peak allocated/reserved CUDA memory and host RSS are within safe limits.
6. No OOM, non-finite loss/parameters, or missing worker acknowledgement.

#### Code-level readiness assessment:

The training worker (`training/worker.py`) correctly:
- Checks `torch.cuda.is_available()` before using CUDA and raises a clear error if unavailable.
- Performs a CUDA matmul probe (8×8 matrix) before training starts.
- Reports the GPU name and CUDA runtime via structured log events.
- Reports `torch.cuda.max_memory_allocated()` and `torch.cuda.memory_reserved()` in snapshots.
- Refuses to silently fall back to CPU (PROJECT.md rule #9).

The monitoring module (`monitoring.py`) correctly:
- Searches for `nvidia-smi` including the WSL-specific path `/usr/lib/wsl/lib/nvidia-smi`.
- Reports GPU name, utilization, memory used/total.
- Names the reason when GPU readings are unavailable.

**Assessment:** The code is correctly prepared for CUDA acceptance. The blocker is solely the absence of target hardware in this sandbox.

## 5. Work Package 3: WSLg and Desktop Acceptance

### Status: BLOCKED (no display server / WSLg in this environment)

The WSLg desktop acceptance cannot be performed because:

1. No display server is present (DISPLAY and WAYLAND_DISPLAY are empty).
2. No WSLg runtime (this is a cloud container, not WSL2).
3. Missing `libGL.so.1` prevents native Qt initialization.
4. Native Qt tests fail at import with `libGL.so.1: cannot open shared object file`.

#### What must be verified on the target machine:

1. `rocket-ai desktop` opens a native Qt window in WSLg.
2. All six routes (Dashboard, Training, Monitoring, Models, Configuration, Visualization) are usable.
3. Window close and reopen reconnects to the same backend without starting training.
4. Native save dialog works for report export.
5. Native cancel dialog leaves files unchanged.
6. No Qt platform plugin failures, WebEngine issues, or rendering failures.
7. Process cleanup: closing the window does not stop the backend or worker.

#### Code-level readiness assessment:

The desktop module (`desktop.py`) correctly:
- Imports PySide6 first to surface missing native dependencies immediately.
- Starts/connects to a detached loopback backend.
- Uses off-the-record QWebEngineProfile (capability never persisted).
- Injects the capability token via URL fragment (consumed and removed by client).
- Has NO `aboutToQuit` → stop backend/worker connection (window close does not stop training).
- Handles native save/download with proper cancel behavior.
- Restricts navigation to same-origin only.
- Performs deterministic Qt object destruction order.

The CI (GitHub Actions, Ubuntu 24.04, Xvfb) previously passed both native tests:
- Window close/reopen during a paused worker with unchanged counters.
- Detached backend bootstrap surviving window closure.

**Assessment:** The code is correctly prepared for WSLg acceptance. The CI pass on Ubuntu/Xvfb demonstrates the desktop code works with a display. The blocker is the absence of an interactive WSLg environment in this sandbox.

## 6. Work Package 4: Thirty-Minute Stability Test

### Status: BLOCKED (requires CUDA training and WSLg desktop)

The 30-minute stability test requires:
1. A working CUDA environment (for real GPU training).
2. A working WSLg desktop (for GUI reconnect cycles).
3. The ability to run actual RocketSim/RLGym training for 30 minutes.

None of these prerequisites are available in this sandbox.

#### What must be verified on the target machine:

Per the roadmap gate N1:

1. Run a 30-minute operator-supervised soak using bounded runs.
2. Perform at least 3 GUI reconnects and 3 pause/resume cycles.
3. Maintain measured update/lifecycle progress throughout.
4. Explicitly stop at least once, verify all manifests.
5. Corrupt a **disposable test copy** of a newest checkpoint and demonstrate older-valid fallback.
6. Interrupt a disposable service process, verify no automatic restart.
7. Record maximum/median API health latency, peak worker/simulation RSS, VRAM, checkpoint size/growth.
8. No orphaned process groups after normal shutdown.
9. No invalid manifest offered for recovery.

## 7. Work Package 5: Failure Analysis and Remediation

### Status: No target-host failures to analyze (tests not executed)

No defects were found in the software-level checks. The existing test suite (98 Python + 5 browser) passes completely.

## 8. Test Results Summary

### All verifiable tests:

| Test suite | Expected | Actual | Status |
|---|---|---|---|
| Ruff lint | 0 errors | 0 errors | ✅ PASS |
| Python compile | 0 errors | 0 errors | ✅ PASS |
| Non-desktop Python (unit + integration) | 98 pass | 98 pass, 43.07s | ✅ PASS |
| Native Qt desktop | Environment-blocked | 2 failed (libGL.so.1) | ⚠️ ENVIRONMENT-BLOCKED |
| JavaScript syntax | Pass | Pass | ✅ PASS |
| Browser lifecycle (Chromium) | 5 pass | 5 pass, 38.7s | ✅ PASS |
| Doctor --probe | Exit 1 (no CUDA) | Exit 1 (no CUDA), sim passed | ✅ EXPECTED |

The desktop test failure is the **same environment-blocked result** as Phase 2 verification (`docs/VERIFICATION.md` §2). It is not a regression.

### Phase 2 CI parity:

The Phase 2 CI run (Ubuntu 24.04, Python 3.12.15, Xvfb) achieved:
- 98 non-desktop: 0 failures, 0 skips
- 2 native Qt: 0 failures, 0 skips
- 5 browser: 0 failures, 0 skips

The codebase has not changed since that verified state.

## 9. Verdict by Work Package

| Work Package | Verdict | Reason |
|---|---|---|
| WP1: Environment & Dependencies | **PARTIALLY COMPLETE** | All verifiable checks pass; CUDA/GPU verification requires target hardware |
| WP2: CUDA Training Acceptance | **BLOCKED** | No NVIDIA GPU in sandbox |
| WP3: WSLg & Desktop Acceptance | **BLOCKED** | No display server / WSLg in sandbox |
| WP4: 30-Minute Stability Test | **BLOCKED** | Requires CUDA + WSLg |
| WP5: Failure Analysis | **N/A** | No target-host failures (tests not executed) |

## 10. Overall Phase 3 Verdict

**Phase 3 is BLOCKED.** The target machine (Windows + WSL2 + RTX 4060 Ti + WSLg) is not accessible from this sandbox.

Software-level acceptance confirms:
- The codebase is intact and matches the Phase 2 verified state.
- All 98 Python tests and 5 browser tests pass.
- Dependencies are correctly locked and installable.
- The doctor probe correctly reports CUDA unavailable (sandbox) and simulation passed.
- Native Qt tests are environment-blocked (same as Phase 2, not a regression).

Hardware-level acceptance requires the operator to run the following on the target machine:

### Target-Machine Checklist (for operator execution):

```bash
# 1. Environment verification
wsl --status
nvidia-smi
cat /usr/lib/wsl/lib/libcuda.so.1 | head -c 4  # should show ELF magic
uv sync --frozen --all-extras --python 3.12
uv run --no-sync rocket-ai doctor --probe
# Expected: exit 0, CUDA available, matmul passed, simulation passed

# 2. CUDA training acceptance
uv run --no-sync rocket-ai desktop
# In the GUI: start a bounded CUDA plan (e.g., 32768 steps)
# Monitor: GPU utilization, VRAM consumption, training throughput
# Verify: no silent CPU fallback, real GPU execution

# 3. WSLg desktop acceptance
# Verify: all six views render correctly
# Close and reopen window during active run
# Test native save/cancel dialogs

# 4. Thirty-minute stability test
# Run a bounded CUDA experiment for 30+ minutes
# Perform 3+ pause/resume cycles
# Perform 3+ GUI reconnect cycles
# Monitor: VRAM, RSS, throughput, heartbeat, logs

# 5. Recovery verification
# Stop explicitly, verify checkpoint manifests
# Recover into a new linked CUDA run
# Verify config/package equality and valid manifests
```

## 11. Changed Files and Uncommitted Work

No source files were modified. No uncommitted work exists.

| File | Status |
|---|---|
| `docs/PHASE3_ACCEPTANCE.md` | **NEW** — this acceptance report |

## 12. Recommended Next Steps

1. **Operator must execute the target-machine checklist** on their Windows/WSL2 machine with the RTX 4060 Ti.
2. Record the actual `nvidia-smi` output, `doctor --probe` result, and training metrics.
3. If CUDA training succeeds on the target, proceed to the 30-minute stability test.
4. If any hardware-specific defect is found, return to this branch with the exact error logs for narrowly scoped remediation.
5. Do not merge any PR until the full Phase 3 acceptance criteria are met on the actual target hardware.

## 13. Safety Compliance

This report adheres to all Phase 3 safety boundaries:

- ✅ No PRs merged, no pushes, no destructive Git operations performed.
- ✅ No training started without user confirmation.
- ✅ No checkpoints deleted or experiments promoted.
- ✅ No fabricated test results — all results are actual command output.
- ✅ No weakening of tests or safety checks.
- ✅ Historical documentation preserved.
- ✅ Blockers documented honestly rather than claims of acceptance.
