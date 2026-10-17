# Failure handling, checkpoints and recovery

Authority: [PROJECT.md](../PROJECT.md). This is a **trusted-local** application; recovery is a deliberate user decision, not an autonomous restart policy.

## 1. Know what disconnected

| Symptom | Meaning / safe first action |
|---|---|
| GUI closed or connection banner | Backend/worker may still be running. Reopen/connect to the same URL/workspace. Do not launch a second run to “recover the window.” |
| `pausing` / `stopping` | Request is pending at a completed PPO update boundary. Inspect heartbeat/progress/logs; do not assume an instantaneous snapshot. |
| `failed` | Worker exception, unexpected exit, watchdog or launch failure. Read Monitoring and the run error first. No automatic restart occurs. |
| `interrupted` after backend restart | Previous service ended without a terminal acknowledgement. Existing artifacts are retained, pending reviews expired, and recovery requires a fresh proposal. |
| Worker lease still held | Prior worker is cleaning up after supervisor loss. Wait for EOF-driven stop/lease release. Never delete a live lock file to force a duplicate worker. |
| CUDA unavailable | Explicit requested device could not initialize. No CPU fallback happened. Fix Windows driver/WSL passthrough, or deliberately approve a separate bounded CPU test plan. |
| Corrupt checkpoint | Failed manifest/schema/file/hash/path checks. Inspect older valid inventory items; do not deserialize or promote the corrupt one. |

The backend owns a process-group supervisor and a 120-second health deadline. A healthy acknowledged pause is exempt from progress timeout, not heartbeat timeout; merely requesting pause does not exempt a wedged worker. On shutdown it waits up to 20 seconds, then contains the group if necessary. These are safety deadlines, not latency or throughput guarantees.

## 2. Graceful stop first

1. Training → **Stop run**. Wait for terminal acknowledgement and worker release.
2. Check Models for a valid checkpoint at the last acknowledged update; inspect logs if publication failed.
3. To stop the backend itself, Ctrl+C its foreground terminal or SIGTERM the correct backend process. GUI close intentionally does not do this.
4. Wait for application shutdown. Uvicorn re-raises the original termination signal after cleanup, so SIGTERM return `-15` / shell 143 is compatible with success. Durable `stopped` state and `backend_stopped` log provide evidence.

If a simulation or learning update throws, the adapter does **not** serialize partially updated parameters as the new recovery checkpoint. Previous completed, committed checkpoints remain authoritative.

## 3. Explicit checkpoint recovery

1. Reconnect/start the backend for the same data directory. A restart does not launch a worker.
2. Read the interrupted/failed run and Monitoring error. Fix the underlying issue before retrying.
3. Models → choose a checkpoint with **valid** integrity and the correct immutable run config. The latest-valid recovery action can fall back to an older verified checkpoint if the newest is corrupt.
4. Review **Recover**. The proposal names the exact run/checkpoint, rationale, expected impact and simulator-reset behavior. Nothing starts yet.
5. Confirm only after reviewing. A **new run ID** links to the original run/checkpoint. Original metadata, weights and history remain intact.
6. Verify real restored counters/logs and the new run's checkpoints. Stop explicitly when appropriate.

Eligibility: original run must be stopped/failed/interrupted, not active/completed; its selected checkpoint must be below its original step budget. Configuration hash, installed training package versions, Python major/minor and the recorded training/config/storage source hash must match (both on proposal/confirmation and again in the worker). A completed budget is not extended through “recovery”; a changed budget/strategy needs a separately reviewed experiment. There is no implicit cross-version checkpoint converter. Missing historical source-profile metadata is also refused; restore the recorded implementation instead of silently resuming changed strategy code.

If there is no valid checkpoint (for example CUDA rejected before initialization), recovery is disabled. Correct the cause and explicitly create a new run. No initialized model or learned progress is fabricated.

### Precisely what resumes

Restored: actor/critic weights, both Adam optimizer states, experience buffer, GAE running statistics, learner iteration/timestep/update counters, Python/NumPy/buffer/PyTorch RNGs (CUDA RNGs for CUDA), completed-episode count/history.

Reset: simulator arena/episode state, in-flight trajectories, partial episode returns and process-local monotonic timing. Recovery is **learner recovery at a completed update**, not bit-exact continuation of a match. Same seed does not imply bitwise equivalence across hardware/drivers, and GPU recovery is a separate target-host check.

Real CPU integration tests compare actual restored tensors, optimizer contents, buffer arrays, GAE statistics, RNGs and counters against the source checkpoint. Unit storage byte fixtures are not evidence of learning-state recovery.

## 4. Checkpoint publication and corruption

Each checkpoint is immutable through the application. The writer creates `.pending-<id>`, flushes/hashes all required files, validates the manifest, fsyncs directories, atomically renames to `step-<step>-<id>`, then atomically updates `latest.json`.

- Interrupted staging is never offered for recovery and is not automatically deleted by a cleanup daemon.
- A broken `latest.json` is not fatal; complete manifests are scanned and verified.
- A corrupt newest checkpoint does not invalidate older good ones.
- Missing files, changed bytes, unexpected files, invalid manifest shapes/types, ID/step mismatch, symlinks or unsafe paths yield a corrupt inventory entry, not a blind load.
- Integrity caches avoid repeated full hashing of unchanged files; all consequential target actions recheck fresh integrity and proposal manifest binding.
- SHA-256 proves consistency with the local manifest, **not provenance**. The buffer includes upstream pickle data. Never load an untrusted or externally supplied checkpoint into this process.

Tests inject storage errors and kill disposable service/worker processes. They do not certify every filesystem, network volume, sudden power failure or storage device. Use Linux storage and keep valuable backups.

## 5. Model designation and deletion

There is **no automatic retention** and no computed “best policy.” A manual designation changes a pointer, not weights or evaluation evidence. Replacing it requires a reviewed, explicit proposal; the previous designation is logged.

Deletion:

- Is offered only for a valid idle inventory item.
- Names the exact run/checkpoint/step and approximate payload bytes in its impact.
- Requires review and confirmation; cancel leaves files unchanged.
- Cannot delete the designated best or the **sole valid** recovery checkpoint of a run.
- Is permanent and not a backup mechanism. Deleting a newer checkpoint while preserving an older one can lose recoverable progress; review the selected step carefully.

Corrupt/pending artifacts are not automatically pruned and cannot bypass integrity protection through the deletion endpoint. If manual quarantine/cleanup is necessary, stop the service/worker, back up valuable state and have the operator explicitly decide what to move/delete. Never edit SQLite or remove random run directories while a service is active.

## 6. Backup and reporting

The UI report exports measured status, inventory metadata and up to the first 1,000 durable events. It does not contain model payloads, the access key or an evaluation result. Raw upstream traces/logs may contain host paths; review reports before sharing. The full event history remains in SQLite; a bounded report is **not** a full log or backup.

For a simple consistent backup:

1. Stop the run and backend gracefully; verify worker/backend leases are free.
2. Copy the **entire workspace**, including SQLite state/sidecars if present, immutable configs/metadata and all desired checkpoint directories, to trusted backup storage.
3. Keep the capability private and protect backups. The key is a local secret, not something to commit or upload with reports.
4. Restore into a separate directory and verify inventory first. Starting that backend only marks unfinished runs interrupted; it does not start training.

There is no online backup wizard, schema migration system, model-import UI or automatic disk pruning in this phase. Do not copy only `latest.json`, only weights, or an actively changing SQLite file and call it a recoverable backup.

## 7. Security and escalation

- Normal service binding is loopback. Capability + same-origin checks are not a substitute for Internet-facing account management/TLS/multi-user isolation.
- Do not publish `browser-url` output or use `--preview-readonly` on valuable private workspaces. The hosted preview used during development is an empty separate workspace.
- Match native GUI `--data-dir` and backend URL; a key for another workspace must fail, not trigger a new training run.
- A duplicate-ownership error is actionable: inspect the correct process and reconnect/wait. Do not remove lock files as an unlock procedure.
- Forced group termination is the last containment option. Inspect logs and valid checkpoints before any confirmed restart; never claim the uncommitted update was recovered.
