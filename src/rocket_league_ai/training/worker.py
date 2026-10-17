"""Internal subprocess entry point. Only an explicitly confirmed or direct run launches this module."""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

from ..checkpoints import CheckpointStore
from ..config import TrainingConfig
from ..metadata import require_recovery_compatible
from ..persistence import Lease
from ..store import checked_id
from .protocol import WorkerChannel

# None of the control-plane imports above load NumPy or Torch. This must precede
# their lazy imports below AND be inherited by the spawned environment processes.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    channel = WorkerChannel()
    coordinator = None
    try:
        with Lease(run_dir.parents[1] / "active-worker.lock"):
            config = TrainingConfig.model_validate_json((run_dir / "config.json").read_text())
            import numpy as np
            import torch

            torch.set_num_threads(1)
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            if config.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA was explicitly selected, but PyTorch cannot access a CUDA device. "
                        "Check the Windows NVIDIA driver and WSL 2 GPU passthrough with rocket-ai doctor --probe. "
                        "No CPU fallback occurred. For a bounded correctness test, explicitly approve a CPU configuration."
                    )
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.ones((8, 8), device=config.device).matmul(torch.ones((8, 8), device=config.device))
                torch.cuda.synchronize()
                if config.pin_memory:
                    torch.empty(1, pin_memory=True)
                channel.emit("log", level="info", message=f"CUDA device: {torch.cuda.get_device_name(0)} (TF32+cuDNN enabled); runtime {torch.version.cuda}")
            else:
                channel.emit("log", level="warning", message="Explicit CPU correctness run; not a CUDA/target-machine benchmark")
            from .adapter import ControlledCoordinator

            checkpoint = None
            resume = run_dir / "resume.json"
            if resume.is_file():
                parent = json.loads(resume.read_text())
                parent_dir = run_dir.parents[1] / "runs" / checked_id(parent["run_id"])
                checkpoint_store = CheckpointStore(parent_dir)
                allow_evolution = bool(parent.get("allow_config_evolution"))
                expected_hash = None if allow_evolution else config.digest()
                checkpoint_store.require_valid(parent["checkpoint_id"], expected_hash)
                if not allow_evolution:
                    parent_metadata = json.loads((parent_dir / "metadata.json").read_text())
                    require_recovery_compatible(parent_metadata)
                checkpoint = checkpoint_store.path(parent["checkpoint_id"])
            # Retain a partially constructed instance so cleanup can see an initialized env interface.
            coordinator = ControlledCoordinator.__new__(ControlledCoordinator)
            coordinator.__init__(config, run_dir, channel, checkpoint)
            status = coordinator.run_controlled()
            coordinator.cleanup()
            coordinator = None
            channel.emit("state", status=status, message=f"Run {status}; last valid checkpoint retained")
        return 0
    except BaseException as error:
        traceback.print_exc()
        message = f"{type(error).__name__}: {error}"
        if isinstance(error, ImportError):
            message += ". Install the locked training extra: uv sync --frozen --extra training"
        channel.emit("failure", message=message)
        return 1
    finally:
        if coordinator is not None and hasattr(coordinator, "env_process_interface"):
            try:
                coordinator.cleanup()
            except BaseException:
                traceback.print_exc()
        channel.done.set()


if __name__ == "__main__":
    sys.exit(main())
