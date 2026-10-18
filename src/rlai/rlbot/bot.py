"""RLBot v5 bot process.

Loads a trained checkpoint once and maps the policy's discrete action onto
RLBot's ``ControllerState``.  All observation construction goes through the
same adapter that is parity-tested against the simulator.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Support running this file directly from the deployment folder.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402
from rlbot import flat  # noqa: E402
from rlbot.managers import Bot  # noqa: E402

from rlai.actions import controller_dict  # noqa: E402
from rlai.checkpoints import resolve_model_path, stable_model  # noqa: E402
from rlai.config import data_dir  # noqa: E402
from rlai.modes import get_mode  # noqa: E402
from rlai.obs import version_for_obs_size  # noqa: E402
from rlai.policy import PolicyConfig, TorchPolicy  # noqa: E402
from rlai.rlbot.obs_adapter import packet_to_observation  # noqa: E402


def _resolve_model() -> Path:
    explicit = os.environ.get("RLAI_MODEL")
    models_dir = data_dir() / "models"
    if explicit:
        return resolve_model_path(models_dir, explicit)
    stable = stable_model(models_dir)
    if stable:
        return models_dir / stable["path"]
    raise FileNotFoundError(
        "No model configured. Set RLAI_MODEL to a checkpoint path or promote a model to stable."
    )


def load_policy(path: Path, device: str = "cpu") -> TorchPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = PolicyConfig.from_dict(payload.get("policy_config"))
    policy = TorchPolicy.from_config(config, device=device)
    policy.model.load_state_dict(payload["model"])
    policy.model.eval()
    return policy


def decide_action(
    policy: TorchPolicy,
    packet: flat.GamePacket,
    index: int,
    last_action: int,
    spec,
    obs_version: int | None = None,
) -> tuple[int, dict]:
    """One deterministic policy step: packet -> observation -> action."""
    version = version_for_obs_size(policy.model.obs_size) if obs_version is None else obs_version
    obs, _ = packet_to_observation(packet, index, last_action, spec, version)
    action, _, _ = policy.act(obs, deterministic=True)
    return int(action), controller_dict(int(action))


class RocketLeagueAIBot(Bot):
    def initialize(self) -> None:
        self.mode = os.environ.get("RLAI_MODE", "soccer-1v1")
        self.spec = get_mode(self.mode)
        self.tick_skip = int(os.environ.get("RLAI_TICK_SKIP", "8"))
        self.model_path = _resolve_model()
        self.policy = load_policy(self.model_path)
        self.obs_version = version_for_obs_size(self.policy.model.obs_size)
        self.last_action = 0
        self.tick = 0
        self.decisions = 0
        self.logger.info(
            "Rocket League AI ready | model=%s mode=%s obs_version=%d tick_skip=%d team=%s index=%s",
            self.model_path,
            self.mode,
            self.obs_version,
            self.tick_skip,
            self.team,
            self.index,
        )

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        match_phase = packet.match_info.match_phase
        if match_phase != flat.MatchPhase.Active:
            return flat.ControllerState()
        if self.tick % self.tick_skip == 0:
            self.last_action, _ = decide_action(
                self.policy,
                packet,
                self.index,
                self.last_action,
                self.spec,
                self.obs_version,
            )
            self.decisions += 1
        self.tick += 1
        payload = controller_dict(self.last_action)
        return flat.ControllerState(**payload)


def main() -> None:
    RocketLeagueAIBot("rlai/agent").run()


if __name__ == "__main__":
    main()
