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

from rlbot import flat  # noqa: E402
from rlbot.managers import Bot  # noqa: E402

from rlai.actions import controller_dict  # noqa: E402
from rlai.checkpoints import resolve_model_path, stable_model, validate_checkpoint  # noqa: E402
from rlai.checkpoints import load_policy as _shared_load_policy  # noqa: E402
from rlai.config import data_dir  # noqa: E402
from rlai.modes import get_mode  # noqa: E402
from rlai.obs import mirror_observation, version_for_obs_size  # noqa: E402
from rlai.policy import TorchPolicy  # noqa: E402
from rlai.rlbot.obs_adapter import packet_to_observation  # noqa: E402

BLUE_TEAM = 0
ORANGE_TEAM = 1


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


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


def team_from_name(name: str | int | None) -> int:
    """``"blue"``/``0`` -> 0, ``"orange"``/``1`` -> 1."""
    if isinstance(name, int):
        return ORANGE_TEAM if name else BLUE_TEAM
    text = str(name or "blue").strip().lower()
    if text in ("orange", "1"):
        return ORANGE_TEAM
    if text in ("blue", "0"):
        return BLUE_TEAM
    raise ValueError(f"unknown side {name!r}; use blue or orange")


def load_policy(path: Path | str, device: str = "cpu", *, weights_only: bool = True) -> TorchPolicy:
    """Load a checkpoint into a policy.

    Delegates to :func:`rlai.checkpoints.load_policy`, which refuses files that
    would execute code through ``pickle`` unless the caller explicitly passes
    ``weights_only=False`` for a trusted legacy file.
    """
    return _shared_load_policy(path, device=device, weights_only=weights_only)


def decide_action(
    policy: TorchPolicy,
    packet: flat.GamePacket,
    index: int,
    last_action: int,
    spec,
    obs_version: int | None = None,
    *,
    mirror: bool = False,
    deterministic: bool = True,
) -> tuple[int, dict]:
    """One policy step: packet -> observation -> (optional mirror) -> action.

    ``mirror=True`` rewrites the observation as seen from the opposite side
    (180 deg about the vertical axis) so a checkpoint that trained on blue can
    play orange.  Actions are car-relative and are *not* transformed; this is the
    same convention the side-balanced evaluation uses and is covered by
    ``tests/test_side_balance.py`` and ``tests/test_obs_mirror.py``.
    """
    version = version_for_obs_size(policy.model.obs_size) if obs_version is None else obs_version
    obs, _ = packet_to_observation(packet, index, last_action, spec, version)
    if mirror:
        obs = mirror_observation(obs, version)
    action, _, _ = policy.act(obs, deterministic=deterministic)
    return int(action), controller_dict(int(action))


class RocketLeagueAIBot(Bot):
    #: Class-level defaults so a partially constructed bot (tests, or an RLBot
    #: version that calls get_output before initialize) still behaves sanely.
    mirror = False
    deterministic = True
    errors = 0
    tick_skip = 8
    obs_version = 2
    last_action = 0
    tick = 0
    decisions = 0
    trained_side = "blue"

    def initialize(self) -> None:
        self.mode = os.environ.get("RLAI_MODE", "soccer-1v1")
        self.spec = get_mode(self.mode)
        self.tick_skip = max(1, int(os.environ.get("RLAI_TICK_SKIP", "8")))
        self.model_path = _resolve_model()
        self.policy = load_policy(self.model_path)
        self.obs_version = version_for_obs_size(self.policy.model.obs_size)
        self.deterministic = _flag("RLAI_DETERMINISTIC", True)
        self.trained_side = os.environ.get("RLAI_TRAINED_SIDE", "blue")
        self.mirror = _flag("RLAI_MIRROR", True) and (
            int(self.team) != team_from_name(self.trained_side)
        )
        self.last_action = 0
        self.tick = 0
        self.decisions = 0
        self.errors = 0
        self._check_checkpoint()
        self.logger.info(
            "Rocket League AI ready | model=%s mode=%s obs_version=%d tick_skip=%d team=%s "
            "index=%s trained_side=%s mirror=%s",
            self.model_path,
            self.mode,
            self.obs_version,
            self.tick_skip,
            self.team,
            self.index,
            self.trained_side,
            self.mirror,
        )

    def _check_checkpoint(self) -> None:
        """Refuse to play with a checkpoint that cannot be validated."""
        validation = validate_checkpoint(
            self.model_path,
            mode=None,
            obs_version=self.obs_version,
            load=False,
        )
        if not validation.ok:
            raise RuntimeError(
                f"checkpoint {self.model_path} is not usable: {'; '.join(validation.reasons)}"
            )
        if validation.mode and validation.mode != self.spec.id:
            self.logger.warning(
                "checkpoint was trained in %s but RLAI_MODE is %s; the observation layout "
                "matches, but behaviour was never measured in this mode",
                validation.mode,
                self.spec.id,
            )
        if validation.step is not None:
            self.logger.info(
                "loaded checkpoint at step %s (run %s, label %s)",
                validation.step,
                validation.run_id or "?",
                validation.label or "?",
            )

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        match_phase = packet.match_info.match_phase
        if match_phase != flat.MatchPhase.Active:
            return flat.ControllerState()
        if self.tick % self.tick_skip == 0:
            try:
                self.last_action, _ = decide_action(
                    self.policy,
                    packet,
                    self.index,
                    self.last_action,
                    self.spec,
                    self.obs_version,
                    mirror=self.mirror,
                    deterministic=self.deterministic,
                )
                self.decisions += 1
            except Exception as exc:  # keep the match alive, say what happened
                self.errors += 1
                if self.errors <= 5 or self.errors % 300 == 0:
                    self.logger.error(
                        "decision %d failed (%s: %s); repeating the previous action",
                        self.decisions + 1,
                        type(exc).__name__,
                        exc,
                    )
        self.tick += 1
        payload = controller_dict(self.last_action)
        return flat.ControllerState(**payload)


def main() -> None:
    RocketLeagueAIBot("rlai/agent").run()


if __name__ == "__main__":
    main()
