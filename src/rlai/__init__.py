"""Rocket League AI — self-training Rocket League agent.

A local-first application that trains discrete-action PPO policies in a
headless RocketSim environment, evaluates them reproducibly, manages model
checkpoints and can deploy a policy through RLBot v5.
"""

from .version import __version__

__all__ = ["__version__"]
