"""
Gymnasium integration: `SmartCerEnv` and its registration as `SmartCer-v1`.

Usage:
    >>> import gymnasium as gym
    >>> import smartcer  # registers SmartCer-v1
    >>> env = gym.make('SmartCer-v1', scenario='path/to/scenario')
"""

from gymnasium.envs.registration import register, registry

from smartcer.gym.cer_env import SmartCerEnv


ENV_ID = 'SmartCer-v1'


def register_envs() -> None:
    """
    Register `SmartCer-v1` with Gymnasium (idempotent).

    Called on `import smartcer` and through the `gymnasium.envs` entry point (used by Gymnasium
    versions before 1.0, which load entry points on import).
    """
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point='smartcer.gym.cer_env:SmartCerEnv')


__all__ = ['ENV_ID', 'SmartCerEnv', 'register_envs']
