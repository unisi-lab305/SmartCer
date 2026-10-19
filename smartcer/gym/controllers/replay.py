"""
Replay controller: plays the optimal actions of the MILP oracle (docs/refactor/REFACTOR_v1.md
phase 7).
"""

from typing import Any

import gymnasium as gym
import numpy as np

from smartcer.gym.controllers.base import Controller


class ReplayController(Controller):
    """
    Plays the optimal actions a* of the environment's oracle (`env.oracle`).

    The actions are float64 and are not clipped by the controller: an oracle action outside the
    dynamic bounds of the environment is clipped and penalised by the environment, which is what
    the replay check (`check_replay`) detects.
    """

    name = 'replay'

    def __init__(self, env: gym.Env) -> None:
        """
        Bind the controller to an environment with an oracle.

        Args:
            env: A `SmartCerEnv` created with `oracle`, possibly wrapped.

        Raises:
            TypeError: If the unwrapped environment is not a `SmartCerEnv`.
            ValueError: If the environment has no oracle.
        """
        super().__init__(env)
        if self.env.oracle is None:
            raise ValueError('ReplayController needs an environment created with an oracle')

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Oracle actions of the step `env.current_step`.

        Args:
            observation: Latest observation (not used).
            info: Latest info (not used).

        Returns:
            np.ndarray: a* of the step, float64, in `storage_node_ids` order.
        """
        return self.env.oracle.action(self.env.current_step)
