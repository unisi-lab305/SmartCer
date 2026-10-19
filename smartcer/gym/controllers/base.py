"""
Base class of the controllers of `SmartCerEnv`: baselines that choose one action per battery.

Adapted from storage-node-environment (`storage_node_env/gym/controllers/base.py`): continuous
actions in [-1, 1] in `storage_node_ids` order instead of a discrete index. A controller is bound to
an environment and may read its community (data and battery states), unlike an agent, which sees
only the observation.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from smartcer.gym.cer_env import SmartCerEnv


class Controller(ABC):
    """
    Chooses the actions of a `SmartCerEnv`, one per battery.

    Attributes:
        name: Controller name, used for the output folders of a run.
        env: The unwrapped environment the controller acts on.
    """

    name: ClassVar[str]

    def __init__(self, env: gym.Env) -> None:
        """
        Bind the controller to an environment.

        Args:
            env: A `SmartCerEnv`, possibly wrapped.

        Raises:
            TypeError: If the unwrapped environment is not a `SmartCerEnv`.
        """
        unwrapped = env.unwrapped
        if not isinstance(unwrapped, SmartCerEnv):
            raise TypeError(f'{type(self).__name__} needs a SmartCerEnv, got '
                            f'{type(unwrapped).__name__}')
        self.env = unwrapped

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.env.community.name!r})'

    def reset(self) -> None:
        """
        Prepare a new episode; called after `env.reset()`. Stateless controllers do nothing.
        """

    @abstractmethod
    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Actions for the step `env.current_step`.

        Args:
            observation: Latest observation of the environment.
            info: Latest info of the environment.

        Returns:
            np.ndarray: One action per battery in [-1, 1], float32, in `storage_node_ids` order.
        """
