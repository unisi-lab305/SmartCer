"""
Naive controller: batteries always idle.

Adapted from storage-node-environment (`storage_node_env/gym/controllers/naive.py`): the neutral
action is 0 in the continuous action space.
"""

from typing import Any

import numpy as np

from smartcer.gym.controllers.base import Controller


class NaiveController(Controller):
    """
    Zero action for every battery: the community without battery control.

    With zero actions the batteries only lose their self-discharge, and the clipping penalty is 0.
    """

    name = 'naive'

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Zero actions.

        Args:
            observation: Not used.
            info: Not used.

        Returns:
            np.ndarray: Zeros, one per battery.
        """
        return np.zeros(len(self.env.storage_node_ids), dtype=np.float32)
