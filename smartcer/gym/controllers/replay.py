"""
Replay controller: plays the optimal actions of the MILP oracle (docs/refactor/REFACTOR_v1.md
phase 7).
"""

from typing import Any

import gymnasium as gym
import numpy as np

from smartcer.gym.controllers.base import Controller


# largest violation of a dynamic bound, as energy on the node side (kWh), that the replay clips
# into the bound without penalty: the round-off of a MILP solver
REPLAY_TOLERANCE_KWH = 1e-6


class ReplayController(Controller):
    """
    Plays the optimal actions a* of the environment's oracle (`env.oracle`).

    An oracle action beyond a dynamic bound of the environment by at most 1e-6 kWh
    (`REPLAY_TOLERANCE_KWH`, the energy (a* - A^upper) P^cha ΔT or (-a* - A^lower) P^dis ΔT on the
    node side) is moved onto the bound, so that the environment does not penalise it, and counted.
    Larger violations are played unchanged: the environment clips and penalises them, which is
    what the replay check (`check_replay`) detects. The actions are float64.

    Attributes:
        clipped_rows: Actions (step and battery) of the episode moved onto a bound.
        clipped_max_kwh: Largest violation moved onto a bound in the episode (kWh).
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
        community = self.env.community
        self._nodes = [community.node(i) for i in self.env.storage_node_ids]
        self.clipped_rows = 0
        self.clipped_max_kwh = 0.0

    def reset(self) -> None:
        """
        Start a new episode: the counts of clipped actions restart from 0.
        """
        self.clipped_rows = 0
        self.clipped_max_kwh = 0.0

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Oracle actions of the step `env.current_step`, with violations of the dynamic bounds up to
        `REPLAY_TOLERANCE_KWH` moved onto the bounds.

        Args:
            observation: Latest observation (not used).
            info: Latest info (not used).

        Returns:
            np.ndarray: a* of the step, float64, in `storage_node_ids` order.
        """
        t = self.env.current_step
        actions = self.env.oracle.action(t)
        for u, node in enumerate(self._nodes):
            a_upper, a_lower = node.get_bounds(t)
            battery, action = node.battery, actions[u]
            if action > a_upper:
                excess, bound = (action - a_upper) * battery.power_charge_max, a_upper
            elif action < -a_lower:
                excess, bound = (-a_lower - action) * battery.power_discharge_max, -a_lower
            else:
                continue
            excess *= node.delta_t
            if excess <= REPLAY_TOLERANCE_KWH:
                actions[u] = bound
                self.clipped_rows += 1
                self.clipped_max_kwh = max(self.clipped_max_kwh, excess)
        return actions
