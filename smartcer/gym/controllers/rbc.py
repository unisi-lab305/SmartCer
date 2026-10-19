"""
Rule-based controller of [M24] Algorithm 1 (docs/FORMULATION.md section 13).

Replaces the 0.6.2 `StandardRBC`, which divided the energy by the capacity instead of the power
(defect S6).
"""

from typing import Any

import numpy as np

from smartcer.gym.controllers.base import Controller


class RBCController(Controller):
    """
    Charges with the node's surplus and discharges to cover its deficit, computed in energy.

    For every battery node u at step t, with d = g_{u,t} - l_{u,t} and s the state of charge at
    the end of the previous step:

    - d ≥ 0: charge energy min(d, S^max - s), action = energy / (P^cha Δt);
    - d < 0: discharge energy min(-d, s - S^min), action = -energy / (P^dis Δt).

    The action is limited to [-1, 1] (the action space); the environment then clips it to
    [-A^lower, A^upper] and penalises the excess as for any other action. The controller reads
    g and l of step t from the scenario, as in [M24], whatever `observe_current_step` is. A
    producer (l = 0) never discharges.
    """

    name = 'rbc'

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Actions of section 13 for the step `env.current_step`.

        Args:
            observation: Not used.
            info: Not used.

        Returns:
            np.ndarray: One action per battery in [-1, 1].
        """
        t = self.env.current_step
        community = self.env.community
        actions = np.zeros(len(self.env.storage_node_ids), dtype=np.float32)
        for i, node_id in enumerate(self.env.storage_node_ids):
            node = community.node(node_id)
            battery = node.battery
            surplus = float(node.generation_kwh[t] - node.load_kwh[t])
            if surplus >= 0.0:
                energy = min(surplus, battery.capacity - battery.soc_kwh)
                action = energy / (battery.power_charge_max * node.delta_t)
            else:
                energy = min(-surplus, battery.soc_kwh - battery.soc_min_kwh)
                action = -energy / (battery.power_discharge_max * node.delta_t)
            actions[i] = min(1.0, max(-1.0, action))
        return actions
