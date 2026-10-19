"""
Rule-based controller of [M24] Algorithm 1, per node or on the community balance
(docs/FORMULATION.md section 13).

Replaces the 0.6.2 `StandardRBC`, which divided the energy by the capacity instead of the power
(defect S6).
"""

from typing import Any, Literal

import gymnasium as gym
import numpy as np

from smartcer.core.nodes import Node
from smartcer.gym.controllers.base import Controller


RBC_MODES = ('node', 'community')


class RBCController(Controller):
    """
    Charges with surplus and discharges to cover deficit, within the dynamic bounds.

    Every action lies in [-A^lower_{u,t}, A^upper_{u,t}] (FORMULATION section 6): the energies
    the rule asks for are limited by the power limits, the charge cap, the efficiencies, S^min,
    S^max and the self-discharge, as for the environment's clipping, so the clipping penalty is
    always 0 (the actions are float64 for this reason). Below, E^cha_u = A^upper_{u,t} P^cha_u Δt
    and E^dis_u = A^lower_{u,t} P^dis_u Δt are the energies (node side) the battery can take and
    deliver in step t.

    - `node`, for every battery node u with d = g_{u,t} - l_{u,t}: d ≥ 0 charges min(d, E^cha_u),
      d < 0 discharges min(-d, E^dis_u); a producer with g_{u,t} = 0 discharges E^dis_u (the
      0.6.2 rule; with d = g it would never discharge). Each node only looks at itself.
    - `community`, with the balance B_t = Σ_u g_{u,t} - Σ_u l_{u,t} over all nodes: B_t > 0
      charges min(B_t, Σ E^cha) split pro rata to E^cha_u; B_t < 0 discharges min(-B_t, Σ E^dis)
      split pro rata to E^dis_u. This is the class A reference baseline.

    The controller reads g and l of step t from the scenario, as in [M24], whatever
    `observe_current_step` is.

    Attributes:
        mode: `node` or `community`.
    """

    def __init__(self, env: gym.Env, mode: Literal['node', 'community'] = 'community') -> None:
        """
        Bind the controller to an environment.

        Args:
            env: A `SmartCerEnv`, possibly wrapped.
            mode: `node` (each battery follows its own node) or `community` (the batteries
                follow the community balance). Defaults to `community`.

        Raises:
            TypeError: If the unwrapped environment is not a `SmartCerEnv`.
            ValueError: If the mode is unknown.
        """
        super().__init__(env)
        if mode not in RBC_MODES:
            raise ValueError(f'unknown RBC mode {mode!r}; available: {list(RBC_MODES)}')
        self.mode = mode
        self.name = f'rbc_{mode}'

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """
        Actions of section 13 for the step `env.current_step`.

        Args:
            observation: Not used.
            info: Not used.

        Returns:
            np.ndarray: One action per battery in [-A^lower, A^upper], float64: the environment
            applies it exactly, while a float32 action equal to a bound may exceed it by about
            1e-8 and be penalised.
        """
        t = self.env.current_step
        nodes = [self.env.community.node(node_id) for node_id in self.env.storage_node_ids]
        bounds = [node.get_bounds(t) for node in nodes]
        if self.mode == 'node':
            actions = [self._node_action(node, t, a_upper, a_lower)
                       for node, (a_upper, a_lower) in zip(nodes, bounds)]
        else:
            actions = self._community_actions(nodes, t, bounds)
        return np.array(actions, dtype=np.float64)

    @staticmethod
    def _node_action(node: Node, t: int, a_upper: float, a_lower: float) -> float:
        """
        Action of mode `node` for one battery node.

        Args:
            node: The battery node.
            t: Step index.
            a_upper: A^upper of the step.
            a_lower: A^lower of the step.

        Returns:
            float: The action in [-a_lower, a_upper].
        """
        battery = node.battery
        generation, load = float(node.generation_kwh[t]), float(node.load_kwh[t])
        if node.node_type == 'producer' and generation == 0.0:
            return -a_lower
        surplus = generation - load
        if surplus >= 0.0:
            return min(surplus / (battery.power_charge_max * node.delta_t), a_upper)
        return -min(-surplus / (battery.power_discharge_max * node.delta_t), a_lower)

    def _community_actions(self, nodes: list[Node], t: int,
                           bounds: list[tuple[float, float]]) -> list[float]:
        """
        Actions of mode `community`: every battery charges (discharges) the same fraction of the
        energy it can take (deliver), so that the batteries together cover the balance B_t.

        Args:
            nodes: The battery nodes, in action order.
            t: Step index.
            bounds: (A^upper, A^lower) per battery node.

        Returns:
            list[float]: The actions, each in [-A^lower, A^upper].
        """
        balance = sum(float(node.generation_kwh[t] - node.load_kwh[t])
                      for node in self.env.community.nodes)
        if balance > 0.0:
            available = sum(a_upper * node.battery.power_charge_max * node.delta_t
                            for node, (a_upper, _) in zip(nodes, bounds))
            share = min(1.0, balance / available) if available > 0.0 else 0.0
            return [share * a_upper for a_upper, _ in bounds]
        if balance < 0.0:
            available = sum(a_lower * node.battery.power_discharge_max * node.delta_t
                            for node, (_, a_lower) in zip(nodes, bounds))
            share = min(1.0, -balance / available) if available > 0.0 else 0.0
            return [-share * a_lower for _, a_lower in bounds]
        return [0.0] * len(nodes)
