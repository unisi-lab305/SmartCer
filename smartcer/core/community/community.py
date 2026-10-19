"""
A community of nodes stepped together (docs/FORMULATION.md sections 7-10).

The community sends each battery action to its node, collects the node flows and computes shared
energy, incentive, welfare and the mean constraint penalty K_t. It is what the Gymnasium
environment wraps.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from smartcer.core.community.incentives import IncentiveModel
from smartcer.core.community.shared_energy import shared_energy
from smartcer.core.community.welfare import WelfareBreakdown, welfare_breakdown
from smartcer.core.nodes import Node, NodeStep


@dataclass(frozen=True, slots=True)
class CommunityStep:
    """
    Record of one community step.

    Attributes:
        t: Step index.
        nodes: One record per node, in the community's node order.
        welfare: Welfare breakdown of the step.
        penalty: Mean constraint penalty over the nodes with a battery,
            K_t = Σ_{u ∈ U_b} k_u / |U_b| (0 without batteries).
    """

    t: int
    nodes: tuple[NodeStep, ...]
    welfare: WelfareBreakdown
    penalty: float


class Community:
    """
    Nodes, incentive model and time axis of a scenario.

    Attributes:
        name: Scenario name.
        nodes: Nodes in a fixed order.
        node_ids: Node identifiers in the same order.
        storage_node_ids: Identifiers of the nodes with a battery, in node order; this is the order
            of the actions.
        incentive: Incentive model.
        timestamps: Start time of each step, numpy datetime64 array.
        delta_t: Duration of a step (hours).
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        incentive: IncentiveModel,
        timestamps: np.ndarray,
        delta_t: float,
        name: str = '',
    ) -> None:
        """
        Create a community.

        Args:
            nodes: Nodes, all with the same number of steps and time step.
            incentive: Incentive model with the same number of steps.
            timestamps: Start time of each step.
            delta_t: Duration of a step (hours).
            name: Scenario name.

        Raises:
            ValueError: If there are no nodes, node ids repeat, or steps and time steps disagree.
        """
        if not nodes:
            raise ValueError('a community needs at least one node')
        node_ids = [node.node_id for node in nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f'node ids must be unique, got {node_ids}')
        n_steps = len(timestamps)
        for node in nodes:
            if node.n_steps != n_steps:
                raise ValueError(f'node {node.node_id!r} has {node.n_steps} steps, the community '
                                 f'has {n_steps}')
            if node.delta_t != delta_t:
                raise ValueError(f'node {node.node_id!r} has delta_t {node.delta_t}, the '
                                 f'community has {delta_t}')
        if incentive.n_steps != n_steps:
            raise ValueError(f'the incentive model has {incentive.n_steps} steps, the community '
                             f'has {n_steps}')

        self.name = name
        self.nodes = tuple(nodes)
        self.node_ids = tuple(node_ids)
        self.storage_node_ids = tuple(node.node_id for node in nodes if node.battery is not None)
        self.incentive = incentive
        self.timestamps = np.asarray(timestamps)
        self.delta_t = float(delta_t)
        self._action_index = {node_id: i for i, node_id in enumerate(self.storage_node_ids)}

    def __repr__(self) -> str:
        return (f'Community({self.name!r}, {len(self.nodes)} nodes, '
                f'{len(self.storage_node_ids)} batteries, {self.n_steps} steps)')

    @property
    def n_steps(self) -> int:
        """Number of steps of the scenario."""
        return len(self.timestamps)

    def node(self, node_id: str) -> Node:
        """
        Node by identifier.

        Args:
            node_id: Node identifier.

        Returns:
            Node: The node.

        Raises:
            KeyError: If the node does not exist.
        """
        try:
            return self.nodes[self.node_ids.index(node_id)]
        except ValueError:
            raise KeyError(f'unknown node {node_id!r}; nodes: {list(self.node_ids)}') from None

    def reset(self) -> None:
        """
        Restore every battery's initial state of charge.
        """
        for node in self.nodes:
            node.reset()

    def step(self, t: int, actions: ArrayLike | None = None) -> CommunityStep:
        """
        Apply one action per battery in step t and compute the community results.

        Args:
            t: Step index.
            actions: One action in [-1, 1] per battery, in `storage_node_ids` order; None leaves
                every battery idle.

        Returns:
            CommunityStep: Node records, welfare breakdown and mean penalty.

        Raises:
            ValueError: If the number of actions differs from the number of batteries.
        """
        if actions is None:
            action_list = [0.0] * len(self.storage_node_ids)
        else:
            action_list = np.asarray(actions, dtype=np.float64).reshape(-1).tolist()
            if len(action_list) != len(self.storage_node_ids):
                raise ValueError(f'expected {len(self.storage_node_ids)} actions (one per battery, '
                                 f'{list(self.storage_node_ids)}), got {len(action_list)}')

        records = []
        for node in self.nodes:
            index = self._action_index.get(node.node_id)
            records.append(node.step(t, None if index is None else action_list[index]))

        export = np.array([r.export_kwh for r in records])
        import_ = np.array([r.import_kwh for r in records])
        shared = shared_energy(export, import_)
        welfare = welfare_breakdown(
            revenue=sum(r.revenue for r in records),
            cost=sum(r.cost for r in records),
            storage_cost=sum(r.storage_cost for r in records),
            incentive=self.incentive.compute(t, shared, export),
            shared_energy_kwh=shared,
            export_kwh=float(export.sum()),
            import_kwh=float(import_.sum()),
        )
        penalties = [r.penalty for r, node in zip(records, self.nodes) if node.battery is not None]
        penalty = sum(penalties) / len(penalties) if penalties else 0.0
        return CommunityStep(t=t, nodes=tuple(records), welfare=welfare, penalty=penalty)
