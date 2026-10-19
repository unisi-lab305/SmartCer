"""
Template of the reward calculators (docs/FORMULATION.md section 11).

Adapted from storage-node-environment (`storage_node_env/gym/rewards/base.py`): the reward is a
main term minus the weighted constraint penalty. The collective variant gives
R_t = M_t - σ K_t; the per-node variant gives r_{u,t} = m_{u,t} - σ k_{u,t} for every node with a
battery, reported in `info`, and R_t = Σ_u r_{u,t}. No reward scaling (defect S9).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from smartcer.core.community import CommunityStep
from smartcer.core.nodes import NodeStep


@dataclass(frozen=True, slots=True)
class NodeReward:
    """
    Reward of one battery node (per-node variant).

    Attributes:
        main: Main term m_{u,t}.
        penalty: Weighted penalty σ k_{u,t}.
        total: r_{u,t} = main - penalty.
    """

    main: float
    penalty: float
    total: float


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """
    Reward of one step.

    Attributes:
        main: Main term: M_t (collective) or Σ_u m_{u,t} (per node).
        penalty: Weighted penalty: σ K_t (collective) or Σ_u σ k_{u,t} (per node).
        total: The scalar reward R_t = main - penalty.
        nodes: Battery node id → `NodeReward` (per-node variant), None for the collective one.
    """

    main: float
    penalty: float
    total: float
    nodes: dict[str, NodeReward] | None = None


class RewardCalculator(ABC):
    """
    Reward as a main term minus σ times the constraint penalty.

    Subclasses implement `collective_main` and, if they support the per-node variant,
    `node_main`.

    Attributes:
        name: Reward type name (class attribute).
        supports_per_node: Whether the per-node variant exists (class attribute).
        penalty_coefficient: σ.
        per_node: Whether the per-node variant is used.
        storage_node_ids: Identifiers of the battery nodes, in action order.
    """

    name: ClassVar[str]
    supports_per_node: ClassVar[bool] = True

    def __init__(self, storage_node_ids: Sequence[str], penalty_coefficient: float = 10.0,
                 per_node: bool = False) -> None:
        """
        Create the calculator.

        Args:
            storage_node_ids: Identifiers of the battery nodes.
            penalty_coefficient: σ, ≥ 0.
            per_node: Use the per-node variant.

        Raises:
            ValueError: If σ is negative or not finite, or the per-node variant is not supported.
        """
        if not (np.isfinite(penalty_coefficient) and penalty_coefficient >= 0):
            raise ValueError(f'penalty_coefficient must be ≥ 0, got {penalty_coefficient!r}')
        if per_node and not self.supports_per_node:
            raise ValueError(f'reward {self.name!r} has no per-node variant')
        self.storage_node_ids = tuple(storage_node_ids)
        self.penalty_coefficient = float(penalty_coefficient)
        self.per_node = bool(per_node)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(penalty_coefficient={self.penalty_coefficient}, '
                f'per_node={self.per_node})')

    @abstractmethod
    def collective_main(self, step: CommunityStep) -> float:
        """
        Main term M_t of the collective variant.

        Args:
            step: Community step record.

        Returns:
            float: M_t.
        """

    def node_main(self, node: NodeStep) -> float:
        """
        Main term m_{u,t} of the per-node variant.

        Args:
            node: Node step record.

        Returns:
            float: m_{u,t}.

        Raises:
            NotImplementedError: For reward types without a per-node variant.
        """
        raise NotImplementedError(f'reward {self.name!r} has no per-node variant')

    def compute(self, step: CommunityStep) -> RewardBreakdown:
        """
        Reward of a step: main term minus σ times the constraint penalty.

        Args:
            step: Community step record.

        Returns:
            RewardBreakdown: Main term, weighted penalty, total and, per node, their values.
        """
        sigma = self.penalty_coefficient
        if not self.per_node:
            main = self.collective_main(step)
            penalty = sigma * step.penalty
            return RewardBreakdown(main=main, penalty=penalty, total=main - penalty)
        records = {r.node_id: r for r in step.nodes}
        nodes = {}
        for node_id in self.storage_node_ids:
            main = self.node_main(records[node_id])
            penalty = sigma * records[node_id].penalty
            nodes[node_id] = NodeReward(main=main, penalty=penalty, total=main - penalty)
        main = sum(r.main for r in nodes.values())
        penalty = sum(r.penalty for r in nodes.values())
        return RewardBreakdown(main=main, penalty=penalty, total=main - penalty, nodes=nodes)
