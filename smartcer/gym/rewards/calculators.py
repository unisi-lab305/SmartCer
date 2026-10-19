"""
Reward types of SmartCer-v1 (docs/FORMULATION.md section 11).

- `welfare` (default): M_t = W_t, collective only;
- `profit`: m_{u,t} = π^egr e^gri - π^igr i^gri - storage cost of node u; its collective variant
  adds the incentive, M_t = Σ_u m_{u,t} + I_t = W_t;
- `energy` (0.6.2 legacy): m_{u,t} = -(i^gri_{u,t})²; collective M_t = Σ_u m_{u,t} over all nodes.
"""

from smartcer.core.community import CommunityStep
from smartcer.core.nodes import NodeStep
from smartcer.gym.rewards.base import RewardCalculator


class WelfareReward(RewardCalculator):
    """
    R_t = W_t - σ K_t ([M24] eq. 8).
    """

    name = 'welfare'
    supports_per_node = False

    def collective_main(self, step: CommunityStep) -> float:
        """
        Social welfare W_t (section 10).

        Args:
            step: Community step record.

        Returns:
            float: W_t.
        """
        return step.welfare.total


class ProfitReward(RewardCalculator):
    """
    Net profit of each node; the collective variant adds the incentive (and equals the welfare).
    """

    name = 'profit'

    def node_main(self, node: NodeStep) -> float:
        """
        Net profit of a node: revenue - cost - storage cost.

        Args:
            node: Node step record.

        Returns:
            float: m_{u,t}.
        """
        return node.revenue - node.cost - node.storage_cost

    def collective_main(self, step: CommunityStep) -> float:
        """
        Σ_u m_{u,t} over all nodes plus the incentive I_t.

        Args:
            step: Community step record.

        Returns:
            float: M_t.
        """
        return sum(self.node_main(r) for r in step.nodes) + step.welfare.incentive


class EnergyReward(RewardCalculator):
    """
    0.6.2 energy reward: minus the squared energy imported by each node.
    """

    name = 'energy'

    def node_main(self, node: NodeStep) -> float:
        """
        -(i^gri_{u,t})².

        Args:
            node: Node step record.

        Returns:
            float: m_{u,t}.
        """
        return -node.import_kwh ** 2

    def collective_main(self, step: CommunityStep) -> float:
        """
        Σ_u -(i^gri_{u,t})² over all nodes.

        Args:
            step: Community step record.

        Returns:
            float: M_t.
        """
        return sum(self.node_main(r) for r in step.nodes)
