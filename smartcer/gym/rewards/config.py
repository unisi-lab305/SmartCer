"""
Registry of the reward types and factory of the reward calculators.

Adapted from storage-node-environment (`storage_node_env/gym/rewards/config.py`): the factory reads
the registry instead of an if/elif chain; the registry is read-only.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from smartcer.gym.rewards.base import RewardCalculator
from smartcer.gym.rewards.calculators import EnergyReward, ProfitReward, WelfareReward


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Entry of the reward registry.

    Attributes:
        name: Reward type name.
        description: Objective of the reward.
        calculator: Calculator class.
    """

    name: str
    description: str
    calculator: type[RewardCalculator]


REWARD_REGISTRY = MappingProxyType({
    config.name: config for config in (
        RewardConfig('welfare', 'social welfare W_t minus σ K_t (default)', WelfareReward),
        RewardConfig('profit', 'net profit per node; collective variant plus the incentive',
                     ProfitReward),
        RewardConfig('energy', '0.6.2 legacy: minus the squared imported energy',
                     EnergyReward),
    )
})
REWARD_TYPES = tuple(REWARD_REGISTRY)


def make_reward(reward_type: str, storage_node_ids: Sequence[str],
                penalty_coefficient: float = 10.0, per_node: bool = False) -> RewardCalculator:
    """
    Create the calculator of a registered reward type.

    Args:
        reward_type: Name in `REWARD_REGISTRY`.
        storage_node_ids: Identifiers of the battery nodes.
        penalty_coefficient: σ, ≥ 0.
        per_node: Use the per-node variant.

    Returns:
        RewardCalculator: The calculator.

    Raises:
        ValueError: For an unknown type, an invalid σ or an unsupported per-node variant.
    """
    if reward_type not in REWARD_REGISTRY:
        raise ValueError(f'unknown reward {reward_type!r}; available: {REWARD_TYPES}')
    calculator = REWARD_REGISTRY[reward_type].calculator
    return calculator(storage_node_ids, penalty_coefficient=penalty_coefficient,
                      per_node=per_node)
