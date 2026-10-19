"""
Unit tests of the reward calculators (`smartcer.gym.rewards`, docs/FORMULATION.md section 11).

Hand values at 10:00 of day 1 of the synthetic scenario with actions (pv_bat -0.5, factory_bat
+0.5): pv_bat sits at S^min, so its discharge is clipped to 0 (k = 0.5) and it exports its 10 kWh
at 0.10 €/kWh (profit 1.0 €); factory_bat charges 5 kWh and exports 20 - 10 - 5 = 5 kWh (profit
0.5 €, k = 0); house_1 imports 2 kWh and shop 1 kWh; W = 1.125 €, K = 0.25.
"""

from pathlib import Path

import numpy as np
import pytest

from smartcer.core.community import CommunityStep
from smartcer.data import load_scenario
from smartcer.gym.rewards import (
    REWARD_REGISTRY,
    REWARD_TYPES,
    EnergyReward,
    ProfitReward,
    RewardCalculator,
    WelfareReward,
    make_reward,
)


pytestmark = pytest.mark.unit

STORAGE = ('pv_bat', 'factory_bat')


@pytest.fixture
def step_10(mini_cer_v1_path: Path) -> CommunityStep:
    """
    Community step at 10:00 of day 1 with actions (-0.5, 0.5), after nine idle steps.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.

    Returns:
        CommunityStep: The step record.
    """
    community = load_scenario(mini_cer_v1_path)
    for t in range(10):
        community.step(t, np.zeros(2))
    return community.step(10, np.array([-0.5, 0.5]))


def test_welfare_reward_is_welfare_minus_sigma_k(step_10: CommunityStep) -> None:
    """
    Test R = W - σ K = 1.125 - 10 × 0.25.

    Args:
        step_10: Community step record.
    """
    reward = WelfareReward(STORAGE).compute(step_10)

    assert (reward.main, reward.penalty) == pytest.approx((1.125, 2.5))
    assert reward.total == pytest.approx(-1.375) and reward.nodes is None


def test_profit_reward_collective_equals_welfare(step_10: CommunityStep) -> None:
    """
    Test that the collective profit (node profits plus incentive) equals the welfare.

    Args:
        step_10: Community step record.
    """
    profit = ProfitReward(STORAGE, penalty_coefficient=3.0).compute(step_10)
    welfare = WelfareReward(STORAGE, penalty_coefficient=3.0).compute(step_10)

    assert profit.total == pytest.approx(welfare.total)


def test_profit_reward_per_node(step_10: CommunityStep) -> None:
    """
    Test the per-node profit: r_u = profit_u - σ k_u for the battery nodes, R = Σ_u r_u.

    Args:
        step_10: Community step record.
    """
    reward = ProfitReward(STORAGE, per_node=True).compute(step_10)

    assert list(reward.nodes) == list(STORAGE)
    pv, factory = reward.nodes['pv_bat'], reward.nodes['factory_bat']
    assert (pv.main, pv.penalty, pv.total) == pytest.approx((1.0, 5.0, -4.0))
    assert (factory.main, factory.penalty, factory.total) == pytest.approx((0.5, 0.0, 0.5))
    assert (reward.main, reward.penalty, reward.total) == pytest.approx((1.5, 5.0, -3.5))


def test_profit_reward_includes_storage_cost(mini_cer_v1_path: Path) -> None:
    """
    Test that the node profit subtracts the storage cost π^sto × throughput.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    for t in range(10):
        community.step(t, np.zeros(2))
    factory = community.node('factory_bat')
    factory.battery.usage_cost = 0.01
    step = community.step(10, np.array([0.0, 0.5]))

    reward = ProfitReward(STORAGE, per_node=True).compute(step)

    # throughput on the cell side: 0.9 × 5 kWh
    assert reward.nodes['factory_bat'].main == pytest.approx(0.5 - 0.01 * 4.5)


def test_energy_reward(step_10: CommunityStep) -> None:
    """
    Test the 0.6.2 energy reward: collective -Σ_u i_u² over all nodes (house_1 2 kWh, shop
    1 kWh: -5) minus σ K; per node -i_u² of the battery nodes (both export: 0) minus σ k_u.

    Args:
        step_10: Community step record.
    """
    collective = EnergyReward(STORAGE).compute(step_10)
    per_node = EnergyReward(STORAGE, per_node=True).compute(step_10)

    assert (collective.main, collective.total) == pytest.approx((-5.0, -7.5))
    assert {i: r.total for i, r in per_node.nodes.items()} == pytest.approx(
        {'pv_bat': -5.0, 'factory_bat': 0.0})
    assert per_node.total == pytest.approx(-5.0)


def test_template_method_subtracts_the_weighted_penalty(step_10: CommunityStep) -> None:
    """
    Test the template of RewardCalculator with a minimal subclass: main term minus σ K.

    Args:
        step_10: Community step record.
    """
    class ConstantReward(RewardCalculator):
        name = 'constant'

        def collective_main(self, step: CommunityStep) -> float:
            return 1.0

    reward = ConstantReward(STORAGE, penalty_coefficient=2.0).compute(step_10)

    assert reward.total == pytest.approx(1.0 - 2.0 * 0.25)
    with pytest.raises(NotImplementedError, match='per-node'):
        ConstantReward(STORAGE, per_node=True).compute(step_10)


def test_registry_and_factory() -> None:
    """
    Test the read-only registry and the factory: unknown types, the welfare per-node variant and a
    negative σ raise ValueError.
    """
    assert REWARD_TYPES == ('welfare', 'profit', 'energy')
    assert isinstance(make_reward('energy', STORAGE, 1.0, per_node=True), EnergyReward)
    with pytest.raises(TypeError):
        REWARD_REGISTRY['other'] = REWARD_REGISTRY['welfare']  # type: ignore[index]
    with pytest.raises(ValueError, match='unknown reward'):
        make_reward('cost', STORAGE)
    with pytest.raises(ValueError, match='no per-node variant'):
        make_reward('welfare', STORAGE, per_node=True)
    with pytest.raises(ValueError, match='penalty_coefficient'):
        make_reward('profit', STORAGE, penalty_coefficient=float('nan'))
