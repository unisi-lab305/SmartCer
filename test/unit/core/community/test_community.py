"""
Unit tests of `smartcer.core.community.Community`.
"""

import numpy as np
import pytest

from smartcer.core.community import Community, CommunityStep, UniformIncentive
from smartcer.core.nodes import Node
from smartcer.core.storage import Battery


pytestmark = pytest.mark.unit

TIMESTAMPS = np.array(['2022-01-03T10:00', '2022-01-03T11:00'], dtype='datetime64[ns]')


def make_community(storage_cost: float = 0.0) -> Community:
    """
    Two-step community: producer with an empty 10 kWh / 5 kW battery (η = 1) generating 6 kWh,
    consumer loading 2 kWh, uniform incentive 0.1 €/kWh; sell 0.1, buy 0.3 €/kWh.

    Args:
        storage_cost: Battery usage cost (€/kWh).

    Returns:
        Community: The community.
    """
    battery = Battery(capacity=10.0, power_charge_max=5.0, power_discharge_max=5.0,
                      efficiency_charge=1.0, efficiency_discharge=1.0, usage_cost=storage_cost)
    ones = np.ones(2)
    producer = Node('pv', 'producer', 1.0, 6 * ones, 0 * ones, 0.1 * ones, 0 * ones, battery)
    consumer = Node('home', 'consumer', 1.0, 0 * ones, 2 * ones, 0 * ones, 0.3 * ones)
    return Community([producer, consumer], UniformIncentive(0.1, 2), TIMESTAMPS, 1.0, 'unit')


def test_idle_step_welfare() -> None:
    """
    Test the idle step: export 6, import 2, shared 2 → W = 0.6 - 0.6 + 0.2 = 0.2.
    """
    result = make_community().step(0)

    assert isinstance(result, CommunityStep)
    assert result.welfare.shared_energy_kwh == 2.0
    assert result.welfare.total == pytest.approx(0.2)
    assert result.penalty == 0.0


def test_actions_follow_storage_node_order_and_penalty_is_mean() -> None:
    """
    Test a charge of 0.8 (4 kWh): export 2, shared 2; then a discharge of 1.0 from 4 kWh is
    clipped to 0.8 with k = 0.2, and K_t is the mean over the one battery.
    """
    community = make_community(storage_cost=0.01)
    charged = community.step(0, [0.8])
    discharged = community.step(1, np.array([-1.0]))

    assert community.storage_node_ids == ('pv',)
    assert charged.welfare.export_kwh == pytest.approx(2.0)
    assert charged.welfare.storage_cost == pytest.approx(0.04)
    assert charged.welfare.total == pytest.approx(0.2 - 0.6 + 0.2 - 0.04)
    assert discharged.penalty == pytest.approx(0.2)
    assert discharged.nodes[0].discharge_kwh == pytest.approx(4.0)


def test_reset_and_validation() -> None:
    """
    Test reset(), node lookup, and errors on a wrong number of actions or mismatched steps.
    """
    community = make_community()
    community.step(0, [1.0])
    community.reset()

    assert community.node('pv').battery.soc_kwh == 0.0
    with pytest.raises(ValueError, match='expected 1 actions'):
        community.step(0, [0.1, 0.2])
    with pytest.raises(KeyError, match='unknown node'):
        community.node('missing')
    with pytest.raises(ValueError, match='steps'):
        Community(community.nodes, UniformIncentive(0.1, 3), TIMESTAMPS, 1.0)
