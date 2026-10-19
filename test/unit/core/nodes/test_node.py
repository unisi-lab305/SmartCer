"""
Unit tests of `smartcer.core.nodes.Node` (docs/FORMULATION.md sections 6 and 8).
"""

from typing import Any

import numpy as np
import pytest

from smartcer.core.nodes import Node, NodeStep
from smartcer.core.storage import Battery


pytestmark = pytest.mark.unit


def make_battery(**overrides: Any) -> Battery:
    """
    Battery of 10 kWh, 5 kW, η = 1, empty, overridable.

    Args:
        **overrides: Constructor arguments that replace the defaults.

    Returns:
        Battery: The battery.
    """
    params = {'capacity': 10.0, 'power_charge_max': 5.0, 'power_discharge_max': 5.0,
              'efficiency_charge': 1.0, 'efficiency_discharge': 1.0}
    return Battery(**{**params, **overrides})


def make_node(node_type: str = 'prosumer', generation: float = 6.0, load: float = 2.0,
              battery: Battery | None = None, delta_t: float = 1.0) -> Node:
    """
    Node with two identical steps: sell price 0.1, buy price 0.3 €/kWh.

    Args:
        node_type: Node type.
        generation: Generation per step (kWh).
        load: Load per step (kWh).
        battery: Optional battery.
        delta_t: Duration of a step (hours).

    Returns:
        Node: The node.
    """
    return Node(node_id='n', node_type=node_type, delta_t=delta_t,
                generation_kwh=np.full(2, generation), load_kwh=np.full(2, load),
                sell_price=np.full(2, 0.1), buy_price=np.full(2, 0.3), battery=battery)


def test_node_without_battery_flows_and_economics() -> None:
    """
    Test n = g - l, export and import, revenue and cost: g 6, l 2 → export 4, revenue 0.4.
    """
    result = make_node().step(0)

    assert isinstance(result, NodeStep)
    assert (result.net_kwh, result.export_kwh, result.import_kwh) == (4.0, 4.0, 0.0)
    assert result.revenue == pytest.approx(0.4)
    assert result.cost == 0.0
    assert result.soc_kwh is None and result.penalty == 0.0


def test_node_with_battery_charges_from_its_generation() -> None:
    """
    Test n = g - l - p^cha Δt: g 6, l 2, full charge of 5 kWh → import 1 kWh at 0.3 €/kWh.
    """
    result = make_node(battery=make_battery()).step(0, 1.0)

    assert result.charge_kwh == pytest.approx(5.0)
    assert result.net_kwh == pytest.approx(-1.0)
    assert (result.export_kwh, result.import_kwh) == (0.0, pytest.approx(1.0))
    assert result.cost == pytest.approx(0.3)
    assert result.soc_kwh == pytest.approx(5.0)


def test_discharge_adds_to_export() -> None:
    """
    Test n = g - l + p^dis Δt: g 6, l 2, discharge 3 kWh from a full battery → export 7 kWh.
    """
    node = make_node(battery=make_battery(soc_initial=1.0))
    result = node.step(0, -0.6)

    assert result.discharge_kwh == pytest.approx(3.0)
    assert result.export_kwh == pytest.approx(7.0)
    assert result.revenue == pytest.approx(0.7)


@pytest.mark.parametrize(('charge_limit', 'expected_cap'), [('production', 6.0), ('surplus', 4.0)])
def test_charge_cap_follows_charge_limit(charge_limit: str, expected_cap: float) -> None:
    """
    Test the charge cap: production c = g = 6 kWh, surplus c = max(0, g - l) = 4 kWh.

    Args:
        charge_limit: Battery charge rule.
        expected_cap: Expected cap (kWh).
    """
    battery = make_battery(capacity=100.0, power_charge_max=50.0, charge_limit=charge_limit)
    node = make_node(battery=battery)

    assert node.charge_cap(0) == expected_cap
    assert node.get_bounds(0)[0] == pytest.approx(expected_cap / 50.0)
    assert node.step(0, 1.0).charge_kwh == pytest.approx(expected_cap)


def test_s8_consumer_battery_cannot_charge() -> None:
    """
    Test S8: a consumer with a battery steps without error; under the production rule its charge
    cap is 0, so a full charge is clipped to 0 with penalty 1, while discharge works.
    """
    node = make_node('consumer', generation=0.0, battery=make_battery(soc_initial=0.5))
    charged = node.step(0, 1.0)
    discharged = node.step(1, -0.2)

    assert charged.charge_kwh == 0.0 and charged.penalty == pytest.approx(1.0)
    assert charged.import_kwh == pytest.approx(2.0)
    assert discharged.discharge_kwh == pytest.approx(1.0)
    assert discharged.import_kwh == pytest.approx(1.0)


def test_s5_energy_scales_with_time_step() -> None:
    """
    Test S5: with Δt = 0.25 h a full charge takes P^cha · Δt = 1.25 kWh from the node.
    """
    node = make_node(battery=make_battery(), delta_t=0.25)

    assert node.step(0, 1.0).charge_kwh == pytest.approx(1.25)


def test_reset_restores_the_battery() -> None:
    """
    Test that reset() restores the initial state of charge.
    """
    node = make_node(battery=make_battery())
    node.step(0, 1.0)
    node.reset()

    assert node.battery.soc_kwh == 0.0


def test_invalid_inputs_raise() -> None:
    """
    Test the validation: unknown type, consumer with generation, negative load, action on a node
    without battery, step outside the data.
    """
    with pytest.raises(ValueError, match='type'):
        make_node('storage')
    with pytest.raises(ValueError, match='consumer has no generation'):
        make_node('consumer', generation=1.0)
    with pytest.raises(ValueError, match='load_kwh is negative'):
        make_node(load=-1.0)
    with pytest.raises(ValueError, match='no battery'):
        make_node().step(0, 0.5)
    with pytest.raises(IndexError, match='step 2'):
        make_node().step(2)
