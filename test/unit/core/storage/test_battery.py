"""
Unit tests of `smartcer.core.storage.Battery` (docs/FORMULATION.md sections 4-7 and 10).

Expected values are computed by hand from the formulation.
"""

import math
from typing import Any

import pytest

from smartcer.core.storage import Battery, BatteryStep


pytestmark = pytest.mark.unit


def make_battery(**overrides: Any) -> Battery:
    """
    Battery of the hand cases: 10 kWh, 5 kW, η = 0.95, soc_min = 0, overridable.

    Args:
        **overrides: Constructor arguments that replace the defaults.

    Returns:
        Battery: The battery.
    """
    params = {
        'capacity': 10.0, 'power_charge_max': 5.0, 'power_discharge_max': 5.0,
        'efficiency_charge': 0.95, 'efficiency_discharge': 0.95, 'soc_min': 0.0,
    }
    return Battery(**{**params, **overrides})


# ==================================
# Hand cases of the acceptance criteria
# ==================================


def test_full_charge_from_half() -> None:
    """
    Test that from 5 kWh a full charge adds 4.75 kWh and takes 5 kWh from the node.
    """
    result = make_battery(soc_initial=0.5).step(1.0, delta_t=1.0, charge_cap_kwh=None)

    assert result.soc_kwh == pytest.approx(5.0 + 4.75)
    assert result.charge_kwh == pytest.approx(5.0)
    assert result.p_cha_kw == pytest.approx(5.0)
    assert result.penalty == 0.0


def test_full_discharge_from_eight() -> None:
    """
    Test that from 8 kWh a full discharge removes 5 / 0.95 = 5.263 kWh and delivers 5 kWh.
    """
    result = make_battery(soc_initial=0.8).step(-1.0, delta_t=1.0, charge_cap_kwh=None)

    assert result.soc_kwh == pytest.approx(8.0 - 5.0 / 0.95)
    assert 8.0 - result.soc_kwh == pytest.approx(5.263, abs=1e-3)
    assert result.discharge_kwh == pytest.approx(5.0)
    assert result.penalty == 0.0


def test_full_discharge_from_half_is_clipped() -> None:
    """
    Test that from 5 kWh a full discharge is clipped to a = -0.95: 4.75 kWh to the node,
    state of charge 0, penalty 0.05.
    """
    result = make_battery(soc_initial=0.5).step(-1.0, delta_t=1.0, charge_cap_kwh=None)

    assert result.a_lower == pytest.approx(0.95)
    assert result.action_applied == pytest.approx(-0.95)
    assert result.discharge_kwh == pytest.approx(4.75)
    assert result.soc_kwh == pytest.approx(0.0, abs=1e-12)
    assert result.penalty == pytest.approx(0.05)


@pytest.mark.parametrize('delta_t', [0.25, 1.0])
@pytest.mark.parametrize('charge_cap_kwh', [None, 0.3, 0.6])
def test_actions_at_bounds_reach_limits(delta_t: float, charge_cap_kwh: float | None) -> None:
    """
    Test that a = A^upper and a = -A^lower reach the binding limit without crossing it.

    From 9.5 kWh with η = 0.9 the capacity headroom (0.5 / 0.9 = 0.556 kWh from the node) binds
    the charge, unless the 0.3 kWh charge cap binds first; from 1.5 kWh with soc_min 1 kWh the
    minimum state of charge binds the discharge.

    Args:
        delta_t: Duration of the step (hours).
        charge_cap_kwh: Charge cap (kWh), or None.
    """
    battery = make_battery(soc_min=0.1, soc_initial=0.95, efficiency_charge=0.9)
    a_upper, _ = battery.get_bounds(delta_t, charge_cap_kwh)
    charged = battery.step(a_upper, delta_t, charge_cap_kwh)
    headroom_binds = charge_cap_kwh is None or 0.5 / 0.9 < charge_cap_kwh
    expected_soc = 10.0 if headroom_binds else 9.5 + 0.9 * charge_cap_kwh

    assert charged.penalty == 0.0
    assert charged.soc_kwh == pytest.approx(expected_soc, abs=1e-9)
    assert charged.soc_kwh <= battery.capacity

    battery = make_battery(soc_min=0.1, soc_initial=0.15, efficiency_discharge=0.9)
    _, a_lower = battery.get_bounds(delta_t, charge_cap_kwh)
    discharged = battery.step(-a_lower, delta_t, charge_cap_kwh)

    assert discharged.penalty == 0.0
    assert discharged.soc_kwh == pytest.approx(1.0, abs=1e-9)
    assert discharged.soc_kwh >= battery.soc_min_kwh


# ==================================
# Bounds, charge cap, self-discharge, cost
# ==================================


def test_bounds_follow_the_formulation() -> None:
    """
    Test A^upper and A^lower on a case where the efficiency and soc_min terms bind.

    6 kWh, soc_min 1 kWh, η = 0.8, Δt = 0.5 h, P = 5 kW, from 5 kWh:
    A^upper = min(5, (6 - 5) / (0.8 · 0.5)) / 5 = 2.5 / 5 = 0.5;
    A^lower = min(5, 0.8 · (5 - 1) / 0.5) / 5 = 5 / 5 = 1.
    """
    battery = make_battery(capacity=6.0, soc_min=1.0 / 6.0, soc_initial=5.0 / 6.0,
                           efficiency_charge=0.8, efficiency_discharge=0.8)

    assert battery.get_bounds(0.5, None) == pytest.approx((0.5, 1.0))
    assert battery.soc_kwh == pytest.approx(5.0)


def test_charge_cap_limits_the_charge() -> None:
    """
    Test that with a 2 kWh charge cap a full charge takes 2 kWh from the node, penalty 0.6.
    """
    result = make_battery(soc_initial=0.2).step(1.0, delta_t=1.0, charge_cap_kwh=2.0)

    assert result.a_upper == pytest.approx(0.4)
    assert result.charge_kwh == pytest.approx(2.0)
    assert result.penalty == pytest.approx(0.6)
    assert result.soc_kwh == pytest.approx(2.0 + 0.95 * 2.0)


def test_zero_charge_cap_forbids_charging() -> None:
    """
    Test that a zero charge cap gives A^upper = 0 and leaves the state of charge unchanged.
    """
    result = make_battery(soc_initial=0.5).step(0.3, delta_t=1.0, charge_cap_kwh=0.0)

    assert result.a_upper == 0.0
    assert result.action_applied == 0.0
    assert result.penalty == pytest.approx(0.3)
    assert result.soc_kwh == pytest.approx(5.0)


def test_self_discharge_is_applied_before_the_step_and_clamped() -> None:
    """
    Test s̃ = max(S^min, (1 - α) s): 5 → 4.5 kWh with α = 0.1, and 2 kWh stays at S^min = 2 kWh.
    """
    battery = make_battery(soc_initial=0.5, alpha=0.1)
    _, a_lower = battery.get_bounds(1.0, None)

    assert a_lower == pytest.approx(0.95 * 4.5 / 5.0)
    assert battery.step(0.0, 1.0, None).soc_kwh == pytest.approx(4.5)

    at_minimum = make_battery(soc_min=0.2, alpha=0.1)

    assert at_minimum.step(0.0, 1.0, None).soc_kwh == pytest.approx(2.0)


def test_self_discharge_is_defined_per_hour() -> None:
    """
    Test α per hour: four idle steps of 0.25 h lose as much as one step of 1 h (5 → 4.5 kWh), and
    one step of 0.25 h keeps (1 - 0.1)^0.25 of the charge.
    """
    quarter = make_battery(soc_initial=0.5, alpha=0.1)
    first = quarter.step(0.0, 0.25, None)
    for _ in range(3):
        last = quarter.step(0.0, 0.25, None)

    assert first.soc_kwh == pytest.approx(5.0 * 0.9 ** 0.25)
    assert last.soc_kwh == pytest.approx(4.5)
    assert make_battery(soc_initial=0.5, alpha=0.1).step(0.0, 1.0, None).soc_kwh == pytest.approx(
        last.soc_kwh)


def test_throughput_and_storage_cost() -> None:
    """
    Test the cell throughput and π^sto · throughput for a full charge and a full discharge.

    Charge 5 kW for 1 h: 0.95 · 5 = 4.75 kWh; discharge 5 kW for 1 h: 5 / 0.95 kWh;
    π^sto = 0.01 €/kWh.
    """
    battery = make_battery(soc_initial=0.5, usage_cost=0.01)
    charged = battery.step(1.0, 1.0, None)
    discharged = battery.step(-1.0, 1.0, None)

    assert charged.throughput_kwh == pytest.approx(4.75)
    assert charged.storage_cost == pytest.approx(0.0475)
    assert discharged.throughput_kwh == pytest.approx(5.0 / 0.95)
    assert discharged.storage_cost == pytest.approx(0.01 * 5.0 / 0.95)


def test_step_record_and_reset() -> None:
    """
    Test the step record type, the default initial state of charge and reset().
    """
    battery = make_battery(soc_min=0.2)
    result = battery.step(0.5, 1.0, None)

    assert isinstance(result, BatteryStep)
    assert result.action_requested == 0.5
    assert battery.soc_kwh == pytest.approx(2.0 + 0.95 * 2.5)

    battery.reset()

    assert battery.soc_kwh == pytest.approx(2.0)


# ==================================
# Regression tests of 0.6.2 defects
# ==================================


def test_s4_discharge_keeps_soc_above_soc_min() -> None:
    """
    Test S4: 10 kWh, 5 kW, soc_min 1 kWh, η = 0.9, from 2 kWh, full discharge ends at 1 kWh
    (0.6.2 ended at 0.889 kWh) and delivers 0.9 kWh to the node.
    """
    result = make_battery(soc_min=0.1, soc_initial=0.2, efficiency_discharge=0.9).step(
        -1.0, 1.0, None)

    assert result.soc_kwh == pytest.approx(1.0, abs=1e-12)
    assert result.discharge_kwh == pytest.approx(0.9)


def test_s4_discharge_with_zero_soc_min_stays_non_negative() -> None:
    """
    Test S4: with soc_min = 0, from 1 kWh, a full discharge ends at 0 kWh without raising
    (0.6.2 computed 1 - 1 / 0.9 < 0 and stopped the process).
    """
    result = make_battery(soc_initial=0.1, efficiency_discharge=0.9).step(-1.0, 1.0, None)

    assert result.soc_kwh == pytest.approx(0.0, abs=1e-12)
    assert result.soc_kwh >= 0.0


def test_s5_energy_scales_with_time_step() -> None:
    """
    Test S5 at battery level: a full charge for 15 minutes takes P^cha · 0.25 h from the node.
    """
    result = make_battery(capacity=100.0, power_charge_max=10.0).step(1.0, 0.25, None)

    assert result.charge_kwh == pytest.approx(2.5)
    assert result.soc_kwh == pytest.approx(0.95 * 2.5)


# ==================================
# Validation
# ==================================


@pytest.mark.parametrize(
    ('overrides', 'name'),
    [
        ({'capacity': 0.0}, 'capacity'),
        ({'capacity': math.inf}, 'capacity'),
        ({'power_charge_max': -1.0}, 'power_charge_max'),
        ({'power_discharge_max': 0.0}, 'power_discharge_max'),
        ({'efficiency_charge': 0.0}, 'efficiency_charge'),
        ({'efficiency_discharge': 1.01}, 'efficiency_discharge'),
        ({'soc_min': 1.0}, 'soc_min'),
        ({'soc_min': 0.3, 'soc_initial': 0.2}, 'soc_initial'),
        ({'alpha': 1.0}, 'alpha'),
        ({'usage_cost': -0.01}, 'usage_cost'),
        ({'capacity': '10'}, 'capacity'),
        ({'charge_limit': 'grid'}, 'charge_limit'),
    ],
)
def test_invalid_parameters_raise(overrides: dict[str, Any], name: str) -> None:
    """
    Test that invalid parameters raise ValueError naming the parameter.

    Args:
        overrides: Invalid constructor arguments.
        name: Parameter expected in the message.
    """
    with pytest.raises(ValueError, match=name):
        make_battery(**overrides)


@pytest.mark.parametrize(
    ('action', 'delta_t', 'charge_cap_kwh', 'name'),
    [
        (1.5, 1.0, None, 'action'),
        (math.nan, 1.0, None, 'action'),
        (0.5, 0.0, None, 'delta_t'),
        (0.5, 1.0, -1.0, 'charge_cap_kwh'),
    ],
)
def test_invalid_step_inputs_raise(
    action: float, delta_t: float, charge_cap_kwh: float | None, name: str
) -> None:
    """
    Test that invalid step inputs raise ValueError and leave the state unchanged.

    Args:
        action: Requested action.
        delta_t: Duration of the step (hours).
        charge_cap_kwh: Charge cap (kWh), or None.
        name: Input expected in the message.
    """
    battery = make_battery(soc_initial=0.5)

    with pytest.raises(ValueError, match=name):
        battery.step(action, delta_t, charge_cap_kwh)
    assert battery.soc_kwh == 5.0
