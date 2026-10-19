"""
Unit tests of `Battery.from_config` with the 1.0 and the 0.6.2 storage keys
(docs/FORMULATION.md section 3, docs/refactor/REFACTOR_v1.md section 6).
"""

from typing import Any
import warnings

import pytest

from smartcer.core.storage import Battery


pytestmark = pytest.mark.unit

LEATHER_STORAGE = {
    'capacity': 100, 'rated_power': 30, 'soc_min': 0.1, 'efficiency': 0.98, 'alfa': 0.0,
    'beta': 0.0,
}


def test_from_config_v1_keys() -> None:
    """
    Test that a 1.0 storage section sets every parameter and emits no warning.
    """
    config = {
        'capacity': 50.0, 'power_charge_max': 10.0, 'power_discharge_max': 20.0,
        'efficiency_charge': 0.95, 'efficiency_discharge': 0.9, 'soc_min': 0.1,
        'soc_initial': 0.5, 'alpha': 0.001, 'usage_cost': 0.01, 'charge_limit': 'surplus',
    }

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        battery = Battery.from_config(config)

    for key, value in config.items():
        assert getattr(battery, key) == value, key
    assert battery.soc_kwh == pytest.approx(25.0)


def test_from_config_v1_defaults() -> None:
    """
    Test the defaults: soc_min 0, soc_initial = soc_min, alpha 0, usage_cost 0, 'production'.
    """
    battery = Battery.from_config({
        'capacity': 10.0, 'power_charge_max': 5.0, 'power_discharge_max': 5.0,
        'efficiency_charge': 0.9, 'efficiency_discharge': 0.9,
    })

    assert (battery.soc_min, battery.soc_initial, battery.alpha) == (0.0, 0.0, 0.0)
    assert (battery.usage_cost, battery.charge_limit) == (0.0, 'production')


def test_from_config_legacy_keys_are_mapped_with_warning() -> None:
    """
    Test the 0.6.2 mapping on the Leather storage section: rated_power → both power limits,
    efficiency → both efficiencies, alfa → alpha, beta ignored, soc_initial = soc_min.
    """
    with pytest.warns(DeprecationWarning, match='rated_power') as record:
        battery = Battery.from_config(LEATHER_STORAGE)

    message = str(record[0].message)
    assert 'efficiency' in message and 'alfa' in message and 'beta ignored' in message
    assert (battery.power_charge_max, battery.power_discharge_max) == (30.0, 30.0)
    assert (battery.efficiency_charge, battery.efficiency_discharge) == (0.98, 0.98)
    assert battery.alpha == 0.0
    assert battery.soc_min_kwh == pytest.approx(10.0)
    assert battery.soc_kwh == pytest.approx(10.0)


def test_from_config_beta_is_ignored() -> None:
    """
    Test that a non-zero 0.6.2 degradation coefficient has no effect (decision D7).
    """
    config = {**LEATHER_STORAGE, 'beta': 0.5}

    with pytest.warns(DeprecationWarning, match='beta ignored'):
        battery = Battery.from_config(config)
    for _ in range(10):
        battery.step(1.0, 1.0, None)
        battery.step(-1.0, 1.0, None)

    assert battery.capacity == 100.0


@pytest.mark.parametrize(
    ('config', 'match'),
    [
        ({**LEATHER_STORAGE, 'power_charge_max': 30}, 'conflicts'),
        ({**LEATHER_STORAGE, 'capactiy': 100}, 'unknown storage keys'),
        ({'capacity': 10, 'power_charge_max': 5, 'power_discharge_max': 5}, 'missing'),
        ({**LEATHER_STORAGE, 'efficiency': 1.5}, 'efficiency_charge'),
    ],
)
def test_from_config_errors(config: dict[str, Any], match: str) -> None:
    """
    Test that conflicting, unknown, missing and invalid keys raise ValueError.

    Args:
        config: Invalid storage section.
        match: Text expected in the message.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(ValueError, match=match):
            Battery.from_config(config)
