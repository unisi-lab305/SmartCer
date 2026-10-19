"""
Tests of the forecast protocol and of `PerfectForecast` on the synthetic scenario.
"""

from pathlib import Path

import numpy as np
import pytest

from smartcer.data import load_scenario
from smartcer.forecasts import FORECAST_PROVIDERS, ForecastProvider, PerfectForecast


pytestmark = pytest.mark.unit


def test_perfect_forecast_returns_the_true_future(mini_cer_v1_path: Path) -> None:
    """
    Test that the perfect forecast of steps t..t+H-1 is the data of every node and the totals of
    the nodes without a battery: pv_bat generates 10 kWh in hours 8-15, factory_bat consumes
    10 kWh in hours 8-17 and 4 kWh otherwise, a producer has zero load and a consumer zero
    generation; house_1 and shop generate 5 kWh in hours 8-15 (shop) and consume 2 + 6 = 8 kWh.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    forecast = PerfectForecast(community)
    forecast.reset(np.random.default_rng(0))
    values = forecast.get(6, 4)

    assert set(values) == {f'{n}/{q}' for n in (*community.node_ids, 'community')
                           for q in ('generation', 'load')}
    np.testing.assert_array_equal(values['pv_bat/generation'], [0.0, 0.0, 10.0, 10.0])
    np.testing.assert_array_equal(values['pv_bat/load'], [0.0] * 4)
    np.testing.assert_array_equal(values['factory_bat/load'], [4.0, 4.0, 10.0, 10.0])
    np.testing.assert_array_equal(values['house_1/generation'], [0.0] * 4)
    np.testing.assert_array_equal(values['community/generation'], [0.0, 0.0, 5.0, 5.0])
    np.testing.assert_array_equal(values['community/load'], [8.0] * 4)
    for node in community.nodes:
        np.testing.assert_array_equal(values[f'{node.node_id}/load'], node.load_kwh[6:10])


def test_perfect_forecast_returns_copies(mini_cer_v1_path: Path) -> None:
    """
    Test that changing a returned forecast does not change the scenario data.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    PerfectForecast(community).get(8, 2)['pv_bat/generation'][:] = -1.0

    assert community.node('pv_bat').generation_kwh[8] == 10.0


@pytest.mark.parametrize(('t', 'horizon'), [(0, 0), (-1, 2), (47, 2), (40, 9)])
def test_perfect_forecast_rejects_steps_outside_the_period(mini_cer_v1_path: Path, t: int,
                                                           horizon: int) -> None:
    """
    Test that a horizon below 1 and forecast steps outside the 48 steps raise.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        t: First forecast step.
        horizon: Forecast horizon.
    """
    forecast = PerfectForecast(load_scenario(mini_cer_v1_path))

    with pytest.raises(ValueError, match='horizon|leaves the period'):
        forecast.get(t, horizon)


def test_perfect_forecast_satisfies_the_protocol(mini_cer_v1_path: Path) -> None:
    """
    Test that `PerfectForecast` is a `ForecastProvider` and is registered as 'perfect'.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)

    assert isinstance(PerfectForecast(community), ForecastProvider)
    assert isinstance(FORECAST_PROVIDERS['perfect'](community), PerfectForecast)
    assert not isinstance(object(), ForecastProvider)
