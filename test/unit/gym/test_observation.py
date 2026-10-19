"""
Unit tests of the observation encoders and of the `meco` preset (`smartcer.gym.observation`).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from smartcer.data import load_scenario
from smartcer.gym.observation import ObservationBuilder
from smartcer.gym.observation.encoders import cyclical, min_max, one_hot


pytestmark = pytest.mark.unit


def test_encoders() -> None:
    """
    Test cyclical (sin, cos), one-hot and min-max scaling (constant series → 0).
    """
    assert cyclical([0, 6], 24) == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]), abs=1e-12)
    assert one_hot([0, 2], 3).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert min_max([2.0, 4.0], 2.0, 6.0).tolist() == [0.0, 0.5]
    assert min_max([3.0, 3.0], 3.0, 3.0).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match='class indices'):
        one_hot([7], 7)


def test_meco_layout_covers_the_vector_in_order(mini_cer_v1_path: Path) -> None:
    """
    Test the meco layout of the synthetic scenario: time features, then per battery node prices
    (buy prices differ between nodes, so no global pair), g, l, SoC and bounds; slices contiguous.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path))
    names = list(builder.layout)
    slices = list(builder.layout.values())

    assert names[:3] == ['time/month', 'time/hour', 'time/day_of_week']
    assert not builder.global_prices
    assert names[3:10] == [f'pv_bat/{f}' for f in ('sell_price', 'buy_price', 'generation', 'load',
                                                   'soc', 'a_lower', 'a_upper')]
    assert slices[0].start == 0 and slices[-1].stop == builder.dim == 25
    assert all(a.stop == b.start for a, b in zip(slices, slices[1:]))


def test_meco_first_observation_values(mini_cer_v1_path: Path) -> None:
    """
    Test the first observation (Monday 2022-01-03 00:00): January (0, 1), hour 0 (0, 1), Monday,
    pv_bat at S^min (SoC 0), A^upper 0 (no generation), A^lower 0 (empty battery).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path))
    obs = builder.build(0)

    assert obs.dtype == np.float32 and builder.space.contains(obs)
    assert obs[builder.layout['time/month']] == pytest.approx([0.0, 1.0])
    assert obs[builder.layout['time/hour']] == pytest.approx([0.0, 1.0])
    assert obs[builder.layout['time/day_of_week']].tolist() == [1, 0, 0, 0, 0, 0, 0]
    for feature in ('soc', 'a_lower', 'a_upper'):
        assert obs[builder.layout[f'pv_bat/{feature}']] == pytest.approx([0.0]), feature


def test_global_price_pair_when_all_nodes_share_prices(
        write_scenario: Callable[..., Path]) -> None:
    """
    Test that identical sell and buy price series give one global, not normalised, price pair.

    Args:
        write_scenario: Scenario factory.
    """
    text = 'datetime,production,consumption,sell_price,buy_price\n' + ''.join(
        f'2022-01-03 0{h}:00:00,{h},1,0.1{h},0.3\n' for h in range(3))
    storage = {'capacity': 10, 'power_charge_max': 5, 'power_discharge_max': 5,
               'efficiency_charge': 1.0, 'efficiency_discharge': 1.0}
    nodes = {'a': {'type': 'prosumer', 'csv': 'a.csv', 'storage': storage},
             'b': {'type': 'prosumer', 'csv': 'a.csv'}}
    builder = ObservationBuilder(load_scenario(write_scenario(nodes, {'a.csv': text})))
    obs = builder.build(2)

    assert builder.global_prices
    assert obs[builder.layout['price/sell']] == pytest.approx([0.12])
    assert obs[builder.layout['price/buy']] == pytest.approx([0.3])
    assert 'a/sell_price' not in builder.layout
    assert builder.space.high[builder.layout['price/buy']] > builder.space.low[
        builder.layout['price/buy']]


def test_unknown_preset_raises(mini_cer_v1_path: Path) -> None:
    """
    Test that an unknown observation preset raises ValueError.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    with pytest.raises(ValueError, match='unknown observation preset'):
        ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended')
