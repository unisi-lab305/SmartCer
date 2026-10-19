"""
Unit tests of the observation encoders and of the `meco` and `extended` presets
(`smartcer.gym.observation`).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from smartcer.data import load_scenario
from smartcer.gym.observation import MinMaxNormalizer, ObservationBuilder
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
        ObservationBuilder(load_scenario(mini_cer_v1_path), preset='paper')


# =========================
# Extended preset, one test per feature
# =========================


def test_extended_layout_sections(mini_cer_v1_path: Path) -> None:
    """
    Test the sections of the extended preset with lookback 2: time (with the holiday flag), no
    global prices, community aggregates, per battery node block, empty forecast section; the
    sections list every layout entry in vector order.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended', lookback=2)

    assert builder.sections['time'] == ['time/month', 'time/hour', 'time/day_of_week',
                                        'time/holiday']
    assert builder.sections['price'] == [] and builder.sections['forecast'] == []
    assert builder.sections['community'] == ['community/generation', 'community/load']
    assert builder.sections['nodes'][:9] == [f'pv_bat/{f}' for f in (
        'sell_price', 'buy_price', 'generation', 'load', 'generation_lookback', 'load_lookback',
        'soc', 'a_lower', 'a_upper')]
    ordered = [name for names in builder.sections.values() for name in names]
    assert ordered == list(builder.layout)
    assert builder.layout['pv_bat/load_lookback'].stop - builder.layout[
        'pv_bat/load_lookback'].start == 2
    assert builder.dim == 12 + 2 + 2 * 11


def test_time_month_hour_and_day_of_week(mini_cer_v1_path: Path) -> None:
    """
    Test the time features at 06:00 of Tuesday 2022-01-04 (t = 30): January (0, 1), hour 6
    (1, 0), Tuesday one-hot.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended')
    obs = builder.build(30)

    assert obs[builder.layout['time/month']] == pytest.approx([0.0, 1.0])
    assert obs[builder.layout['time/hour']] == pytest.approx([1.0, 0.0], abs=1e-7)
    assert obs[builder.layout['time/day_of_week']].tolist() == [0, 1, 0, 0, 0, 0, 0]


def test_time_holiday_flag(write_scenario: Callable[..., Path]) -> None:
    """
    Test the Italian holiday flag across midnight of 6 January 2022 (Epifania), and on 2 June
    (Festa della Repubblica) with its month encoding.

    Args:
        write_scenario: Scenario factory.
    """
    header = 'datetime,production,consumption,sell_price,buy_price\n'
    stamps = ['2022-01-05 22:00:00', '2022-01-05 23:00:00', '2022-01-06 00:00:00',
              '2022-01-06 01:00:00']
    text = header + ''.join(f'{s},1,1,0.1,0.3\n' for s in stamps)
    storage = {'capacity': 10, 'power_charge_max': 5, 'power_discharge_max': 5,
               'efficiency_charge': 1.0, 'efficiency_discharge': 1.0}
    nodes = {'a': {'type': 'prosumer', 'csv': 'a.csv', 'storage': storage}}
    builder = ObservationBuilder(load_scenario(write_scenario(nodes, {'a.csv': text})),
                                 preset='extended')
    flags = [builder.build(t)[builder.layout['time/holiday']][0] for t in range(4)]
    june = header + '2022-06-02 12:00:00,1,1,0.1,0.3\n2022-06-02 13:00:00,1,1,0.1,0.3\n'
    republic_day = ObservationBuilder(load_scenario(write_scenario(nodes, {'a.csv': june})),
                                      preset='extended')
    obs = republic_day.build(0)

    assert flags == [0.0, 0.0, 1.0, 1.0]
    assert obs[republic_day.layout['time/holiday']] == pytest.approx([1.0])
    assert obs[republic_day.layout['time/month']] == pytest.approx(
        [np.sin(2 * np.pi * 5 / 12), np.cos(2 * np.pi * 5 / 12)])


def test_node_prices(mini_cer_v1_path: Path) -> None:
    """
    Test the per-node prices, not normalised: pv_bat sells at 0.10 €/kWh at 10:00 of day 1 and
    has buy price 0 (producer); factory_bat buys at 0.30 €/kWh.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended')
    obs = builder.build(10)

    assert obs[builder.layout['pv_bat/sell_price']] == pytest.approx([0.10])
    assert obs[builder.layout['pv_bat/buy_price']] == pytest.approx([0.0])
    assert obs[builder.layout['factory_bat/buy_price']] == pytest.approx([0.30])


def test_community_aggregates(mini_cer_v1_path: Path) -> None:
    """
    Test the aggregates of the nodes without a battery (house_1, shop): generation 0 or 5 kWh →
    0 or 1; load 2 + 6 = 8 kWh at every step, a constant series → 0.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended')

    assert builder.build(0)[builder.layout['community/generation']] == pytest.approx([0.0])
    assert builder.build(10)[builder.layout['community/generation']] == pytest.approx([1.0])
    assert builder.build(10)[builder.layout['community/load']] == pytest.approx([0.0])
    assert builder.normalizer.bounds['community/load'] == (8.0, 8.0)


def test_node_generation_and_load(mini_cer_v1_path: Path) -> None:
    """
    Test g and l min-max scaled: factory_bat load 10 kWh at 10:00 (max 10, min 4) → 1, 4 kWh at
    00:00 → 0; pv_bat generation 10 kWh at 10:00 → 1.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended')

    assert builder.build(10)[builder.layout['factory_bat/load']] == pytest.approx([1.0])
    assert builder.build(0)[builder.layout['factory_bat/load']] == pytest.approx([0.0])
    assert builder.build(10)[builder.layout['pv_bat/generation']] == pytest.approx([1.0])


@pytest.mark.parametrize(('observe_current_step', 't', 'expected'), [
    (True, 10, [1.0, 1.0, 0.0]),   # g of steps 9, 8, 7
    (False, 10, [1.0, 0.0, 0.0]),  # g of steps 8, 7, 6 (the observed g is that of step 9)
    (True, 1, [0.0, 0.0, 0.0]),    # step 0, then before the first row
])
def test_lookback(mini_cer_v1_path: Path, observe_current_step: bool, t: int,
                  expected: list[float]) -> None:
    """
    Test the lookback of pv_bat generation (10 kWh in hours 8-15), most recent first, with both
    information timings and zeros before the first row of the data.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        observe_current_step: Information timing.
        t: Step.
        expected: Expected lookback values.
    """
    builder = ObservationBuilder(load_scenario(mini_cer_v1_path), preset='extended', lookback=3,
                                 observe_current_step=observe_current_step)

    assert builder.build(t)[builder.layout['pv_bat/generation_lookback']] == pytest.approx(
        expected)


def test_soc_and_bounds_follow_the_battery(mini_cer_v1_path: Path) -> None:
    """
    Test SoC and bounds after factory_bat charges 5 kWh at 10:00 of day 1 (from S^min 4 kWh to
    8.5 kWh): SoC (8.5 - 4) / 36 = 0.125, A^lower 0.9 × 4.5 / 10 = 0.405, A^upper 1 at 11:00.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    builder = ObservationBuilder(community, preset='extended')
    for t in range(10):
        community.step(t, np.zeros(2))
    community.step(10, np.array([0.0, 0.5]))
    obs = builder.build(11)

    assert obs[builder.layout['factory_bat/soc']] == pytest.approx([0.125])
    assert obs[builder.layout['factory_bat/a_lower']] == pytest.approx([0.405])
    assert obs[builder.layout['factory_bat/a_upper']] == pytest.approx([1.0])


def test_reserved_prefix_and_missing_normalizer_series_raise(
        mini_cer_v1_path: Path, write_scenario: Callable[..., Path]) -> None:
    """
    Test that a node named 'community' and a normaliser without the community's series raise.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        write_scenario: Scenario factory.
    """
    text = ('datetime,production,consumption,sell_price,buy_price\n'
            '2022-01-03 00:00:00,1,1,0.1,0.3\n')
    nodes = {'community': {'type': 'prosumer', 'csv': 'a.csv'}}
    with pytest.raises(ValueError, match='reserved'):
        ObservationBuilder(load_scenario(write_scenario(nodes, {'a.csv': text})))
    with pytest.raises(ValueError, match='no bounds'):
        ObservationBuilder(load_scenario(mini_cer_v1_path),
                           normalizer=MinMaxNormalizer({'pv_bat/generation': (0.0, 10.0)}))
