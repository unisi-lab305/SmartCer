"""
Unit tests of the KPIs and the Fit Score (`smartcer.gym.metrics`) on hand-computed tables.
"""

import pandas as pd
import pytest

from smartcer.gym.metrics import compute_kpis, daily_welfare, fit_score


pytestmark = pytest.mark.unit


def test_fit_score_is_100_on_identical_series() -> None:
    """
    Test F = 100 when the candidate equals the reference.
    """
    series = [3.5, -1.0, 2.25, 8.0]

    assert fit_score(series, series) == 100.0


def test_fit_score_hand_example() -> None:
    """
    Test T1 = (1, 2, 3), T2 = (1, 2, 4): Σ (T2 - T1)² = 1, Σ (T1 - 2)² = 2, F = (1 - 1/2) · 100 = 50;
    the mean of T1 as candidate gives F = 0, and a farther candidate a negative F.
    """
    assert fit_score([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(50.0)
    assert fit_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert fit_score([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-300.0)


@pytest.mark.parametrize(('reference', 'candidate', 'message'), [
    ([1.0, 2.0], [1.0], 'same length'),
    ([], [], 'non-empty'),
    ([1.0, float('nan')], [1.0, 2.0], 'finite'),
    ([2.0, 2.0], [1.0, 2.0], 'constant reference'),
])
def test_fit_score_invalid_input_raises(reference: list[float], candidate: list[float],
                                        message: str) -> None:
    """
    Test the errors of the Fit Score.

    Args:
        reference: Reference series.
        candidate: Candidate series.
        message: Expected error text.
    """
    with pytest.raises(ValueError, match=message):
        fit_score(reference, candidate)


def test_daily_welfare_sums_each_calendar_day() -> None:
    """
    Test the daily welfare of three hourly steps across midnight.
    """
    table = pd.DataFrame({'datetime': pd.to_datetime(['2022-01-03 22:00', '2022-01-03 23:00',
                                                      '2022-01-04 00:00']),
                          'welfare': [1.0, 2.5, -4.0]})

    daily = daily_welfare(table)

    assert daily.tolist() == [3.5, -4.0]
    assert [str(day) for day in daily.index] == ['2022-01-03', '2022-01-04']
    with pytest.raises(ValueError, match='welfare'):
        daily_welfare(table.drop(columns='welfare'))


def community_table() -> pd.DataFrame:
    """
    Two-step community table: exports 10 and 0 kWh, imports 4 and 6 kWh, shared 4 and 0 kWh.

    Returns:
        pd.DataFrame: The table.
    """
    return pd.DataFrame({
        'reward': [1.0, -2.5], 'welfare': [1.5, -1.0], 'revenue': [1.0, 0.0],
        'cost': [0.8, 1.2], 'storage_cost': [0.1, 0.2], 'incentive': [1.4, 0.4],
        'shared_energy_kwh': [4.0, 0.0], 'export_kwh': [10.0, 0.0], 'import_kwh': [4.0, 6.0],
        'penalty': [0.0, 0.15],
    })


def node_table(throughput: list[float], penalty: list[float]) -> pd.DataFrame:
    """
    Two-step node table.

    Args:
        throughput: Cell throughput per step (kWh).
        penalty: k per step.

    Returns:
        pd.DataFrame: The table, with charge = throughput of step 1 and discharge of step 2.
    """
    return pd.DataFrame({'charge_kwh': [throughput[0], 0.0], 'discharge_kwh': [0.0, throughput[1]],
                         'throughput_kwh': throughput, 'penalty': penalty})


def test_kpis_hand_values() -> None:
    """
    Test the KPIs of the two-step tables: sums, rates (4 / 10 and 4 / 10), equivalent full cycles
    (battery a: 10 kWh through a 10 kWh cell = 0.5 cycles; b: 4 kWh through 20 kWh = 0.1; total
    14 / 60), cycles on the usable capacity (a: S^min 2 kWh, 10 / 16; b: S^min 0, 0.1; total
    14 / 56) and penalty statistics.
    """
    nodes = {'a': node_table([6.0, 4.0], [0.0, 0.3]), 'b': node_table([4.0, 0.0], [0.0, 0.0]),
             'c': pd.DataFrame({'penalty': [0.0, 0.0]})}

    kpis = compute_kpis(community_table(), nodes, {'a': 10.0, 'b': 20.0}, {'a': 2.0})

    assert kpis['steps'] == 2
    assert kpis['welfare'] == pytest.approx(0.5)
    assert kpis['revenue'] - kpis['cost'] - kpis['storage_cost'] + kpis['incentive'] == \
        pytest.approx(kpis['welfare'])
    assert kpis['reward'] == pytest.approx(-1.5)
    assert kpis['shared_energy_kwh'] == 4.0
    assert kpis['sharing_rate'] == pytest.approx(0.4)
    assert kpis['coverage_rate'] == pytest.approx(0.4)
    assert kpis['throughput_kwh'] == pytest.approx(14.0)
    assert kpis['equivalent_full_cycles'] == pytest.approx(14.0 / 60.0)
    assert kpis['efc_usable'] == pytest.approx(14.0 / 56.0)
    assert kpis['penalty_mean'] == pytest.approx(0.075)
    assert kpis['penalty_max'] == pytest.approx(0.15)
    assert kpis['penalty_steps'] == 1
    assert kpis['batteries']['a'] == {
        'charge_kwh': 6.0, 'discharge_kwh': 4.0, 'throughput_kwh': 10.0,
        'equivalent_full_cycles': 0.5, 'efc_usable': pytest.approx(10.0 / 16.0),
        'penalty_mean': pytest.approx(0.15),
        'penalty_max': pytest.approx(0.3), 'penalty_steps': 1,
    }
    assert kpis['batteries']['b']['equivalent_full_cycles'] == pytest.approx(0.1)
    assert kpis['batteries']['b']['efc_usable'] == pytest.approx(0.1)
    assert set(kpis['batteries']) == {'a', 'b'}


def test_kpis_report_the_gap_from_the_oracle_state_of_charge() -> None:
    """
    Test that a battery table with `soc_gap_kwh` (runs with an oracle) adds the mean and the
    maximum of the absolute gap, by hand: gaps 1 and -3 kWh → mean 2, max 3; a table without the
    column adds neither.
    """
    gap = node_table([6.0, 4.0], [0.0, 0.0]).assign(soc_gap_kwh=[1.0, -3.0])
    nodes = {'a': gap, 'b': node_table([4.0, 0.0], [0.0, 0.0])}

    batteries = compute_kpis(community_table(), nodes, {'a': 10.0, 'b': 20.0})['batteries']

    assert batteries['a']['soc_gap_mean_kwh'] == pytest.approx(2.0)
    assert batteries['a']['soc_gap_max_kwh'] == pytest.approx(3.0)
    assert 'soc_gap_mean_kwh' not in batteries['b'] and 'soc_gap_max_kwh' not in batteries['b']


def test_s12_rates_are_named_after_what_they_measure() -> None:
    """
    Test the replacement of the 0.6.2 KPIs (S12): shared / exported is the sharing rate,
    shared / imported the coverage rate; the 0.6.2 names are gone; zero denominators give 0.
    """
    table = community_table().assign(export_kwh=0.0, import_kwh=0.0)

    kpis = compute_kpis(table, {}, {})

    assert kpis['sharing_rate'] == 0.0 and kpis['coverage_rate'] == 0.0
    assert kpis['equivalent_full_cycles'] == 0.0 and kpis['efc_usable'] == 0.0
    for name in ('self_consumption', 'net_metering', 'energy_balance', 'tip_potential'):
        assert name not in kpis


def test_kpis_invalid_input_raises() -> None:
    """
    Test the errors: missing community column, missing battery table or column, capacity ≤ 0,
    minimum state of charge outside [0, S^max).
    """
    nodes = {'a': node_table([1.0, 1.0], [0.0, 0.0])}
    with pytest.raises(ValueError, match=r"missing columns \['penalty'\]"):
        compute_kpis(community_table().drop(columns='penalty'), nodes, {'a': 10.0})
    with pytest.raises(ValueError, match="no table for battery node 'b'"):
        compute_kpis(community_table(), nodes, {'b': 10.0})
    with pytest.raises(ValueError, match="node table 'a': missing columns"):
        compute_kpis(community_table(), {'a': nodes['a'].drop(columns='charge_kwh')}, {'a': 10.0})
    with pytest.raises(ValueError, match='capacity must be > 0'):
        compute_kpis(community_table(), nodes, {'a': 0.0})
    with pytest.raises(ValueError, match=r'minimum state of charge must be in \[0, 10.0\)'):
        compute_kpis(community_table(), nodes, {'a': 10.0}, {'a': 10.0})
