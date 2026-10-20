"""
Tests of the oracle adapter `read_oracle` (provisional DIISM file format) and of `OracleProvider`
on the synthetic scenario (batteries pv_bat 5 kW and factory_bat 10 kW).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartcer.data import load_scenario
from smartcer.oracle import OracleProvider, OracleWarning, read_oracle


pytestmark = pytest.mark.unit

HEADER = 'datetime,node_id,p_cha_kw,p_dis_kw,soc_kwh'
STEPS = pd.date_range('2022-01-03 00:00', periods=48, freq='h')
# S^ini of the batteries: soc_min 0.1 of 20 and 40 kWh
SOC_INITIAL = {'pv_bat': 2.0, 'factory_bat': 4.0}


def write_oracle(folder: Path, trajectory: list[str] | pd.DataFrame,
                 objective: list[str] | None = None) -> Path:
    """
    Write an oracle folder.

    Args:
        folder: Folder to create.
        trajectory: Lines of `trajectory.csv` (header included) or its table.
        objective: Lines of `objective.csv` (header included), or None for no file.

    Returns:
        Path: The folder.
    """
    folder.mkdir()
    if isinstance(trajectory, pd.DataFrame):
        trajectory.to_csv(folder / 'trajectory.csv', index=False)
    else:
        (folder / 'trajectory.csv').write_text('\n'.join(trajectory) + '\n')
    if objective is not None:
        (folder / 'objective.csv').write_text('\n'.join(objective) + '\n')
    return folder


def full_trajectory(steps: pd.DatetimeIndex = STEPS,
                    nodes: tuple[str, ...] = ('factory_bat', 'pv_bat')) -> pd.DataFrame:
    """
    Idle trajectory with one row per step and node, factory_bat first.

    Args:
        steps: Timestamps.
        nodes: Node ids.

    Returns:
        pd.DataFrame: Columns of the trajectory file, zero powers, state of charge S^ini of the
        scenario (0 for a node without a battery).
    """
    return pd.DataFrame([{'datetime': str(s), 'node_id': n, 'p_cha_kw': 0.0, 'p_dis_kw': 0.0,
                          'soc_kwh': SOC_INITIAL.get(n, 0.0)} for s in steps for n in nodes])


def set_row(table: pd.DataFrame, step: int, node_id: str, **values: float) -> None:
    """
    Set the values of one row of a trajectory table.

    Args:
        table: Trajectory table.
        step: Step index.
        node_id: Node id.
        **values: Column → value.
    """
    row = (table['datetime'] == str(STEPS[step])) & (table['node_id'] == node_id)
    for column, value in values.items():
        table.loc[row, column] = value


def test_read_oracle_pivots_the_trajectory(tmp_path: Path) -> None:
    """
    Test that rows in any order become tables per quantity (sorted timestamps, one column per
    node), that extra columns are ignored, and that the daily objective is read.

    Args:
        tmp_path: Pytest temporary directory.
    """
    folder = write_oracle(tmp_path / 'oracle', [
        HEADER + ',note',
        '2022-01-03 01:00:00,pv_bat,0,1.5,3.0,x',
        '2022-01-03 00:00:00,pv_bat,2.5,0,4.5,y',
        '2022-01-03 01:00:00,factory_bat,10,0,13.0,',
        '2022-01-03 00:00:00,factory_bat,0,0,4.0,',
    ], ['date,objective,extra', '2022-01-04,-3.5,a', '2022-01-03,12.25,b'])

    data = read_oracle(folder)

    assert list(data.p_cha_kw.index) == list(STEPS[:2])
    assert data.p_cha_kw.loc[STEPS[0], 'pv_bat'] == 2.5
    assert data.p_dis_kw.loc[STEPS[1], 'pv_bat'] == 1.5
    assert data.soc_kwh.loc[STEPS[1], 'factory_bat'] == 13.0
    assert [str(d) for d in data.daily_objective.index] == ['2022-01-03', '2022-01-04']
    assert list(data.daily_objective) == [12.25, -3.5]
    assert data.source == str(folder)


def test_read_oracle_objective_is_optional_and_round_off_is_clipped(tmp_path: Path) -> None:
    """
    Test that a folder without `objective.csv` has no daily objective, and that a power of
    -1e-12 kW (solver round-off) is read as 0.

    Args:
        tmp_path: Pytest temporary directory.
    """
    data = read_oracle(write_oracle(tmp_path / 'oracle', [
        HEADER, '2022-01-03 00:00:00,pv_bat,0,-1e-12,4.0']))

    assert data.daily_objective is None
    assert data.p_dis_kw.loc[STEPS[0], 'pv_bat'] == 0.0


@pytest.mark.parametrize(('lines', 'match'), [
    (['datetime,node_id,p_cha_kw,p_dis_kw', '2022-01-03 00:00:00,pv_bat,0,0'],
     r"missing columns \['soc_kwh'\]"),
    ([HEADER], 'no rows'),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,0,0,4', 'not a date,pv_bat,0,0,4'],
     "'datetime' is not a timestamp at line 3"),
    ([HEADER, '2022-01-03 00:00:00+01:00,pv_bat,0,0,4'], 'time zone'),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,abc,0,4'], "'p_cha_kw' is not a finite number at line 2"),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,0,0,nan'], "'soc_kwh' is not a finite number"),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,-0.5,0,4'], "'p_cha_kw' is negative at line 2"),
    ([HEADER, '2022-01-03 00:00:00,,0,0,4'], "'node_id' is empty at line 2"),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,0,0,4', '2022-01-03 00:00:00,pv_bat,1,0,4'],
     "line 3 repeats datetime 2022-01-03 00:00:00 and node 'pv_bat'"),
    ([HEADER, '2022-01-03 00:00:00,pv_bat,0,0,4', '2022-01-03 00:00:00,factory_bat,0,0,4',
      '2022-01-03 01:00:00,pv_bat,0,0,4'],
     r"no row for nodes \['factory_bat'\] at 2022-01-03 01:00:00"),
])
def test_read_oracle_rejects_malformed_trajectories(tmp_path: Path, lines: list[str],
                                                    match: str) -> None:
    """
    Test that every malformed trajectory raises, naming the file, the column and the line.

    Args:
        tmp_path: Pytest temporary directory.
        lines: Lines of the trajectory file.
        match: Expected message.
    """
    folder = write_oracle(tmp_path / 'oracle', lines)

    with pytest.raises(ValueError, match=match):
        read_oracle(folder)


def test_read_oracle_rejects_missing_file_and_repeated_days(tmp_path: Path) -> None:
    """
    Test that a folder without `trajectory.csv` and an objective with a repeated day raise.

    Args:
        tmp_path: Pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match='trajectory.csv missing'):
        read_oracle(tmp_path)
    folder = write_oracle(tmp_path / 'oracle', [HEADER, '2022-01-03 00:00:00,pv_bat,0,0,4'],
                          ['date,objective', '2022-01-03,1', '2022-01-03,2'])
    with pytest.raises(ValueError, match='line 3 repeats the day 2022-01-03'):
        read_oracle(folder)


def test_provider_computes_the_optimal_actions(tmp_path: Path, mini_cer_v1_path: Path) -> None:
    """
    Test a* = p_cha / P^cha - p_dis / P^dis by hand, in action order (pv_bat, factory_bat) although
    the file lists factory_bat first: pv_bat charging 2.5 of 5 kW → 0.5, discharging 1 kW → -0.2;
    factory_bat discharging 10 of 10 kW → -1. The state of charge of `soc_kwh` is that of the
    file, at the end of each step; `soc(t)` is the state at the start of step t: S^ini (2 and
    4 kWh) at t = 0, else the file's state at the end of step t - 1. Rows after the period are
    ignored and the objective is kept for the days of the period.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    table = full_trajectory(STEPS.append(pd.date_range('2022-01-05', periods=3, freq='h')))
    set_row(table, 9, 'pv_bat', p_cha_kw=2.5, soc_kwh=6.25)
    set_row(table, 30, 'pv_bat', p_dis_kw=1.0)
    set_row(table, 20, 'factory_bat', p_dis_kw=10.0, soc_kwh=15.0)
    folder = write_oracle(tmp_path / 'oracle', table, [
        'date,objective', '2022-01-03,10', '2022-01-04,-2', '2022-01-05,99'])

    oracle = OracleProvider.from_folder(folder, load_scenario(mini_cer_v1_path))

    assert oracle.node_ids == ('pv_bat', 'factory_bat')
    assert oracle.actions.shape == (48, 2)
    np.testing.assert_allclose(oracle.action(9), [0.5, 0.0])
    np.testing.assert_allclose(oracle.action(30), [-0.2, 0.0])
    np.testing.assert_allclose(oracle.action(20), [0.0, -1.0])
    assert np.count_nonzero(oracle.actions) == 3
    np.testing.assert_allclose(oracle.soc_kwh[9], [6.25, 4.0])
    np.testing.assert_allclose(oracle.soc_kwh[20], [2.0, 15.0])
    np.testing.assert_allclose(oracle.soc(0), [2.0, 4.0])
    np.testing.assert_allclose(oracle.soc(9), [2.0, 4.0])
    np.testing.assert_allclose(oracle.soc(10), [6.25, 4.0])
    np.testing.assert_allclose(oracle.soc(21), [2.0, 15.0])
    assert list(oracle.daily_objective) == [10.0, -2.0]
    oracle.action(9)[:] = 0.0
    assert oracle.action(9)[0] == 0.5


def test_provider_aligns_with_a_sub_period(tmp_path: Path, mini_cer_v1_path: Path) -> None:
    """
    Test that an oracle of the whole scenario serves a community loaded for the second day, whose
    first step starts from the oracle state at the end of the first day (factory_bat 7 kWh).

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    table = full_trajectory()
    set_row(table, 23, 'factory_bat', soc_kwh=7.0)
    set_row(table, 33, 'factory_bat', p_cha_kw=5.0)
    folder = write_oracle(tmp_path / 'oracle', table, [
        'date,objective', '2022-01-03,10', '2022-01-04,-2'])
    community = load_scenario(mini_cer_v1_path, start='2022-01-04 00:00', end='2022-01-04 23:00')

    oracle = OracleProvider.from_folder(folder, community)

    np.testing.assert_allclose(oracle.action(9), [0.0, 0.5])
    np.testing.assert_allclose(oracle.soc(0), [2.0, 7.0])
    assert list(oracle.daily_objective) == [-2.0]


def test_provider_counts_simultaneous_charge_and_discharge(tmp_path: Path,
                                                          mini_cer_v1_path: Path) -> None:
    """
    Test that rows with p_cha > 0 and p_dis > 0 are counted in the period only and reported with
    one warning naming the count and the first timestamp, and that a* is their net value
    (pv_bat 2 kW in, 1 kW out of 5 kW → 0.2).

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    table = full_trajectory()
    set_row(table, 30, 'factory_bat', p_cha_kw=1.0, p_dis_kw=1.0)
    set_row(table, 12, 'pv_bat', p_cha_kw=2.0, p_dis_kw=1.0)
    set_row(table, 40, 'pv_bat', p_cha_kw=1.0)
    folder = write_oracle(tmp_path / 'oracle', table)

    with pytest.warns(OracleWarning, match='2 rows of the period charge and discharge in the same '
                                           r"step .* first at 2022-01-03 12:00:00 \(node 'pv_bat'\)"):
        oracle = OracleProvider.from_folder(folder, load_scenario(mini_cer_v1_path))
    second_day = load_scenario(mini_cer_v1_path, start='2022-01-04 00:00')
    with pytest.warns(OracleWarning, match='1 rows'):
        assert OracleProvider.from_folder(folder, second_day).simultaneous_rows == 1

    assert oracle.simultaneous_rows == 2
    np.testing.assert_allclose(oracle.action(12), [0.2, 0.0])


@pytest.mark.parametrize(('change', 'match'), [
    ('extra_node', r"not battery nodes \['house_1'\]"),
    ('missing_node', r"missing \['factory_bat'\]"),
    ('missing_step', 'no oracle rows for 1 steps of the period .* first 2022-01-03 05:00:00'),
    ('over_power', "node 'pv_bat' at 2022-01-03 10:00:00: p_cha 6.0 kW"),
    ('missing_day', 'no objective for 1 days of the period, first 2022-01-04'),
])
def test_provider_rejects_inconsistent_oracles(tmp_path: Path, mini_cer_v1_path: Path,
                                               change: str, match: str) -> None:
    """
    Test that the provider rejects an oracle whose nodes are not the battery nodes, that lacks a
    step of the period, whose power exceeds the battery limit, or that lacks the objective of a
    day of the period.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        change: Inconsistency to introduce.
        match: Expected message.
    """
    nodes = {'extra_node': ('factory_bat', 'pv_bat', 'house_1'),
             'missing_node': ('pv_bat',)}.get(change, ('factory_bat', 'pv_bat'))
    table = full_trajectory(nodes=nodes)
    if change == 'missing_step':
        table = table[table['datetime'] != str(STEPS[5])]
    if change == 'over_power':
        set_row(table, 10, 'pv_bat', p_cha_kw=6.0)
    objective = ['date,objective', '2022-01-03,1']
    if change != 'missing_day':
        objective.append('2022-01-04,1')
    folder = write_oracle(tmp_path / 'oracle', table, objective)

    with pytest.raises(ValueError, match=match):
        OracleProvider.from_folder(folder, load_scenario(mini_cer_v1_path))


@pytest.mark.parametrize(('node_id', 'values', 'error'), [
    ('pv_bat', {'p_cha_kw': 2.5, 'soc_kwh': 4.25}, None),
    ('factory_bat', {'p_dis_kw': 1.0, 'soc_kwh': 4.0 - 1.0 / 0.9}, None),
    ('pv_bat', {'soc_kwh': 2.0 + 5e-4}, None),
    ('pv_bat', {'p_cha_kw': 2.5, 'soc_kwh': 4.3}, 2.05),
    ('factory_bat', {'soc_kwh': 4.0 + 2e-3}, 4.002),
])
def test_provider_checks_the_inferred_initial_state_of_charge(
        tmp_path: Path, mini_cer_v1_path: Path, node_id: str, values: dict[str, float],
        error: float | None) -> None:
    """
    Test that the MILP's S^ini is inferred from the first row of the file as
    s - η^cha p^cha Δt + p^dis Δt / η^dis (η = 0.9, Δt = 1 h) and checked against the scenario's
    S^ini (pv_bat 2 kWh, factory_bat 4 kWh) within 1e-3 kWh: pv_bat charging 2.5 kW to 4.25 kWh
    and factory_bat discharging 1 kW to 4 - 1/0.9 kWh start from S^ini; 4.3 kWh after the same
    charge starts from 2.05 kWh and raises; a difference of 5e-4 kWh (a rounded file) is accepted,
    one of 2e-3 kWh is not.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        node_id: Battery node of the changed first row.
        values: Values of the first row.
        error: Inferred S^ini named by the expected error, or None if the oracle is accepted.
    """
    table = full_trajectory()
    set_row(table, 0, node_id, **values)
    folder = write_oracle(tmp_path / 'oracle', table)
    community = load_scenario(mini_cer_v1_path)

    if error is None:
        np.testing.assert_allclose(OracleProvider.from_folder(folder, community).soc(0),
                                   [2.0, 4.0])
    else:
        with pytest.raises(ValueError, match=rf"node '{node_id}': the MILP starts from "
                                             rf'{error:.6f} kWh \(inferred from its first step '
                                             r'2022-01-03 00:00:00\), the scenario from '
                                             'soc_initial 0.1'):
            OracleProvider.from_folder(folder, community)


def test_provider_needs_the_step_before_a_sub_period(tmp_path: Path,
                                                     mini_cer_v1_path: Path) -> None:
    """
    Test that a period starting after the first step of the file needs the row of the step before
    it, whose state of charge is the initial one.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    table = full_trajectory()
    folder = write_oracle(tmp_path / 'oracle', table[table['datetime'] != str(STEPS[23])])
    community = load_scenario(mini_cer_v1_path, start='2022-01-04 00:00')

    with pytest.raises(ValueError, match=r'no row for the step before it \(2022-01-03 23:00:00\)'):
        OracleProvider.from_folder(folder, community)


@pytest.mark.parametrize(('soc_kwh', 'accepted'), [(9.9, True), (10.0, False)])
def test_inferred_initial_state_includes_self_discharge(tmp_path: Path, mini_cer_v1_copy: Path,
                                                        soc_kwh: float, accepted: bool) -> None:
    """
    Test that with self-discharge the inferred state is compared with S^ini after one step of
    self-discharge: pv_bat with S^ini 0.5 × 20 = 10 kWh and α = 0.01 per hour starts the first
    step from 9.9 kWh, so an idle first row at 9.9 kWh is accepted and one at 10 kWh is not.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_copy: Writable copy of the 1.0-format synthetic scenario.
        soc_kwh: State of charge of pv_bat in the idle first row.
        accepted: Whether the oracle is accepted.
    """
    path = mini_cer_v1_copy / 'config.json'
    config = json.loads(path.read_text())
    config['nodes']['pv_bat']['storage'].update(soc_initial=0.5, alpha=0.01)
    path.write_text(json.dumps(config))
    table = full_trajectory()
    set_row(table, 0, 'pv_bat', soc_kwh=soc_kwh)
    folder = write_oracle(tmp_path / 'oracle', table)
    community = load_scenario(mini_cer_v1_copy)

    if accepted:
        np.testing.assert_allclose(OracleProvider.from_folder(folder, community).soc(0),
                                   [10.0, 4.0])
    else:
        with pytest.raises(ValueError, match=r'starts from 10.000000 kWh .* soc_initial 0.5 = '
                                             r'10.000000 kWh \(9.900000 kWh after '
                                             r'self-discharge\)'):
            OracleProvider.from_folder(folder, community)


def test_initial_state_tolerance_is_configurable(tmp_path: Path, mini_cer_v1_path: Path) -> None:
    """
    Test that the tolerance of the S^ini check is an argument (default 1e-3 kWh): a first row 5e-4
    kWh above S^ini passes by default and with 1e-3, fails with 1e-6; a negative or non-finite
    tolerance raises.

    Args:
        tmp_path: Pytest temporary directory.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    table = full_trajectory()
    set_row(table, 0, 'pv_bat', soc_kwh=2.0 + 5e-4)
    folder = write_oracle(tmp_path / 'oracle', table)
    community = load_scenario(mini_cer_v1_path)

    OracleProvider.from_folder(folder, community)
    OracleProvider.from_folder(folder, community, soc_initial_tolerance_kwh=1e-3)
    with pytest.raises(ValueError, match='the MILP starts from 2.000500 kWh'):
        OracleProvider.from_folder(folder, community, soc_initial_tolerance_kwh=1e-6)
    for tolerance in (-1e-3, float('nan')):
        with pytest.raises(ValueError, match='soc_initial_tolerance_kwh must be a finite number'):
            OracleProvider.from_folder(folder, community, soc_initial_tolerance_kwh=tolerance)
