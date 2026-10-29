"""
Replay of the MILP oracle: `info['privileged']`, `ReplayController` and `check_replay`
on a synthetic oracle trajectory (fixture `synthetic_oracle`), and on local oracle files when they
are present in `smartcer/cer_data/<scenario>/oracle/` (skipped otherwise).
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from smartcer.data import TimestampWarning, load_scenario
from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import ReplayController, check_replay, compute_baselines
from smartcer.oracle import OracleProvider, OracleWarning, read_oracle


pytestmark = pytest.mark.integration

CER_DATA_PATH = Path(__file__).parents[3] / 'smartcer' / 'cer_data'
# relative tolerance of the cumulative welfare against the MILP objective 
RTOL = 1e-3


def test_replay_of_a_synthetic_oracle_trajectory(mini_cer_v1_path: Path, synthetic_oracle: Path,
                                                 tmp_path: Path) -> None:
    """
    Test the replay check on a synthetic trajectory: playing a* over the 48 steps
    gives K_t = 0 at every step, the oracle states of charge and a cumulative welfare within
    1e-3 of the objective (here equal up to round-off), and writes the per-day comparison.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)
    output = tmp_path / 'out'

    result = check_replay(env, output)

    # the trajectory charges and discharges: 14 and 44 of the 96 actions
    assert np.count_nonzero(env.oracle.actions > 0) > 10
    assert np.count_nonzero(env.oracle.actions < 0) > 10
    assert result.passed
    assert result.steps == 48
    assert result.penalty_steps == 0 and result.penalty_max <= 1e-9
    assert result.soc_error_max_kwh <= 1e-9
    assert result.simultaneous_rows == 0
    assert result.relative_gap <= 1e-12
    assert result.objective == pytest.approx(result.welfare, rel=1e-12)
    assert result.fit_score == pytest.approx(100.0)
    daily = pd.read_csv(output / 'oracle_daily.csv')
    assert list(daily.columns) == ['date', 'welfare', 'objective', 'difference']
    assert list(daily['date']) == ['2022-01-03', '2022-01-04']
    np.testing.assert_allclose(daily['welfare'], daily['objective'], rtol=1e-12)
    summary = json.loads((output / 'oracle_replay.json').read_text(encoding='utf-8'))
    assert summary['passed'] is True and summary['rtol'] == RTOL
    assert summary['simultaneous_rows'] == 0
    assert summary['run_config']['oracle'] == str(synthetic_oracle)


def test_privileged_info_follows_the_returned_observation(mini_cer_v1_path: Path,
                                                          synthetic_oracle: Path) -> None:
    """
    Test that `info['privileged']` holds the oracle action and state of charge of the step of the
    observation it comes with (the next action; the last step again at the end), with a* computed
    from the file by hand (pv_bat 5 kW, factory_bat 10 kW) and the state of charge at the start
    of the step (S^ini 2 and 4 kWh at step 0, else the file's state at the end of the step
    before), equal to the observed one when the oracle actions are played; no `true_future`
    without forecasts.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
    """
    trajectory = pd.read_csv(synthetic_oracle / 'trajectory.csv', parse_dates=['datetime'])
    trajectory = trajectory.set_index(['datetime', 'node_id']).sort_index()
    power = {'pv_bat': 5.0, 'factory_bat': 10.0}
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)
    _, info = env.reset(seed=0)
    steps = []
    while True:
        t = env.current_step
        privileged = info['privileged']
        stamp = pd.Timestamp(env.community.timestamps[t])
        rows = [trajectory.loc[(stamp, n)] for n in env.storage_node_ids]
        assert privileged['t'] == t
        assert set(privileged) == {'t', 'oracle_action', 'oracle_soc'}
        np.testing.assert_allclose(
            privileged['oracle_action'],
            [r['p_cha_kw'] / power[n] - r['p_dis_kw'] / power[n]
             for r, n in zip(rows, env.storage_node_ids)], atol=1e-15)
        start = [2.0, 4.0] if t == 0 else [
            trajectory.loc[(stamp - pd.Timedelta(hours=1), n), 'soc_kwh']
            for n in env.storage_node_ids]
        np.testing.assert_array_equal(privileged['oracle_soc'], start)
        if len(steps) < 48:
            np.testing.assert_allclose(
                privileged['oracle_soc'],
                [env.community.node(n).battery.soc_kwh for n in env.storage_node_ids], atol=1e-12)
        steps.append(t)
        if len(steps) > 48:
            break
        _, _, _, truncated, info = env.step(privileged['oracle_action'])
        if truncated:
            assert info['t'] == env.current_step == 47

    assert steps == [*range(48), 47]


def test_replay_reports_a_welfare_gap(mini_cer_v1_path: Path, synthetic_oracle: Path,
                                      tmp_path: Path) -> None:
    """
    Test that an objective 1 % above the replayed welfare fails with the default tolerance 1e-3
    and passes with rtol = 0.02.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    path = synthetic_oracle / 'objective.csv'
    objective = pd.read_csv(path)
    objective['objective'] += 0.01 * abs(objective['objective'].sum()) / len(objective)
    objective.to_csv(path, index=False)
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)

    strict = check_replay(env, tmp_path / 'strict')
    loose = check_replay(env, tmp_path / 'loose', rtol=0.02)

    assert not strict.passed and strict.penalty_steps == 0
    assert strict.objective - strict.welfare == pytest.approx(0.01 * abs(strict.welfare))
    assert strict.relative_gap == pytest.approx(
        0.01 * abs(strict.welfare) / abs(strict.objective))
    assert loose.passed


def test_replay_reports_infeasible_oracle_actions(mini_cer_v1_path: Path, synthetic_oracle: Path,
                                                  tmp_path: Path) -> None:
    """
    Test that an oracle charging pv_bat at 01:00, when it generates nothing (A^upper = 0), is
    clipped and penalised by the environment and fails the check. (The first row is left as it
    is: it fixes the MILP's initial state of charge.)

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    path = synthetic_oracle / 'trajectory.csv'
    trajectory = pd.read_csv(path)
    row = (trajectory['datetime'] == '2022-01-03 01:00:00') & (trajectory['node_id'] == 'pv_bat')
    trajectory.loc[row, ['p_cha_kw', 'p_dis_kw']] = [1.0, 0.0]
    trajectory.to_csv(path, index=False)
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)

    result = check_replay(env, tmp_path / 'out')

    assert not result.passed and result.clipped_rows == 0
    assert result.penalty_steps >= 1
    # k = 0.2 for pv_bat, averaged over the two batteries
    assert result.penalty_max == pytest.approx(0.1)


def test_replay_reports_simultaneous_charge_and_discharge(mini_cer_v1_path: Path,
                                                          synthetic_oracle: Path,
                                                          tmp_path: Path) -> None:
    """
    Test that an oracle row with both powers positive is reported with a warning when the oracle
    is loaded and counted in the `ReplayCheck` and in `oracle_replay.json`.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    path = synthetic_oracle / 'trajectory.csv'
    trajectory = pd.read_csv(path)
    row = (trajectory['datetime'] == '2022-01-03 05:00:00') & (trajectory['node_id'] == 'pv_bat')
    trajectory.loc[row, ['p_cha_kw', 'p_dis_kw']] = [0.5, 1.0]
    trajectory.to_csv(path, index=False)

    with pytest.warns(OracleWarning, match='1 rows .* first at 2022-01-03 05:00:00'):
        env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)
    result = check_replay(env, tmp_path / 'out')

    assert result.simultaneous_rows == 1
    summary = json.loads((tmp_path / 'out' / 'oracle_replay.json').read_text(encoding='utf-8'))
    assert summary['simultaneous_rows'] == 1


@pytest.mark.parametrize(('start', 'end'), [('2022-01-03 06:00', None),
                                            (None, '2022-01-04 12:00')])
def test_replay_needs_whole_days(mini_cer_v1_path: Path, synthetic_oracle: Path, tmp_path: Path,
                                 start: str | None, end: str | None) -> None:
    """
    Test that the replay rejects a period that starts after midnight or ends before the last step
    of a day.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
        start: First timestamp of the period.
        end: Last timestamp of the period.
    """
    env = SmartCerEnv(mini_cer_v1_path, start=start, end=end, oracle=synthetic_oracle)

    with pytest.raises(ValueError, match='whole days'):
        check_replay(env, tmp_path / 'out')


def test_replay_needs_an_oracle_and_the_whole_period(mini_cer_v1_path: Path,
                                                     synthetic_oracle: Path,
                                                     tmp_path: Path) -> None:
    """
    Test the checks of `ReplayController`, `check_replay` and of the `oracle` argument, and that
    `compute_baselines` runs the 'replay' controller.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    plain = SmartCerEnv(mini_cer_v1_path)
    with pytest.raises(ValueError, match='created with an oracle'):
        ReplayController(plain)
    with pytest.raises(ValueError, match='created with an oracle'):
        check_replay(plain, tmp_path / 'out')
    short = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle, episode_length=24)
    with pytest.raises(ValueError, match='covers the whole period of 48 steps'):
        check_replay(short, tmp_path / 'out')
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)
    with pytest.raises(ValueError, match='cer_data'):
        check_replay(env, tmp_path / 'cer_data' / 'out')
    with pytest.raises(ValueError, match='rtol'):
        check_replay(env, tmp_path / 'out', rtol=-1.0)
    other = load_scenario(mini_cer_v1_path, start='2022-01-04 00:00')
    with pytest.raises(ValueError, match='built for another community'):
        SmartCerEnv(mini_cer_v1_path, oracle=OracleProvider.from_folder(synthetic_oracle, other))

    kpis = compute_baselines(env, tmp_path / 'baselines', controllers=['replay'])

    assert kpis['replay']['penalty_steps'] == 0
    assert (tmp_path / 'baselines' / 'replay' / 'community.csv').is_file()


def test_replay_of_local_oracle_files(tmp_path: Path) -> None:
    """
    Test the replay on the MILP outputs of every local scenario that has them
    (`smartcer/cer_data/<scenario>/oracle/trajectory.csv`, optional `objective.csv`), over the
    period the oracle covers: K_t = 0 at every step and, with an objective, a cumulative welfare
    within 1e-3. Skipped when no scenario has oracle files.

    Args:
        tmp_path: Pytest temporary directory (receives the per-day comparison of each scenario).
    """
    folders = sorted(p.parent for p in CER_DATA_PATH.glob('*/oracle/trajectory.csv'))
    if not folders:
        pytest.skip('no oracle files in smartcer/cer_data/<scenario>/oracle/')
    for folder in folders:
        index = read_oracle(folder).p_cha_kw.index
        with warnings.catch_warnings():
            # 0.6.2 scenarios report their irregular timestamps
            warnings.simplefilter('ignore', TimestampWarning)
            env = SmartCerEnv(folder.parent, start=index[0], end=index[-1], oracle=folder)

        result = check_replay(env, tmp_path / folder.parent.name, rtol=RTOL)

        assert result.penalty_steps == 0, f'{folder}: {result}'
        assert result.relative_gap is None or result.relative_gap <= RTOL, f'{folder}: {result}'


def idle_night_row(trajectory: pd.DataFrame, node_id: str) -> pd.Series:
    """
    Mask of the first oracle row of a node, after the first step, in a night hour (0-7, no
    generation) with both powers zero.

    Args:
        trajectory: Table of `trajectory.csv`.
        node_id: Battery node.

    Returns:
        pd.Series: Boolean mask selecting one row.
    """
    stamps = pd.to_datetime(trajectory['datetime'])
    idle = ((trajectory['node_id'] == node_id) & (stamps.dt.hour.between(1, 7))
            & (trajectory['p_cha_kw'] == 0.0) & (trajectory['p_dis_kw'] == 0.0))
    first = stamps[idle].min()
    return idle & (stamps == first)


@pytest.mark.parametrize(('p_cha_kw', 'clipped'), [(5e-7, True), (2e-6, False)])
def test_replay_clips_solver_round_off_into_the_bounds(mini_cer_v1_path: Path,
                                                       synthetic_oracle: Path, tmp_path: Path,
                                                       p_cha_kw: float, clipped: bool) -> None:
    """
    Test the replay tolerance: an oracle that charges pv_bat at night (A^upper = 0) with
    5e-7 kW for one hour exceeds the bound by 5e-7 kWh ≤ 1e-6 kWh, so the replay moves it onto the
    bound without penalty, counts it and passes; 2e-6 kWh is a real violation, penalised and not
    counted.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
        p_cha_kw: Charge power written in an idle night row of pv_bat.
        clipped: Whether the replay should clip it.
    """
    path = synthetic_oracle / 'trajectory.csv'
    trajectory = pd.read_csv(path)
    trajectory.loc[idle_night_row(trajectory, 'pv_bat'), 'p_cha_kw'] = p_cha_kw
    trajectory.to_csv(path, index=False)
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)

    result = check_replay(env, tmp_path / 'out')

    summary = json.loads((tmp_path / 'out' / 'oracle_replay.json').read_text(encoding='utf-8'))
    if clipped:
        assert result.passed and result.penalty_steps == 0
        assert result.clipped_rows == summary['clipped_rows'] == 1
        assert result.clipped_max_kwh == pytest.approx(5e-7)
    else:
        assert not result.passed and result.penalty_steps == 1
        assert result.clipped_rows == summary['clipped_rows'] == 0
        assert result.clipped_max_kwh == 0.0


def test_tracker_reports_the_gap_from_the_oracle_state_of_charge(mini_cer_v1_path: Path,
                                                                 synthetic_oracle: Path,
                                                                 tmp_path: Path) -> None:
    """
    Test the agent-oracle gap of the state of charge: the battery tables hold the oracle state at
    the end of every step and the gap from it; replaying a* gives a gap of 0, idle batteries
    (naive) stay at S^ini, so their gap is S^ini minus the oracle state, and the KPIs report its
    absolute mean and maximum per battery. Without an oracle the columns and KPIs are absent.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)
    kpis = compute_baselines(env, tmp_path / 'out', controllers=('replay', 'naive'))

    for node_id, start in (('pv_bat', 2.0), ('factory_bat', 4.0)):
        table = pd.read_csv(tmp_path / 'out' / 'naive' / 'nodes' / f'{node_id}.csv')
        oracle = env.oracle.soc_kwh[:, env.storage_node_ids.index(node_id)]
        np.testing.assert_allclose(table['oracle_soc_kwh'], oracle)
        np.testing.assert_allclose(table['soc_gap_kwh'], start - oracle)
        naive = kpis['naive']['batteries'][node_id]
        assert naive['soc_gap_mean_kwh'] == pytest.approx(np.abs(start - oracle).mean())
        assert naive['soc_gap_max_kwh'] == pytest.approx(np.abs(start - oracle).max())
        assert naive['soc_gap_max_kwh'] > 1.0
        assert kpis['replay']['batteries'][node_id]['soc_gap_max_kwh'] <= 1e-9
    plain = compute_baselines(SmartCerEnv(mini_cer_v1_path), tmp_path / 'plain',
                              controllers=('naive',))
    assert 'soc_gap_mean_kwh' not in plain['naive']['batteries']['pv_bat']
    assert 'soc_gap_kwh' not in pd.read_csv(tmp_path / 'plain' / 'naive' / 'nodes' / 'pv_bat.csv')


def test_replay_starts_at_the_first_step_of_the_oracle_file(mini_cer_v1_path: Path,
                                                             synthetic_oracle: Path,
                                                             tmp_path: Path) -> None:
    """
    Test that the replay rejects a period of whole days that starts after the first step of the
    oracle file (the second day of the synthetic oracle): the replay starts from soc_initial, the
    oracle's state only at its first step.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_path, start='2022-01-04 00:00', oracle=synthetic_oracle)

    with pytest.raises(ValueError, match='the period starts at 2022-01-04 00:00:00, the oracle '
                                         'file at 2022-01-03 00:00:00; set start to 2022-01-03'):
        check_replay(env, tmp_path / 'out')
