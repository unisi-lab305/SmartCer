"""
Replay of the MILP oracle (phase 7): `info['privileged']`, `ReplayController` and `check_replay`
on a synthetic oracle trajectory (fixture `synthetic_oracle`), and on the DIISM files when they
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
from smartcer.oracle import OracleProvider, read_oracle


pytestmark = pytest.mark.integration

CER_DATA_PATH = Path(__file__).parents[3] / 'smartcer' / 'cer_data'
# relative tolerance of the cumulative welfare against the MILP objective (brief, phase 7)
RTOL = 1e-3


def test_replay_of_a_synthetic_oracle_trajectory(mini_cer_v1_path: Path, synthetic_oracle: Path,
                                                 tmp_path: Path) -> None:
    """
    Test the acceptance check of phase 7 on a synthetic trajectory: playing a* over the 48 steps
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
    assert result.relative_gap <= 1e-12
    assert result.objective == pytest.approx(result.welfare, rel=1e-12)
    assert result.fit_score == pytest.approx(100.0)
    daily = pd.read_csv(output / 'oracle_daily.csv')
    assert list(daily.columns) == ['date', 'welfare', 'objective', 'difference']
    assert list(daily['date']) == ['2022-01-03', '2022-01-04']
    np.testing.assert_allclose(daily['welfare'], daily['objective'], rtol=1e-12)
    summary = json.loads((output / 'oracle_replay.json').read_text(encoding='utf-8'))
    assert summary['passed'] is True and summary['rtol'] == RTOL
    assert summary['run_config']['oracle'] == str(synthetic_oracle)


def test_privileged_info_follows_the_returned_observation(mini_cer_v1_path: Path,
                                                          synthetic_oracle: Path) -> None:
    """
    Test that `info['privileged']` holds the oracle action and state of charge of the step of the
    observation it comes with (the next action; the last step again at the end), with a* computed
    from the file by hand (pv_bat 5 kW, factory_bat 10 kW), and no `true_future` without
    forecasts.

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
        np.testing.assert_array_equal(privileged['oracle_soc'], [r['soc_kwh'] for r in rows])
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
    Test that an oracle charging pv_bat at midnight, when it generates nothing (A^upper = 0),
    is clipped and penalised by the environment and fails the check.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    path = synthetic_oracle / 'trajectory.csv'
    trajectory = pd.read_csv(path)
    row = (trajectory['datetime'] == '2022-01-03 00:00:00') & (trajectory['node_id'] == 'pv_bat')
    trajectory.loc[row, ['p_cha_kw', 'p_dis_kw']] = [1.0, 0.0]
    trajectory.to_csv(path, index=False)
    env = SmartCerEnv(mini_cer_v1_path, oracle=synthetic_oracle)

    result = check_replay(env, tmp_path / 'out')

    assert not result.passed
    assert result.penalty_steps >= 1
    # k = 0.2 for pv_bat, averaged over the two batteries
    assert result.penalty_max == pytest.approx(0.1)


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


def test_replay_of_the_diism_oracle(tmp_path: Path) -> None:
    """
    Test the replay on the DIISM MILP outputs of every local scenario that has them
    (`smartcer/cer_data/<scenario>/oracle/trajectory.csv`, optional `objective.csv`), over the
    period the oracle covers: K_t = 0 at every step and, with an objective, a cumulative welfare
    within 1e-3. Skipped when no scenario has oracle files.

    Args:
        tmp_path: Pytest temporary directory (receives the per-day comparison of each scenario).
    """
    folders = sorted(p.parent for p in CER_DATA_PATH.glob('*/oracle/trajectory.csv'))
    if not folders:
        pytest.skip('no DIISM oracle files in smartcer/cer_data/<scenario>/oracle/')
    for folder in folders:
        index = read_oracle(folder).p_cha_kw.index
        with warnings.catch_warnings():
            # 0.6.2 scenarios report their irregular timestamps
            warnings.simplefilter('ignore', TimestampWarning)
            env = SmartCerEnv(folder.parent, start=index[0], end=index[-1], oracle=folder)

        result = check_replay(env, tmp_path / folder.parent.name, rtol=RTOL)

        assert result.penalty_steps == 0, f'{folder}: {result}'
        assert result.relative_gap is None or result.relative_gap <= RTOL, f'{folder}: {result}'
