"""
Normaliser fitted on a training window and observations over a whole year (phase 4 acceptance).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartcer.data import TimestampWarning
from smartcer.gym import SmartCerEnv
from smartcer.gym.observation import MinMaxNormalizer


pytestmark = pytest.mark.integration

STORAGE = {'capacity': 20, 'power_charge_max': 5, 'power_discharge_max': 5,
           'efficiency_charge': 0.9, 'efficiency_discharge': 0.9, 'soc_min': 0.1}
NODES = {'pros_bat': {'type': 'prosumer', 'csv': 'pros_bat.csv', 'storage': STORAGE},
         'pros': {'type': 'prosumer', 'csv': 'pros.csv'}}


def scenario_csv(n_steps: int, seed: int, test_scale: float = 1.0,
                 train_steps: int | None = None) -> dict[str, str]:
    """
    CSV files of a two-prosumer scenario with PV-like generation and random load from 2022-01-01.

    Args:
        n_steps: Hourly rows.
        seed: Seed of the random load.
        test_scale: Factor applied to generation and load after the first `train_steps` rows.
        train_steps: Rows left unchanged by `test_scale`; None for all.

    Returns:
        dict[str, str]: CSV path → text.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range('2022-01-01', periods=n_steps, freq='h')
    season = 1.0 + 0.5 * np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 80) / 365)
    sun = np.clip(np.sin(2 * np.pi * (index.hour.to_numpy() - 6) / 24), 0.0, None)
    files = {}
    for name, peak in (('pros_bat', 8.0), ('pros', 4.0)):
        generation = np.round(peak * season * sun, 6)
        load = np.round(rng.uniform(1.0, 5.0, n_steps), 6)
        if train_steps is not None:
            generation[train_steps:] *= test_scale
            load[train_steps:] *= test_scale
        rows = (f'{t:%Y-%m-%d %H:%M:%S},{g},{c},0.1,0.3\n'
                for t, g, c in zip(index, generation, load))
        files[f'{name}.csv'] = ('datetime,production,consumption,sell_price,buy_price\n'
                                + ''.join(rows))
    return files


def test_normalizer_fitted_on_training_window_ignores_the_test_window(
        write_scenario: Callable[..., Path]) -> None:
    """
    Test that two scenarios equal in the training window (first day) and different afterwards
    (generation and load × 100 on day 2) give the same normaliser and the same training-window
    observations, while the test-window values are clipped into the space and counted in
    `info['obs_clipped']`.

    Args:
        write_scenario: Scenario factory.
    """
    window = ('2022-01-01 00:00', '2022-01-01 23:00')
    envs = []
    for scale in (1.0, 100.0):
        folder = write_scenario(NODES, scenario_csv(48, seed=3, test_scale=scale, train_steps=24))
        envs.append(SmartCerEnv(folder, observation='extended', lookback=2,
                                normalizer_window=window))
    plain, spiked = envs

    assert plain.normalizer.bounds == spiked.normalizer.bounds
    assert plain.normalizer.window == ('2022-01-01T00:00:00', '2022-01-01T23:00:00')
    expected = plain.community.node('pros_bat').generation_kwh[:24]
    assert plain.normalizer.bounds['pros_bat/generation'] == (expected.min(), expected.max())
    observations, clipped = [], []
    for env in envs:
        obs, info = env.reset(seed=0)
        trajectory, counts = [obs], [info['obs_clipped']]
        for _ in range(47):
            obs, *_, info = env.step(np.zeros(1, dtype=np.float32))
            trajectory.append(obs)
            counts.append(info['obs_clipped'])
        observations.append(np.array(trajectory))
        clipped.append(counts)
    assert np.array_equal(observations[0][:24], observations[1][:24])
    assert all(spiked.observation_space.contains(obs) for obs in observations[1])
    assert observations[1][30, spiked.obs_layout['pros_bat/load']] == pytest.approx([1.0])
    # 8 scaled values per step (community g and l, g and l of pros_bat, 2 + 2 lookback); on day
    # 2 the loads are clipped, and at noon also the generations (night generation stays 0)
    assert clipped[0][:24] == clipped[1][:24] == [0] * 24
    assert clipped[1][30] == 4 and clipped[1][36] == 8


def test_saved_normalizer_is_reused_by_an_evaluation_environment(
        mini_cer_v1_path: Path, tmp_path: Path) -> None:
    """
    Test the JSON round trip between a training environment (day 1) and an evaluation
    environment on another period (day 2), which keeps the training bounds.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    train = SmartCerEnv(mini_cer_v1_path, start='2022-01-03 00:00', end='2022-01-03 23:00')
    path = train.normalizer.save(tmp_path / 'normalizer.json')
    evaluation = SmartCerEnv(mini_cer_v1_path, start='2022-01-04 00:00', end='2022-01-04 23:00',
                             observation='extended', normalizer=path)
    obs, _ = evaluation.reset(seed=0)

    assert evaluation.normalizer.bounds == train.normalizer.bounds
    assert evaluation.normalizer.window == ('2022-01-03T00:00:00', '2022-01-03T23:00:00')
    assert evaluation.observation_space.contains(obs)
    assert SmartCerEnv(mini_cer_v1_path, normalizer=MinMaxNormalizer.load(path)).normalizer.bounds \
        == train.normalizer.bounds


def run_year(env: SmartCerEnv, seed: int) -> int:
    """
    Run one episode with random actions and count the observations outside the space.

    Args:
        env: Environment.
        seed: Seed of the reset and of the action space.

    Returns:
        int: Observations outside `env.observation_space` (reset included).
    """
    obs, _ = env.reset(seed=seed)
    outside = int(not env.observation_space.contains(obs))
    truncated = False
    while not truncated:
        obs, _, _, truncated, _ = env.step(env.action_space.sample())
        outside += int(not env.observation_space.contains(obs))
    return outside


@pytest.mark.parametrize(('observation', 'lookback', 'observe_current_step'), [
    ('meco', 0, True), ('extended', 4, False),
])
def test_every_observation_of_a_year_lies_in_the_space(
        write_scenario: Callable[..., Path], observation: str, lookback: int,
        observe_current_step: bool) -> None:
    """
    Test a synthetic year (8,760 hourly steps) with random actions and a normaliser fitted on
    January-March only, so that the summer generation exceeds the training range.

    Args:
        write_scenario: Scenario factory.
        observation: Observation preset.
        lookback: Lookback steps.
        observe_current_step: Information timing.
    """
    folder = write_scenario(NODES, scenario_csv(8760, seed=11))
    env = SmartCerEnv(folder, observation=observation, lookback=lookback,
                      observe_current_step=observe_current_step,
                      normalizer_window=('2022-01-01', '2022-03-31 23:00'))
    generation = env.community.node('pros_bat').generation_kwh

    assert generation.max() > env.normalizer.bounds['pros_bat/generation'][1]
    assert env.episode_length == 8760
    assert run_year(env, seed=5) == 0


def test_every_observation_of_leather_lies_in_the_space(leather_path: Path) -> None:
    """
    Test one year of the local Leather scenario (skipped when missing) with the extended preset,
    lookback 3, causal timing and a normaliser fitted on the first half of 2022.

    Args:
        leather_path: Path of the local Leather scenario.
    """
    with pytest.warns(TimestampWarning):
        env = SmartCerEnv(leather_path, observation='extended', lookback=3,
                          observe_current_step=False,
                          normalizer_window=(None, '2022-06-30 23:00'))

    assert env.episode_length == 8760
    assert run_year(env, seed=7) == 0
