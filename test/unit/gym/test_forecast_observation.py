"""
Tests of the forecast hook of `SmartCerEnv`: the `forecast` section of the observation (sale
prices, community totals, battery nodes), the episode window, `info['privileged']['true_future']`
and the provider reset (phases 7 and 7b).
"""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smartcer.core.community import Community
from smartcer.data import load_scenario
from smartcer.forecasts import PerfectForecast
from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.unit

FORECAST_KEYS = ('community/generation', 'community/load', 'pv_bat/generation', 'pv_bat/load',
                 'factory_bat/generation', 'factory_bat/load')
# the sell price is the same for every exporting node, the buy prices differ: per-node prices
PRICE_FEATURES = ('forecast/pv_bat/sell_price', 'forecast/factory_bat/sell_price')


def true_series(community: Community, key: str) -> np.ndarray:
    """
    True values of a forecast key: a node's series, or the total of the nodes without a battery
    (house_1 and shop).

    Args:
        community: The synthetic community.
        key: `'<node_id>/<quantity>'` or `'community/<quantity>'`.

    Returns:
        np.ndarray: The series (kWh per step).
    """
    node_id, quantity = key.split('/')
    nodes = ('house_1', 'shop') if node_id == 'community' else (node_id,)
    return np.sum([getattr(community.node(n), f'{quantity}_kwh') for n in nodes], axis=0)


class NoisyForecast:
    """
    Test provider: the true values plus uniform noise in [0, 1) kWh drawn from the episode
    generator, or a fixed defect.
    """

    def __init__(self, community: Community, defect: str | None = None) -> None:
        """
        Args:
            community: The community.
            defect: None, 'missing' (no pv_bat/load), 'short' (H - 1 values), 'nan' or 'huge'
                (100 kWh everywhere).
        """
        self.truth = PerfectForecast(community)
        self.defect = defect
        self.rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """
        Args:
            rng: The environment's generator.
        """
        self.rng = rng

    def get(self, t: int, horizon: int) -> dict[str, np.ndarray]:
        """
        Args:
            t: First forecast step.
            horizon: Forecast horizon.

        Returns:
            dict[str, np.ndarray]: Noisy or defective forecasts.
        """
        values = {k: v + self.rng.uniform(size=horizon)
                  for k, v in self.truth.get(t, horizon).items()}
        if self.defect == 'missing':
            del values['pv_bat/load']
        elif self.defect == 'short':
            values['pv_bat/load'] = values['pv_bat/load'][:-1]
        elif self.defect == 'nan':
            values['pv_bat/load'][0] = np.nan
        elif self.defect == 'huge':
            values = {k: np.full(horizon, 100.0) for k in values}
        return values


def test_forecast_section_layout(mini_cer_v1_path: Path) -> None:
    """
    Test that the forecast section closes the vector with H = 3 values per feature: the sell price
    of each battery node (prices are per node in this scenario), the community totals, then
    each battery node in `storage_node_ids` order, generation before load; no purchase price.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3)
    names = [name for name in env.obs_layout if name.startswith('forecast/')]

    assert names == [*PRICE_FEATURES, *(f'forecast/{key}' for key in FORECAST_KEYS)]
    assert env._builder.sections['forecast'] == names
    assert list(env.obs_layout)[-8:] == names
    assert [env.obs_layout[n].stop - env.obs_layout[n].start for n in names] == [3] * 8
    assert env.obs_layout[names[-1]].stop == env.observation_space.shape[0]
    assert env.obs_layout[names[0]].start == env.obs_layout['factory_bat/a_upper'].stop


@pytest.mark.parametrize('observe_current_step', [True, False])
def test_perfect_forecast_puts_the_true_future_in_the_observation(
        mini_cer_v1_path: Path, observe_current_step: bool) -> None:
    """
    Test the acceptance check of phase 7: with `PerfectForecast` and horizon H = 3, every
    observation of an episode holds, at `forecast/<node>/<quantity>` and
    `forecast/community/<quantity>`, the scaled true values of steps t + 1..t + 3
    (`observe_current_step=True`) or t..t + 2 (False), `info['privileged']['true_future']` holds
    the same values in kWh, and `forecast/<node>/sell_price` the raw sell prices of those steps.
    By hand, for t = 6 and True: pv_bat generates 0, 10, 10 kWh in hours 7-9 (maximum 10) → 0, 1,
    1, the community 0, 5, 5 kWh (maximum 5) → 0, 1, 1, and the sell price is 0.2, 0.1, 0.1
    €/kWh. The whole-period
    episode has 44 steps with True and 45 with False, and its terminal observation is that of the
    next step, whose forecasts end on step 47, the last of the data.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        observe_current_step: Information timing.
    """
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3,
                      observe_current_step=observe_current_step)
    community = env.community
    offset = 1 if observe_current_step else 0
    obs, info = env.reset(seed=0)
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(env.episode_length, 2))
    seen = []
    for action in [*actions, None]:
        t = info['privileged']['t']
        first = t + offset
        assert list(info['privileged']['true_future']) == list(FORECAST_KEYS)
        for key in FORECAST_KEYS:
            truth = true_series(community, key)[first:first + 3]
            np.testing.assert_allclose(obs[env.obs_layout[f'forecast/{key}']],
                                       env.normalizer.transform(key, truth), atol=1e-7)
            np.testing.assert_array_equal(info['privileged']['true_future'][key], truth)
        for name in PRICE_FEATURES:
            prices = community.node(name.split('/')[1]).sell_price[first:first + 3]
            np.testing.assert_allclose(obs[env.obs_layout[name]], prices, atol=1e-7)
        if t == 6 and observe_current_step:
            for name in ('forecast/pv_bat/generation', 'forecast/community/generation'):
                np.testing.assert_array_equal(obs[env.obs_layout[name]], [0.0, 1.0, 1.0])
            np.testing.assert_allclose(obs[env.obs_layout['forecast/pv_bat/sell_price']],
                                       [0.2, 0.1, 0.1], atol=1e-7)
        assert env.observation_space.contains(obs)
        seen.append(t)
        if action is not None:
            obs, _, _, _, info = env.step(action)

    length = 44 if observe_current_step else 45
    assert env.episode_length == length
    assert seen == [*range(length), length]


@pytest.mark.parametrize(('observe_current_step', 'usable'), [(True, 44), (False, 45)])
def test_forecast_horizon_shortens_the_episodes(mini_cer_v1_path: Path,
                                                observe_current_step: bool, usable: int) -> None:
    """
    Test that with H = 3 the episodes use steps 0..43 with `observe_current_step=True` (the
    forecasts of step 44, the step after the last episode step, end on step 47) and 0..44 with
    False (those of step 45 end on step 47): the default episode has 44 or 45 steps, one more
    raises, and random starts of 40-step episodes lie in 0..4 or 0..5 and reach the last one.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        observe_current_step: Information timing.
        usable: Steps the episodes can use.
    """
    kept = 48 - usable
    with pytest.raises(ValueError, match=rf'\[1, {usable}\] .* the last {kept} kept for the '
                                         'forecasts'):
        SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3,
                    observe_current_step=observe_current_step, episode_length=usable + 1)
    assert SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3,
                       observe_current_step=observe_current_step).episode_length == usable
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3,
                      observe_current_step=observe_current_step, episode_length=40,
                      random_start=True)

    starts = {env.reset(seed=seed)[1]['episode_start'] for seed in range(200)}

    assert starts == set(range(usable - 40 + 1))


@pytest.mark.parametrize('observe_current_step', [True, False])
def test_terminal_observation_with_forecasts_is_the_next_step(
        mini_cer_v1_path: Path, observe_current_step: bool) -> None:
    """
    Test that with forecasts an episode ending on the last usable step (43 with
    `observe_current_step=True`, 44 with False) returns the observation of the next step, whose
    forecasts are the true values of steps 45..47, the last of the data. (Without forecasts the
    fallback to the last step at the end of the data is tested by
    `test_s10_terminal_observation_comes_from_last_step`.)

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        observe_current_step: Information timing.
    """
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3, episode_length=4,
                      random_start=True, observe_current_step=observe_current_step)
    last = env.community.n_steps - 3 - (2 if observe_current_step else 1)
    seed = next(s for s in range(10_000) if env.reset(seed=s)[1]['episode_start'] == last - 3)
    env.reset(seed=seed)
    for _ in range(4):
        obs, _, _, truncated, info = env.step(np.zeros(2))

    assert truncated and info['t'] == last
    assert info['privileged']['t'] == last + 1
    load = env.community.node('factory_bat').load_kwh[45:48]
    np.testing.assert_array_equal(info['privileged']['true_future']['factory_bat/load'], load)
    np.testing.assert_allclose(obs[env.obs_layout['forecast/factory_bat/load']],
                               env.normalizer.transform('factory_bat/load', load), atol=1e-7)


def test_forecast_provider_is_reset_with_the_episode_generator(mini_cer_v1_path: Path) -> None:
    """
    Test that `reset` passes `env.np_random` to the provider, so that a noisy provider gives the
    same observations for the same seed and different ones for another seed.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    def first_observations(seed: int) -> np.ndarray:
        community = load_scenario(mini_cer_v1_path)
        provider = NoisyForecast(community)
        env = SmartCerEnv(community, forecast=provider, forecast_horizon=4)
        obs, _ = env.reset(seed=seed)
        assert provider.rng is env.np_random
        return np.vstack([obs, env.step(np.zeros(2))[0]])

    np.testing.assert_array_equal(first_observations(3), first_observations(3))
    assert not np.array_equal(first_observations(3), first_observations(4))


def test_clipped_forecasts_are_counted(mini_cer_v1_path: Path) -> None:
    """
    Test that forecasts above the normaliser's range are clipped to 1 and counted in
    `obs_clipped`: 6 series × H = 2 values. The load of pv_bat is constant (0), so it maps to 0
    like the current value (`min_max`), but its values are still counted.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    env = SmartCerEnv(community, forecast=NoisyForecast(community, 'huge'), forecast_horizon=2)

    obs, info = env.reset(seed=0)
    _, _, _, _, step_info = env.step(np.zeros(2))

    assert info['obs_clipped'] == step_info['obs_clipped'] == 12
    np.testing.assert_array_equal(obs[env.obs_layout['forecast/pv_bat/generation']], [1.0, 1.0])
    np.testing.assert_array_equal(obs[env.obs_layout['forecast/pv_bat/load']], [0.0, 0.0])


@pytest.mark.parametrize(('kwargs', 'error', 'match'), [
    ({'forecast_horizon': 2}, ValueError, r'horizon \(2\) needs a forecast provider'),
    ({'forecast': 'perfect'}, ValueError, r'horizon must lie in \[1, 47\]'),
    ({'forecast': 'perfect', 'forecast_horizon': 48}, ValueError, r'got 48'),
    ({'forecast': 'perfect', 'forecast_horizon': 47}, ValueError, 'leaves no step'),
    ({'forecast': 'perfect', 'forecast_horizon': 1.5}, ValueError, 'must be an integer'),
    ({'forecast': 'oracle', 'forecast_horizon': 2}, ValueError, 'unknown forecast provider'),
    ({'forecast': object(), 'forecast_horizon': 2}, TypeError, 'must have reset'),
])
def test_forecast_arguments_are_checked(mini_cer_v1_path: Path, kwargs: dict[str, Any],
                                        error: type[Exception], match: str) -> None:
    """
    Test the checks of `forecast` and `forecast_horizon`.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        kwargs: Environment arguments.
        error: Expected exception.
        match: Expected message.
    """
    with pytest.raises(error, match=match):
        SmartCerEnv(mini_cer_v1_path, **kwargs)


@pytest.mark.parametrize(('defect', 'match'), [
    ('missing', r"returned no \['pv_bat/load'\]"),
    ('short', 'must return 3 values per series'),
    ('nan', 'non-finite values for step 0'),
])
def test_invalid_forecasts_raise(mini_cer_v1_path: Path, defect: str, match: str) -> None:
    """
    Test that a provider that omits a series, returns too few values or non-finite values makes
    the observation raise.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        defect: Defect of the provider.
        match: Expected message.
    """
    community = load_scenario(mini_cer_v1_path)
    env = SmartCerEnv(community, forecast=NoisyForecast(community, defect), forecast_horizon=3)

    with pytest.raises(ValueError, match=match):
        env.reset(seed=0)


def test_run_config_records_the_forecast(mini_cer_v1_path: Path) -> None:
    """
    Test that the run configuration names the provider (registry name or class), its settings
    (`config()` of the MMFE provider, None otherwise) and the horizon, and is JSON-serialisable.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    by_name = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3).run_config()
    by_object = SmartCerEnv(community, forecast=NoisyForecast(community),
                            forecast_horizon=2).run_config()
    plain = SmartCerEnv(mini_cer_v1_path).run_config()

    mmfe = SmartCerEnv(mini_cer_v1_path, forecast='mmfe_standard', forecast_horizon=3).run_config()

    assert (by_name['forecast'], by_name['forecast_horizon']) == ('perfect', 3)
    assert (by_object['forecast'], by_object['forecast_horizon']) == ('NoisyForecast', 2)
    assert by_name['forecast_config'] is None and by_object['forecast_config'] is None
    assert (plain['forecast'], plain['forecast_horizon'], plain['oracle']) == (None, 0, None)
    assert mmfe['forecast'] == 'mmfe_standard'
    assert mmfe['forecast_config'] == {'preset': 'standard', 'sigma_generation': 0.07,
                                       'sigma_load': 0.03, 'rho': 0.5, 'phi': 0.8,
                                       'bias_generation': 0.0, 'bias_load': 0.0}
    json.dumps(mmfe)


def test_global_sale_price_is_forecast_once(write_scenario: Callable[..., Path]) -> None:
    """
    Test that with global prices the forecast section has one `forecast/price_sell`, the raw sell
    prices of the forecast steps, and no purchase price; by hand, the sell price is 0.1 €/kWh
    from 08:00 to 15:00 and 0.2 otherwise.

    Args:
        write_scenario: Factory of 1.0 scenarios (conftest).
    """
    stamps = pd.date_range('2022-01-03', periods=24, freq='h')
    sell = [0.1 if 8 <= s.hour <= 15 else 0.2 for s in stamps]
    pv = ['datetime,production,sell_price'] + [f'{s},{3 if 8 <= s.hour <= 15 else 0},{p}'
                                               for s, p in zip(stamps, sell)]
    home = ['datetime,consumption,buy_price'] + [f'{s},2,0.3' for s in stamps]
    storage = {'capacity': 10, 'power_charge_max': 5, 'power_discharge_max': 5,
               'efficiency_charge': 0.9, 'efficiency_discharge': 0.9}
    folder = write_scenario(
        {'pv': {'type': 'producer', 'csv': 'data/pv.csv', 'storage': storage},
         'home': {'type': 'consumer', 'csv': 'data/home.csv'}},
        {'data/pv.csv': '\n'.join(pv) + '\n', 'data/home.csv': '\n'.join(home) + '\n'})
    env = SmartCerEnv(folder, forecast='perfect', forecast_horizon=4)

    obs, _ = env.reset(seed=0)
    for _ in range(6):
        obs, *_ = env.step(np.zeros(1))

    assert [n for n in env.obs_layout if n.startswith('forecast/')] == [
        'forecast/price_sell', 'forecast/community/load', 'forecast/pv/generation',
        'forecast/pv/load']
    np.testing.assert_allclose(obs[env.obs_layout['forecast/price_sell']], [0.2, 0.1, 0.1, 0.1],
                               atol=1e-7)


def test_mmfe_forecasts_are_reproducible_from_the_seed(mini_cer_v1_path: Path) -> None:
    """
    Test the MMFE provider in the environment: the same seed gives the same observations, another
    seed or a second episode without a seed gives new forecast errors, the rest of the
    observation is unchanged, and `true_future` still holds the true values.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    def episode(env: SmartCerEnv, seed: int | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        obs, info = env.reset(seed=seed)
        observations, infos = [obs], [info]
        for _ in range(10):
            obs, _, _, _, info = env.step(np.zeros(2))
            observations.append(obs)
            infos.append(info)
        return np.vstack(observations), infos

    def make() -> SmartCerEnv:
        return SmartCerEnv(mini_cer_v1_path, forecast='mmfe_noisy', forecast_horizon=6)

    env = make()
    first, infos = episode(env, 3)
    again, _ = episode(env, None)
    forecasts = env._builder._forecast_slice
    other = np.ones(first.shape[1], dtype=bool)
    other[forecasts] = False

    np.testing.assert_array_equal(first, episode(make(), 3)[0])
    assert not np.array_equal(first[:, forecasts], again[:, forecasts])
    assert not np.array_equal(first[:, forecasts], episode(make(), 4)[0][:, forecasts])
    np.testing.assert_array_equal(first[:, other], again[:, other])
    truth = true_series(env.community, 'factory_bat/load')
    for info in infos:
        first_step = info['privileged']['t'] + 1
        np.testing.assert_array_equal(info['privileged']['true_future']['factory_bat/load'],
                                      truth[first_step:first_step + 6])
