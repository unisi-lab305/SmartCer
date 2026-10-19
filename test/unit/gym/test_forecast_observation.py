"""
Tests of the forecast hook of `SmartCerEnv`: the `forecast` section of the observation, the episode
window, `info['privileged']['true_future']` and the provider reset (phase 7).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smartcer.core.community import Community
from smartcer.data import load_scenario
from smartcer.forecasts import PerfectForecast
from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.unit

FORECAST_KEYS = ('pv_bat/generation', 'pv_bat/load', 'factory_bat/generation',
                 'factory_bat/load')


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
    Test that the forecast section closes the vector with H = 3 values per battery node and
    quantity, in `storage_node_ids` order, generation before load.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3)
    names = [name for name in env.obs_layout if name.startswith('forecast/')]

    assert names == [f'forecast/{key}' for key in FORECAST_KEYS]
    assert list(env.obs_layout)[-4:] == names
    assert [env.obs_layout[n].stop - env.obs_layout[n].start for n in names] == [3] * 4
    assert env.obs_layout[names[-1]].stop == env.observation_space.shape[0]
    assert env.obs_layout[names[0]].start == env.obs_layout['factory_bat/a_upper'].stop


@pytest.mark.parametrize('observe_current_step', [True, False])
def test_perfect_forecast_puts_the_true_future_in_the_observation(
        mini_cer_v1_path: Path, observe_current_step: bool) -> None:
    """
    Test the acceptance check of phase 7: with `PerfectForecast` and horizon H = 3, every
    observation of an episode holds, at `forecast/<node>/<quantity>`, the scaled true values of
    steps t + 1..t + 3 (`observe_current_step=True`) or t..t + 2 (False), and
    `info['privileged']['true_future']` holds the same values in kWh. By hand, for t = 6 and
    True: pv_bat generates 0, 10, 10 kWh in hours 7-9 (maximum 10) → 0, 1, 1.

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
        t = env.current_step
        assert info['privileged']['t'] == t
        first = t + offset
        for key in FORECAST_KEYS:
            node_id, quantity = key.split('/')
            truth = getattr(community.node(node_id), f'{quantity}_kwh')[first:first + 3]
            np.testing.assert_allclose(obs[env.obs_layout[f'forecast/{key}']],
                                       env.normalizer.transform(key, truth), atol=1e-7)
            np.testing.assert_array_equal(info['privileged']['true_future'][key], truth)
        if t == 6 and observe_current_step:
            np.testing.assert_array_equal(obs[env.obs_layout['forecast/pv_bat/generation']],
                                          [0.0, 1.0, 1.0])
        assert env.observation_space.contains(obs)
        seen.append(t)
        if action is not None:
            obs, _, _, _, info = env.step(action)

    assert env.episode_length == 45
    # the terminal observation comes from the last step again
    assert seen == [*range(45), 44]


def test_forecast_horizon_shortens_the_episodes(mini_cer_v1_path: Path) -> None:
    """
    Test that with H = 3 the episodes use steps 0..44 only: the default episode has 45 steps, 46
    steps raise, and random starts of 40-step episodes lie in 0..5 and reach 5.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    with pytest.raises(ValueError, match=r'\[1, 45\] .* the last 3 kept for the forecasts'):
        SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3, episode_length=46)
    env = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3,
                      episode_length=40, random_start=True)

    starts = {env.reset(seed=seed)[1]['episode_start'] for seed in range(200)}

    assert starts == set(range(6))


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
    `obs_clipped`: 4 series × H = 2 values. The load of pv_bat is constant (0), so it maps to 0
    like the current value (`min_max`), but its values are still counted.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    env = SmartCerEnv(community, forecast=NoisyForecast(community, 'huge'), forecast_horizon=2)

    obs, info = env.reset(seed=0)
    _, _, _, _, step_info = env.step(np.zeros(2))

    assert info['obs_clipped'] == step_info['obs_clipped'] == 8
    np.testing.assert_array_equal(obs[env.obs_layout['forecast/pv_bat/generation']], [1.0, 1.0])
    np.testing.assert_array_equal(obs[env.obs_layout['forecast/pv_bat/load']], [0.0, 0.0])


@pytest.mark.parametrize(('kwargs', 'error', 'match'), [
    ({'forecast_horizon': 2}, ValueError, r'horizon \(2\) needs a forecast provider'),
    ({'forecast': 'perfect'}, ValueError, r'horizon must lie in \[1, 47\]'),
    ({'forecast': 'perfect', 'forecast_horizon': 48}, ValueError, r'got 48'),
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
    Test that the run configuration names the provider (registry name or class) and the horizon.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    by_name = SmartCerEnv(mini_cer_v1_path, forecast='perfect', forecast_horizon=3).run_config()
    by_object = SmartCerEnv(community, forecast=NoisyForecast(community),
                            forecast_horizon=2).run_config()
    plain = SmartCerEnv(mini_cer_v1_path).run_config()

    assert (by_name['forecast'], by_name['forecast_horizon']) == ('perfect', 3)
    assert (by_object['forecast'], by_object['forecast_horizon']) == ('NoisyForecast', 2)
    assert (plain['forecast'], plain['forecast_horizon'], plain['oracle']) == (None, 0, None)
