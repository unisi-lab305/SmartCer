"""
Tests of `MMFEForecast`: the error structure of the martingale model of forecast evolution
(variance linear in the lead time, consistent issues, correlation between nodes, AR(1) over the
target steps), the photovoltaic and load transforms, the presets and the reproducibility.

The statistical tests draw the errors of many episodes (seeds) on a flat scenario
(`flat_community`) and compare empirical moments with their theoretical values, with tolerances of
at least 5 standard errors.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smartcer.core.community import Community
from smartcer.data import load_scenario
from smartcer.forecasts import (
    FORECAST_PROVIDERS,
    MMFE_PRESETS,
    ForecastProvider,
    MMFEForecast,
    MMFEParameters,
    PerfectForecast,
)


pytestmark = pytest.mark.unit

HORIZON = 12
SIGMA = 0.02
SEEDS = range(2000)


@pytest.fixture
def flat_community(write_scenario: Callable[..., Path]) -> Community:
    """
    Three prosumers a, b, c with a battery, 48 hourly steps from 2022-01-03: generation 5 kWh at
    every hour of day 1 and 10 kWh of day 2 (reference profile 7.5 kWh, peak 10 kWh), load
    10 kWh. On day 1 neither bound of the generation forecast is reached in practice.

    Args:
        write_scenario: Factory of 1.0 scenarios (conftest).

    Returns:
        Community: The community.
    """
    stamps = pd.date_range('2022-01-03', periods=48, freq='h')
    lines = ['datetime,production,consumption,buy_price,sell_price'] + [
        f'{s},{5 if s.day == 3 else 10},10,0.3,0.1' for s in stamps]
    storage = {'capacity': 10, 'power_charge_max': 5, 'power_discharge_max': 5,
               'efficiency_charge': 0.9, 'efficiency_discharge': 0.9}
    nodes = {n: {'type': 'prosumer', 'csv': f'data/{n}.csv', 'storage': storage} for n in 'abc'}
    folder = write_scenario(nodes, {f'data/{n}.csv': '\n'.join(lines) + '\n' for n in 'abc'})
    return load_scenario(folder)


def relative_errors(values: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """
    Errors of forecasts of day 1 of the flat scenario relative to their scale: (ŷ - 5) / 7.5 for
    generation, ŷ / 10 - 1 for load.

    Args:
        values: Forecasts by key.
        keys: Keys, in order.

    Returns:
        np.ndarray: Shape (len(keys), H).
    """
    return np.array([(values[k] - 5.0) / 7.5 if k.endswith('/generation')
                     else values[k] / 10.0 - 1.0 for k in keys])


def errors_by_lead(community: Community, parameters: MMFEParameters,
                   target: int = HORIZON) -> np.ndarray:
    """
    Relative errors of the forecasts of one target step issued at lead times H, H - 1, ..., 1 in
    one episode per seed (issues moving forward, as in an episode).

    Args:
        community: The flat community.
        parameters: Error parameters.
        target: Target step, on day 1.

    Returns:
        np.ndarray: errors[seed, h - 1, series].
    """
    provider = MMFEForecast(community, parameters)
    out = np.empty((len(SEEDS), HORIZON, len(provider.keys)))
    for i, seed in enumerate(SEEDS):
        provider.reset(np.random.default_rng(seed))
        for lead in range(HORIZON, 0, -1):
            values = provider.get(target - lead + 1, HORIZON)
            out[i, lead - 1] = relative_errors(values, provider.keys)[:, lead - 1]
    return out


def test_error_variance_grows_linearly_with_the_lead_time(flat_community: Community) -> None:
    """
    Test that the variance of the error at lead time h is h σ² (σ = 0.02, relative to the
    reference profile for generation and to the load for load), for h = 1..12, with zero mean.

    Args:
        flat_community: Flat scenario.
    """
    errors = errors_by_lead(flat_community, MMFEParameters(SIGMA, SIGMA, rho=0.5, phi=0.6))
    lead = np.arange(1, HORIZON + 1)

    variance = errors.transpose(1, 0, 2).reshape(HORIZON, -1).var(axis=1)

    np.testing.assert_allclose(variance / (lead * SIGMA ** 2), 1.0, atol=0.1)
    assert np.all(np.abs(errors.mean(axis=(0, 2))) <= 5 * SIGMA * np.sqrt(lead / len(SEEDS)))
    slope, intercept = np.polyfit(lead, variance, 1)
    assert slope == pytest.approx(SIGMA ** 2, rel=0.1)
    assert abs(intercept) < 0.5 * SIGMA ** 2


def test_successive_issues_converge_to_the_true_value(flat_community: Community) -> None:
    """
    Test that the issues for the same target are consistent: from lead time h to h - 1 the error
    changes by one innovation (variance σ², not the (2h - 1) σ² of independent draws), so the
    error shrinks towards 0 as the target approaches, with the smallest error at lead time 1 and
    the true value observed at lead time 0. A request repeated with the same step gives the same
    forecasts.

    Args:
        flat_community: Flat scenario.
    """
    errors = errors_by_lead(flat_community, MMFEParameters(SIGMA, SIGMA, rho=0.5, phi=0.6))
    # error at lead 0 is 0: the observation holds the true value
    padded = np.concatenate([np.zeros((len(SEEDS), 1, errors.shape[2])), errors], axis=1)

    increments = np.diff(padded, axis=1).transpose(1, 0, 2).reshape(HORIZON, -1).var(axis=1)
    mean_abs = np.abs(errors).mean(axis=(0, 2))

    np.testing.assert_allclose(increments / SIGMA ** 2, 1.0, atol=0.1)
    assert np.all(np.diff(mean_abs) > 0.0)
    assert mean_abs[0] == pytest.approx(SIGMA * np.sqrt(2.0 / np.pi), rel=0.1)
    provider = MMFEForecast(flat_community, MMFEParameters(SIGMA, SIGMA))
    provider.reset(np.random.default_rng(0))
    first = provider.get(5, HORIZON)
    later = provider.get(9, HORIZON)
    for key in provider.keys:
        np.testing.assert_array_equal(provider.get(5, HORIZON)[key], first[key])
        np.testing.assert_array_equal(provider.get(9, HORIZON)[key], later[key])


@pytest.mark.parametrize('rho', [0.0, 0.5, 0.9])
def test_errors_of_the_nodes_correlate_with_rho(flat_community: Community, rho: float) -> None:
    """
    Test that the errors of two nodes for the same target and lead time have correlation ρ, for
    generation and for load, and that generation and load errors are uncorrelated.

    Args:
        flat_community: Flat scenario.
        rho: Share of the common factor.
    """
    provider = MMFEForecast(flat_community, MMFEParameters(SIGMA, SIGMA, rho=rho, phi=0.6))
    errors = np.empty((len(SEEDS), len(provider.keys)))
    for i, seed in enumerate(SEEDS):
        provider.reset(np.random.default_rng(seed))
        errors[i] = relative_errors(provider.get(1, HORIZON), provider.keys)[:, -1]
    column = {key: errors[:, i] for i, key in enumerate(provider.keys)}

    for quantity in ('generation', 'load'):
        for first, second in (('a', 'b'), ('a', 'c'), ('b', 'c')):
            corr = np.corrcoef(column[f'{first}/{quantity}'], column[f'{second}/{quantity}'])[0, 1]
            assert corr == pytest.approx(rho, abs=0.07)
    assert np.corrcoef(column['a/generation'], column['a/load'])[0, 1] == pytest.approx(0.0,
                                                                                         abs=0.1)


def test_common_factor_is_ar1_over_the_target_steps(flat_community: Community) -> None:
    """
    Test that the lead-1 errors of adjacent target steps of a series have correlation ρ φ (the
    common factor is AR(1) over the target steps, the own part independent): ρ = 0.8, φ = 0.5 →
    0.4; two steps apart ρ φ² = 0.2.

    Args:
        flat_community: Flat scenario.
    """
    provider = MMFEForecast(flat_community, MMFEParameters(SIGMA, SIGMA, rho=0.8, phi=0.5))
    errors = np.empty((len(SEEDS), 3, len(provider.keys)))
    for i, seed in enumerate(SEEDS):
        provider.reset(np.random.default_rng(seed))
        for k, t in enumerate((5, 6, 7)):
            errors[i, k] = relative_errors(provider.get(t, HORIZON), provider.keys)[:, 0]

    pooled = errors.transpose(1, 0, 2).reshape(3, -1)

    assert np.corrcoef(pooled[0], pooled[1])[0, 1] == pytest.approx(0.4, abs=0.07)
    assert np.corrcoef(pooled[0], pooled[2])[0, 1] == pytest.approx(0.2, abs=0.07)


def test_bias_accumulates_with_the_lead_time(flat_community: Community) -> None:
    """
    Test the transforms and the bias without noise, by hand: with b = 0.01 for generation and
    -0.02 for load, the forecast at lead time h is 5 + 7.5 · 0.01 h kWh (reference profile 7.5)
    and 10 (1 - 0.02 h) kWh.

    Args:
        flat_community: Flat scenario.
    """
    provider = MMFEForecast(flat_community, MMFEParameters(0.0, 0.0, bias_generation=0.01,
                                                           bias_load=-0.02))
    provider.reset(np.random.default_rng(0))
    values = provider.get(1, HORIZON)
    lead = np.arange(1, HORIZON + 1)

    np.testing.assert_allclose(values['b/generation'], 5.0 + 0.075 * lead)
    np.testing.assert_allclose(values['c/load'], 10.0 * (1.0 - 0.02 * lead))


def test_photovoltaic_forecast_is_zero_at_night_and_within_the_peak(mini_cer_v1_path: Path) -> None:
    """
    Test the photovoltaic transform on the synthetic scenario with the noisy preset: the
    generation forecasts (pv_bat, factory_bat, community) are exactly 0 in the hours without
    generation (0-7 and 16-23, reference profile 0), lie in [0, peak] (10, 20 and 5 kWh) and differ
    from the true values in daylight; load forecasts are never negative.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    provider = MMFEForecast(community, 'noisy')
    hours = pd.DatetimeIndex(community.timestamps).hour.to_numpy()
    truth = PerfectForecast(community)
    peak = {'pv_bat': 10.0, 'factory_bat': 20.0, 'community': 5.0}
    differs = False
    for seed in range(20):
        provider.reset(np.random.default_rng(seed))
        for t in range(community.n_steps - HORIZON + 1):
            values, true = provider.get(t, HORIZON), truth.get(t, HORIZON)
            night = (hours[t:t + HORIZON] < 8) | (hours[t:t + HORIZON] > 15)
            for owner, top in peak.items():
                forecast = values[f'{owner}/generation']
                np.testing.assert_array_equal(forecast[night], 0.0)
                assert np.all((forecast >= 0.0) & (forecast <= top))
                differs |= not np.array_equal(forecast, true[f'{owner}/generation'])
            assert all(np.all(values[k] >= 0.0) for k in provider.keys if k.endswith('/load'))
    assert differs


def test_perfect_preset_returns_the_true_values(mini_cer_v1_path: Path) -> None:
    """
    Test that the preset 'perfect' (σ = 0, no bias) returns the true values of the forecast
    series: generation and load of the battery nodes and the community totals.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    provider = MMFEForecast(community, 'perfect')
    provider.reset(np.random.default_rng(0))
    truth = PerfectForecast(community).get(10, 5)

    values = provider.get(10, 5)

    assert provider.keys == ('pv_bat/generation', 'pv_bat/load', 'factory_bat/generation',
                             'factory_bat/load', 'community/generation', 'community/load')
    for key in provider.keys:
        np.testing.assert_array_equal(values[key], truth[key])


def test_noise_is_reproducible_and_resampled_at_every_reset(mini_cer_v1_path: Path) -> None:
    """
    Test that the forecasts depend only on the generator passed to `reset`: the same seed gives the
    same forecasts, another seed and a second reset with the same generator give new ones.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)

    def episode(rng: np.random.Generator) -> np.ndarray:
        provider.reset(rng)
        return np.concatenate([np.concatenate(list(provider.get(t, 6).values()))
                               for t in range(10, 20)])

    provider = MMFEForecast(community, 'standard')
    generator = np.random.default_rng(3)
    first, second = episode(generator), episode(generator)

    np.testing.assert_array_equal(first, episode(np.random.default_rng(3)))
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, episode(np.random.default_rng(4)))


@pytest.mark.parametrize(('kwargs', 'match'), [
    ({'sigma_generation': -0.1, 'sigma_load': 0.0}, 'sigma_generation must be ≥ 0'),
    ({'sigma_generation': 0.1, 'sigma_load': float('nan')}, 'sigma_load must be a finite number'),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'rho': 1.5}, r'rho must lie in \[0, 1\]'),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'phi': 1.0}, r'phi must lie in \(-1, 1\)'),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'bias_load': '0'}, 'bias_load must be a finite'),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'sigma_overrides': {'a/load': -0.1}},
     r"sigma_overrides\['a/load'\] must be ≥ 0"),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'sigma_overrides': {'a/load': None}},
     r"sigma_overrides\['a/load'\] must be a finite number"),
    ({'sigma_generation': 0.1, 'sigma_load': 0.1, 'sigma_overrides': [0.1]},
     'sigma_overrides must map forecast keys'),
])
def test_parameters_are_checked(kwargs: dict[str, Any], match: str) -> None:
    """
    Test the checks of `MMFEParameters`.

    Args:
        kwargs: Parameters.
        match: Expected message.
    """
    with pytest.raises(ValueError, match=match):
        MMFEParameters(**kwargs)


def test_provider_checks_its_use(mini_cer_v1_path: Path) -> None:
    """
    Test that an unknown preset, a request before `reset`, a horizon beyond the episode's first
    one, a horizon below 1 and steps outside the period raise.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)
    with pytest.raises(ValueError, match='unknown MMFE preset'):
        MMFEForecast(community, 'calm')
    provider = MMFEForecast(community)
    with pytest.raises(RuntimeError, match='reset'):
        provider.get(0, 3)
    provider.reset(np.random.default_rng(0))
    provider.get(0, 3)
    provider.get(1, 2)
    with pytest.raises(ValueError, match='horizon 4 exceeds the horizon 3'):
        provider.get(2, 4)
    with pytest.raises(ValueError, match='at least 1'):
        provider.get(2, 0)
    with pytest.raises(ValueError, match='leaves the period'):
        provider.get(46, 3)
    provider.reset(np.random.default_rng(0))
    assert len(provider.get(0, 4)['pv_bat/load']) == 4


def test_presets_and_registry(mini_cer_v1_path: Path) -> None:
    """
    Test that the presets are registered as 'mmfe_<preset>', that the provider satisfies the
    protocol and that `config()` names the preset and the parameters.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    community = load_scenario(mini_cer_v1_path)

    assert set(MMFE_PRESETS) == {'perfect', 'standard', 'noisy'}
    for name, parameters in MMFE_PRESETS.items():
        provider = FORECAST_PROVIDERS[f'mmfe_{name}'](community)
        assert isinstance(provider, MMFEForecast) and isinstance(provider, ForecastProvider)
        assert provider.parameters == parameters
        assert provider.config() == {'preset': name, **vars(parameters)}
    custom = MMFEForecast(community, MMFEParameters(0.1, 0.2))
    assert custom.config()['preset'] is None and custom.config()['sigma_load'] == 0.2


def test_sigma_overrides_set_the_sigma_of_one_series(flat_community: Community,
                                                     mini_cer_v1_path: Path) -> None:
    """
    Test the per-series σ: with σ = 0.02 for both quantities and an override of 0.05 for the load
    of node a, the lead-12 error of a/load has standard deviation 0.05 √12 and the other loads
    0.02 √12 (2000 seeds); with σ = 0 and an override for `community/load` on the synthetic
    scenario only that series differs from the truth. Unknown keys raise, and the parameters keep
    a copy of the overrides.

    Args:
        flat_community: Flat scenario.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    provider = MMFEForecast(flat_community, MMFEParameters(SIGMA, SIGMA, rho=0.5, phi=0.6,
                                                           sigma_overrides={'a/load': 0.05}))
    errors = np.empty((len(SEEDS), len(provider.keys)))
    for i, seed in enumerate(SEEDS):
        provider.reset(np.random.default_rng(seed))
        errors[i] = relative_errors(provider.get(1, HORIZON), provider.keys)[:, -1]
    std = dict(zip(provider.keys, errors.std(axis=0)))

    assert std['a/load'] == pytest.approx(0.05 * np.sqrt(HORIZON), rel=0.1)
    assert std['b/load'] == pytest.approx(SIGMA * np.sqrt(HORIZON), rel=0.1)
    assert std['a/generation'] == pytest.approx(SIGMA * np.sqrt(HORIZON), rel=0.1)
    community = load_scenario(mini_cer_v1_path)
    overrides = {'community/load': 0.05}
    parameters = MMFEParameters(0.0, 0.0, sigma_overrides=overrides)
    overrides['community/load'] = 0.0
    provider = MMFEForecast(community, parameters)
    provider.reset(np.random.default_rng(0))
    values, truth = provider.get(10, 6), PerfectForecast(community).get(10, 6)
    assert parameters.sigma_overrides == {'community/load': 0.05}
    assert not np.array_equal(values['community/load'], truth['community/load'])
    for key in provider.keys:
        if key != 'community/load':
            np.testing.assert_array_equal(values[key], truth[key])
    with pytest.raises(ValueError, match=r"sigma_overrides: \['house_1/load'\] are not forecast "
                                         'keys'):
        MMFEForecast(community, MMFEParameters(0.1, 0.1, sigma_overrides={'house_1/load': 0.1}))
