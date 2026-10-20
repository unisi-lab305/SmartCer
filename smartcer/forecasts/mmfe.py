"""
Forecasts with simulated errors following the martingale model of forecast evolution (MMFE).

For every forecast series i and target step τ, H innovations ε_{i,τ,1}, ..., ε_{i,τ,H} are drawn
once per episode. The forecast of τ issued with lead time h (h steps before the step τ is
observed) carries the first h of them:

    e_{i,τ}(h) = ε_{i,τ,1} + ... + ε_{i,τ,h}

so that the error variance grows linearly with the lead time and successive issues for the same
target are consistent: when the lead time drops from h to h - 1 the innovation ε_{i,τ,h} is
revealed and leaves the error; at lead time 0 the true value is observed. Each innovation is

    ε_{i,τ,j} = b_q + σ_q (sqrt(ρ) z_{q,τ,j} + sqrt(1 - ρ) ξ_{i,τ,j})

with q the quantity of the series (generation or load), b_q an optional bias, ξ independent
standard normal draws per series and z a standard normal factor common to all series of the
quantity, AR(1) over the target steps: z_{q,τ,j} = φ z_{q,τ-1,j} + sqrt(1 - φ²) η_{q,τ,j}. Two
series of the same quantity therefore have error correlation ρ, and the errors of adjacent target
steps are correlated through φ.

The error enters the forecast ŷ of the true value y as follows:

- generation (photovoltaic): ŷ = min(max(y + r_τ e, 0), y^max), with r_τ the reference profile of
  the series (its mean over the steps of the same calendar month at the same time of day) and
  y^max the largest value of the series; at night r_τ = 0 and the forecast is 0;
- load: ŷ = max(y (1 + e), 0), a multiplicative error.

The innovations of an episode are drawn from a generator spawned from the one passed to `reset`
(`env.np_random`), so the same seed gives the same forecasts; they are drawn lazily, in blocks of
target steps, as the episode moves forward.
"""

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from smartcer.core.community import Community
from smartcer.forecasts.base import check_steps, forecast_series


# target steps drawn at once when an episode needs new ones
CHUNK_STEPS = 168


@dataclass(frozen=True)
class MMFEParameters:
    """
    Parameters of the MMFE forecast errors.

    Attributes:
        sigma_generation: σ of the generation innovations, as a fraction of the reference profile.
        sigma_load: σ of the load innovations, as a fraction of the load.
        rho: Share ρ of the innovation variance due to the factor common to all series of a
            quantity, in [0, 1].
        phi: AR(1) coefficient φ of the common factor over the target steps, in (-1, 1).
        bias_generation: Mean b of a generation innovation, as a fraction of the reference
            profile; the mean error at lead time h is h b.
        bias_load: Mean b of a load innovation, as a fraction of the load.
    """

    sigma_generation: float
    sigma_load: float
    rho: float = 0.0
    phi: float = 0.0
    bias_generation: float = 0.0
    bias_load: float = 0.0

    def __post_init__(self) -> None:
        """
        Check the parameters.

        Raises:
            ValueError: If a parameter is not a finite number or lies outside its range.
        """
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value):
                raise ValueError(f'MMFE parameter {name} must be a finite number, got {value!r}')
        for name in ('sigma_generation', 'sigma_load'):
            if getattr(self, name) < 0.0:
                raise ValueError(f'MMFE parameter {name} must be ≥ 0, got {getattr(self, name)}')
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f'MMFE parameter rho must lie in [0, 1], got {self.rho}')
        if not -1.0 < self.phi < 1.0:
            raise ValueError(f'MMFE parameter phi must lie in (-1, 1), got {self.phi}')


# provisional values, to be calibrated (docs/CONFIG.md section 6)
MMFE_PRESETS: MappingProxyType[str, MMFEParameters] = MappingProxyType({
    'perfect': MMFEParameters(sigma_generation=0.0, sigma_load=0.0),
    'standard': MMFEParameters(sigma_generation=0.07, sigma_load=0.03, rho=0.5, phi=0.8),
    'noisy': MMFEParameters(sigma_generation=0.15, sigma_load=0.08, rho=0.7, phi=0.9),
})


def reference_profile(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Mean of each series over the steps of the same calendar month at the same time of day.

    Args:
        values: Series, shape (n_series, n_steps).
        timestamps: Step timestamps, shape (n_steps,).

    Returns:
        np.ndarray: The reference profile of every step, same shape as `values`.
    """
    index = pd.DatetimeIndex(timestamps)
    group = index.month * 1440 + index.hour * 60 + index.minute
    return pd.DataFrame(values.T).groupby(group).transform('mean').to_numpy().T


class MMFEForecast:
    """
    Forecast provider with MMFE errors (module docstring) on the series of `forecast_series`:
    generation and load of every battery node and the community totals.

    Attributes:
        parameters: The error parameters.
        preset: Name of the preset in `MMFE_PRESETS`, or None for custom parameters.
        keys: Forecast keys, in order.
        n_steps: Number of steps of the community period.
    """

    def __init__(self, community: Community,
                 parameters: MMFEParameters | str = 'standard') -> None:
        """
        Read the series and compute their reference profiles.

        Args:
            community: The community whose data are forecast.
            parameters: Error parameters, or the name of a preset of `MMFE_PRESETS`.

        Raises:
            ValueError: If the preset is unknown.
        """
        self.preset = None
        if isinstance(parameters, str):
            if parameters not in MMFE_PRESETS:
                raise ValueError(f'unknown MMFE preset {parameters!r}; available: '
                                 f'{list(MMFE_PRESETS)}')
            self.preset, parameters = parameters, MMFE_PRESETS[parameters]
        self.parameters = parameters
        series = forecast_series(community)
        self.keys = tuple(series)
        self.n_steps = community.n_steps
        self._truth = np.vstack(list(series.values()))
        generation = np.array([key.endswith('/generation') for key in self.keys])
        # error scale: the reference profile for generation, the value itself for load
        self._scale = self._truth.copy()
        if generation.any():
            self._scale[generation] = reference_profile(self._truth[generation],
                                                        community.timestamps)
        self._upper = np.where(generation, self._truth.max(axis=1), np.inf)
        p = parameters
        self._quantity = np.where(generation, 0, 1)
        self._sigma = np.where(generation, p.sigma_generation, p.sigma_load)
        self._bias = np.where(generation, p.bias_generation, p.bias_load)
        self._rng: np.random.Generator | None = None
        self._depth = 0
        self._forecasts = np.empty((len(self.keys), 0, 0))
        self._lo = self._hi = 0
        self._factor: np.ndarray | None = None

    def __repr__(self) -> str:
        name = self.preset if self.preset is not None else 'custom'
        return f'MMFEForecast({name!r}, {len(self.keys)} series, {self.n_steps} steps)'

    def config(self) -> dict[str, Any]:
        """
        The provider settings, JSON-serialisable, for the run configuration.

        Returns:
            dict[str, Any]: `preset` (or None) and the parameters.
        """
        return {'preset': self.preset, **asdict(self.parameters)}

    def reset(self, rng: np.random.Generator) -> None:
        """
        Start an episode: new innovations, drawn from a generator spawned from `rng`.

        Args:
            rng: The environment's random generator.
        """
        self._rng = rng.spawn(1)[0]
        self._depth = 0
        self._lo = self._hi = 0
        self._factor = None

    def get(self, t: int, horizon: int) -> dict[str, np.ndarray]:
        """
        Forecasts of the steps t, ..., t + horizon - 1, with lead times 1, ..., horizon.

        The innovations of a target step are drawn the first time it is forecast in the episode,
        so later issues for it are consistent, as long as the issues move forward in time (a
        request before the steps drawn so far, or after a gap, starts a new chain of draws).

        Args:
            t: First forecast step.
            horizon: Number of steps H, ≥ 1, at most the horizon of the first request of the
                episode.

        Returns:
            dict[str, np.ndarray]: `keys` → H forecast values (kWh per step).

        Raises:
            RuntimeError: If called before `reset`.
            ValueError: If the horizon is not positive or exceeds the episode's, or the steps
                leave the period.
        """
        if self._rng is None:
            raise RuntimeError('call reset(rng) before get()')
        check_steps(t, horizon, self.n_steps)
        if self._depth == 0:
            self._depth = horizon
            if self._forecasts.shape[1:] != (self.n_steps, horizon):
                self._forecasts = np.empty((len(self.keys), self.n_steps, horizon))
        elif horizon > self._depth:
            raise ValueError(f'horizon {horizon} exceeds the horizon {self._depth} of the first '
                             f'forecast of the episode')
        if not self._lo <= t <= self._hi:
            self._lo = self._hi = t
            self._factor = None
        end = t + horizon
        if end > self._hi:
            self._draw(self._hi, min(self.n_steps, max(end, self._hi + CHUNK_STEPS)))
        lead = np.arange(horizon)
        return dict(zip(self.keys, self._forecasts[:, t + lead, lead]))

    def _draw(self, first: int, stop: int) -> None:
        """
        Draw the innovations of the target steps first, ..., stop - 1 and store their forecasts
        at every lead time.

        Args:
            first: First target step, equal to the end of the steps drawn so far.
            stop: End of the target steps (exclusive).
        """
        p, rng, depth = self.parameters, self._rng, self._depth
        count = stop - first
        shocks = rng.standard_normal((count, 2, depth))
        factor = np.empty_like(shocks)
        previous = self._factor if self._factor is not None \
            else rng.standard_normal((2, depth))
        weight = math.sqrt(1.0 - p.phi ** 2)
        for k in range(count):
            previous = p.phi * previous + weight * shocks[k]
            factor[k] = previous
        self._factor = previous
        own = rng.standard_normal((len(self.keys), count, depth))
        common = factor[:, self._quantity, :].transpose(1, 0, 2)
        innovations = self._bias[:, None, None] + self._sigma[:, None, None] * (
            math.sqrt(p.rho) * common + math.sqrt(1.0 - p.rho) * own)
        error = np.cumsum(innovations, axis=2)
        steps = slice(first, stop)
        self._forecasts[:, steps, :] = np.clip(
            self._truth[:, steps, None] + self._scale[:, steps, None] * error,
            0.0, self._upper[:, None, None])
        self._hi = stop
