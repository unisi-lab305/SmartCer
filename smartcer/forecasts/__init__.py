"""
Forecast providers for the observation of `SmartCerEnv` (`forecast` and `forecast_horizon`).
"""

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType

from smartcer.core.community import Community
from smartcer.forecasts.base import (
    COMMUNITY_PREFIX,
    FORECAST_QUANTITIES,
    ForecastProvider,
    check_steps,
    community_totals,
    forecast_series,
)
from smartcer.forecasts.mmfe import MMFE_PRESETS, MMFEForecast, MMFEParameters
from smartcer.forecasts.perfect import PerfectForecast


# provider name → factory taking the community; 'mmfe_<preset>' for every preset of MMFE_PRESETS
FORECAST_PROVIDERS: Mapping[str, Callable[[Community], ForecastProvider]] = MappingProxyType({
    'perfect': PerfectForecast,
    **{f'mmfe_{name}': partial(MMFEForecast, parameters=name) for name in MMFE_PRESETS},
})


__all__ = [
    'COMMUNITY_PREFIX',
    'FORECAST_PROVIDERS',
    'FORECAST_QUANTITIES',
    'MMFE_PRESETS',
    'ForecastProvider',
    'MMFEForecast',
    'MMFEParameters',
    'PerfectForecast',
    'check_steps',
    'community_totals',
    'forecast_series',
]
