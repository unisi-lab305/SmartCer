"""
Forecast providers for the observation of `SmartCerEnv` (`forecast` and `forecast_horizon`).
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from smartcer.core.community import Community
from smartcer.forecasts.base import FORECAST_QUANTITIES, ForecastProvider
from smartcer.forecasts.perfect import PerfectForecast


# provider name → factory taking the community
FORECAST_PROVIDERS: Mapping[str, Callable[[Community], ForecastProvider]] = MappingProxyType({
    'perfect': PerfectForecast,
})


__all__ = ['FORECAST_PROVIDERS', 'FORECAST_QUANTITIES', 'ForecastProvider', 'PerfectForecast']
