"""
Perfect forecast: the true future values of the scenario data.
"""

import numpy as np

from smartcer.core.community import Community
from smartcer.forecasts.base import FORECAST_QUANTITIES


class PerfectForecast:
    """
    Forecast provider that returns the true generation and load of every node.

    Generation is zero for a consumer and load is zero for a producer, as in the node data.

    Attributes:
        n_steps: Number of steps of the community period.
    """

    def __init__(self, community: Community) -> None:
        """
        Read the series of every node.

        Args:
            community: The community whose data are forecast.
        """
        self.n_steps = community.n_steps
        self._keys = tuple(f'{node.node_id}/{quantity}' for node in community.nodes
                           for quantity in FORECAST_QUANTITIES)
        # one row per key, so that a forecast is a single copy
        self._data = np.vstack([getattr(node, f'{quantity}_kwh') for node in community.nodes
                                for quantity in FORECAST_QUANTITIES])

    def __repr__(self) -> str:
        return f'PerfectForecast({len(self._keys)} series, {self.n_steps} steps)'

    def reset(self, rng: np.random.Generator) -> None:
        """
        Start an episode; a perfect forecast is deterministic and ignores the generator.

        Args:
            rng: The environment's random generator.
        """

    def get(self, t: int, horizon: int) -> dict[str, np.ndarray]:
        """
        True values of the steps t, ..., t + horizon - 1.

        Args:
            t: First forecast step.
            horizon: Number of steps H, ≥ 1.

        Returns:
            dict[str, np.ndarray]: `'<node_id>/generation'` and `'<node_id>/load'` of every node
            → copies of the H values (kWh per step).

        Raises:
            ValueError: If the horizon is not positive or the steps leave the period.
        """
        if horizon < 1:
            raise ValueError(f'horizon must be at least 1, got {horizon}')
        if t < 0 or t + horizon > self.n_steps:
            raise ValueError(f'forecast of steps {t}..{t + horizon - 1} leaves the period of '
                             f'{self.n_steps} steps')
        return dict(zip(self._keys, self._data[:, t:t + horizon].copy()))
