"""
Perfect forecast: the true future values of the scenario data.
"""

import numpy as np

from smartcer.core.community import Community
from smartcer.forecasts.base import FORECAST_QUANTITIES, check_steps, community_totals


class PerfectForecast:
    """
    Forecast provider that returns the true generation and load of every node and the community
    totals of the nodes without a battery (`community_totals`).

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
        series = {f'{node.node_id}/{quantity}': getattr(node, f'{quantity}_kwh')
                  for node in community.nodes for quantity in FORECAST_QUANTITIES}
        series.update(community_totals(community))
        self._keys = tuple(series)
        # one row per key, so that a forecast is a single copy
        self._data = np.vstack(list(series.values()))

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
            and the community totals → copies of the H values (kWh per step).

        Raises:
            ValueError: If the horizon is not positive or the steps leave the period.
        """
        check_steps(t, horizon, self.n_steps)
        return dict(zip(self._keys, self._data[:, t:t + horizon].copy()))
