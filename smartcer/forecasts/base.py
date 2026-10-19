"""
Protocol of the forecast providers: future generation and load of the nodes.

Forecasts enter only the observation (docs/refactor/REFACTOR_v1.md section 4). A provider is reset
at every episode with the environment's random generator (`env.np_random`), so that a stochastic
provider, e.g. one that simulates forecast errors, is reproducible from the episode seed.
"""

from typing import Protocol, runtime_checkable

import numpy as np


# quantities a provider forecasts for every node, in kWh per step
FORECAST_QUANTITIES = ('generation', 'load')


@runtime_checkable
class ForecastProvider(Protocol):
    """
    Source of forecasts of generation and load, keyed `'<node_id>/<quantity>'`.
    """

    def reset(self, rng: np.random.Generator) -> None:
        """
        Start an episode.

        Args:
            rng: The environment's random generator (`env.np_random`).
        """

    def get(self, t: int, horizon: int) -> dict[str, np.ndarray]:
        """
        Forecasts of the steps t, ..., t + horizon - 1.

        The environment asks for the steps after the last observed one (docs/FORMULATION.md
        section 12): t + 1, ..., t + H with `observe_current_step=True`, t, ..., t + H - 1 with
        False; the k-th value (k = 0, 1, ...) is therefore the forecast of lead time k + 1 steps.

        Args:
            t: First forecast step.
            horizon: Number of steps H, ≥ 1.

        Returns:
            dict[str, np.ndarray]: `'<node_id>/<quantity>'` → H values (kWh per step), for the
            quantities of `FORECAST_QUANTITIES`.
        """
