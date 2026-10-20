"""
Protocol of the forecast providers: future generation and load of the nodes and of the community.

Forecasts enter only the observation (docs/refactor/REFACTOR_v1.md section 4). A provider is reset
at every episode with the environment's random generator (`env.np_random`), so that a stochastic
provider, e.g. one that simulates forecast errors, is reproducible from the episode seed.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from smartcer.core.community import Community


# quantities a provider forecasts for every node, in kWh per step
FORECAST_QUANTITIES = ('generation', 'load')
# key prefix of the totals of the nodes without a battery
COMMUNITY_PREFIX = 'community'


def community_totals(community: Community) -> dict[str, np.ndarray]:
    """
    Total generation and load of the nodes without a battery (e.g. aggregated consumers).

    Args:
        community: The community.

    Returns:
        dict[str, np.ndarray]: `community/generation` (producers and prosumers) and
        `community/load` (consumers and prosumers), in kWh per step, each only if such nodes exist.
    """
    others = [n for n in community.nodes if n.battery is None]
    generators = [n.generation_kwh for n in others if n.node_type != 'consumer']
    loads = [n.load_kwh for n in others if n.node_type != 'producer']
    totals: dict[str, np.ndarray] = {}
    if generators:
        totals[f'{COMMUNITY_PREFIX}/generation'] = np.sum(generators, axis=0)
    if loads:
        totals[f'{COMMUNITY_PREFIX}/load'] = np.sum(loads, axis=0)
    return totals


def forecast_series(community: Community) -> dict[str, np.ndarray]:
    """
    The series the observation forecasts, over the whole community period.

    Args:
        community: The community.

    Returns:
        dict[str, np.ndarray]: `<node_id>/generation` and `<node_id>/load` of every battery node,
        in `storage_node_ids` order, then the totals of `community_totals` (kWh per step).
    """
    series: dict[str, np.ndarray] = {}
    for node_id in community.storage_node_ids:
        node = community.node(node_id)
        for quantity in FORECAST_QUANTITIES:
            series[f'{node_id}/{quantity}'] = getattr(node, f'{quantity}_kwh')
    series.update(community_totals(community))
    return series


def check_steps(t: int, horizon: int, n_steps: int) -> None:
    """
    Check that the forecast steps t, ..., t + horizon - 1 lie in the period.

    Args:
        t: First forecast step.
        horizon: Number of steps.
        n_steps: Number of steps of the period.

    Raises:
        ValueError: If the horizon is not positive or the steps leave the period.
    """
    if horizon < 1:
        raise ValueError(f'horizon must be at least 1, got {horizon}')
    if t < 0 or t + horizon > n_steps:
        raise ValueError(f'forecast of steps {t}..{t + horizon - 1} leaves the period of '
                         f'{n_steps} steps')


@runtime_checkable
class ForecastProvider(Protocol):
    """
    Source of forecasts of generation and load, keyed `'<node_id>/<quantity>'` for a node and
    `'community/<quantity>'` for the totals of the nodes without a battery.
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
            dict[str, np.ndarray]: Key → H values (kWh per step). The environment needs at least
            the keys of `forecast_series`: generation and load of every battery node and the
            community totals.
        """
