"""
Observation schema of `SmartCerEnv`: a flat `Box` with a layout feature name → slice.

Each environment builds its own schema from the community and the run arguments (preset, timing,
lookback, normaliser). Sections, in this order:

- `time`: `time/month`, `time/hour` cyclical (sin, cos); `time/day_of_week` one-hot (Monday
  first); `extended` only: `time/holiday` (1 on Italian national holidays, `holidays` package;
  no regional or local holidays);
- `price`: `price/sell`, `price/buy`, one global pair, not normalised, when every node that sells
  has the same sell price series and every node that buys has the same buy price series;
  otherwise the prices are `<node_id>/sell_price` and `<node_id>/buy_price` inside each battery
  node's block (a documented deviation from [M24], needed for scenarios such as Leather);
- `community` (`extended` only): `community/generation` and `community/load`, the total
  generation and load of the nodes without a battery (each only if such nodes have it);
- `nodes`: per battery node, in `storage_node_ids` order, [prices], `<node_id>/generation`,
  `<node_id>/load`, [`extended` with lookback L: `<node_id>/generation_lookback`,
  `<node_id>/load_lookback`, L values each, most recent first], `<node_id>/soc` (scaled to
  [S^min, S^max] → [0, 1]), `<node_id>/a_lower`, `<node_id>/a_upper` (in [0, 1]);
- `forecast`: reserved for the forecasts (`forecast/<node_id>/<quantity>`), empty for now.

Preset `meco` is [M24] Tab. II. Generation and load (also the aggregates and the lookback) are
min-max scaled with a `MinMaxNormalizer` fitted on a training window and clipped to [0, 1]; the
number of clipped values of each step's observation is in `ObservationBuilder.clipped`.

With `observe_current_step=True` the observation used to choose the action of step t holds g, l,
prices and bounds of step t; with False it holds g and l of step t - 1, the prices of step t, and
an upper bound computed without the charge cap, so that it does not reveal g of step t
(docs/FORMULATION.md section 12). The lookback holds the L steps before the observed g and l.
Values before the first row of the data are 0.
"""

from typing import Any

from gymnasium import spaces
import holidays
import numpy as np
import pandas as pd

from smartcer.core.community import Community
from smartcer.gym.observation.encoders import cyclical, one_hot
from smartcer.gym.observation.normalizers import MinMaxNormalizer


OBSERVATION_PRESETS = ('meco', 'extended')
SECTIONS = ('time', 'price', 'community', 'nodes', 'forecast')
# layout prefixes that node ids must not use
RESERVED_PREFIXES = ('time', 'price', 'community', 'forecast')


def scaled_series(community: Community) -> dict[str, np.ndarray]:
    """
    The series that the observation min-max scales, over the whole community period.

    Args:
        community: The community.

    Returns:
        dict[str, np.ndarray]: `<node_id>/generation` and `<node_id>/load` of every battery node,
        and `community/generation`, `community/load` (totals of the nodes without a battery that
        have generation or load).
    """
    series: dict[str, np.ndarray] = {}
    for node_id in community.storage_node_ids:
        node = community.node(node_id)
        series[f'{node_id}/generation'] = node.generation_kwh
        series[f'{node_id}/load'] = node.load_kwh
    others = [n for n in community.nodes if n.battery is None]
    generators = [n.generation_kwh for n in others if n.node_type != 'consumer']
    loads = [n.load_kwh for n in others if n.node_type != 'producer']
    if generators:
        series['community/generation'] = np.sum(generators, axis=0)
    if loads:
        series['community/load'] = np.sum(loads, axis=0)
    return series


def window_mask(timestamps: np.ndarray, window: tuple[Any, Any] | None) -> np.ndarray:
    """
    Steps whose timestamp lies in a window, both ends inclusive.

    Args:
        timestamps: Step timestamps.
        window: (first, last) timestamp, either None for an open end; None for every step.

    Returns:
        np.ndarray: Boolean mask.

    Raises:
        ValueError: If the window is malformed or contains no step.
    """
    index = pd.DatetimeIndex(timestamps)
    if window is None:
        return np.ones(len(index), dtype=bool)
    if len(window) != 2:
        raise ValueError(f'window must be a pair (first, last), got {window!r}')
    first, last = window
    mask = np.ones(len(index), dtype=bool)
    if first is not None:
        mask &= index >= pd.Timestamp(first)
    if last is not None:
        mask &= index <= pd.Timestamp(last)
    if not mask.any():
        raise ValueError(f'window {window!r} contains no step of the period {index[0]} → '
                         f'{index[-1]}')
    return mask


def fit_normalizer(community: Community,
                   window: tuple[Any, Any] | None = None) -> MinMaxNormalizer:
    """
    Fit the observation normaliser on the steps of a training window.

    Only the values of the window are read, so data outside it (e.g. a test period) do not
    affect the bounds.

    Args:
        community: The community.
        window: (first, last) timestamp, both inclusive; None for the whole period.

    Returns:
        MinMaxNormalizer: Bounds of `scaled_series(community)` over the window.
    """
    mask = window_mask(community.timestamps, window)
    used = pd.DatetimeIndex(community.timestamps[mask])
    series = {name: values[mask] for name, values in scaled_series(community).items()}
    return MinMaxNormalizer.fit(series, window=(used[0].isoformat(), used[-1].isoformat()),
                                scenario=community.name)


def lagged(series: np.ndarray, lag: int) -> np.ndarray:
    """
    Series shifted by `lag` steps: value of step t - lag at step t, 0 before the first row.

    Args:
        series: Values.
        lag: Shift, ≥ 0.

    Returns:
        np.ndarray: Shifted values.
    """
    if lag == 0:
        return series
    out = np.zeros_like(series)
    out[lag:] = series[:-lag]
    return out


class ObservationBuilder:
    """
    Builds the flat observation of a community for a step.

    Attributes:
        preset: Observation preset.
        observe_current_step: Information timing (docs/FORMULATION.md section 12).
        lookback: Past steps of g and l per battery node (`extended` only).
        normalizer: Min-max normaliser of generation and load.
        layout: Feature name → slice of the observation vector.
        sections: Section name → feature names, in order (`forecast` is empty).
        space: The observation space, `Box(low, high, (dim,), float32)`.
        global_prices: True if the prices are one global pair.
        clipped: Per step, the number of scaled values of the observation that lie outside the
            normaliser's training range and are clipped to [0, 1] (the zeros before the first
            row of the data are not counted).
    """

    def __init__(self, community: Community, preset: str = 'meco',
                 observe_current_step: bool = True, lookback: int = 0,
                 normalizer: MinMaxNormalizer | None = None) -> None:
        """
        Precompute the step-independent features of every step.

        Args:
            community: The community.
            preset: Observation preset, 'meco' or 'extended'.
            observe_current_step: Information timing.
            lookback: Past steps of g and l per battery node, ≥ 0 (> 0 only with 'extended').
            normalizer: Normaliser of generation and load; None fits one on the whole period.

        Raises:
            ValueError: If the preset is unknown, the lookback is invalid, a node id is a reserved
                layout prefix or the normaliser lacks a series.
        """
        if preset not in OBSERVATION_PRESETS:
            raise ValueError(f'unknown observation preset {preset!r}; available: '
                             f'{OBSERVATION_PRESETS}')
        if not (isinstance(lookback, (int, np.integer)) and lookback >= 0):
            raise ValueError(f'lookback must be an integer ≥ 0, got {lookback!r}')
        if lookback and preset != 'extended':
            raise ValueError(f'lookback is available with the extended preset only, got preset '
                             f'{preset!r}')
        reserved = [i for i in community.node_ids if i in RESERVED_PREFIXES]
        if reserved:
            raise ValueError(f'node ids {reserved} are reserved observation prefixes '
                             f'{RESERVED_PREFIXES}')
        series = scaled_series(community)
        if normalizer is None:
            normalizer = fit_normalizer(community)
        normalizer.require(series)
        self.preset = preset
        self.observe_current_step = observe_current_step
        self.lookback = int(lookback)
        self.normalizer = normalizer
        self._community = community
        self._battery_nodes = [community.node(i) for i in community.storage_node_ids]
        extended = preset == 'extended'
        lag = 0 if observe_current_step else 1

        columns: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        self.layout: dict[str, slice] = {}
        self.sections: dict[str, list[str]] = {name: [] for name in SECTIONS}
        self.clipped = np.zeros(community.n_steps, dtype=np.int64)

        def add(section: str, name: str, values: np.ndarray, low: float | np.ndarray,
                high: float | np.ndarray) -> slice:
            values = values.reshape(len(values), -1)
            start = sum(c.shape[1] for c in columns)
            width = values.shape[1]
            columns.append(values)
            low = np.broadcast_to(np.asarray(low, dtype=np.float64), (width,))
            high = np.broadcast_to(np.asarray(high, dtype=np.float64), (width,))
            # a constant series (e.g. a fixed tariff) would give low == high, which Gymnasium flags
            lows.append(low)
            highs.append(np.where(high > low, high, low + 1.0))
            self.layout[name] = slice(start, start + width)
            self.sections[section].append(name)
            return self.layout[name]

        def add_scaled(section: str, name: str, key: str, lags: list[int]) -> None:
            values = np.column_stack([lagged(series[key], k) for k in lags])
            # the zeros before the first row are padding, not clipped data
            outside = normalizer.outside_range(key, series[key])
            self.clipped += np.column_stack([lagged(outside, k) for k in lags]).sum(axis=1)
            add(section, name, normalizer.transform(key, values), 0.0, 1.0)

        timestamps = pd.DatetimeIndex(community.timestamps)
        add('time', 'time/month', cyclical(timestamps.month - 1, 12), -1.0, 1.0)
        add('time', 'time/hour', cyclical(timestamps.hour + timestamps.minute / 60.0, 24),
            -1.0, 1.0)
        add('time', 'time/day_of_week', one_hot(timestamps.dayofweek, 7), 0.0, 1.0)
        if extended:
            calendar = holidays.country_holidays('IT', years=sorted(set(timestamps.year)))
            flag = np.array([day in calendar for day in timestamps.date], dtype=np.float64)
            add('time', 'time/holiday', flag, 0.0, 1.0)

        sellers = [n for n in community.nodes if n.node_type != 'consumer']
        buyers = [n for n in community.nodes if n.node_type != 'producer']
        self.global_prices = (
            all(np.array_equal(n.sell_price, sellers[0].sell_price) for n in sellers)
            and all(np.array_equal(n.buy_price, buyers[0].buy_price) for n in buyers)
        ) if sellers and buyers else False
        if self.global_prices:
            for name, prices in (('price/sell', sellers[0].sell_price),
                                 ('price/buy', buyers[0].buy_price)):
                add('price', name, prices, prices.min(), prices.max())

        if extended:
            for key in ('community/generation', 'community/load'):
                if key in series:
                    add_scaled('community', key, key, [lag])

        self._dynamic: list[tuple[int, int, int]] = []
        past = [lag + k for k in range(1, self.lookback + 1)]
        for node in self._battery_nodes:
            prefix = node.node_id
            if not self.global_prices:
                for name, prices in (('sell_price', node.sell_price),
                                     ('buy_price', node.buy_price)):
                    add('nodes', f'{prefix}/{name}', prices, prices.min(), prices.max())
            for quantity in ('generation', 'load'):
                add_scaled('nodes', f'{prefix}/{quantity}', f'{prefix}/{quantity}', [lag])
            for quantity in ('generation', 'load') if past else ():
                add_scaled('nodes', f'{prefix}/{quantity}_lookback', f'{prefix}/{quantity}', past)
            # placeholders, filled at every step from the battery state
            zeros = np.zeros(community.n_steps)
            soc = add('nodes', f'{prefix}/soc', zeros, 0.0, 1.0)
            a_lower = add('nodes', f'{prefix}/a_lower', zeros, 0.0, 1.0)
            a_upper = add('nodes', f'{prefix}/a_upper', zeros, 0.0, 1.0)
            self._dynamic.append((soc.start, a_lower.start, a_upper.start))

        self._static = np.hstack(columns)
        low = np.concatenate(lows).astype(np.float32)
        high = np.concatenate(highs).astype(np.float32)
        self._low, self._high = low, high
        self.space = spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def dim(self) -> int:
        """Length of the observation vector."""
        return self._static.shape[1]

    def build(self, t: int) -> np.ndarray:
        """
        Observation for choosing the action of step t, from the current battery states.

        Args:
            t: Step index.

        Returns:
            np.ndarray: Float32 vector inside `space`.
        """
        obs = self._static[t].copy()
        for node, (i_soc, i_lower, i_upper) in zip(self._battery_nodes, self._dynamic):
            battery = node.battery
            span = battery.capacity - battery.soc_min_kwh
            obs[i_soc] = (battery.soc_kwh - battery.soc_min_kwh) / span
            if self.observe_current_step:
                a_upper, a_lower = node.get_bounds(t)
            else:
                a_upper, a_lower = battery.get_bounds(node.delta_t, None)
            obs[i_lower], obs[i_upper] = a_lower, a_upper
        # float32 rounding of the data bounds must not push a value outside the space
        return np.clip(obs.astype(np.float32), self._low, self._high)
