"""
Observation schema of `SmartCerEnv`: a flat `Box` with a layout feature name → slice.

Preset `meco` ([M24] Tab. II, docs/FORMULATION.md section 12):

- `time/month`, `time/hour`: cyclical (sin, cos); `time/day_of_week`: one-hot (Monday first);
- `price/sell`, `price/buy`: one global pair, not normalised, when every node that sells has the
  same sell price series and every node that buys has the same buy price series; otherwise
  `<node_id>/sell_price` and `<node_id>/buy_price` inside each battery node's block (a documented
  deviation from [M24], needed for scenarios such as Leather);
- per battery node, in `storage_node_ids` order: `<node_id>/generation` and `<node_id>/load`
  (min-max over the scenario period), `<node_id>/soc` (scaled to [S^min, S^max] → [0, 1]),
  `<node_id>/a_lower` and `<node_id>/a_upper` (in [0, 1]).

With `observe_current_step=True` the observation used to choose the action of step t holds g, l,
prices and bounds of step t; with False it holds g and l of step t - 1 (0 at the first row of the
data), the prices of step t, and an upper bound computed without the charge cap, so that it does
not reveal g of step t.
"""

import numpy as np
import pandas as pd
from gymnasium import spaces

from smartcer.core.community import Community
from smartcer.gym.observation.encoders import cyclical, min_max, one_hot


OBSERVATION_PRESETS = ('meco',)
# layout prefixes that node ids must not use
RESERVED_PREFIXES = ('time', 'price', 'forecast')


class ObservationBuilder:
    """
    Builds the flat observation of a community for a step.

    Attributes:
        preset: Observation preset.
        observe_current_step: Information timing (docs/FORMULATION.md section 12).
        layout: Feature name → slice of the observation vector.
        space: The observation space, `Box(low, high, (dim,), float32)`.
        global_prices: True if the prices are one global pair.
    """

    def __init__(self, community: Community, preset: str = 'meco',
                 observe_current_step: bool = True) -> None:
        """
        Precompute the step-independent features of every step.

        Args:
            community: The community (its data define the scaling).
            preset: Observation preset; only 'meco' exists (the 'extended' preset comes later).
            observe_current_step: Information timing.

        Raises:
            ValueError: If the preset is unknown or a node id is a reserved layout prefix.
        """
        if preset not in OBSERVATION_PRESETS:
            raise ValueError(f'unknown observation preset {preset!r}; available: '
                             f'{OBSERVATION_PRESETS}')
        reserved = [i for i in community.node_ids if i in RESERVED_PREFIXES]
        if reserved:
            raise ValueError(f'node ids {reserved} are reserved observation prefixes '
                             f'{RESERVED_PREFIXES}')
        self.preset = preset
        self.observe_current_step = observe_current_step
        self._community = community
        self._battery_nodes = [community.node(i) for i in community.storage_node_ids]

        columns: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        self.layout: dict[str, slice] = {}

        def add(name: str, values: np.ndarray, low: float | np.ndarray,
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
            return self.layout[name]

        timestamps = pd.DatetimeIndex(community.timestamps)
        add('time/month', cyclical(timestamps.month - 1, 12), -1.0, 1.0)
        add('time/hour', cyclical(timestamps.hour + timestamps.minute / 60.0, 24), -1.0, 1.0)
        add('time/day_of_week', one_hot(timestamps.dayofweek, 7), 0.0, 1.0)

        sellers = [n for n in community.nodes if n.node_type != 'consumer']
        buyers = [n for n in community.nodes if n.node_type != 'producer']
        self.global_prices = (
            all(np.array_equal(n.sell_price, sellers[0].sell_price) for n in sellers)
            and all(np.array_equal(n.buy_price, buyers[0].buy_price) for n in buyers)
        ) if sellers and buyers else False
        if self.global_prices:
            for name, series in (('price/sell', sellers[0].sell_price),
                                 ('price/buy', buyers[0].buy_price)):
                add(name, series, series.min(), series.max())

        self._dynamic: list[tuple[int, int, int]] = []
        for node in self._battery_nodes:
            prefix = node.node_id
            if not self.global_prices:
                prices = (('sell_price', node.sell_price), ('buy_price', node.buy_price))
                for name, series in prices:
                    add(f'{prefix}/{name}', series, series.min(), series.max())
            for name, series in (('generation', node.generation_kwh), ('load', node.load_kwh)):
                observed = series if observe_current_step else np.concatenate([[0.0], series[:-1]])
                add(f'{prefix}/{name}', min_max(observed, series.min(), series.max()), 0.0, 1.0)
            # placeholders, filled at every step from the battery state
            soc = add(f'{prefix}/soc', np.zeros(community.n_steps), 0.0, 1.0)
            a_lower = add(f'{prefix}/a_lower', np.zeros(community.n_steps), 0.0, 1.0)
            a_upper = add(f'{prefix}/a_upper', np.zeros(community.n_steps), 0.0, 1.0)
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
