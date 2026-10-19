"""
Community node of docs/FORMULATION.md sections 2, 6 and 8.

A node is a producer (generation only), a consumer (load only) or a prosumer (both), with an
optional battery. Its data are per-step energies in kWh and prices in €/kWh, preloaded as arrays by
`smartcer.data.loader`; `step(t, action)` applies the battery action of step t and returns the
node's flows and economics.

Adapted from storage-node-environment (`storage_node_env/core/nodes/base.py`): optional battery,
one class for the three node types, data passed as arrays (the CSV reading and its time-step
validation live in `smartcer.data.loader`), no `print`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from smartcer.core.storage import Battery


NODE_TYPES = ('producer', 'consumer', 'prosumer')


@dataclass(frozen=True, slots=True)
class NodeStep:
    """
    Record of one node step (docs/FORMULATION.md sections 4-8 and 10).

    For a node without a battery the battery fields are zero and `soc_kwh` is None.

    Attributes:
        node_id: Node identifier.
        t: Step index.
        generation_kwh: Generation g (kWh).
        load_kwh: Load l (kWh).
        p_cha_kw: Charge power p^cha, node side (kW).
        p_dis_kw: Discharge power p^dis, node side (kW).
        charge_kwh: Energy supplied by the node to its battery, p^cha · Δt (kWh).
        discharge_kwh: Energy delivered by the battery to the node, p^dis · Δt (kWh).
        net_kwh: Net energy exchanged with the grid n = g - l + p^dis Δt - p^cha Δt (kWh).
        export_kwh: Energy fed into the grid e^gri = max(0, n) (kWh).
        import_kwh: Energy withdrawn from the grid i^gri = max(0, -n) (kWh).
        revenue: π^egr · e^gri (€).
        cost: π^igr · i^gri (€).
        throughput_kwh: Energy entering or leaving the cell (kWh).
        storage_cost: π^sto · throughput (€).
        penalty: Constraint penalty k of the battery action (section 7).
        soc_kwh: State of charge at the end of the step (kWh), None without a battery.
        action_requested: Requested battery action.
        action_applied: Battery action after clipping.
        a_upper: Upper action bound A^upper of the step.
        a_lower: Lower action bound A^lower of the step.
    """

    node_id: str
    t: int
    generation_kwh: float
    load_kwh: float
    p_cha_kw: float
    p_dis_kw: float
    charge_kwh: float
    discharge_kwh: float
    net_kwh: float
    export_kwh: float
    import_kwh: float
    revenue: float
    cost: float
    throughput_kwh: float
    storage_cost: float
    penalty: float
    soc_kwh: float | None
    action_requested: float
    action_applied: float
    a_upper: float
    a_lower: float


class Node:
    """
    Producer, consumer or prosumer with an optional battery.

    Attributes:
        node_id: Node identifier.
        node_type: 'producer', 'consumer' or 'prosumer'.
        delta_t: Duration of a step (hours).
        generation_kwh: Generation per step g (kWh), zero for a consumer.
        load_kwh: Load per step l (kWh), zero for a producer.
        sell_price: Sell price π^egr per step (€/kWh); zero for a consumer without the column.
        buy_price: Buy price π^igr per step (€/kWh); zero for a producer.
        battery: The node's battery, or None.
        photovoltaic: TIP parameters of the node's plant (`tip_base`, `tip_threshold`, `fi_zone`,
            `pnrr`, ...), or None.
        category: Free-text category from the scenario (0.6.2 `Type`, e.g. 'factory'), or None.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        delta_t: float,
        generation_kwh: np.ndarray,
        load_kwh: np.ndarray,
        sell_price: np.ndarray,
        buy_price: np.ndarray,
        battery: Battery | None = None,
        photovoltaic: dict[str, Any] | None = None,
        category: str | None = None,
    ) -> None:
        """
        Create a node from per-step arrays.

        Args:
            node_id: Node identifier.
            node_type: 'producer', 'consumer' or 'prosumer'.
            delta_t: Duration of a step (hours), > 0.
            generation_kwh: Generation per step (kWh), finite and ≥ 0; all zero for a consumer.
            load_kwh: Load per step (kWh), finite and ≥ 0; all zero for a producer.
            sell_price: Sell price per step (€/kWh), finite.
            buy_price: Buy price per step (€/kWh), finite.
            battery: Optional battery.
            photovoltaic: Optional TIP parameters of the node's plant.
            category: Optional free-text category.

        Raises:
            ValueError: If the type is unknown, delta_t is not positive, the arrays are not 1-D of
                equal length, or contain non-finite or negative energies, or generation (load) is
                non-zero for a consumer (producer).
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f'node {node_id!r}: type must be one of {NODE_TYPES}, got '
                             f'{node_type!r}')
        if not (np.isfinite(delta_t) and delta_t > 0):
            raise ValueError(f'node {node_id!r}: delta_t must be a positive number of hours, got '
                             f'{delta_t!r}')
        arrays = {
            'generation_kwh': generation_kwh, 'load_kwh': load_kwh,
            'sell_price': sell_price, 'buy_price': buy_price,
        }
        converted = {}
        for name, values in arrays.items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or len(array) == 0:
                raise ValueError(f'node {node_id!r}: {name} must be a non-empty 1-D array')
            bad = np.flatnonzero(~np.isfinite(array))
            if bad.size:
                raise ValueError(f'node {node_id!r}: {name} is not finite at step {bad[0]}')
            converted[name] = array
        n_steps = len(converted['generation_kwh'])
        for name, array in converted.items():
            if len(array) != n_steps:
                raise ValueError(f'node {node_id!r}: {name} has {len(array)} steps, '
                                 f'generation_kwh has {n_steps}')
        for name in ('generation_kwh', 'load_kwh'):
            negative = np.flatnonzero(converted[name] < 0)
            if negative.size:
                raise ValueError(f'node {node_id!r}: {name} is negative at step {negative[0]} '
                                 f'({converted[name][negative[0]]!r})')
        if node_type == 'consumer' and np.any(converted['generation_kwh'] != 0):
            raise ValueError(f'node {node_id!r}: a consumer has no generation')
        if node_type == 'producer' and np.any(converted['load_kwh'] != 0):
            raise ValueError(f'node {node_id!r}: a producer has no load')

        self.node_id = node_id
        self.node_type = node_type
        self.delta_t = float(delta_t)
        self.generation_kwh = converted['generation_kwh']
        self.load_kwh = converted['load_kwh']
        self.sell_price = converted['sell_price']
        self.buy_price = converted['buy_price']
        self.battery = battery
        self.photovoltaic = photovoltaic
        self.category = category

        # python floats for fast scalar access in step()
        self._g = self.generation_kwh.tolist()
        self._l = self.load_kwh.tolist()
        self._sell = self.sell_price.tolist()
        self._buy = self.buy_price.tolist()

    def __repr__(self) -> str:
        battery = 'battery' if self.battery else 'no battery'
        return f'Node({self.node_id!r}, {self.node_type}, {self.n_steps} steps, {battery})'

    @property
    def n_steps(self) -> int:
        """Number of steps of the node's data."""
        return len(self._g)

    def _check_t(self, t: int) -> None:
        """
        Check a step index.

        Args:
            t: Step index.

        Raises:
            IndexError: If t is outside [0, n_steps).
        """
        if not 0 <= t < len(self._g):
            raise IndexError(f'node {self.node_id!r}: step {t} outside [0, {len(self._g)})')

    def charge_cap(self, t: int) -> float:
        """
        Charge cap c of step t (docs/FORMULATION.md section 6, decision D6).

        `production`: c = g; `surplus`: c = max(0, g - l).

        Args:
            t: Step index.

        Returns:
            float: c (kWh); 0 for a node without a battery.
        """
        self._check_t(t)
        if self.battery is None:
            return 0.0
        if self.battery.charge_limit == 'surplus':
            return max(0.0, self._g[t] - self._l[t])
        return self._g[t]

    def get_bounds(self, t: int) -> tuple[float, float]:
        """
        Dynamic action bounds of step t for the current state of charge.

        Args:
            t: Step index.

        Returns:
            tuple[float, float]: (a_upper, a_lower) in [0, 1]; (0, 0) without a battery.
        """
        if self.battery is None:
            self._check_t(t)
            return 0.0, 0.0
        return self.battery.get_bounds(self.delta_t, self.charge_cap(t))

    def step(self, t: int, action: float | None = None) -> NodeStep:
        """
        Apply the battery action of step t and compute the node's flows and economics.

        Args:
            t: Step index.
            action: Battery action in [-1, 1]; None (or 0) leaves the battery idle. Must be None or
                0 for a node without a battery.

        Returns:
            NodeStep: Flows, economics and battery record of the step.

        Raises:
            IndexError: If t is outside [0, n_steps).
            ValueError: If a node without a battery receives a non-zero action, or the battery
                rejects the action.
        """
        self._check_t(t)
        generation, load = self._g[t], self._l[t]
        if self.battery is None:
            if action:
                raise ValueError(f'node {self.node_id!r} has no battery but received action '
                                 f'{action!r}')
            p_cha = p_dis = charge = discharge = throughput = storage_cost = penalty = 0.0
            requested = applied = a_upper = a_lower = 0.0
            soc = None
        else:
            result = self.battery.step(0.0 if action is None else action, self.delta_t,
                                       self.charge_cap(t))
            p_cha, p_dis = result.p_cha_kw, result.p_dis_kw
            charge, discharge = result.charge_kwh, result.discharge_kwh
            throughput, storage_cost = result.throughput_kwh, result.storage_cost
            penalty, soc = result.penalty, result.soc_kwh
            requested, applied = result.action_requested, result.action_applied
            a_upper, a_lower = result.a_upper, result.a_lower

        net = generation - load + discharge - charge
        export = net if net > 0.0 else 0.0
        import_ = -net if net < 0.0 else 0.0
        return NodeStep(
            node_id=self.node_id, t=t, generation_kwh=generation, load_kwh=load, p_cha_kw=p_cha,
            p_dis_kw=p_dis, charge_kwh=charge, discharge_kwh=discharge, net_kwh=net,
            export_kwh=export, import_kwh=import_, revenue=self._sell[t] * export,
            cost=self._buy[t] * import_, throughput_kwh=throughput, storage_cost=storage_cost,
            penalty=penalty, soc_kwh=soc, action_requested=requested, action_applied=applied,
            a_upper=a_upper, a_lower=a_lower,
        )

    def reset(self) -> None:
        """
        Restore the battery's initial state of charge (the data are not reloaded).
        """
        if self.battery is not None:
            self.battery.reset()
