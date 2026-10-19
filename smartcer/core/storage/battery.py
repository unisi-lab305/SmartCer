"""
Battery model of docs/FORMULATION.md sections 4-7.

Dynamics of [S22] eq. 6-8 with both power limits on the node side: in charge the node supplies
p_cha · Δt and the cell stores η_cha · p_cha · Δt; in discharge the cell releases p_dis · Δt / η_dis
and the node receives p_dis · Δt. The dynamic action bounds are derived from these dynamics, so an
action inside them never drives the state of charge outside [S_min, S_max]. No degradation.

Ported from storage-node-environment (`storage_node_env/core/storage/battery.py`) and adapted:
discharge limit on the node side, bounds with η and S_min, charge cap passed by the node,
action-based API, storage usage cost, exceptions instead of assertions.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, ClassVar
import warnings

from smartcer.core.storage.base import StorageSystem


CHARGE_LIMITS = ('production', 'surplus')

# tolerance on the state of charge for floating-point rounding at the limits (kWh)
SOC_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class BatteryStep:
    """
    Record of one battery step (docs/FORMULATION.md sections 4-7 and 10).

    Attributes:
        action_requested: Action a requested by the agent, in [-1, 1].
        action_applied: Action after clipping to [-a_lower, a_upper].
        a_upper: Upper action bound A^upper of the step, in [0, 1].
        a_lower: Lower action bound A^lower of the step, in [0, 1].
        p_cha_kw: Charge power p^cha, node side (kW).
        p_dis_kw: Discharge power p^dis, node side (kW).
        charge_kwh: Energy supplied by the node to the battery, p^cha · Δt (kWh).
        discharge_kwh: Energy delivered by the battery to the node, p^dis · Δt (kWh).
        throughput_kwh: Energy entering or leaving the cell,
            η^cha · p^cha · Δt + p^dis · Δt / η^dis (kWh).
        storage_cost: Storage usage cost π^sto · throughput (€).
        penalty: Constraint penalty k of the action (section 7), ≥ 0.
        soc_kwh: State of charge at the end of the step (kWh).
    """

    action_requested: float
    action_applied: float
    a_upper: float
    a_lower: float
    p_cha_kw: float
    p_dis_kw: float
    charge_kwh: float
    discharge_kwh: float
    throughput_kwh: float
    storage_cost: float
    penalty: float
    soc_kwh: float


class Battery(StorageSystem):
    """
    Battery with node-side power limits, efficiencies, minimum state of charge and self-discharge.

    The state of charge s is kept in kWh; `soc_min` and `soc_initial` are given as fractions of the
    capacity. Each step: self-discharge s̃ = max(S_min, (1 - α) · s), bounds from s̃, clipping of
    the action, update s = s̃ + η_cha · p_cha · Δt - p_dis · Δt / η_dis.

    Attributes:
        capacity: Capacity S^max (kWh).
        power_charge_max: Maximum charge power P^cha, node side (kW).
        power_discharge_max: Maximum discharge power P^dis, node side (kW).
        efficiency_charge: Charge efficiency η^cha, in (0, 1].
        efficiency_discharge: Discharge efficiency η^dis, in (0, 1].
        soc_min: Minimum state of charge as a fraction of capacity, in [0, 1).
        soc_initial: Initial state of charge as a fraction of capacity, in [soc_min, 1].
        alpha: Self-discharge per step α, in [0, 1).
        usage_cost: Unit cost of storage usage π^sto (€/kWh of cell throughput), ≥ 0.
        charge_limit: Charge rule used by the node to compute the charge cap: 'production'
            (c = g) or 'surplus' (c = max(0, g - l)); the battery only stores it.
        soc_min_kwh: Minimum state of charge S^min (kWh).
        soc_initial_kwh: Initial state of charge S^ini (kWh).
        soc_kwh: Current state of charge s (kWh).

    Example:
        >>> battery = Battery(capacity=10.0, power_charge_max=5.0, power_discharge_max=5.0,
        ...                   efficiency_charge=0.95, efficiency_discharge=0.95, soc_initial=0.5)
        >>> result = battery.step(action=1.0, delta_t=1.0, charge_cap_kwh=None)
        >>> round(result.soc_kwh, 2), round(result.charge_kwh, 2)
        (9.75, 5.0)
    """

    CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({
        'capacity', 'power_charge_max', 'power_discharge_max', 'efficiency_charge',
        'efficiency_discharge', 'soc_min', 'soc_initial', 'alpha', 'usage_cost', 'charge_limit',
    })
    # 0.6.2 key → 1.0 keys it maps to (docs/refactor/REFACTOR_v1.md section 6)
    LEGACY_KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        'rated_power': ('power_charge_max', 'power_discharge_max'),
        'efficiency': ('efficiency_charge', 'efficiency_discharge'),
        'alfa': ('alpha',),
        'beta': (),
    }

    def __init__(
        self,
        capacity: float,
        power_charge_max: float,
        power_discharge_max: float,
        efficiency_charge: float,
        efficiency_discharge: float,
        soc_min: float = 0.0,
        soc_initial: float | None = None,
        alpha: float = 0.0,
        usage_cost: float = 0.0,
        charge_limit: str = 'production',
    ) -> None:
        """
        Create a battery and set its state of charge to the initial value.

        Args:
            capacity: Capacity S^max (kWh), > 0.
            power_charge_max: Maximum charge power P^cha, node side (kW), > 0.
            power_discharge_max: Maximum discharge power P^dis, node side (kW), > 0.
            efficiency_charge: Charge efficiency η^cha, in (0, 1].
            efficiency_discharge: Discharge efficiency η^dis, in (0, 1].
            soc_min: Minimum state of charge as a fraction of capacity, in [0, 1). Defaults to 0.
            soc_initial: Initial state of charge as a fraction of capacity, in [soc_min, 1].
                Defaults to soc_min.
            alpha: Self-discharge per step, in [0, 1). Defaults to 0.
            usage_cost: Unit cost of storage usage (€/kWh of cell throughput), ≥ 0. Defaults to 0.
            charge_limit: 'production' (default) or 'surplus', see docs/FORMULATION.md section 6.

        Raises:
            ValueError: If a parameter is not finite or outside its range.
        """
        _check_number('capacity', capacity, low=0.0, low_open=True)
        _check_number('power_charge_max', power_charge_max, low=0.0, low_open=True)
        _check_number('power_discharge_max', power_discharge_max, low=0.0, low_open=True)
        _check_number('efficiency_charge', efficiency_charge, low=0.0, low_open=True, high=1.0)
        _check_number('efficiency_discharge', efficiency_discharge, low=0.0, low_open=True,
                      high=1.0)
        _check_number('soc_min', soc_min, low=0.0, high=1.0, high_open=True)
        if soc_initial is None:
            soc_initial = soc_min
        _check_number('soc_initial', soc_initial, low=soc_min, high=1.0)
        _check_number('alpha', alpha, low=0.0, high=1.0, high_open=True)
        _check_number('usage_cost', usage_cost, low=0.0)
        if charge_limit not in CHARGE_LIMITS:
            raise ValueError(
                f'charge_limit must be one of {CHARGE_LIMITS}, got {charge_limit!r}'
            )

        self.capacity = float(capacity)
        self.power_charge_max = float(power_charge_max)
        self.power_discharge_max = float(power_discharge_max)
        self.efficiency_charge = float(efficiency_charge)
        self.efficiency_discharge = float(efficiency_discharge)
        self.soc_min = float(soc_min)
        self.soc_initial = float(soc_initial)
        self.alpha = float(alpha)
        self.usage_cost = float(usage_cost)
        self.charge_limit = charge_limit

        self.soc_min_kwh = self.soc_min * self.capacity
        self.soc_initial_kwh = self.soc_initial * self.capacity
        self.soc_kwh = self.soc_initial_kwh

    def __repr__(self) -> str:
        return (
            f'Battery(capacity={self.capacity:g} kWh, power_charge_max={self.power_charge_max:g} '
            f'kW, power_discharge_max={self.power_discharge_max:g} kW, '
            f'efficiency_charge={self.efficiency_charge:g}, '
            f'efficiency_discharge={self.efficiency_discharge:g}, soc_min={self.soc_min:g}, '
            f'alpha={self.alpha:g}, usage_cost={self.usage_cost:g}, '
            f'charge_limit={self.charge_limit!r}, soc_kwh={self.soc_kwh:g})'
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'Battery':
        """
        Create a battery from a scenario `storage` section, in the 1.0 or the 0.6.2 format.

        1.0 keys: see `CONFIG_KEYS` (docs/FORMULATION.md section 3). 0.6.2 keys are mapped with a
        DeprecationWarning: `rated_power` → both power limits, `efficiency` → both efficiencies,
        `alfa` → `alpha`; `beta` (degradation) is read and ignored (decision D7); `soc_min` is a
        fraction in both formats.

        Args:
            config: Storage section of a node.

        Returns:
            Battery: The battery.

        Raises:
            ValueError: On unknown keys, on a 0.6.2 key combined with the 1.0 key it maps to, on a
                missing required key, or on invalid values.
        """
        unknown = sorted(set(config) - cls.CONFIG_KEYS - set(cls.LEGACY_KEYS))
        if unknown:
            raise ValueError(
                f'unknown storage keys {unknown}; valid keys: {sorted(cls.CONFIG_KEYS)} '
                f'(0.6.2: {sorted(cls.LEGACY_KEYS)})'
            )

        params = {key: value for key, value in config.items() if key in cls.CONFIG_KEYS}
        legacy = [key for key in cls.LEGACY_KEYS if key in config]
        for key in legacy:
            for target in cls.LEGACY_KEYS[key]:
                if target in params:
                    raise ValueError(
                        f'storage key {key!r} (0.6.2) conflicts with {target!r} (1.0): use only '
                        f'the 1.0 key'
                    )
                params[target] = config[key]
        if legacy:
            mapped = ', '.join(
                f'{key} → {" and ".join(cls.LEGACY_KEYS[key])}' if cls.LEGACY_KEYS[key]
                else f'{key} ignored (degradation is not modelled)'
                for key in legacy
            )
            warnings.warn(
                f'0.6.2 storage keys are deprecated: {mapped}', DeprecationWarning, stacklevel=2
            )

        required = ('capacity', 'power_charge_max', 'power_discharge_max', 'efficiency_charge',
                    'efficiency_discharge')
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f'missing storage keys {missing}')
        return cls(**params)

    def _soc_after_self_discharge(self) -> float:
        """
        State of charge after self-discharge, s̃ = max(S_min, (1 - α) · s).

        Returns:
            float: s̃ (kWh).
        """
        if self.alpha == 0.0:
            return self.soc_kwh
        return max(self.soc_min_kwh, (1.0 - self.alpha) * self.soc_kwh)

    def _bounds(self, soc_kwh: float, delta_t: float,
                charge_cap_kwh: float | None) -> tuple[float, float]:
        """
        Action bounds of docs/FORMULATION.md section 6 from a given state of charge.

        Args:
            soc_kwh: State of charge s̃ after self-discharge (kWh).
            delta_t: Duration of the step (hours).
            charge_cap_kwh: Charge cap c (kWh), or None for no cap.

        Returns:
            tuple[float, float]: (a_upper, a_lower), both in [0, 1].
        """
        p_upper = min(self.power_charge_max,
                      (self.capacity - soc_kwh) / (self.efficiency_charge * delta_t))
        if charge_cap_kwh is not None:
            p_upper = min(p_upper, charge_cap_kwh / delta_t)
        p_lower = min(self.power_discharge_max,
                      self.efficiency_discharge * (soc_kwh - self.soc_min_kwh) / delta_t)
        return (max(0.0, p_upper / self.power_charge_max),
                max(0.0, p_lower / self.power_discharge_max))

    def get_bounds(self, delta_t: float, charge_cap_kwh: float | None) -> tuple[float, float]:
        """
        Dynamic action bounds A^upper, A^lower for the coming step (docs/FORMULATION.md section 6).

        Computed from the current state of charge after self-discharge; the state is not changed.

        Args:
            delta_t: Duration of the step (hours), > 0.
            charge_cap_kwh: Maximum energy the node can supply for charging in the step (kWh),
                ≥ 0; None means no cap.

        Returns:
            tuple[float, float]: (a_upper, a_lower), both in [0, 1]; feasible actions lie in
            [-a_lower, a_upper].

        Raises:
            ValueError: If delta_t or charge_cap_kwh is invalid.
        """
        _check_step_inputs(delta_t, charge_cap_kwh)
        return self._bounds(self._soc_after_self_discharge(), delta_t, charge_cap_kwh)

    def step(self, action: float, delta_t: float, charge_cap_kwh: float | None) -> BatteryStep:
        """
        Clip the action to its bounds, apply it for one step and update the state of charge.

        Args:
            action: Requested normalised action in [-1, 1]; positive charges, negative discharges.
            delta_t: Duration of the step (hours), > 0.
            charge_cap_kwh: Maximum energy the node can supply for charging in the step (kWh),
                ≥ 0; None means no cap.

        Returns:
            BatteryStep: Requested and applied action, bounds, powers, energies, cost, penalty and
            the new state of charge.

        Raises:
            ValueError: If the action is not finite or outside [-1, 1], or if delta_t or
                charge_cap_kwh is invalid.
            RuntimeError: If the new state of charge leaves [S_min, S_max] by more than the
                rounding tolerance (an implementation error).
        """
        action = float(action)
        if not (math.isfinite(action) and -1.0 <= action <= 1.0):
            raise ValueError(f'action must be a finite number in [-1, 1], got {action}')
        _check_step_inputs(delta_t, charge_cap_kwh)

        soc_start = self._soc_after_self_discharge()
        a_upper, a_lower = self._bounds(soc_start, delta_t, charge_cap_kwh)

        # clipping and penalty (section 7)
        if action > 0.0:
            applied = min(action, a_upper)
            penalty = action - applied
        elif action < 0.0:
            applied = max(action, -a_lower)
            penalty = applied - action
        else:
            applied, penalty = 0.0, 0.0

        p_cha = max(0.0, applied) * self.power_charge_max
        p_dis = max(0.0, -applied) * self.power_discharge_max
        stored = self.efficiency_charge * p_cha * delta_t
        released = p_dis * delta_t / self.efficiency_discharge
        soc_end = soc_start + stored - released

        if not (self.soc_min_kwh - SOC_TOLERANCE <= soc_end <= self.capacity + SOC_TOLERANCE):
            raise RuntimeError(
                f'state of charge {soc_end!r} kWh outside [{self.soc_min_kwh}, {self.capacity}] '
                f'kWh after action {action!r} (applied {applied!r}) from {soc_start!r} kWh'
            )
        self.soc_kwh = min(self.capacity, max(self.soc_min_kwh, soc_end))

        throughput = stored + released
        return BatteryStep(
            action_requested=action,
            action_applied=applied,
            a_upper=a_upper,
            a_lower=a_lower,
            p_cha_kw=p_cha,
            p_dis_kw=p_dis,
            charge_kwh=p_cha * delta_t,
            discharge_kwh=p_dis * delta_t,
            throughput_kwh=throughput,
            storage_cost=self.usage_cost * throughput,
            penalty=penalty,
            soc_kwh=self.soc_kwh,
        )

    def reset(self) -> None:
        """
        Restore the initial state of charge S^ini.
        """
        self.soc_kwh = self.soc_initial_kwh


def _check_number(name: str, value: Any, low: float | None = None, high: float | None = None,
                  low_open: bool = False, high_open: bool = False) -> None:
    """
    Check that a parameter is a finite real number inside an interval.

    Args:
        name: Parameter name, used in the error message.
        value: Value to check.
        low: Lower bound, or None.
        high: Upper bound, or None.
        low_open: True if the lower bound is excluded.
        high_open: True if the upper bound is excluded.

    Raises:
        ValueError: If the value is not a finite real number or lies outside the interval.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    below = low is not None and (value <= low if low_open else value < low)
    above = high is not None and (value >= high if high_open else value > high)
    if below or above:
        left = '(' if low_open else '['
        right = ')' if high_open else ']'
        low_text = '-inf' if low is None else low
        high_text = 'inf' if high is None else high
        raise ValueError(f'{name} must lie in {left}{low_text}, {high_text}{right}, got {value!r}')


def _check_step_inputs(delta_t: float, charge_cap_kwh: float | None) -> None:
    """
    Check the time step and the charge cap of a step.

    Args:
        delta_t: Duration of the step (hours).
        charge_cap_kwh: Charge cap (kWh), or None.

    Raises:
        ValueError: If delta_t is not a finite positive number or the cap is negative or not finite.
    """
    if not (math.isfinite(delta_t) and delta_t > 0.0):
        raise ValueError(f'delta_t must be a finite positive number of hours, got {delta_t!r}')
    if charge_cap_kwh is not None and not (math.isfinite(charge_cap_kwh) and charge_cap_kwh >= 0.0):
        raise ValueError(
            f'charge_cap_kwh must be None or a finite number ≥ 0, got {charge_cap_kwh!r}'
        )
