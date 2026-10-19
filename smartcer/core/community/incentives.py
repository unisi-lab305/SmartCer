"""
Incentive on shared energy (docs/FORMULATION.md section 9): models `uniform` and `tip`.

The pure functions compute one value; the model classes hold the per-step data of a scenario
(unit values or per-plant TIP coefficients) and are used by `smartcer.core.community.Community`.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


# zonal price below which the TIP variable part grows (€/kWh)
TIP_PRICE_REFERENCE = 0.180
# maximum PNRR capital grant (%)
PNRR_MAX = 40.0


def tip_coefficient(
    zonal_price: ArrayLike,
    tip_base: float,
    tip_threshold: float,
    fi_zone: float,
    pnrr: float,
) -> float | np.ndarray:
    """
    Per-plant TIP coefficient
    c = (min(TIP_th, TIP_base + max(0, 0.180 - P^z)) + f_zone) · (1 - 0.5 · PNRR / 40).

    Args:
        zonal_price: Zonal price P^z (€/kWh), a scalar or one value per step; SmartCer uses the
            node's sell price.
        tip_base: Base TIP (€/kWh).
        tip_threshold: Maximum of the base plus variable part (€/kWh).
        fi_zone: Geographic correction (€/kWh).
        pnrr: Capital grant (%), in [0, 40].

    Returns:
        float | np.ndarray: Coefficient (€/kWh), a float for a scalar zonal price.

    Raises:
        ValueError: If pnrr is outside [0, 40].
    """
    if not 0.0 <= pnrr <= PNRR_MAX:
        raise ValueError(f'pnrr must lie in [0, {PNRR_MAX:g}] %, got {pnrr!r}')
    variable = np.maximum(0.0, TIP_PRICE_REFERENCE - np.asarray(zonal_price, dtype=np.float64))
    coefficient = (np.minimum(tip_threshold, tip_base + variable) + fi_zone) * (
        1.0 - 0.5 * pnrr / PNRR_MAX)
    return float(coefficient) if np.ndim(coefficient) == 0 else coefficient


def uniform_incentive(shared_kwh: float, unit_value: float) -> float:
    """
    Incentive of the `uniform` model, I = π^inc · E^sh.

    Args:
        shared_kwh: Shared energy (kWh).
        unit_value: Unit incentive π^inc (€/kWh).

    Returns:
        float: Incentive (€).
    """
    return unit_value * shared_kwh


def tip_incentive(shared_kwh: float, export_kwh: ArrayLike, coefficients: ArrayLike) -> float:
    """
    Incentive of the `tip` model, I = E^sh / Σ_u e^gri_u · Σ_u c_u e^gri_u (0 if nothing is
    exported).

    The fraction of exported energy that is shared is computed over all exporting nodes; nodes
    without a photovoltaic section have c_u = 0.

    Args:
        shared_kwh: Shared energy (kWh).
        export_kwh: Energy fed into the grid by each node (kWh).
        coefficients: TIP coefficient of each node (€/kWh), 0 for nodes without a plant.

    Returns:
        float: Incentive (€).
    """
    export = np.asarray(export_kwh, dtype=np.float64)
    total_export = float(export.sum())
    if total_export <= 0.0:
        return 0.0
    return shared_kwh / total_export * float(np.dot(coefficients, export))


class IncentiveModel(ABC):
    """
    Incentive model of a scenario: the incentive of each step from shared and exported energy.

    Attributes:
        model: Model name, 'uniform' or 'tip'.
        n_steps: Number of steps of the per-step data.
    """

    model: str
    n_steps: int

    @abstractmethod
    def compute(self, t: int, shared_kwh: float, export_kwh: np.ndarray) -> float:
        """
        Incentive of step t.

        Args:
            t: Step index.
            shared_kwh: Shared energy of the step (kWh).
            export_kwh: Energy fed into the grid by each node, in the community's node order (kWh).

        Returns:
            float: Incentive (€).
        """


class UniformIncentive(IncentiveModel):
    """
    `uniform` model with a constant or per-step unit value π^inc_t.

    Attributes:
        unit_value: π^inc per step (€/kWh), shape (n_steps,).
    """

    model = 'uniform'

    def __init__(self, unit_value: float | ArrayLike, n_steps: int) -> None:
        """
        Create the model.

        Args:
            unit_value: π^inc (€/kWh), a constant or one finite value per step.
            n_steps: Number of steps.

        Raises:
            ValueError: If the per-step values do not have n_steps elements or are not finite.
        """
        values = np.asarray(unit_value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(n_steps, float(values))
        if values.shape != (n_steps,):
            raise ValueError(f'unit_value must have {n_steps} steps, got shape {values.shape}')
        if not np.all(np.isfinite(values)):
            raise ValueError('unit_value must be finite')
        self.n_steps = n_steps
        self.unit_value = values
        self._values = values.tolist()

    def __repr__(self) -> str:
        return f'UniformIncentive({self.n_steps} steps)'

    def compute(self, t: int, shared_kwh: float, export_kwh: np.ndarray) -> float:
        """
        Incentive of step t, π^inc_t · E^sh_t.

        Args:
            t: Step index.
            shared_kwh: Shared energy of the step (kWh).
            export_kwh: Energy fed into the grid by each node (kWh); not used.

        Returns:
            float: Incentive (€).
        """
        return uniform_incentive(shared_kwh, self._values[t])


class TipIncentive(IncentiveModel):
    """
    `tip` model (the 0.6.2 per-plant model) with per-step, per-node coefficients.

    Attributes:
        coefficients: TIP coefficient per step and node (€/kWh), shape (n_steps, n_nodes), 0 for
            nodes without a plant.
    """

    model = 'tip'

    def __init__(self, coefficients: ArrayLike) -> None:
        """
        Create the model.

        Args:
            coefficients: Array of shape (n_steps, n_nodes) in the community's node order.

        Raises:
            ValueError: If the array is not 2-D or not finite.
        """
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f'coefficients must have shape (steps, nodes), got {values.shape}')
        if not np.all(np.isfinite(values)):
            raise ValueError('coefficients must be finite')
        self.n_steps = values.shape[0]
        self.coefficients = values

    def __repr__(self) -> str:
        return f'TipIncentive({self.n_steps} steps, {self.coefficients.shape[1]} nodes)'

    def potential(self, t: int, export_kwh: np.ndarray) -> float:
        """
        Potential TIP of step t, Σ_u c_u e^gri_u (the 0.6.2 `tip_potential`).

        Args:
            t: Step index.
            export_kwh: Energy fed into the grid by each node (kWh).

        Returns:
            float: Potential TIP (€).
        """
        return float(np.dot(self.coefficients[t], export_kwh))

    def compute(self, t: int, shared_kwh: float, export_kwh: np.ndarray) -> float:
        """
        Incentive of step t, E^sh / Σ e^gri · Σ c e^gri.

        Args:
            t: Step index.
            shared_kwh: Shared energy of the step (kWh).
            export_kwh: Energy fed into the grid by each node (kWh).

        Returns:
            float: Incentive (€).
        """
        return tip_incentive(shared_kwh, export_kwh, self.coefficients[t])
