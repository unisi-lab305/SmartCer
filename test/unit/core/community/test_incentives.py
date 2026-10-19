"""
Unit tests of `smartcer.core.community.incentives` (docs/FORMULATION.md section 9).

The TIP coefficient cases are those of the 0.6.2 characterisation tests
(`test/unit/v062/test_tip_coefficient.py`), re-pointed to the new function.
"""

import numpy as np
import pytest

from smartcer.core.community import (
    TipIncentive,
    UniformIncentive,
    tip_coefficient,
    tip_incentive,
    uniform_incentive,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ('tip_threshold', 'tip_base', 'zonal_price', 'fi_zone', 'pnrr', 'expected'),
    [
        # pnrr 0, zonal price below 0.180, threshold binding: min(0.12, 0.16) + 0.01
        (0.12, 0.08, 0.10, 0.01, 0.0, 0.13),
        # pnrr 30, zonal price above 0.180, threshold not binding: (0.08 + 0.004) * 0.625
        (0.12, 0.08, 0.20, 0.004, 30.0, 0.0525),
        # pnrr 40, zonal price below 0.180, threshold not binding: (min(0.12, 0.11) + 0.01) * 0.5
        (0.12, 0.08, 0.15, 0.01, 40.0, 0.06),
        # pnrr 40, zonal price 0, threshold binding: (min(0.11, 0.25) + 0) * 0.5
        (0.11, 0.07, 0.0, 0.0, 40.0, 0.055),
        # pnrr 30, zonal price exactly 0.180: (0.07 + 0.004) * 0.625
        (0.11, 0.07, 0.18, 0.004, 30.0, 0.04625),
    ],
)
def test_tip_coefficient_hand_cases(tip_threshold: float, tip_base: float, zonal_price: float,
                                    fi_zone: float, pnrr: float, expected: float) -> None:
    """
    Test the TIP coefficient on hand-computed values.

    Args:
        tip_threshold: TIP threshold (€/kWh).
        tip_base: Base TIP (€/kWh).
        zonal_price: Zonal price (€/kWh).
        fi_zone: Zonal correction (€/kWh).
        pnrr: Capital grant (%).
        expected: Hand-computed coefficient (€/kWh).
    """
    coefficient = tip_coefficient(zonal_price, tip_base=tip_base, tip_threshold=tip_threshold,
                                  fi_zone=fi_zone, pnrr=pnrr)

    assert isinstance(coefficient, float)
    assert coefficient == pytest.approx(expected, abs=1e-12)


def test_tip_coefficient_is_vectorised() -> None:
    """
    Test one coefficient per step for an array of zonal prices.
    """
    coefficients = tip_coefficient(np.array([0.10, 0.15, 0.20]), tip_base=0.08,
                                   tip_threshold=0.12, fi_zone=0.01, pnrr=0.0)

    assert coefficients == pytest.approx([0.13, 0.12, 0.09])


@pytest.mark.parametrize('pnrr', [-1.0, 41.0])
def test_tip_coefficient_rejects_pnrr_outside_range(pnrr: float) -> None:
    """
    Test that a capital grant outside [0, 40] % raises ValueError.

    Args:
        pnrr: Invalid capital grant (%).
    """
    with pytest.raises(ValueError, match='pnrr'):
        tip_coefficient(0.1, tip_base=0.08, tip_threshold=0.12, fi_zone=0.01, pnrr=pnrr)


def test_incentive_functions() -> None:
    """
    Test I = π E^sh and I = E^sh / Σ e · Σ c e on the mini_cer step at 10:00 of day 1:
    exports 10, 0, 10, 0 kWh, coefficients 0.13, 0, 0.065, 0.0975, shared 3 kWh → 0.2925 €.
    """
    export = np.array([10.0, 0.0, 10.0, 0.0])
    coefficients = np.array([0.13, 0.0, 0.065, 0.0975])

    assert uniform_incentive(3.0, 0.1) == pytest.approx(0.3)
    assert tip_incentive(3.0, export, coefficients) == pytest.approx(0.2925)
    assert tip_incentive(0.0, np.zeros(4), coefficients) == 0.0


def test_models_coincide_with_equal_coefficients() -> None:
    """
    Test that with c_u = π^inc for every node the tip and uniform models give the same incentive.
    """
    export = np.array([4.0, 1.0, 0.0])
    tip = TipIncentive(np.full((2, 3), 0.11))
    uniform = UniformIncentive(0.11, n_steps=2)

    assert tip.compute(1, 2.5, export) == pytest.approx(uniform.compute(1, 2.5, export))
    assert tip.potential(1, export) == pytest.approx(0.11 * 5.0)


def test_uniform_model_per_step_values_and_validation() -> None:
    """
    Test a per-step unit value and the shape validation of the models.
    """
    model = UniformIncentive([0.1, 0.2], n_steps=2)

    assert model.compute(1, 3.0, np.zeros(1)) == pytest.approx(0.6)
    with pytest.raises(ValueError, match='3 steps'):
        UniformIncentive([0.1, 0.2], n_steps=3)
    with pytest.raises(ValueError, match='shape'):
        TipIncentive(np.zeros(3))


def test_tip_export_without_plant_lowers_the_average_coefficient() -> None:
    """
    Test the 0.6.2 reading of FORMULATION section 9: 4 kWh exported by a plant (c = 0.1) and
    4 kWh by a consumer battery (c = 0), shared 2 kWh → I = 2 / 8 · 0.4 = 0.1 € instead of 0.2 €.
    """
    with_battery_export = tip_incentive(2.0, np.array([4.0, 4.0]), np.array([0.1, 0.0]))
    plant_only = tip_incentive(2.0, np.array([4.0, 0.0]), np.array([0.1, 0.0]))

    assert with_battery_export == pytest.approx(0.1)
    assert plant_only == pytest.approx(0.2)
