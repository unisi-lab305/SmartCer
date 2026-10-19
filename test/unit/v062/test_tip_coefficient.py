"""
Characterisation tests of the 0.6.2 TIP coefficient `smartcer.cer_utils.get_tip_coefficient`.

Expected values are computed by hand from docs/FORMULATION.md section 9:
c = (min(tip_th, tip_base + max(0, 0.180 - pi_zone)) + fi_zone) * (1 - 0.5 * pnrr / 40).
"""

import pytest

from smartcer.cer_utils import get_tip_coefficient


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ('tip_th', 'tip_base', 'pi_zone', 'fi_zone', 'pnrr', 'expected'),
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
def test_tip_coefficient_hand_cases(
    tip_th: float, tip_base: float, pi_zone: float, fi_zone: float, pnrr: float, expected: float
) -> None:
    """
    Test that the TIP coefficient matches hand-computed values.

    Args:
        tip_th: TIP threshold (€/kWh).
        tip_base: Base TIP (€/kWh).
        pi_zone: Zonal price (€/kWh).
        fi_zone: Zonal correction factor (€/kWh).
        pnrr: Capital grant percentage.
        expected: Hand-computed coefficient (€/kWh).
    """
    coefficient = get_tip_coefficient(
        tip_th=tip_th, tip_base=tip_base, pi_zone=pi_zone, fi_zone=fi_zone, pnrr=pnrr
    )

    assert coefficient == pytest.approx(expected, abs=1e-12)


def test_tip_coefficient_rejects_pnrr_above_40() -> None:
    """
    Test that a capital grant above 40 % is rejected (0.6.2 raises AssertionError).
    """
    with pytest.raises(AssertionError, match='40%'):
        get_tip_coefficient(tip_th=0.12, tip_base=0.08, pi_zone=0.10, fi_zone=0.01, pnrr=41.0)
