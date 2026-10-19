"""
Unit tests of shared energy and welfare (docs/FORMULATION.md sections 9-10).
"""

import numpy as np
import pytest

from smartcer.core.community import WelfareBreakdown, shared_energy, welfare_breakdown


pytestmark = pytest.mark.unit


def test_shared_energy_is_min_of_totals() -> None:
    """
    Test E^sh = min(Σ e, Σ i) for one step (float) and per step for 2-D input.
    """
    assert shared_energy([10.0, 0.0, 10.0], [0.0, 2.0, 1.0]) == 3.0
    assert shared_energy([1.0, 0.5], [0.0, 4.0]) == 1.5
    per_step = shared_energy(np.array([[10.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 3.0],
                                                                           [2.0, 0.0]]))

    assert per_step.tolist() == [3.0, 0.0]


def test_welfare_breakdown_total() -> None:
    """
    Test W = revenue - cost - storage cost + incentive: 2 - 0.7 - 0.05 + 0.2925 = 1.5425.
    """
    welfare = welfare_breakdown(revenue=2.0, cost=0.7, storage_cost=0.05, incentive=0.2925,
                                shared_energy_kwh=3.0, export_kwh=20.0, import_kwh=3.0)

    assert isinstance(welfare, WelfareBreakdown)
    assert welfare.total == pytest.approx(1.5425)
    assert (welfare.shared_energy_kwh, welfare.export_kwh, welfare.import_kwh) == (3.0, 20.0, 3.0)
