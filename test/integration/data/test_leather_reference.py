"""
Leather 2022 with idle batteries: the new loader, nodes and community against the 0.6.2 reference
values written in phase 0 by `scripts/make_reference_v062.py` (in git history).

Skipped when the local scenario or the reference files are missing (both gitignored).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smartcer.data import TimestampWarning, load_scenario


pytestmark = [pytest.mark.integration, pytest.mark.slow]

ABS_TOL = 1e-6
# 0.6.2 community column → value computed from a new CommunityStep
SERIES = {
    'imported_energy': lambda step, community: step.welfare.import_kwh,
    'exported_energy': lambda step, community: step.welfare.export_kwh,
    'selfconsumption': lambda step, community: step.welfare.shared_energy_kwh,
    'energy_expense': lambda step, community: step.welfare.cost,
    'energy_revenue': lambda step, community: step.welfare.revenue,
    'tip_potential': lambda step, community: community.incentive.potential(
        step.t, np.array([r.export_kwh for r in step.nodes])),
    'tip_actual': lambda step, community: step.welfare.incentive,
}


@pytest.fixture
def leather_idle(leather_path: Path, local_fixtures_path: Path) -> dict[str, Any]:
    """
    Idle Leather 2022 run with the new API and the 0.6.2 reference series.

    Args:
        leather_path: Local Leather scenario.
        local_fixtures_path: Folder of the local reference values.

    Returns:
        dict[str, Any]: 'community', 'new' (DataFrame of SERIES) and 'reference' (0.6.2 CSV).
    """
    reference_path = local_fixtures_path / 'leather_2022_community.csv'
    if not reference_path.is_file():
        pytest.skip('Leather reference missing: run scripts/make_reference_v062.py from commit '
                    'd878969 (removed in phase 3)')
    # Leather has three nodes with timestamps shifted by one hour (aligned by position as in 0.6.2)
    with pytest.warns(TimestampWarning, match='cartotecnica'):
        community = load_scenario(leather_path, start='2022-01-01 00:00:00',
                                  end='2022-12-31 23:00:00')
    steps = [community.step(t) for t in range(community.n_steps)]
    new = pd.DataFrame({name: [f(step, community) for step in steps] for name, f in SERIES.items()})
    return {'community': community, 'new': new, 'reference': pd.read_csv(reference_path)}


def test_leather_idle_matches_v062_hourly_and_annual(leather_idle: dict[str, Any]) -> None:
    """
    Test every hourly community series and its annual sum against 0.6.2 within 1e-6.

    Args:
        leather_idle: Idle run and reference.
    """
    new, reference = leather_idle['new'], leather_idle['reference']

    assert len(new) == len(reference) == 8760
    assert np.array_equal(pd.to_datetime(reference['datetime']).to_numpy(),
                          leather_idle['community'].timestamps)
    for column in SERIES:
        assert new[column].to_numpy() == pytest.approx(reference[column].to_numpy(), abs=ABS_TOL), \
            column
        assert new[column].sum() == pytest.approx(reference[column].sum(), abs=ABS_TOL), column
