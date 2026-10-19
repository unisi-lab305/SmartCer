"""
Consistency of the local 0.6.2 reference values written by `scripts/make_reference_v062.py`.

Skipped when `test/fixtures/local/` is missing (it is gitignored and absent in CI). Phase 2 compares
the new community functions against the same files.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest


pytestmark = pytest.mark.integration

NAME = 'leather_2022'


@pytest.fixture
def reference(local_fixtures_path: Path) -> Dict[str, Any]:
    """
    Load the Leather 2022 reference values.

    Args:
        local_fixtures_path: Folder of the local reference values.

    Returns:
        Dict[str, Any]: 'community' (hourly DataFrame) and 'kpis' (content of the JSON file).
    """
    csv_path = local_fixtures_path / f'{NAME}_community.csv'
    json_path = local_fixtures_path / f'{NAME}_kpis.json'
    if not (csv_path.is_file() and json_path.is_file()):
        pytest.skip(f'{NAME} reference files missing: run scripts/make_reference_v062.py')
    return {'community': pd.read_csv(csv_path), 'kpis': json.loads(json_path.read_text())}


def test_reference_covers_one_hourly_year(reference: Dict[str, Any]) -> None:
    """
    Test that the hourly series cover 2022 with 8760 sorted, unique timestamps.

    Args:
        reference: Leather 2022 reference values.
    """
    timestamps = pd.to_datetime(reference['community']['datetime'])

    assert len(timestamps) == reference['kpis']['metadata']['steps'] == 8760
    assert timestamps.is_monotonic_increasing and timestamps.is_unique
    assert (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_reference_annual_sums_match_hourly_series(reference: Dict[str, Any]) -> None:
    """
    Test that the annual sums in the JSON file equal the sums of the hourly series.

    Args:
        reference: Leather 2022 reference values.
    """
    community = reference['community']
    for column, annual_sum in reference['kpis']['annual_sums'].items():
        assert community[column].sum() == pytest.approx(annual_sum, rel=1e-12), column


def test_reference_hourly_identities(reference: Dict[str, Any]) -> None:
    """
    Test shared = min(exported, imported) and actual TIP = shared / exported * potential TIP.

    Args:
        reference: Leather 2022 reference values.
    """
    community = reference['community']
    shared = community[['exported_energy', 'imported_energy']].min(axis=1)
    exported = community['exported_energy']
    ratio = (shared / exported.where(exported > 0)).fillna(0.0)

    assert community['selfconsumption'].to_numpy() == pytest.approx(shared.to_numpy(), abs=1e-9)
    assert community['tip_actual'].to_numpy() == pytest.approx(
        (ratio * community['tip_potential']).to_numpy(), abs=1e-9
    )
