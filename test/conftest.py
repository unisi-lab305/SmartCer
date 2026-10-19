"""
Shared fixtures for all tests (unit and integration).

The synthetic scenario `test/fixtures/mini_cer/` (0.6.2 format; `test/fixtures/mini_cer_v1/` is
the same scenario in the 1.0 format) has 4 nodes and
48 hourly rows (2022-01-03 00:00 to 2022-01-04 23:00):

- `pv_bat` (producer, battery 20 kWh / 5 kW): generation 10 kWh in hours 8-15, PNRR 0 %;
- `house_1` (consumer, reads `house.csv` through the `_N` suffix rule): load 2 kWh, buy price 0.25;
- `factory_bat` (prosumer, battery 40 kWh / 10 kW): generation 20 kWh in hours 8-15, load 10 kWh in
  hours 8-17 and 4 kWh otherwise, buy price 0.30, PNRR 40 %;
- `shop` (prosumer, no battery): generation 5 kWh in hours 8-15, load 6 kWh, buy price 0.20,
  PNRR 20 %.

All exporting nodes share the sell price 0.10 €/kWh in hours 8-15 of day 1, 0.15 €/kWh in hours
8-15 of day 2 and 0.20 €/kWh otherwise. Every node with a `Photovoltaic` section has
`tip_base` 0.08, `tip_threshold` 0.12 and `fi_zone` 0.01 €/kWh; batteries have `soc_min` 0.1 and
efficiency 0.9.
"""

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from smartcer.data import load_scenario


FIXTURES_PATH = Path(__file__).parent / 'fixtures'
MINI_CER_PATH = FIXTURES_PATH / 'mini_cer'
MINI_CER_V1_PATH = FIXTURES_PATH / 'mini_cer_v1'
LOCAL_FIXTURES_PATH = FIXTURES_PATH / 'local'
CER_DATA_PATH = Path(__file__).parents[1] / 'smartcer' / 'cer_data'


# =================================
# Fixtures for the synthetic scenario
# =================================


@pytest.fixture
def mini_cer_path() -> Path:
    """
    Path of the committed synthetic scenario.

    Returns:
        Path: Folder `test/fixtures/mini_cer/` (read it, do not modify it).
    """
    return MINI_CER_PATH


@pytest.fixture
def mini_cer_v1_path() -> Path:
    """
    Path of the synthetic scenario in the 1.0 format (same nodes, data and parameters).

    Returns:
        Path: Folder `test/fixtures/mini_cer_v1/` (read it, do not modify it).
    """
    return MINI_CER_V1_PATH


@pytest.fixture
def mini_cer_v1_copy(tmp_path: Path) -> Path:
    """
    Writable copy of the 1.0-format synthetic scenario.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path: Folder of the copy.
    """
    return Path(shutil.copytree(MINI_CER_V1_PATH, tmp_path / 'mini_cer_v1'))


@pytest.fixture
def write_scenario(tmp_path: Path) -> Callable[..., Path]:
    """
    Factory writing a 1.0-format scenario into the temporary directory.

    The returned function takes the `nodes` section, a mapping from CSV path (relative to the
    scenario) to CSV text, and optional top-level keys (defaults: delta_t 1, units kWh, uniform
    incentive 0.1 €/kWh).

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Callable[..., Path]: Function returning the scenario folder.
    """
    def write(nodes: Dict[str, Any], csv_files: Dict[str, str], **top: Any) -> Path:
        folder = tmp_path / 'scenario'
        folder.mkdir(exist_ok=True)
        config = {'format_version': '1.0', 'delta_t': 1.0, 'units': 'kWh',
                  'incentive': {'model': 'uniform', 'unit_value': 0.1}, **top, 'nodes': nodes}
        (folder / 'config.json').write_text(json.dumps(config))
        for name, text in csv_files.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text)
        return folder

    return write


@pytest.fixture
def synthetic_oracle(tmp_path: Path) -> Path:
    """
    Oracle folder of the 1.0-format synthetic scenario, in the provisional format of
    `smartcer.oracle` (`trajectory.csv`, `objective.csv`).

    The trajectory stands in for the MILP output: every battery takes, at every step, a random
    fraction of its dynamic bound (A^upper or -A^lower, seed 7), so it is never penalised; the
    objective is the welfare of that run per day. The trajectory rows are shuffled.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path: Folder `<tmp_path>/oracle`.
    """
    community = load_scenario(MINI_CER_V1_PATH)
    community.reset()
    rng = np.random.default_rng(7)
    rows, welfare = [], []
    for t in range(community.n_steps):
        actions = []
        for node_id in community.storage_node_ids:
            a_upper, a_lower = community.node(node_id).get_bounds(t)
            fraction = rng.uniform()
            actions.append(fraction * a_upper if rng.uniform() < 0.5 else -fraction * a_lower)
        result = community.step(t, np.array(actions))
        stamp = str(pd.Timestamp(community.timestamps[t]))
        rows += [{'datetime': stamp, 'node_id': r.node_id, 'p_cha_kw': r.p_cha_kw,
                  'p_dis_kw': r.p_dis_kw, 'soc_kwh': r.soc_kwh}
                 for r in result.nodes if r.soc_kwh is not None]
        welfare.append(result.welfare.total)
    folder = tmp_path / 'oracle'
    folder.mkdir()
    trajectory = pd.DataFrame(rows).sample(frac=1.0, random_state=7)
    trajectory.to_csv(folder / 'trajectory.csv', index=False)
    days = pd.DatetimeIndex(community.timestamps).date
    objective = pd.Series(welfare).groupby(days).sum()
    pd.DataFrame({'date': [str(d) for d in objective.index], 'objective': objective.to_numpy()}
                 ).to_csv(folder / 'objective.csv', index=False)
    return folder


# =========================
# Fixtures for local data
# =========================


@pytest.fixture
def local_fixtures_path() -> Path:
    """
    Folder of the local reference values (gitignored), written in phase 0 by
    `scripts/make_reference_v062.py` (removed in phase 3, available in git history).

    Returns:
        Path: Folder `test/fixtures/local/`; the test is skipped when it is missing.
    """
    if not LOCAL_FIXTURES_PATH.is_dir():
        pytest.skip('local reference values missing: run scripts/make_reference_v062.py from '
                    'commit d878969 (removed in phase 3)')
    return LOCAL_FIXTURES_PATH


@pytest.fixture
def leather_path() -> Path:
    """
    Folder of the local Leather scenario (gitignored, read-only).

    Returns:
        Path: `smartcer/cer_data/Leather/`; the test is skipped when it is missing.
    """
    path = CER_DATA_PATH / 'Leather'
    if not (path / 'config.json').is_file():
        pytest.skip('local scenario smartcer/cer_data/Leather missing')
    return path
