"""
Shared fixtures for all tests (unit and integration).

The synthetic scenario `test/fixtures/mini_cer/` is in the 0.6.2 format and has 4 nodes and
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

import collections
from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np
import pytest

from smartcer.cer_env import SmartCer


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
def mini_cer_copy(tmp_path: Path) -> Path:
    """
    Writable copy of the synthetic scenario, for tests that change its configuration or data.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path: Folder of the copy.
    """
    return Path(shutil.copytree(MINI_CER_PATH, tmp_path / 'mini_cer'))


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


# =================================
# Fixtures for the 0.6.2 environment
# =================================


@pytest.fixture
def v062_run_config() -> Dict[str, Any]:
    """
    Run configuration of the 0.6.2 environment covering the whole synthetic scenario.

    Collective `profit` reward with penalty and no reward scaler.

    Returns:
        Dict[str, Any]: The `cer_config` argument of `smartcer.cer_env.SmartCer`.
    """
    return {
        'simulation': {
            'start': '2022-01-03 00:00:00',
            'end': '2022-01-04 23:00:00',
            'seed': 0,
        },
        'reward': {
            'type': 'profit',
            'single_agent': False,
            'constraints': {'penalty': True, 'coefficient': 10.0, 'max_ratio': 0.99},
            'scaler': {'apply': False},
        },
    }


@pytest.fixture
def v062_env(mini_cer_path: Path, v062_run_config: Dict[str, Any]) -> SmartCer:
    """
    0.6.2 environment on the synthetic scenario, reset with seed 0.

    Args:
        mini_cer_path: Path of the synthetic scenario.
        v062_run_config: Run configuration of the 0.6.2 environment.

    Returns:
        SmartCer: The environment, ready for its first step.
    """
    env = SmartCer(cer_path=str(mini_cer_path), cer_config=v062_run_config)
    env.reset(seed=0)
    return env


@pytest.fixture
def v062_zero_actions(v062_env: SmartCer) -> 'collections.OrderedDict[str, np.ndarray]':
    """
    Zero (idle) action for every battery of the 0.6.2 environment.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.

    Returns:
        collections.OrderedDict[str, np.ndarray]: Action dictionary accepted by `SmartCer.step()`.
    """
    return collections.OrderedDict(
        (node_id, np.array([0.0], dtype=np.float32)) for node_id in v062_env.storage_nodes_ids
    )


# =========================
# Fixtures for local data
# =========================


@pytest.fixture
def local_fixtures_path() -> Path:
    """
    Folder of the local reference values (gitignored, written by `scripts/make_reference_v062.py`).

    Returns:
        Path: Folder `test/fixtures/local/`; the test is skipped when it is missing.
    """
    if not LOCAL_FIXTURES_PATH.is_dir():
        pytest.skip('local reference values missing: run scripts/make_reference_v062.py')
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
