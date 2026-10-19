"""
Baseline runs on the synthetic scenario: output files, KPIs against hand calculations and the
protection of the scenario folders (S18).
"""

import json
from pathlib import Path

import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import compute_baselines


pytestmark = pytest.mark.integration


def test_baselines_write_the_run_outputs(mini_cer_v1_path: Path, tmp_path: Path) -> None:
    """
    Test the output folder of `compute_baselines`: per controller the community table, one table
    per node, the KPIs and the run configuration; a summary of the KPIs at the top.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    results = compute_baselines(env, tmp_path / 'baselines', seed=0)

    assert list(results) == ['naive', 'rbc']
    summary = json.loads((tmp_path / 'baselines' / 'kpis.json').read_text(encoding='utf-8'))
    assert summary['rbc']['welfare'] == pytest.approx(results['rbc']['welfare'])
    for name in results:
        folder = tmp_path / 'baselines' / name
        assert (folder / 'community.csv').is_file()
        assert len((folder / 'community.csv').read_text().splitlines()) == 49
        assert len(list((folder / 'nodes').glob('*.csv'))) == 4
        run_config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        assert run_config['controller'] == name
        assert run_config['scenario_path'] == str(mini_cer_v1_path)


def test_kpis_of_the_synthetic_scenario_match_hand_calculations(mini_cer_v1_path: Path,
                                                                tmp_path: Path) -> None:
    """
    Test the KPIs of the synthetic scenario (docstring of test/conftest.py) over its 48 hours.

    Idle batteries (naive), per day: hours 8-15 export 10 + 10 kWh and import 2 + 1 kWh (shared 3);
    hours 16-17 import 2 + 10 + 6 kWh; the other 14 hours 2 + 4 + 6 kWh. Exported 320 kWh,
    imported 8·3·2 + 2·18·2 + 14·12·2 = 456 kWh, shared 48 kWh. Revenue 160 · 0.10 + 160 · 0.15 = 40 €;
    cost 96 · 0.25 + 152 · 0.30 + 208 · 0.20 = 111.2 €; TIP incentive 1.5 (c_pv + c_factory) per
    hour of hours 8-15: 1.5 · (0.13 + 0.065) · 8 + 1.5 · (0.12 + 0.06) · 8 = 4.5 €.

    Rule-based: pv_bat charges 5 kWh (4.5 in the cell) for 4 hours until full (2 → 20 kWh) and,
    being a producer, never discharges: charge 20 kWh, throughput 18 kWh, 18 / 40 = 0.45 cycles.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    results = compute_baselines(SmartCerEnv(mini_cer_v1_path), tmp_path / 'baselines')
    naive, rbc = results['naive'], results['rbc']

    assert naive['export_kwh'] == pytest.approx(320.0)
    assert naive['import_kwh'] == pytest.approx(456.0)
    assert naive['shared_energy_kwh'] == pytest.approx(48.0)
    assert naive['sharing_rate'] == pytest.approx(48.0 / 320.0)
    assert naive['coverage_rate'] == pytest.approx(48.0 / 456.0)
    assert naive['revenue'] == pytest.approx(40.0)
    assert naive['cost'] == pytest.approx(111.2)
    assert naive['incentive'] == pytest.approx(4.5)
    assert naive['welfare'] == pytest.approx(40.0 - 111.2 + 4.5)
    assert naive['reward'] == pytest.approx(naive['welfare'])
    assert naive['throughput_kwh'] == 0.0 and naive['penalty_steps'] == 0
    pv = rbc['batteries']['pv_bat']
    assert pv['charge_kwh'] == pytest.approx(20.0)
    assert pv['discharge_kwh'] == 0.0
    assert pv['throughput_kwh'] == pytest.approx(18.0)
    assert pv['equivalent_full_cycles'] == pytest.approx(0.45)
    assert pv['penalty_steps'] == 0
    assert rbc['welfare'] > naive['welfare']


def test_s18_baselines_refuse_scenario_folders(mini_cer_v1_copy: Path, tmp_path: Path) -> None:
    """
    Test that results are never written into the scenario folder or a cer_data folder (0.6.2
    wrote them into cer_data/<scenario>/simulations/), and that unknown controllers raise.

    Args:
        mini_cer_v1_copy: Writable copy of the synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_copy)
    with pytest.raises(ValueError, match='inside the scenario folder'):
        compute_baselines(env, mini_cer_v1_copy / 'simulations')
    with pytest.raises(ValueError, match='inside a cer_data folder'):
        compute_baselines(env, tmp_path / 'cer_data' / 'mine' / 'simulations')
    with pytest.raises(ValueError, match=r"unknown controllers \['ppo'\]"):
        compute_baselines(env, tmp_path / 'out', controllers=['naive', 'ppo'])
    assert not (mini_cer_v1_copy / 'simulations').exists()
    assert not (tmp_path / 'cer_data').exists()
    assert not (tmp_path / 'out').exists()
