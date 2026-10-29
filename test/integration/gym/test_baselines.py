"""
Baseline runs on the synthetic scenario: output files, KPIs against hand calculations, the
community mode of the rule-based controller and the protection of the scenario folders (S18).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import RBCController, compute_baselines, run_episode


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

    assert list(results) == ['naive', 'rbc_community']
    summary = json.loads((tmp_path / 'baselines' / 'kpis.json').read_text(encoding='utf-8'))
    assert summary['rbc_community']['welfare'] == \
        pytest.approx(results['rbc_community']['welfare'])
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

    Rule-based, mode `node`, every day: pv_bat charges 5 kWh (4.5 in the cell) in hours 8-11
    until full (2 → 20 kWh) and, with g = 0, discharges 5, 5, 5 and 0.9 · 1.333 = 1.2 kWh in
    hours 16-19 (cell 20 → 2 kWh): charge 40 kWh, discharge 32.4 kWh, throughput 72 kWh in two
    days, 72 / 40 = 1.8 cycles on S^max and 72 / 36 = 2 on S^max - S^min. factory_bat charges its
    10 kWh surplus in hours 8-11 (4 → 40 kWh) and covers its load with 10, 10, 4, 4, 4, 0.4 kWh in
    hours 16-21: charge 80 kWh, discharge 64.8 kWh, throughput 144 kWh. Exports drop by 15 kWh in
    hours 8-11 (-6 € on day 1, -9 € on day 2) and the 16.2 kWh pv_bat exports at 0.20 €/kWh are
    all shared: revenue 40 - 15 + 6.48 = 31.48 €, shared 48 + 32.4 = 80.4 kWh; factory_bat
    imports 64.8 kWh less: cost 111.2 - 0.30 · 64.8 = 91.76 €.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    results = compute_baselines(SmartCerEnv(mini_cer_v1_path), tmp_path / 'baselines',
                                controllers=('naive', 'rbc_node'))
    naive, rbc = results['naive'], results['rbc_node']

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
    pv, factory = rbc['batteries']['pv_bat'], rbc['batteries']['factory_bat']
    assert pv['charge_kwh'] == pytest.approx(40.0)
    assert pv['discharge_kwh'] == pytest.approx(32.4)
    assert pv['throughput_kwh'] == pytest.approx(72.0)
    assert pv['equivalent_full_cycles'] == pytest.approx(1.8)
    assert pv['efc_usable'] == pytest.approx(2.0)
    assert factory['charge_kwh'] == pytest.approx(80.0)
    assert factory['discharge_kwh'] == pytest.approx(64.8)
    assert factory['throughput_kwh'] == pytest.approx(144.0)
    assert rbc['revenue'] == pytest.approx(31.48)
    assert rbc['cost'] == pytest.approx(91.76)
    assert rbc['shared_energy_kwh'] == pytest.approx(80.4)
    assert rbc['welfare'] == pytest.approx(rbc['revenue'] - rbc['cost'] + rbc['incentive'])
    assert rbc['penalty_steps'] == 0


def test_community_rbc_follows_the_community_balance(mini_cer_v1_path: Path) -> None:
    """
    Test mode `community` on the synthetic scenario: in every step the batteries together charge
    at most the surplus B_t = Σ g - Σ l of all nodes and discharge at most the deficit -B_t, they
    cover it fully whenever they can, and no action is clipped.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    tracker = run_episode(env, RBCController(env, mode='community'))
    nodes = tracker.node_tables()
    balance = sum(table['generation_kwh'] - table['load_kwh'] for table in nodes.values())
    batteries = [nodes[node_id] for node_id in env.storage_node_ids]
    charge = sum(table['charge_kwh'] for table in batteries)
    discharge = sum(table['discharge_kwh'] for table in batteries)
    storages = [env.community.node(node_id).battery for node_id in env.storage_node_ids]
    delta_t = env.community.delta_t
    # energies the batteries can take and deliver in each step (node side)
    upper = sum(table['a_upper'] * battery.power_charge_max * delta_t
                for table, battery in zip(batteries, storages))
    lower = sum(table['a_lower'] * battery.power_discharge_max * delta_t
                for table, battery in zip(batteries, storages))

    assert (charge <= balance.clip(lower=0.0) + 1e-9).all()
    assert (discharge <= (-balance).clip(lower=0.0) + 1e-9).all()
    assert np.allclose(charge, np.minimum(balance.clip(lower=0.0), upper))
    assert np.allclose(discharge, np.minimum((-balance).clip(lower=0.0), lower))
    assert discharge.sum() > 0.0 and charge.sum() > 0.0
    assert tracker.kpis()['penalty_steps'] == 0


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
