"""
Integration tests of loader, nodes and community on the synthetic scenario (described in
`test/conftest.py`), in the 0.6.2 format (`mini_cer`) and the 1.0 format (`mini_cer_v1`).

The first tests are the phase 0 characterisation tests (`test/integration/v062/
test_characterisation.py`) re-pointed to the new API, with the `tip` incentive model.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from smartcer.core.community import Community
from smartcer.data import load_scenario


pytestmark = pytest.mark.integration

ABS_TOL = 1e-9


def run_idle(community: Community, steps: int) -> list[Any]:
    """
    Step the community with idle batteries.

    Args:
        community: Community, freshly loaded or reset.
        steps: Number of steps from t = 0.

    Returns:
        list[Any]: The CommunityStep records.
    """
    return [community.step(t) for t in range(steps)]


# ==================================
# Characterisation tests re-pointed to the new API
# ==================================


@pytest.mark.parametrize(
    ('t', 'expected_flows', 'expected_shared', 'expected_tip'),
    [
        (0, {'pv_bat': (0, 0), 'house_1': (0, 2), 'factory_bat': (0, 4), 'shop': (0, 6)}, 0.0, 0.0),
        (10, {'pv_bat': (10, 0), 'house_1': (0, 2), 'factory_bat': (10, 0), 'shop': (0, 1)}, 3.0,
         0.2925),
        (16, {'pv_bat': (0, 0), 'house_1': (0, 2), 'factory_bat': (0, 10), 'shop': (0, 6)}, 0.0,
         0.0),
        (34, {'pv_bat': (10, 0), 'house_1': (0, 2), 'factory_bat': (10, 0), 'shop': (0, 1)}, 3.0,
         0.27),
    ],
)
def test_zero_action_step_matches_hand_calculation(
    mini_cer_path: Path, t: int, expected_flows: Dict[str, Tuple[float, float]],
    expected_shared: float, expected_tip: float,
) -> None:
    """
    Test exported and imported energy per node, shared energy and TIP incentive of one idle step
    (same hand values as the 0.6.2 characterisation test).

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
        t: Step index checked.
        expected_flows: Hand-computed node id → (exported kWh, imported kWh).
        expected_shared: Hand-computed shared energy (kWh).
        expected_tip: Hand-computed TIP incentive (€).
    """
    result = run_idle(load_scenario(mini_cer_path), t + 1)[t]

    flows = {r.node_id: (r.export_kwh, r.import_kwh) for r in result.nodes}
    assert flows.keys() == expected_flows.keys()
    for node_id, expected in expected_flows.items():
        assert flows[node_id] == pytest.approx(expected, abs=ABS_TOL), node_id
    assert result.welfare.shared_energy_kwh == pytest.approx(expected_shared, abs=ABS_TOL)
    assert result.welfare.incentive == pytest.approx(expected_tip, abs=ABS_TOL)


def test_shared_energy_and_tip_identities(mini_cer_path: Path) -> None:
    """
    Test, over the idle run, shared = min(total exported, total imported) and
    TIP incentive = shared / exported × Σ coefficient × exported.

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
    """
    community = load_scenario(mini_cer_path)
    for result in run_idle(community, community.n_steps):
        export = np.array([r.export_kwh for r in result.nodes])
        imported = sum(r.import_kwh for r in result.nodes)
        shared = min(export.sum(), imported)
        potential = community.incentive.potential(result.t, export)
        expected = shared / export.sum() * potential if export.sum() > 0 else 0.0

        assert result.welfare.shared_energy_kwh == pytest.approx(shared, abs=ABS_TOL)
        assert result.welfare.incentive == pytest.approx(expected, abs=ABS_TOL)


def test_energy_balance_per_node_random_run(mini_cer_path: Path) -> None:
    """
    Test g + discharge + import = l + charge + export for every node and step of a random run.

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
    """
    community = load_scenario(mini_cer_path)
    rng = np.random.default_rng(42)
    charged = discharged = 0.0
    for t in range(community.n_steps):
        result = community.step(t, rng.uniform(-1.0, 1.0, len(community.storage_node_ids)))
        for r in result.nodes:
            inflow = r.generation_kwh + r.discharge_kwh + r.import_kwh
            outflow = r.load_kwh + r.charge_kwh + r.export_kwh
            assert inflow == pytest.approx(outflow, abs=ABS_TOL), f'{r.node_id}, step {t}'
            charged += r.charge_kwh
            discharged += r.discharge_kwh

    # the run must have used the batteries, otherwise the identity is trivial
    assert charged > 0 and discharged > 0


# ==================================
# Welfare by hand, both incentive models
# ==================================


def uniform_copy(folder: Path, unit_value: float) -> Path:
    """
    Switch a copy of the 1.0 synthetic scenario to the uniform model.

    Args:
        folder: Writable copy of `mini_cer_v1`.
        unit_value: π^inc (€/kWh).

    Returns:
        Path: The same folder.
    """
    config_path = folder / 'config.json'
    config = json.loads(config_path.read_text())
    config['incentive'] = {'model': 'uniform', 'unit_value': unit_value}
    config_path.write_text(json.dumps(config))
    return folder


@pytest.mark.parametrize('model', ['tip', 'uniform'])
def test_welfare_matches_hand_calculation(mini_cer_v1_copy: Path, model: str) -> None:
    """
    Test W at 10:00 of day 1, idle and with actions, for both incentive models.

    Idle: revenue 10 · 0.1 + 10 · 0.1 = 2, cost 2 · 0.25 + 1 · 0.2 = 0.7, shared 3 kWh;
    tip I = 3 / 20 · (0.13 · 10 + 0.065 · 10) = 0.2925 → W = 1.5925; uniform 0.1 → W = 1.6.
    Actions (pv_bat -0.5 at S^min, factory_bat +0.5): pv_bat clipped to 0 with k = 0.5,
    factory_bat charges 5 kWh and exports 5; revenue 1.5, cost 0.7, shared 3 of 15 exported;
    tip I = 3 / 15 · (0.13 · 10 + 0.065 · 5) = 0.325 → W = 1.125; uniform → W = 1.1; K = 0.25.

    Args:
        mini_cer_v1_copy: Writable copy of the 1.0 synthetic scenario.
        model: Incentive model.
    """
    path = mini_cer_v1_copy if model == 'tip' else uniform_copy(mini_cer_v1_copy, 0.1)
    community = load_scenario(path)
    idle = run_idle(community, 11)[10]
    community.reset()
    run_idle(community, 10)
    acted = community.step(10, [-0.5, 0.5])

    expected_idle, expected_acted = (1.5925, 1.125) if model == 'tip' else (1.6, 1.1)
    assert idle.welfare.revenue == pytest.approx(2.0)
    assert idle.welfare.cost == pytest.approx(0.7)
    assert idle.welfare.total == pytest.approx(expected_idle)
    assert acted.welfare.revenue == pytest.approx(1.5)
    assert acted.welfare.export_kwh == pytest.approx(15.0)
    assert acted.welfare.total == pytest.approx(expected_acted)
    assert acted.penalty == pytest.approx(0.25)


# ==================================
# 1.0 format vs 0.6.2 format, consumer battery
# ==================================


def test_v1_format_gives_the_same_results_as_v062(mini_cer_path: Path,
                                                  mini_cer_v1_path: Path) -> None:
    """
    Test that the synthetic scenario in the 1.0 format gives exactly the same node records,
    welfare and penalties as in the 0.6.2 format, over a random 48 h run.

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    legacy, v1 = load_scenario(mini_cer_path), load_scenario(mini_cer_v1_path)

    assert v1.node_ids == legacy.node_ids
    assert v1.storage_node_ids == legacy.storage_node_ids
    assert np.array_equal(v1.timestamps, legacy.timestamps)
    rng = np.random.default_rng(7)
    for t in range(legacy.n_steps):
        actions = rng.uniform(-1.0, 1.0, len(legacy.storage_node_ids))
        assert v1.step(t, actions) == legacy.step(t, actions), f'step {t}'


def test_consumer_with_battery_loads_and_steps(mini_cer_v1_copy: Path) -> None:
    """
    Test that a consumer with a battery loads and steps: under D6 it cannot charge (penalty 1 for
    a full charge), but it can discharge what it holds.

    Args:
        mini_cer_v1_copy: Writable copy of the 1.0 synthetic scenario.
    """
    config_path = mini_cer_v1_copy / 'config.json'
    config = json.loads(config_path.read_text())
    config['nodes']['house_1']['storage'] = {
        'capacity': 10, 'power_charge_max': 2, 'power_discharge_max': 2, 'efficiency_charge': 1.0,
        'efficiency_discharge': 1.0, 'soc_initial': 0.5,
    }
    config_path.write_text(json.dumps(config))
    community = load_scenario(mini_cer_v1_copy)

    assert community.storage_node_ids == ('pv_bat', 'house_1', 'factory_bat')
    charged = community.step(0, [0.0, 1.0, 0.0]).nodes[1]
    discharged = community.step(1, [0.0, -1.0, 0.0]).nodes[1]

    assert charged.charge_kwh == 0.0 and charged.penalty == pytest.approx(1.0)
    assert discharged.discharge_kwh == pytest.approx(2.0)
    assert discharged.import_kwh == pytest.approx(0.0)
