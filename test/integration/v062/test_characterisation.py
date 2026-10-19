"""
Characterisation tests of the 0.6.2 environment `smartcer.cer_env.SmartCer` on the synthetic
scenario `test/fixtures/mini_cer/` (described in `test/conftest.py`).

They record the current behaviour and must keep passing until the 0.6.2 modules are removed; in
phase 2 they are re-pointed to the new community functions.
"""

import collections
from typing import Dict, Tuple

import numpy as np
import pytest

from smartcer.cer_env import SmartCer


pytestmark = pytest.mark.integration

ABS_TOL = 1e-9


def _node_flows(env: SmartCer, t: int) -> Dict[str, Tuple[float, float]]:
    """
    Exported and imported energy of every node in step t, as recorded by the 0.6.2 environment.

    Args:
        env: 0.6.2 environment after at least t + 1 steps.
        t: Step index.

    Returns:
        Dict[str, Tuple[float, float]]: Node id → (exported kWh, imported kWh).
    """
    flows = {}
    for node_id, node in env.nodes.items():
        exported = node.exported_energies[t] if hasattr(node, 'exported_energies') else 0.0
        imported = node.imported_energies[t] if hasattr(node, 'imported_energies') else 0.0
        flows[node_id] = (float(exported), float(imported))
    return flows


@pytest.mark.parametrize(
    ('t', 'expected_flows', 'expected_shared', 'expected_tip'),
    [
        # day 1, 00:00: no generation, every load is imported
        (
            0,
            {'pv_bat': (0, 0), 'house_1': (0, 2), 'factory_bat': (0, 4), 'shop': (0, 6)},
            0.0,
            0.0,
        ),
        # day 1, 10:00, sell price 0.10: coefficients 0.13 (pv_bat) and 0.13 * 0.5 (factory_bat)
        # actual TIP = 3 / 20 * (0.13 * 10 + 0.065 * 10)
        (
            10,
            {'pv_bat': (10, 0), 'house_1': (0, 2), 'factory_bat': (10, 0), 'shop': (0, 1)},
            3.0,
            0.2925,
        ),
        # day 1, 16:00: no generation, factory load 10
        (
            16,
            {'pv_bat': (0, 0), 'house_1': (0, 2), 'factory_bat': (0, 10), 'shop': (0, 6)},
            0.0,
            0.0,
        ),
        # day 2, 10:00, sell price 0.15: coefficients 0.12 (pv_bat) and 0.12 * 0.5 (factory_bat)
        # actual TIP = 3 / 20 * (0.12 * 10 + 0.06 * 10)
        (
            34,
            {'pv_bat': (10, 0), 'house_1': (0, 2), 'factory_bat': (10, 0), 'shop': (0, 1)},
            3.0,
            0.27,
        ),
    ],
)
def test_zero_action_step_matches_hand_calculation(
    v062_env: SmartCer,
    v062_zero_actions: 'collections.OrderedDict[str, np.ndarray]',
    t: int,
    expected_flows: Dict[str, Tuple[float, float]],
    expected_shared: float,
    expected_tip: float,
) -> None:
    """
    Test exported and imported energy per node, shared energy and actual TIP of one step.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
        t: Step index checked.
        expected_flows: Hand-computed node id → (exported kWh, imported kWh).
        expected_shared: Hand-computed shared energy (kWh).
        expected_tip: Hand-computed actual TIP (€).
    """
    for _ in range(t + 1):
        v062_env.step(v062_zero_actions)

    flows = _node_flows(v062_env, t)
    assert flows.keys() == expected_flows.keys()
    for node_id, expected in expected_flows.items():
        assert flows[node_id] == pytest.approx(expected, abs=ABS_TOL), node_id
    assert v062_env.cer_energy_selfcons[t] == pytest.approx(expected_shared, abs=ABS_TOL)
    assert v062_env.cer_actual_tip[t] == pytest.approx(expected_tip, abs=ABS_TOL)


def test_shared_energy_and_tip_identities(
    v062_env: SmartCer, v062_zero_actions: 'collections.OrderedDict[str, np.ndarray]'
) -> None:
    """
    Test, over the whole idle run, shared = min(total exported, total imported) and
    actual TIP = shared / exported * sum_i(coefficient_i * exported_i).

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
    """
    tip_nodes = [node for node in v062_env.nodes.values() if hasattr(node, 'tip_coefficients')
                 and hasattr(node, 'exported_energies')]
    for t in range(v062_env.sim_hours):
        v062_env.step(v062_zero_actions)
        flows = _node_flows(v062_env, t)
        exported = sum(e for e, _ in flows.values())
        imported = sum(i for _, i in flows.values())
        shared = min(exported, imported)
        potential = sum(n.tip_coefficients[t] * n.exported_energies[t] for n in tip_nodes)
        expected_tip = shared / exported * potential if exported > 0 else 0.0

        assert v062_env.cer_energy_selfcons[t] == pytest.approx(shared, abs=ABS_TOL)
        assert v062_env.cer_actual_tip[t] == pytest.approx(expected_tip, abs=ABS_TOL)


def test_energy_balance_per_node_random_run(v062_env: SmartCer) -> None:
    """
    Test production + discharge + import = consumption + charge + export for every node and step
    of a random 48 h run.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
    """
    rng = np.random.default_rng(42)
    for _ in range(v062_env.sim_hours):
        actions = collections.OrderedDict(
            (node_id, rng.uniform(-1.0, 1.0, size=1).astype(np.float32))
            for node_id in v062_env.storage_nodes_ids
        )
        v062_env.step(actions)

    n_steps = v062_env.sim_hours
    for node_id, node in v062_env.nodes.items():
        production = getattr(node, 'production', np.zeros(n_steps))
        consumption = getattr(node, 'consumption', np.zeros(n_steps))
        charge = node.storage.actual_charges if node.storage else [0.0] * n_steps
        discharge = node.storage.actual_discharges if node.storage else [0.0] * n_steps
        for t in range(n_steps):
            exported, imported = _node_flows(v062_env, t)[node_id]
            inflow = production[t] + discharge[t] + imported
            outflow = consumption[t] + charge[t] + exported
            assert inflow == pytest.approx(outflow, abs=ABS_TOL), f'{node_id}, step {t}'

    # the run must have used the batteries, otherwise the identity is trivial
    assert sum(v062_env.nodes['factory_bat'].storage.actual_charges) > 0
    assert sum(v062_env.nodes['factory_bat'].storage.actual_discharges) > 0
