"""
Unit tests of the baseline controllers (`smartcer.gym.controllers`): the naive controller and the
rule-based controller of docs/FORMULATION.md section 13 in its two modes, on hand cases.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import NaiveController, RBCController, run_episode


pytestmark = pytest.mark.unit

HEADER = 'datetime,production,consumption,sell_price,buy_price'


def storage(capacity: float, power: float, efficiency: float = 0.9, soc_min: float = 0.0,
            soc_initial: float | None = None) -> dict[str, Any]:
    """
    1.0 storage section with equal charge and discharge limits and efficiencies.

    Args:
        capacity: S^max (kWh).
        power: P^cha = P^dis (kW).
        efficiency: η^cha = η^dis.
        soc_min: S^min as a fraction of capacity.
        soc_initial: S^ini as a fraction of capacity; None for soc_min.

    Returns:
        dict[str, Any]: The storage section.
    """
    section = {'capacity': capacity, 'power_charge_max': power, 'power_discharge_max': power,
               'efficiency_charge': efficiency, 'efficiency_discharge': efficiency,
               'soc_min': soc_min}
    if soc_initial is not None:
        section['soc_initial'] = soc_initial
    return section


def prosumer_env(write_scenario: Callable[..., Path], rows: list[tuple[float, float]],
                 battery: dict[str, Any], node_type: str = 'prosumer',
                 delta_t: float = 1.0) -> SmartCerEnv:
    """
    Environment with one battery node and the given (generation, load) rows.

    Args:
        write_scenario: Scenario factory.
        rows: (generation, load) per step, kWh.
        battery: Storage section.
        node_type: 'producer', 'consumer' or 'prosumer'.
        delta_t: Time step (hours).

    Returns:
        SmartCerEnv: The environment, whole period in one episode.
    """
    minutes = int(delta_t * 60)
    lines = [f'2022-01-03 {(i * minutes) // 60:02d}:{(i * minutes) % 60:02d}:00,{g},{c},0.1,0.3'
             for i, (g, c) in enumerate(rows)]
    nodes = {'bat': {'type': node_type, 'csv': 'bat.csv', 'storage': battery}}
    folder = write_scenario(nodes, {'bat.csv': '\n'.join([HEADER, *lines]) + '\n'},
                            delta_t=delta_t)
    return SmartCerEnv(folder)


def run(env: SmartCerEnv, mode: str = 'node') -> list[dict[str, Any]]:
    """
    One episode with the rule-based controller.

    Args:
        env: Environment with the battery node `bat`.
        mode: RBC mode.

    Returns:
        list[dict[str, Any]]: Per step, the action chosen and the node info of the step.
    """
    controller = RBCController(env, mode=mode)
    obs, info = env.reset(seed=0)
    steps, done = [], False
    while not done:
        action = controller.act(obs, info)
        obs, _, _, done, info = env.step(action)
        steps.append({'action': float(action[0]), **info['nodes']['bat']})
    return steps


def test_naive_controller_returns_zero_actions(mini_cer_v1_path: Path) -> None:
    """
    Test the naive controller: zeros of the action space's shape and dtype.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    obs, info = env.reset(seed=0)
    action = NaiveController(gym.wrappers.RecordEpisodeStatistics(env)).act(obs, info)

    assert action.tolist() == [0.0, 0.0]
    assert env.action_space.contains(action)


def test_s6_rbc_executes_intended_energy(write_scenario: Callable[..., Path]) -> None:
    """
    Test the 0.6.2 defect S6 case: surplus 40 kWh, 30 kW, 100 kWh, battery at S^min 10 kWh. The
    intended charge is 40 kWh, i.e. action 40 / 30 limited by A^upper = 1, which executes 30 kWh
    (0.6.2: action 0.3, 9 kWh).

    Args:
        write_scenario: Scenario factory.
    """
    env = prosumer_env(write_scenario, [(40.0, 0.0)], storage(100, 30, soc_min=0.1),
                       node_type='producer')
    step, = run(env)

    assert step['action'] == 1.0
    assert step['charge_kwh'] == pytest.approx(30.0)
    assert step['penalty'] == 0.0


def test_rbc_charges_surplus_and_discharges_deficit(write_scenario: Callable[..., Path]) -> None:
    """
    Test mode `node` on a prosumer with 10 kWh, 5 kW, η = 0.9, S^min = 0, Δt = 1 h:

    - surplus 3 kWh, empty battery: action 3 / 5 = 0.6, 3 kWh charged, state of charge 2.7 kWh;
    - deficit 2 kWh: action -2 / 5 = -0.4, 2 kWh to the node, no import, state of charge
      2.7 - 2 / 0.9 = 0.478 kWh;
    - deficit 2 kWh: the cell can deliver only η · 0.478 = 0.430 kWh to the node, so the action
      is -A^lower = -0.0860 and the battery ends empty, without penalty.

    Args:
        write_scenario: Scenario factory.
    """
    env = prosumer_env(write_scenario, [(3.0, 0.0), (0.0, 2.0), (0.0, 2.0)], storage(10, 5))
    charge, discharge, last = run(env)

    assert charge['action'] == pytest.approx(0.6)
    assert charge['charge_kwh'] == pytest.approx(3.0)
    assert charge['soc_kwh'] == pytest.approx(2.7)
    assert charge['export_kwh'] == pytest.approx(0.0)
    assert discharge['action'] == pytest.approx(-0.4)
    assert discharge['discharge_kwh'] == pytest.approx(2.0)
    assert discharge['import_kwh'] == pytest.approx(0.0)
    soc = 2.7 - 2.0 / 0.9
    assert discharge['soc_kwh'] == pytest.approx(soc)
    assert last['action'] == pytest.approx(-0.9 * soc / 5, rel=1e-6)
    assert last['discharge_kwh'] == pytest.approx(0.9 * soc, rel=1e-6)
    assert last['soc_kwh'] == pytest.approx(0.0, abs=1e-6)
    assert [step['penalty'] for step in (charge, discharge, last)] == [0.0, 0.0, 0.0]


def test_rbc_respects_headroom_and_time_step(write_scenario: Callable[..., Path]) -> None:
    """
    Test the headroom of the dynamic bounds (battery at 9 of 10 kWh with η = 0.9, surplus 3 kWh:
    the node can supply (10 - 9) / 0.9 = 1.111 kWh, action 0.2222, the battery ends full) and a
    15-minute step (surplus 1 kWh with 5 kW: action 1 / (5 · 0.25) = 0.8, equal to A^upper).

    Args:
        write_scenario: Scenario factory.
    """
    full = prosumer_env(write_scenario, [(3.0, 0.0)], storage(10, 5, soc_initial=0.9))
    step, = run(full)
    assert step['action'] == pytest.approx(1.0 / 0.9 / 5, rel=1e-6)
    assert step['charge_kwh'] == pytest.approx(1.0 / 0.9, rel=1e-6)
    assert step['soc_kwh'] == pytest.approx(10.0)
    assert step['penalty'] == 0.0

    quarter = prosumer_env(write_scenario, [(1.0, 0.0)], storage(10, 5), delta_t=0.25)
    step, = run(quarter)
    assert step['action'] == pytest.approx(0.8)
    assert step['charge_kwh'] == pytest.approx(1.0)
    assert step['penalty'] == 0.0


def test_rbc_node_mode_producer_and_consumer(write_scenario: Callable[..., Path]) -> None:
    """
    Test the node types in mode `node`: a producer charges with its generation by day (2 kWh,
    action 0.4, 1.8 kWh in the cell) and, with g = 0, discharges all it can (η · 1.8 = 1.62 kWh,
    action -0.324, exported); a consumer has d = -l < 0, so it only discharges.

    Args:
        write_scenario: Scenario factory.
    """
    producer = prosumer_env(write_scenario, [(2.0, 0.0), (0.0, 0.0)], storage(10, 5),
                            node_type='producer')
    day, night = run(producer)
    assert day['action'] == pytest.approx(0.4)
    assert night['action'] == pytest.approx(-0.324)
    assert night['discharge_kwh'] == pytest.approx(1.62)
    assert night['export_kwh'] == pytest.approx(1.62)
    assert night['soc_kwh'] == pytest.approx(0.0, abs=1e-6)
    assert night['penalty'] == 0.0

    consumer = prosumer_env(write_scenario, [(0.0, 1.0)], storage(10, 5, soc_initial=0.5),
                            node_type='consumer')
    step, = run(consumer)
    assert step['action'] == pytest.approx(-0.2)
    assert step['discharge_kwh'] == pytest.approx(1.0)
    assert step['import_kwh'] == pytest.approx(0.0)


def test_rbc_community_mode_splits_the_balance(write_scenario: Callable[..., Path]) -> None:
    """
    Test mode `community` on a prosumer `a` (10 kWh, 10 kW), a producer `b` (20 kWh, 10 kW), both
    with η = 1 and S^min = 0, and a consumer `c` without battery. Per step (g_a, l_a, g_b, l_c):

    - (8, 2, 4, 4): B = 6; a can take min(10, 8, 10) = 8 kWh, b min(10, 4, 20) = 4: both charge
      6 / 12 = half of it, 4 and 2 kWh (actions 0.4, 0.2);
    - (0, 1, 0, 2): B = -3; a can deliver 4, b 2: both discharge half, 2 and 1 kWh;
    - (0, 0, 0, 10): B = -10 > 3 available: both empty (actions -0.2, -0.1);
    - (20, 0, 10, 5): B = 25 > 20 available: both charge at full power;
    - (5, 5, 0, 0): B = 0: idle.

    Args:
        write_scenario: Scenario factory.
    """
    rows = [(8, 2, 4, 4), (0, 1, 0, 2), (0, 0, 0, 10), (20, 0, 10, 5), (5, 5, 0, 0)]
    stamps = [f'2022-01-03 {i:02d}:00:00' for i in range(len(rows))]
    files = {
        'a.csv': [f'{d},{g},{l},0.1,0.3' for d, (g, l, _, _) in zip(stamps, rows)],
        'b.csv': [f'{d},{g},0,0.1,0.3' for d, (_, _, g, _) in zip(stamps, rows)],
        'c.csv': [f'{d},0,{l},0.1,0.3' for d, (_, _, _, l) in zip(stamps, rows)],
    }
    nodes = {'a': {'type': 'prosumer', 'csv': 'a.csv', 'storage': storage(10, 10, 1.0)},
             'b': {'type': 'producer', 'csv': 'b.csv', 'storage': storage(20, 10, 1.0)},
             'c': {'type': 'consumer', 'csv': 'c.csv'}}
    folder = write_scenario(nodes, {name: '\n'.join([HEADER, *lines]) + '\n'
                                    for name, lines in files.items()})
    env = SmartCerEnv(folder)
    controller = RBCController(env)
    obs, info = env.reset(seed=0)
    actions, penalties, done = [], [], False
    while not done:
        action = controller.act(obs, info)
        obs, _, _, done, info = env.step(action)
        actions.append(action.tolist())
        penalties.append(info['penalty'])

    assert controller.mode == 'community' and controller.name == 'rbc_community'
    assert np.allclose(actions, [[0.4, 0.2], [-0.2, -0.1], [-0.2, -0.1], [1.0, 1.0], [0.0, 0.0]])
    assert penalties == [0.0] * len(rows)


def test_rbc_mode_is_checked(mini_cer_v1_path: Path) -> None:
    """
    Test the mode argument: the controller name follows it, and an unknown mode raises.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    assert RBCController(env, mode='node').name == 'rbc_node'
    with pytest.raises(ValueError, match="unknown RBC mode 'global'"):
        RBCController(env, mode='global')  # type: ignore[arg-type]


def test_controllers_check_their_environment(mini_cer_v1_path: Path) -> None:
    """
    Test that a controller needs a SmartCerEnv and that run_episode refuses a controller bound to
    another environment.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    with pytest.raises(TypeError, match='needs a SmartCerEnv'):
        NaiveController(gym.make('CartPole-v1'))
    env, other = SmartCerEnv(mini_cer_v1_path), SmartCerEnv(mini_cer_v1_path)
    with pytest.raises(ValueError, match='bound to another environment'):
        run_episode(env, NaiveController(other))
