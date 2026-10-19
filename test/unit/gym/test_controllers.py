"""
Unit tests of the baseline controllers (`smartcer.gym.controllers`): the naive controller and the
rule-based controller of docs/FORMULATION.md section 13, on hand cases.
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


def run(env: SmartCerEnv) -> list[dict[str, Any]]:
    """
    One episode with the rule-based controller.

    Args:
        env: Environment.

    Returns:
        list[dict[str, Any]]: Per step, the actions chosen and the node info of the step.
    """
    controller = RBCController(env)
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
    intended charge is min(40, 90) = 40 kWh, i.e. action 40 / 30 limited to 1, which executes
    30 kWh (0.6.2: action 0.3, 9 kWh).

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
    Test section 13 on a prosumer with 10 kWh, 5 kW, η = 0.9, S^min = 0, Δt = 1 h:

    - surplus 3 kWh, empty battery: action 3 / 5 = 0.6, 3 kWh charged, state of charge 2.7 kWh;
    - deficit 2 kWh: action -2 / 5 = -0.4, 2 kWh to the node, no import, state of charge
      2.7 - 2 / 0.9 = 0.478 kWh;
    - deficit 2 kWh: the rule asks min(2, 0.478) = 0.478 kWh (action -0.0956) on the node side,
      but the cell can deliver only η · 0.478 = 0.430 kWh: the environment clips the action to
      -A^lower = -0.0860 with penalty k = 0.0096, as for any other action.

    Args:
        write_scenario: Scenario factory.
    """
    env = prosumer_env(write_scenario, [(3.0, 0.0), (0.0, 2.0), (0.0, 2.0)], storage(10, 5))
    charge, discharge, clipped = run(env)

    assert charge['action'] == pytest.approx(0.6)
    assert charge['charge_kwh'] == pytest.approx(3.0)
    assert charge['soc_kwh'] == pytest.approx(2.7)
    assert charge['export_kwh'] == pytest.approx(0.0)
    assert discharge['action'] == pytest.approx(-0.4)
    assert discharge['discharge_kwh'] == pytest.approx(2.0)
    assert discharge['import_kwh'] == pytest.approx(0.0)
    soc = 2.7 - 2.0 / 0.9
    assert discharge['soc_kwh'] == pytest.approx(soc)
    assert clipped['action'] == pytest.approx(-soc / 5, rel=1e-6)
    assert clipped['action_applied'] == pytest.approx(-0.9 * soc / 5, rel=1e-6)
    assert clipped['penalty'] == pytest.approx(0.1 * soc / 5, rel=1e-6)
    assert clipped['soc_kwh'] == pytest.approx(0.0, abs=1e-9)


def test_rbc_respects_headroom_and_time_step(write_scenario: Callable[..., Path]) -> None:
    """
    Test the headroom S^max - s (battery at 9 of 10 kWh, surplus 3 kWh: 1 kWh, action 0.2) and a
    15-minute step (surplus 1 kWh with 5 kW: action 1 / (5 · 0.25) = 0.8).

    Args:
        write_scenario: Scenario factory.
    """
    full = prosumer_env(write_scenario, [(3.0, 0.0)], storage(10, 5, soc_initial=0.9))
    step, = run(full)
    assert step['action'] == pytest.approx(0.2)
    assert step['charge_kwh'] == pytest.approx(1.0)
    assert step['soc_kwh'] == pytest.approx(9.9)
    assert step['penalty'] == 0.0

    quarter = prosumer_env(write_scenario, [(1.0, 0.0)], storage(10, 5), delta_t=0.25)
    step, = run(quarter)
    assert step['action'] == pytest.approx(0.8)
    assert step['charge_kwh'] == pytest.approx(1.0)


def test_rbc_producer_never_discharges_and_consumer_never_charges(
        write_scenario: Callable[..., Path]) -> None:
    """
    Test the node types: a producer has d = g ≥ 0, so it charges by day and stays idle at night;
    a consumer has d = -l < 0, so it only discharges.

    Args:
        write_scenario: Scenario factory.
    """
    producer = prosumer_env(write_scenario, [(2.0, 0.0), (0.0, 0.0)], storage(10, 5),
                            node_type='producer')
    day, night = run(producer)
    assert day['action'] == pytest.approx(0.4)
    assert night['action'] == 0.0

    consumer = prosumer_env(write_scenario, [(0.0, 1.0)], storage(10, 5, soc_initial=0.5),
                            node_type='consumer')
    step, = run(consumer)
    assert step['action'] == pytest.approx(-0.2)
    assert step['discharge_kwh'] == pytest.approx(1.0)
    assert step['import_kwh'] == pytest.approx(0.0)


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
