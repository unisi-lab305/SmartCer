"""
Known defects of SmartCer 0.6.2 (docs/refactor/REFACTOR_v1.md section 2), unit level.

Every test states the correct behaviour and is marked `xfail(strict=True)`: it fails on the 0.6.2
code, and an unexpected pass makes the suite fail. Before the 0.6.2 modules are removed in phase 3,
each test is replaced by a regression test on the new API.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from smartcer.cer_models.energy_models import Battery
from smartcer.cer_models.node_models import Producer
from smartcer.cer_reward.reward import Reward
from smartcer.storage_controllers.rbc_controller import StandardRBC


pytestmark = pytest.mark.unit


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S2: reward type cost is rejected')
def test_s2_reward_type_cost_is_accepted(v062_run_config: Dict[str, Any]) -> None:
    """
    Test that the reward type 'cost' used in the README and in saved configurations is accepted.

    Args:
        v062_run_config: Run configuration of the 0.6.2 environment.
    """
    reward_config = {**v062_run_config['reward'], 'type': 'cost'}

    reward = Reward(nodes={}, **reward_config)

    assert reward.type == 'cost'


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S4: discharge limit on node side')
def test_s4_discharge_keeps_soc_above_soc_min() -> None:
    """
    Test that a full discharge near the minimum does not drive the state of charge below S^min.

    10 kWh, 5 kW, soc_min 1 kWh, efficiency 0.9, from 2 kWh: 0.6.2 delivers 1 kWh to the node and
    removes 1 / 0.9 kWh from the cell, ending at 0.889 kWh < 1 kWh.
    """
    battery = Battery(capacity=10.0, rated_power=5.0, soc_min=0.1, efficiency=0.9, alfa=0.0,
                      beta=0.0)
    battery.soc = 2.0

    battery.charge(-5.0, production=0.0)

    assert battery.soc >= battery.soc_min - 1e-9


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S4: negative state of charge')
def test_s4_discharge_with_zero_soc_min_keeps_soc_non_negative() -> None:
    """
    Test that with soc_min = 0 a full discharge leaves a non-negative state of charge without
    raising.

    10 kWh, 5 kW, efficiency 0.9, from 1 kWh: 0.6.2 computes 1 - 1 / 0.9 < 0 and its internal
    assertion fails.
    """
    battery = Battery(capacity=10.0, rated_power=5.0, soc_min=0.0, efficiency=0.9, alfa=0.0,
                      beta=0.0)
    battery.soc = 1.0

    battery.charge(-5.0, production=0.0)

    assert battery.soc >= 0.0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S5: no time step')
def test_s5_charge_energy_scales_with_time_step(tmp_path: Path) -> None:
    """
    Test that on 15-minute data a full charge stores rated power x 0.25 h on the node side.

    Args:
        tmp_path: Pytest temporary directory.
    """
    rows = ['datetime,production,price_exp'] + [
        f'2022-01-03 00:{minute:02d}:00,100,0.1' for minute in (0, 15, 30, 45)
    ]
    (tmp_path / 'pv.csv').write_text('\n'.join(rows) + '\n')
    battery = Battery(capacity=100.0, rated_power=10.0, soc_min=0.0, efficiency=1.0, alfa=0.0,
                      beta=0.0)
    producer = Producer(
        data_dir=str(tmp_path), sim_period=('2022-01-03 00:00:00', '2022-01-03 00:45:00'),
        node_id='pv', node_type='plant', storage=battery, pv_power=100, pnrr=0.0, tip_base=0.08,
        tip_threshold=0.12, fi_zone=0.01, pz='CNOR',
    )

    producer.set_storage(1.0)

    assert battery.actual_charges[-1] == pytest.approx(10.0 * 0.25)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S6: RBC normalises by capacity')
def test_s6_rbc_action_executes_intended_energy() -> None:
    """
    Test that the rule-based controller charges the intended energy.

    Surplus 40 kWh, rated power 30 kW, capacity 100 kWh, empty battery: the intended charge is
    30 kWh, so the action must be 1.0; 0.6.2 returns 30 / 100 = 0.3, executed as 9 kWh.
    """
    controller = StandardRBC(storage_node={'pv': None}, producer_ids=['pv'], prosumer_ids=[])
    state = {'cer': {}, 'pv': {'production': 40.0, 'soc': 10.0, 'soc_min': 10.0,
                               'capacity': 100.0, 'rated_power': 30.0}}

    action = float(controller.choose_actions(state)['pv'][0])

    assert action * state['pv']['rated_power'] == pytest.approx(30.0)
