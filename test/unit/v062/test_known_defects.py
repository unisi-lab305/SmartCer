"""
Known defect S6 of SmartCer 0.6.2 (docs/refactor/REFACTOR_v1.md section 2).

The other 0.6.2 defects were replaced by regression tests on the new API (`test_s<n>_*`) and
removed with the 0.6.2 modules in phase 3. This test stays until `smartcer/storage_controllers/` is
replaced by the new rule-based controller in phase 5: it is marked `xfail(strict=True)`, so an
unexpected pass makes the suite fail.
"""

import pytest

from smartcer.storage_controllers.rbc_controller import StandardRBC


pytestmark = pytest.mark.unit


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
