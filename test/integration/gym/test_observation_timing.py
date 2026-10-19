"""
Information timing `observe_current_step` (docs/FORMULATION.md section 12, defect S7 of 0.6.2).

pv_bat (mini_cer_v1) generates 0 kWh until 07:00 and 10 kWh from 08:00 (min-max scale 0-10 kWh);
its battery is at S^min = 2 kWh, so A^upper = min(5, g / 1 h, (20 - 2) / 0.9) / 5.
"""

from pathlib import Path

import numpy as np
import pytest

from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.integration

IDLE = np.zeros(2, dtype=np.float32)


def observe_at(path: Path, t: int, observe_current_step: bool) -> tuple[np.ndarray, SmartCerEnv]:
    """
    Observation used to choose the action of step t after idle steps.

    Args:
        path: Scenario folder.
        t: Step index.
        observe_current_step: Information timing.

    Returns:
        tuple[np.ndarray, SmartCerEnv]: The observation and the environment.
    """
    env = SmartCerEnv(path, observe_current_step=observe_current_step)
    obs, _ = env.reset(seed=0)
    for _ in range(t):
        obs, *_ = env.step(IDLE)
    return obs, env


@pytest.mark.parametrize(('t', 'expected_g', 'expected_a_upper'), [(0, 0.0, 0.0), (8, 1.0, 1.0)])
def test_current_step_is_observed(mini_cer_v1_path: Path, t: int, expected_g: float,
                                  expected_a_upper: float) -> None:
    """
    Test observe_current_step=True: g of step t and A^upper with the charge cap g_t.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        t: Step index.
        expected_g: Normalised generation of step t.
        expected_a_upper: A^upper of step t.
    """
    obs, env = observe_at(mini_cer_v1_path, t, True)

    assert obs[env.obs_layout['pv_bat/generation']] == pytest.approx([expected_g])
    assert obs[env.obs_layout['pv_bat/a_upper']] == pytest.approx([expected_a_upper])


@pytest.mark.parametrize(('t', 'expected_g'), [(0, 0.0), (8, 0.0), (9, 1.0)])
def test_s7_causal_observation_hides_current_generation(mini_cer_v1_path: Path, t: int,
                                                        expected_g: float) -> None:
    """
    Test S7 with observe_current_step=False: g of step t - 1 (0 at the first row), and A^upper
    without the charge cap (1.0 even at night), while the applied bound still uses g_t.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        t: Step index.
        expected_g: Normalised generation of step t - 1.
    """
    obs, env = observe_at(mini_cer_v1_path, t, False)

    assert obs[env.obs_layout['pv_bat/generation']] == pytest.approx([expected_g])
    assert obs[env.obs_layout['pv_bat/a_upper']] == pytest.approx([1.0])
    if t == 0:
        _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert info['nodes']['pv_bat']['action_applied'] == 0.0
        assert info['nodes']['pv_bat']['penalty'] == pytest.approx(1.0)


def test_prices_of_the_current_step_are_observed_in_both_modes(mini_cer_v1_path: Path) -> None:
    """
    Test that the sell price of step t is observed with both timings (0.10 at 08:00, 0.20 at
    07:00 of day 1).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    for observe_current_step in (True, False):
        obs, env = observe_at(mini_cer_v1_path, 8, observe_current_step)

        assert obs[env.obs_layout['pv_bat/sell_price']] == pytest.approx([0.10])
