"""
Known defects of SmartCer 0.6.2 (docs/refactor/REFACTOR_v1.md section 2), environment level.

Every test states the correct behaviour and is marked `xfail(strict=True)`: it fails on the 0.6.2
code, and an unexpected pass makes the suite fail. Before the 0.6.2 modules are removed in phase 3,
each test is replaced by a regression test on the new API.
"""

import collections
import json
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from smartcer.cer_env import SmartCer


pytestmark = pytest.mark.integration

ZeroActions = 'collections.OrderedDict[str, np.ndarray]'


# ==================================
# S3: Gymnasium compliance
# ==================================


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: step returns 6 items')
def test_s3_step_returns_five_items(v062_env: SmartCer, v062_zero_actions: ZeroActions) -> None:
    """
    Test that step() returns (obs, reward, terminated, truncated, info).

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
    """
    assert len(v062_env.step(v062_zero_actions)) == 5


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: reward is a list')
def test_s3_reward_is_float(v062_env: SmartCer, v062_zero_actions: ZeroActions) -> None:
    """
    Test that the reward is a float, not the list [reward, penalty, signal].

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
    """
    assert isinstance(v062_env.step(v062_zero_actions)[1], float)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: truncated is None')
def test_s3_truncated_is_bool(v062_env: SmartCer, v062_zero_actions: ZeroActions) -> None:
    """
    Test that the truncated flag is a bool.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
    """
    assert isinstance(v062_env.step(v062_zero_actions)[3], bool)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: undeclared observation keys')
def test_s3_observation_lies_in_observation_space(v062_env: SmartCer) -> None:
    """
    Test that the reset observation is contained in the declared observation space.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
    """
    obs, _ = v062_env.reset(seed=0)

    assert v062_env.observation_space.contains(obs)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason='S3: OHESpace not flattenable')
def test_s3_observation_space_is_flattenable(v062_env: SmartCer) -> None:
    """
    Test that gymnasium can compute the flat dimension of the observation space.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
    """
    assert gym.spaces.flatdim(v062_env.observation_space) > 0


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: check_env fails')
def test_s3_gymnasium_check_env_passes(v062_env: SmartCer) -> None:
    """
    Test that the Gymnasium environment checker accepts the environment.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
    """
    check_env(v062_env)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S3: gym.make env fails on reset')
def test_s3_gym_make_reset_and_step(mini_cer_path: Path, v062_run_config: Dict[str, Any]) -> None:
    """
    Test that the registered environment can be created with gym.make, reset and stepped.

    Args:
        mini_cer_path: Path of the synthetic scenario.
        v062_run_config: Run configuration of the 0.6.2 environment.
    """
    env = gym.make('SmartCer-v0', cer_path=str(mini_cer_path), cer_config=v062_run_config)

    env.reset(seed=0)
    env.step(env.action_space.sample())


# ==================================
# S7-S10
# ==================================


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S7: implicit perfect nowcast')
def test_s7_causal_observation_hides_current_generation(
    mini_cer_path: Path, v062_run_config: Dict[str, Any], v062_zero_actions: ZeroActions
) -> None:
    """
    Test that with observe_current_step False the observation of step t holds g of step t - 1.

    pv_bat generates 0 kWh at 07:00 and 10 kWh at 08:00: before choosing the action of step 8 the
    agent must see 0 kWh. 0.6.2 has no such option and always shows g of step t.

    Args:
        mini_cer_path: Path of the synthetic scenario.
        v062_run_config: Run configuration of the 0.6.2 environment.
        v062_zero_actions: Idle action for every battery.
    """
    v062_run_config['observe_current_step'] = False
    env = SmartCer(cer_path=str(mini_cer_path), cer_config=v062_run_config)
    _, info = env.reset(seed=0)
    for _ in range(8):
        _, _, _, _, info, _ = env.step(v062_zero_actions)

    assert info['pv_bat']['production'] == pytest.approx(0.0)


@pytest.mark.xfail(strict=True, raises=KeyError, reason='S8: consumer battery, KeyError production')
def test_s8_consumer_with_battery_loads_and_steps(
    mini_cer_copy: Path, v062_run_config: Dict[str, Any]
) -> None:
    """
    Test that a consumer with a battery loads and steps, and that its battery is controlled.

    Args:
        mini_cer_copy: Writable copy of the synthetic scenario.
        v062_run_config: Run configuration of the 0.6.2 environment.
    """
    config_path = mini_cer_copy / 'config.json'
    config = json.loads(config_path.read_text())
    config['Consumers']['house_1']['Storage'] = {
        'capacity': 10, 'rated_power': 2, 'soc_min': 0.1, 'efficiency': 0.9, 'alfa': 0.0,
        'beta': 0.0,
    }
    config_path.write_text(json.dumps(config))

    env = SmartCer(cer_path=str(mini_cer_copy), cer_config=v062_run_config)
    env.reset(seed=0)
    env.step(env.action_space.sample())

    assert 'house_1' in env.storage_nodes_ids


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S9: scaler not reset')
def test_s9_identical_episodes_give_identical_rewards(
    mini_cer_path: Path, v062_run_config: Dict[str, Any], v062_zero_actions: ZeroActions
) -> None:
    """
    Test that two episodes with the same seed and actions give the same rewards.

    With the 0.6.2 running reward scaler the statistics carry over between episodes (and the first
    scaled reward is always 0). Whole episodes are compared: the idle night steps all have the same
    raw reward, which the scaler maps to 0 in both episodes.

    Args:
        mini_cer_path: Path of the synthetic scenario.
        v062_run_config: Run configuration of the 0.6.2 environment.
        v062_zero_actions: Idle action for every battery.
    """
    v062_run_config['reward']['scaler'] = {'apply': True}
    env = SmartCer(cer_path=str(mini_cer_path), cer_config=v062_run_config)

    episodes = []
    for _ in range(2):
        env.reset(seed=0)
        episodes.append([env.step(v062_zero_actions)[1][2] for _ in range(env.sim_hours)])

    assert episodes[0] == pytest.approx(episodes[1])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S10: terminal observation wraps')
def test_s10_terminal_observation_comes_from_last_step(
    v062_env: SmartCer, v062_zero_actions: ZeroActions
) -> None:
    """
    Test that the terminal observation describes the last valid step (23:00), not step 0.

    Args:
        v062_env: 0.6.2 environment on the synthetic scenario.
        v062_zero_actions: Idle action for every battery.
    """
    terminated, info = False, {}
    while not terminated:
        _, _, terminated, _, info, _ = v062_env.step(v062_zero_actions)

    assert info['cer']['hour'] == 23
