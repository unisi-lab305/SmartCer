"""
Unit tests of the 0.6.2 run configuration mapping (`smartcer.gym.run_config`, defect S2).
"""

from pathlib import Path
from typing import Any

import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.run_config import map_v062_run_config


pytestmark = pytest.mark.unit


def legacy_config(**reward: Any) -> dict[str, Any]:
    """
    0.6.2 run configuration as in the 0.6.2 README, with overridable reward keys.

    Args:
        **reward: Reward keys that replace the defaults.

    Returns:
        dict[str, Any]: The configuration.
    """
    return {
        'simulation': {'start': '2022-01-03 00:00:00', 'end': '2022-01-03 23:00:00',
                       'delta_input': '1h', 'delta_output': 'YE', 'seed': 42},
        'reward': {'type': 'cost', 'single_agent': False,
                   'constraints': {'penalty': True, 'coefficient': 0.3, 'max_ratio': 0.99},
                   'scaler': {'apply': True}, **reward},
    }


def test_s2_reward_type_cost_is_mapped_to_welfare() -> None:
    """
    Test S2: reward.type 'cost' (README and saved configurations) maps to the welfare reward, with
    the period, seed and penalty coefficient, and the unused keys reported in the warning.
    """
    with pytest.warns(DeprecationWarning, match='keys not used') as record:
        kwargs = map_v062_run_config(legacy_config())

    assert kwargs == {'reward': 'welfare', 'penalty_coefficient': 0.3,
                      'start': '2022-01-03 00:00:00', 'end': '2022-01-03 23:00:00', 'seed': 42}
    message = str(record[0].message)
    assert 'simulation.delta_input' in message and 'reward.scaler.apply' in message
    assert 'reward.constraints.max_ratio' in message


def test_disabled_penalty_maps_to_zero_coefficient() -> None:
    """
    Test that reward.constraints.penalty False gives σ = 0.
    """
    constraints = {'penalty': False, 'coefficient': 0.3}
    with pytest.warns(DeprecationWarning):
        kwargs = map_v062_run_config(legacy_config(type='profit', constraints=constraints))

    assert kwargs['penalty_coefficient'] == 0.0


@pytest.mark.parametrize(('reward', 'expected'), [
    ({'type': 'profit', 'single_agent': False}, {'reward': 'welfare'}),
    ({'type': 'cost', 'single_agent': True}, {'reward': 'profit', 'reward_per_node': True}),
    ({'type': 'profit', 'single_agent': True}, {'reward': 'profit', 'reward_per_node': True}),
    ({'type': 'energy', 'single_agent': False}, {'reward': 'energy'}),
    ({'type': 'energy', 'single_agent': True}, {'reward': 'energy', 'reward_per_node': True}),
])
def test_reward_types_and_single_agent_are_mapped(reward: dict[str, Any],
                                                  expected: dict[str, Any]) -> None:
    """
    Test the mapping of reward.type and reward.single_agent (answer of 2026-10-19): energy → the
    energy reward, single_agent True → per-node rewards, with a DeprecationWarning.

    Args:
        reward: Reward keys.
        expected: Expected reward arguments.
    """
    with pytest.warns(DeprecationWarning, match='single_agent' if reward['single_agent']
                      else 'mapped') as record:
        kwargs = map_v062_run_config(legacy_config(**reward))

    assert {k: v for k, v in kwargs.items() if k.startswith('reward')} == expected
    assert len(record) == 1


def test_unknown_reward_type_raises() -> None:
    """
    Test that an unknown 0.6.2 reward type raises ValueError.
    """
    with pytest.raises(ValueError, match='unknown 0.6.2 reward type'):
        map_v062_run_config(legacy_config(type='reward'))


def test_single_agent_config_gives_per_node_rewards_summing_to_the_reward(
        mini_cer_path: Path) -> None:
    """
    Test that a 0.6.2 single_agent profit configuration gives one reward per battery node in info,
    and a scalar reward equal to their sum.

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
    """
    with pytest.warns(DeprecationWarning):
        env = SmartCerEnv.from_v062_config(mini_cer_path,
                                           legacy_config(type='profit', single_agent=True))
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    _, reward, _, _, info = env.step(env.action_space.sample())

    nodes = info['reward']['nodes']
    assert list(nodes) == ['pv_bat', 'factory_bat']
    assert reward == pytest.approx(sum(r['total'] for r in nodes.values()))


def test_from_v062_config_builds_the_environment(mini_cer_path: Path) -> None:
    """
    Test SmartCerEnv.from_v062_config on the 0.6.2 synthetic scenario: one day, σ = 0.3, and the
    seed applied by the first reset.

    Args:
        mini_cer_path: Path of the 0.6.2-format synthetic scenario.
    """
    with pytest.warns(DeprecationWarning):
        env = SmartCerEnv.from_v062_config(mini_cer_path, legacy_config())
    obs, _ = env.reset()
    first = env.action_space.sample()
    env.reset(seed=42)

    assert env.episode_length == 24 and env.penalty_coefficient == 0.3
    assert (first == env.action_space.sample()).all()
