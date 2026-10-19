"""
Unit tests of the action handling and argument validation of `SmartCerEnv`.
"""

from pathlib import Path

import numpy as np
import pytest

from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.unit


def test_actions_are_clipped_to_the_box_without_penalty(mini_cer_v1_path: Path) -> None:
    """
    Test that actions outside [-1, 1] are clipped to the box before the nodes, without penalty
    for that clip: 3.0 behaves exactly as 1.0 (penalty only from the dynamic bounds).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    results = []
    for value in (1.0, 3.0):
        env = SmartCerEnv(mini_cer_v1_path)
        env.reset(seed=0)
        for _ in range(8):
            env.step(np.zeros(2, dtype=np.float32))
        results.append(env.step(np.array([value, 0.0], dtype=np.float32)))

    assert results[0][1] == results[1][1]
    assert results[1][4]['action'].tolist() == [1.0, 0.0]
    assert results[1][4]['nodes']['pv_bat']['action_requested'] == 1.0
    assert results[1][4]['nodes']['pv_bat']['penalty'] == 0.0


@pytest.mark.parametrize(('action', 'match'), [([np.nan, 0.0], 'finite'),
                                               ([np.inf, 0.0], 'finite'), ([0.0], 'shape')])
def test_invalid_actions_raise(mini_cer_v1_path: Path, action: list[float], match: str) -> None:
    """
    Test that non-finite actions and actions of the wrong shape raise ValueError.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        action: Invalid action.
        match: Text expected in the message.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    env.reset(seed=0)

    with pytest.raises(ValueError, match=match):
        env.step(np.array(action))


def test_step_before_reset_raises(mini_cer_v1_path: Path) -> None:
    """
    Test that step() before reset() raises RuntimeError.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    with pytest.raises(RuntimeError, match='reset'):
        SmartCerEnv(mini_cer_v1_path).step(np.zeros(2))


@pytest.mark.parametrize(('kwargs', 'match'), [
    ({'episode_length': 49}, 'episode_length'),
    ({'episode_length': 0}, 'episode_length'),
    ({'reward': 'cost'}, 'unknown reward'),
    ({'reward_per_node': True}, 'no per-node variant'),
    ({'penalty_coefficient': -1.0}, 'penalty_coefficient'),
    ({'observation': 'paper'}, 'observation preset'),
    ({'lookback': 2}, 'extended preset only'),
    ({'observation': 'extended', 'lookback': -1}, 'lookback'),
    ({'normalizer_window': ('2023-01-01', None)}, 'contains no step'),
    ({'normalizer_window': ('2022-01-03', '2022-01-03 05:00'), 'normalizer': 'x.json'},
     'either normalizer'),
])
def test_invalid_arguments_raise(mini_cer_v1_path: Path, kwargs: dict, match: str) -> None:
    """
    Test the argument validation.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        kwargs: Invalid arguments.
        match: Text expected in the message.
    """
    with pytest.raises(ValueError, match=match):
        SmartCerEnv(mini_cer_v1_path, **kwargs)


def test_random_start_stays_inside_the_period(mini_cer_v1_path: Path) -> None:
    """
    Test that random starts use self.np_random and leave room for the whole episode.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, episode_length=40, random_start=True)
    starts = {env.reset(seed=seed)[1]['episode_start'] for seed in range(30)}

    assert starts <= set(range(9)) and len(starts) > 1
