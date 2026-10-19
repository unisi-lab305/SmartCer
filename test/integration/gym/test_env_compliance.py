"""
Gymnasium and Stable-Baselines3 compliance of `SmartCerEnv` (defect S3 of 0.6.2).
"""

from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import smartcer
from smartcer.gym import ENV_ID, SmartCerEnv


pytestmark = pytest.mark.integration


def test_gymnasium_check_env_passes(mini_cer_v1_path: Path) -> None:
    """
    Test that the Gymnasium environment checker accepts the environment, with fixed and random
    episode starts, both information timings, both presets and a per-node reward.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    check_env(gym.make(ENV_ID, scenario=mini_cer_v1_path).unwrapped)
    check_env(gym.make(ENV_ID, scenario=mini_cer_v1_path, episode_length=12, random_start=True,
                       observe_current_step=False).unwrapped)
    check_env(gym.make(ENV_ID, scenario=mini_cer_v1_path, observation='extended', lookback=3,
                       reward='profit', reward_per_node=True, episode_length=12,
                       random_start=True).unwrapped)


def test_stable_baselines3_check_env_passes(mini_cer_v1_path: Path) -> None:
    """
    Test that the Stable-Baselines3 environment checker accepts the environment.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env_checker = pytest.importorskip('stable_baselines3.common.env_checker')

    env_checker.check_env(SmartCerEnv(mini_cer_v1_path), warn=True)


def test_s3_gym_make_reset_and_step(mini_cer_v1_path: Path) -> None:
    """
    Test S3: `gymnasium.make('SmartCer-v1', ...)` works (with the passive checker), `step()`
    returns five items, a float reward, bool flags and an observation inside the space.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = gym.make('SmartCer-v1', scenario=mini_cer_v1_path)
    obs, info = env.reset(seed=0)
    result = env.step(env.action_space.sample())

    assert smartcer.__version__
    assert env.observation_space.contains(obs) and isinstance(info, dict)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, step_info = result
    assert isinstance(reward, float)
    assert terminated is False and isinstance(truncated, bool)
    assert env.observation_space.contains(next_obs)
    assert set(step_info) >= {'reward', 'welfare', 'penalty', 'nodes', 'timestamp'}


def test_s3_spaces_are_flattenable(mini_cer_v1_path: Path) -> None:
    """
    Test S3: the spaces are standard flat Boxes: one float32 action per battery in [-1, 1].

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)

    assert gym.spaces.flatdim(env.observation_space) == env.observation_space.shape[0] == 25
    assert env.action_space == gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    assert env.storage_node_ids == ('pv_bat', 'factory_bat')
