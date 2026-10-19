"""
Stable-Baselines3 SAC on the synthetic scenario (skipped without stable-baselines3).
"""

from pathlib import Path

import pytest

from smartcer.gym import SmartCerEnv


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_sac_runs_2000_steps(mini_cer_v1_path: Path) -> None:
    """
    Test that SAC trains for 2,000 steps (about 42 episodes of 48 h) without errors.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    sb3 = pytest.importorskip('stable_baselines3')
    env = SmartCerEnv(mini_cer_v1_path, seed=0)
    model = sb3.SAC('MlpPolicy', env, learning_starts=100, batch_size=64, seed=0, device='cpu',
                    verbose=0)

    model.learn(total_timesteps=2_000)

    assert model.num_timesteps == 2_000
