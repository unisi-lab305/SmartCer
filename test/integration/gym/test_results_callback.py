"""
Stable-Baselines3 callback of the results tracker (skipped without Stable-Baselines3).
"""

from pathlib import Path

import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.tracking import ResultsCallback, TrackResults


pytestmark = pytest.mark.integration

sb3 = pytest.importorskip('stable_baselines3')
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def test_callback_logs_and_saves_completed_episodes(mini_cer_v1_path: Path,
                                                   tmp_path: Path) -> None:
    """
    Test SAC for 3 episodes of 8 steps (no gradient step): the callback saves every second
    completed episode with its run configuration and logs the episode KPIs.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = SmartCerEnv(mini_cer_v1_path, episode_length=8, random_start=True)
    vec_env = DummyVecEnv([lambda: TrackResults(env)])
    callback = ResultsCallback(tmp_path / 'training', save_freq=2, run_config=env.run_config())
    model = sb3.SAC('MlpPolicy', vec_env, learning_starts=100, seed=0, device='cpu')

    model.learn(total_timesteps=24, callback=callback)

    assert sorted(p.name for p in (tmp_path / 'training').iterdir()) == ['episode_00002']
    saved = tmp_path / 'training' / 'episode_00002'
    assert {p.name for p in saved.iterdir()} == {'community.csv', 'nodes', 'kpis.json',
                                                 'run_config.json'}
    assert len((saved / 'community.csv').read_text().splitlines()) == 9
    assert 'smartcer/welfare' in model.logger.name_to_value


def test_callback_needs_the_tracking_wrapper(mini_cer_v1_path: Path, tmp_path: Path) -> None:
    """
    Test the errors: an environment without `TrackResults`, a negative save frequency.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    vec_env = DummyVecEnv([lambda: SmartCerEnv(mini_cer_v1_path, episode_length=8)])
    model = sb3.SAC('MlpPolicy', vec_env, learning_starts=100, seed=0, device='cpu')
    with pytest.raises(RuntimeError, match='TrackResults'):
        model.learn(total_timesteps=1, callback=ResultsCallback(tmp_path))
    with pytest.raises(ValueError, match='save_freq'):
        ResultsCallback(tmp_path, save_freq=-1)
