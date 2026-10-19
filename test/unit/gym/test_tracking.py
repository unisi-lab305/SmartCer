"""
Unit tests of the results tracker and its wrapper (`smartcer.gym.tracking`).
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from smartcer.gym import SmartCerEnv
from smartcer.gym.metrics import compute_kpis
from smartcer.gym.tracking import TrackResults


pytestmark = pytest.mark.unit

ACTIONS = np.array([0.5, -0.5], dtype=np.float32)


def tracked_env(path: Path, **kwargs: object) -> TrackResults:
    """
    Synthetic scenario, 3-step episodes starting at 09:00, wrapped with TrackResults.

    Args:
        path: Scenario folder.
        **kwargs: Further environment arguments.

    Returns:
        TrackResults: The wrapped environment.
    """
    return TrackResults(SmartCerEnv(path, start='2022-01-03 09:00', episode_length=3, **kwargs))


def run(env: TrackResults, steps: int) -> list[dict]:
    """
    Reset and take `steps` steps with fixed actions.

    Args:
        env: Wrapped environment.
        steps: Number of steps.

    Returns:
        list[dict]: Info of each step.
    """
    env.reset(seed=0)
    return [env.step(ACTIONS)[-1] for _ in range(steps)]


def test_tracker_tables_hold_the_step_info(mini_cer_v1_path: Path) -> None:
    """
    Test the community table and the node tables of a completed episode against the step info.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = tracked_env(mini_cer_v1_path)
    infos = run(env, 3)
    tracker = env.get_wrapper_attr('tracker')
    community = tracker.community_table()
    nodes = tracker.node_tables()

    assert tracker.episode == 1 and tracker.episodes_completed == 1
    assert list(community.columns) == [
        'datetime', 'episode', 't', 'reward', 'reward_main', 'reward_penalty', 'welfare',
        'revenue', 'cost', 'storage_cost', 'incentive', 'shared_energy_kwh', 'export_kwh',
        'import_kwh', 'penalty', 'obs_clipped']
    assert community['t'].tolist() == [0, 1, 2]
    assert community['datetime'].iloc[0] == pd.Timestamp('2022-01-03 09:00')
    assert community['welfare'].tolist() == [i['welfare']['total'] for i in infos]
    assert community['reward'].tolist() == [i['reward']['total'] for i in infos]
    assert community['penalty'].tolist() == [i['penalty'] for i in infos]
    assert list(nodes) == ['pv_bat', 'house_1', 'factory_bat', 'shop']
    assert nodes['factory_bat']['soc_kwh'].tolist() == \
        [i['nodes']['factory_bat']['soc_kwh'] for i in infos]
    assert 'reward' not in nodes['factory_bat'].columns
    assert tracker.kpis() == compute_kpis(community, nodes, {'pv_bat': 20.0, 'factory_bat': 40.0},
                                          {'pv_bat': 2.0, 'factory_bat': 4.0})


def test_tracker_records_per_node_rewards(mini_cer_v1_path: Path) -> None:
    """
    Test the reward columns of the battery nodes with `reward_per_node`.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = tracked_env(mini_cer_v1_path, reward='profit', reward_per_node=True)
    infos = run(env, 3)
    nodes = env.get_wrapper_attr('tracker').node_tables()

    assert nodes['pv_bat']['reward'].tolist() == \
        [i['reward']['nodes']['pv_bat']['total'] for i in infos]
    assert 'reward' not in nodes['shop'].columns


def test_tracker_keeps_the_last_completed_episode(mini_cer_v1_path: Path) -> None:
    """
    Test that an unfinished episode is not returned, that a reset drops it, and that the tables
    are those of the last completed episode.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = tracked_env(mini_cer_v1_path)
    tracker = env.get_wrapper_attr('tracker')
    run(env, 2)
    with pytest.raises(ValueError, match='no completed episode'):
        tracker.community_table()

    run(env, 3)
    run(env, 1)

    assert tracker.episode == 3 and tracker.episodes_completed == 1
    assert tracker.community_table()['episode'].unique().tolist() == [2]


def test_tracker_saves_tables_kpis_and_run_config(mini_cer_v1_path: Path,
                                                  tmp_path: Path) -> None:
    """
    Test the files written by `save`: CSV tables that read back equal, KPI and run configuration
    JSON.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        tmp_path: Pytest temporary directory.
    """
    env = tracked_env(mini_cer_v1_path)
    run(env, 3)
    tracker = env.get_wrapper_attr('tracker')

    folder = tracker.save(tmp_path / 'run', env.unwrapped.run_config())

    community = pd.read_csv(folder / 'community.csv', parse_dates=['datetime'])
    pd.testing.assert_frame_equal(community, tracker.community_table(), check_dtype=False)
    assert sorted(p.name for p in (folder / 'nodes').iterdir()) == [
        'factory_bat.csv', 'house_1.csv', 'pv_bat.csv', 'shop.csv']
    kpis = json.loads((folder / 'kpis.json').read_text(encoding='utf-8'))
    assert kpis['welfare'] == pytest.approx(tracker.kpis()['welfare'])
    run_config = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
    assert run_config['scenario'] == 'mini_cer_v1'
    assert run_config['start'] == '2022-01-03 09:00:00'
    assert run_config['episode_length'] == 3


def test_track_results_needs_a_smartcer_env() -> None:
    """
    Test that the wrapper refuses another environment.
    """
    with pytest.raises(TypeError, match='needs a SmartCerEnv'):
        TrackResults(gym.make('CartPole-v1'))
