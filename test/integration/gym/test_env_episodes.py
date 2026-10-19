"""
Episodes, reward, info and determinism of `SmartCerEnv` (defects S9 and S10 of 0.6.2).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.integration


def run_episode(env: SmartCerEnv, seed: int,
                actions: np.ndarray | None = None) -> list[tuple[Any, ...]]:
    """
    Run one episode and record every transition.

    Args:
        env: Environment.
        seed: Reset seed.
        actions: Actions per step; None samples them from the (seeded) action space.

    Returns:
        list[tuple[Any, ...]]: (obs, reward, terminated, truncated, t) per step, plus the reset
        observation and first step as the first item.
    """
    obs, info = env.reset(seed=seed)
    trajectory: list[tuple[Any, ...]] = [(obs, info['t'])]
    done, i = False, 0
    while not done:
        action = env.action_space.sample() if actions is None else actions[i]
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append((obs, reward, terminated, truncated, info['t']))
        done, i = terminated or truncated, i + 1
    return trajectory


def assert_same(a: list[tuple[Any, ...]], b: list[tuple[Any, ...]]) -> None:
    """
    Assert that two trajectories are identical.

    Args:
        a: First trajectory.
        b: Second trajectory.
    """
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]


def test_same_seed_and_actions_give_identical_trajectories(mini_cer_v1_path: Path) -> None:
    """
    Test determinism with random starts: same seed → same start, same sampled actions, same
    trajectory; a different seed gives a different start or actions.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, episode_length=12, random_start=True)
    first, second = run_episode(env, seed=3), run_episode(env, seed=3)
    other = run_episode(env, seed=4)

    assert_same(first, second)
    assert not all(np.array_equal(x[0], y[0]) for x, y in zip(first, other))


def test_s9_identical_episodes_give_identical_rewards(mini_cer_v1_path: Path) -> None:
    """
    Test S9: no reward scaling inside the environment, so two episodes with the same actions give
    the same rewards (the first one included).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    actions = np.random.default_rng(0).uniform(-1, 1, (env.episode_length, 2)).astype(np.float32)

    assert_same(run_episode(env, 0, actions), run_episode(env, 1, actions))


def test_episode_length_and_truncation(mini_cer_v1_path: Path) -> None:
    """
    Test that an episode of 5 steps controls every step and ends on its last one with
    truncated=True, terminated=False, and that step() then requires a reset.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, episode_length=5)
    trajectory = run_episode(env, seed=0)

    assert [x[-1] for x in trajectory[1:]] == [0, 1, 2, 3, 4]
    assert [x[3] for x in trajectory[1:]] == [False] * 4 + [True]
    assert not any(x[2] for x in trajectory[1:])
    with pytest.raises(RuntimeError, match='reset'):
        env.step(np.zeros(2, dtype=np.float32))


def test_s10_terminal_observation_comes_from_last_step(mini_cer_v1_path: Path) -> None:
    """
    Test S10: on the last step of the data, the observation returned with truncated=True
    describes that step (23:00 of day 2), not step 0, and holds the final state of charge.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    env.reset(seed=0)
    for _ in range(env.episode_length):
        obs, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    hour = obs[env.obs_layout['time/hour']]
    assert truncated and info['t'] == 47
    assert hour == pytest.approx([np.sin(2 * np.pi * 23 / 24), np.cos(2 * np.pi * 23 / 24)],
                                 abs=1e-6)
    assert obs[env.obs_layout['time/day_of_week']].tolist() == [0, 1, 0, 0, 0, 0, 0]


def start_at(env: SmartCerEnv, start: int) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Reset a random-start environment with the first seed whose episode starts at a given step.

    Args:
        env: Environment with `random_start=True`.
        start: First step wanted.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Reset observation and info.
    """
    seed = next(s for s in range(10_000) if env.reset(seed=s)[1]['episode_start'] == start)
    return env.reset(seed=seed)


def test_terminal_observation_is_the_next_step(mini_cer_v1_path: Path,
                                               synthetic_oracle: Path) -> None:
    """
    Test that a truncated random-start episode of steps 10-21 returns the observation of step
    22 with the final battery states, and the privileged data of step 22.

    With zero actions the states of charge stay at S^ini (no self-discharge), so the terminal
    observation equals the reset observation of an episode starting at 22. With random actions
    every feature but the state of charge and the bounds equals it too, and those come from the
    final states of charge.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, episode_length=12, random_start=True,
                      oracle=synthetic_oracle)
    other = SmartCerEnv(mini_cer_v1_path, episode_length=12, random_start=True,
                        oracle=synthetic_oracle)
    reset_obs, reset_info = start_at(other, 22)
    battery_features = [name for node in env.storage_node_ids
                        for name in (f'{node}/soc', f'{node}/a_lower', f'{node}/a_upper')]
    rest = np.ones(env.observation_space.shape[0], dtype=bool)
    for name in battery_features:
        rest[env.obs_layout[name]] = False

    for actions in (np.zeros((12, 2)), np.random.default_rng(3).uniform(-1, 1, (12, 2))):
        start_at(env, 10)
        for action in actions:
            obs, _, _, truncated, info = env.step(action)

        assert truncated and info['t'] == env.current_step == 21
        assert info['privileged']['t'] == 22
        np.testing.assert_array_equal(info['privileged']['oracle_action'],
                                      reset_info['privileged']['oracle_action'])
        np.testing.assert_array_equal(obs[rest], reset_obs[rest])
        for node_id in env.storage_node_ids:
            node = env.community.node(node_id)
            battery = node.battery
            soc = (battery.soc_kwh - battery.soc_min_kwh) / (battery.capacity - battery.soc_min_kwh)
            a_upper, a_lower = node.get_bounds(22)
            np.testing.assert_allclose(
                [obs[env.obs_layout[f'{node_id}/{name}']][0] for name in ('soc', 'a_lower',
                                                                          'a_upper')],
                [soc, a_lower, a_upper], atol=1e-7)
        if not actions.any():
            np.testing.assert_array_equal(obs, reset_obs)


def test_reward_is_welfare_minus_penalty_and_info(mini_cer_v1_path: Path) -> None:
    """
    Test R_t = W_t - σ K_t and the info content at 10:00 of day 1 (hand values of phase 2:
    W = 1.125 €, K = 0.25, σ = 10 → R = -1.375).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    env.reset(seed=0)
    for _ in range(10):
        env.step(np.zeros(2, dtype=np.float32))
    _, reward, _, _, info = env.step(np.array([-0.5, 0.5], dtype=np.float32))

    assert info['welfare']['total'] == pytest.approx(1.125)
    assert info['penalty'] == pytest.approx(0.25)
    assert reward == pytest.approx(1.125 - 10 * 0.25)
    assert info['reward'] == {'type': 'welfare', 'main': pytest.approx(1.125),
                              'penalty': pytest.approx(2.5), 'total': pytest.approx(reward)}
    assert info['privileged'] == {}
    # the normaliser is fitted on the whole period
    assert info['obs_clipped'] == 0
    assert info['nodes']['pv_bat']['action_applied'] == 0.0
    assert info['nodes']['factory_bat']['soc_kwh'] == pytest.approx(4.0 + 0.9 * 5.0)
    assert info['timestamp'] == np.datetime64('2022-01-03T10:00')


def test_reset_seeds_the_action_space(mini_cer_v1_path: Path) -> None:
    """
    Test that reset(seed) seeds the action space (S17).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path)
    env.reset(seed=5)
    first = env.action_space.sample()
    env.reset(seed=5)

    assert np.array_equal(first, env.action_space.sample())
