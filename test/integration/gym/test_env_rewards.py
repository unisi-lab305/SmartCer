"""
Reward types of `SmartCerEnv` over full episodes of the synthetic scenario.
"""

from pathlib import Path

import numpy as np
import pytest

from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.integration


def run(env: SmartCerEnv, seed: int = 0) -> list[tuple[float, dict]]:
    """
    Run one episode with sampled actions.

    Args:
        env: Environment.
        seed: Seed of the reset and of the action space.

    Returns:
        list[tuple[float, dict]]: Reward and info of every step.
    """
    env.reset(seed=seed)
    steps, truncated = [], False
    while not truncated:
        _, reward, _, truncated, info = env.step(env.action_space.sample())
        steps.append((reward, info))
    return steps


def test_welfare_reward_over_an_episode(mini_cer_v1_path: Path) -> None:
    """
    Test R_t = W_t - σ K_t at every step of a 48 h episode, and the empty privileged channel.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, penalty_coefficient=4.0)
    _, reset_info = env.reset(seed=0)

    assert reset_info['privileged'] == {}
    for reward, info in run(env):
        assert reward == pytest.approx(info['welfare']['total'] - 4.0 * info['penalty'])
        assert info['reward']['type'] == 'welfare' and 'nodes' not in info['reward']
        assert info['privileged'] == {}


def test_collective_profit_equals_welfare(mini_cer_v1_path: Path) -> None:
    """
    Test that the collective profit reward equals the welfare reward on the same trajectory.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    welfare = [r for r, _ in run(SmartCerEnv(mini_cer_v1_path))]
    profit = [r for r, _ in run(SmartCerEnv(mini_cer_v1_path, reward='profit'))]

    assert profit == pytest.approx(welfare)


@pytest.mark.parametrize('reward', ['profit', 'energy'])
def test_per_node_rewards_sum_to_the_reward(mini_cer_v1_path: Path, reward: str) -> None:
    """
    Test the per-node variant: one reward per battery node in info, r_u = m_u - σ k_u, and the
    scalar reward equal to their sum at every step.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        reward: Reward type.
    """
    env = SmartCerEnv(mini_cer_v1_path, reward=reward, reward_per_node=True)

    for total, info in run(env, seed=3):
        nodes = info['reward']['nodes']
        assert list(nodes) == list(env.storage_node_ids)
        assert total == pytest.approx(sum(r['total'] for r in nodes.values()))
        for node_id, r in nodes.items():
            node = info['nodes'][node_id]
            main = (node['revenue'] - node['cost'] - node['storage_cost'] if reward == 'profit'
                    else -node['import_kwh'] ** 2)
            assert (r['main'], r['penalty']) == pytest.approx((main, 10.0 * node['penalty']))


def test_energy_reward_is_minus_squared_imports(mini_cer_v1_path: Path) -> None:
    """
    Test the collective energy reward: -Σ_u (i_u)² over all nodes minus σ K_t.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
    """
    env = SmartCerEnv(mini_cer_v1_path, reward='energy', penalty_coefficient=0.0)

    for reward, info in run(env, seed=1):
        imports = np.array([n['import_kwh'] for n in info['nodes'].values()])
        assert reward == pytest.approx(-np.sum(imports ** 2))
