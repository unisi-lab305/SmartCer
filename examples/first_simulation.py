"""First SmartCer simulation on the synthetic example scenario of the repository.

The scenario ``test/fixtures/mini_cer_v1`` (4 nodes, 2 batteries, 48 hours) is synthetic example
data, available from a source checkout (``git clone`` and ``pip install -e .``). It is not part
of the PyPI package.
"""

from pathlib import Path

import gymnasium as gym

import smartcer  # registers SmartCer-v1
from smartcer.gym.controllers import compute_baselines


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = REPO_ROOT / 'test' / 'fixtures' / 'mini_cer_v1'
OUTPUT_DIR = REPO_ROOT / 'results' / 'first_simulation'


def main() -> None:
    """Run one episode with random actions, then compare the baseline controllers."""
    env = gym.make('SmartCer-v1', scenario=SCENARIO, episode_length=24)
    obs, info = env.reset(seed=0)
    print(f'observation: {obs.shape}, action space: {env.action_space}')

    done = False
    while not done:
        action = env.action_space.sample()  # one action per battery, positive charges
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print('last step welfare:', info['welfare'])
    print('last step penalty:', info['penalty'])

    kpis = compute_baselines(env.unwrapped, OUTPUT_DIR, seed=0)  # naive and rbc_community
    for name, values in kpis.items():
        print(f'{name}: welfare {values["welfare"]:.2f} EUR, '
              f'sharing rate {values["sharing_rate"]:.2f}')
    print(f'tables and KPIs written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
