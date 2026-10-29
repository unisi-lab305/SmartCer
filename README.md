# SmartCer

**SmartCer** is a [Gymnasium](https://gymnasium.farama.org/) environment for Italian Renewable
Energy Communities (CER, *comunità energetiche rinnovabili*). A community is a set of producers,
consumers and prosumers, some of them with a battery. In every hour it earns an incentive on the
shared energy (*energia condivisa*), the minimum between the energy its members feed into the grid
and the energy they take from it. A reinforcement learning agent controls the batteries to
maximise the social welfare of the community: sales, minus purchases and storage costs, plus the
incentive.

> **Beta.** Version 1.0.1 is a beta release: validation against a MILP reference trajectory is in
> progress. The MILP oracle file format and the forecast error presets are provisional.

**Documentation: <https://unisi-lab305.github.io/SmartCer/>**

## Installation

Python 3.12 or newer.

```bash
pip install smartcer
# with Stable-Baselines3, for the training example and the results callback
pip install "smartcer[rl]"
```

From source, for development:

```bash
git clone https://github.com/unisi-lab305/SmartCer
cd SmartCer
pip install -e .[all]       # tests, lint tools and Stable-Baselines3
make test
```

## Quick start

The PyPI package ships no scenario. This example runs on the synthetic scenario
`test/fixtures/mini_cer_v1` of the repository, so it needs a source checkout (`git clone` and
`pip install -e .`). It is the file
[examples/first_simulation.py](https://github.com/unisi-lab305/SmartCer/blob/main/examples/first_simulation.py).

```python
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
```

For your own data, use `gym.make('SmartCer-v1', scenario='path/to/scenario', ...)`; the scenario
format, the observation, reward, forecasts, MILP oracle and baselines are described in the
[documentation](https://unisi-lab305.github.io/SmartCer/). Coming from 0.6.2? See
[Migrating from 0.x](https://unisi-lab305.github.io/SmartCer/migration.html).

## Citation

If you use **SmartCer** in your research, please cite it (GitHub shows a "Cite this repository"
entry from [CITATION.cff](https://github.com/unisi-lab305/SmartCer/blob/main/CITATION.cff)):

```bibtex
@software{smartcer,
    author  = {Guiducci, Leonardo},
    title   = {SmartCer: A Gymnasium Environment for Renewable Energy Communities},
    year    = {2026},
    version = {1.0.1},
    url     = {https://github.com/unisi-lab305/SmartCer},
}
```

If you use the reinforcement learning formulation, please also cite:

```bibtex
@inproceedings{guiducci2023meco,
  author    = {Guiducci, Leonardo and Palma, Giulia and Stentati, Marta and Rizzo, Antonio and Paoletti, Simone},
  title     = {A Reinforcement Learning approach to the management of Renewable Energy Communities},
  booktitle = {2023 12th Mediterranean Conference on Embedded Computing (MECO)},
  year      = {2023},
  pages     = {1--8},
  doi       = {10.1109/MECO58584.2023.10154979}
}
```

## License

This project is licensed under the MIT License. For more details, see the
[LICENSE](https://github.com/unisi-lab305/SmartCer/blob/main/LICENSE) file.
