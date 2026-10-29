# Installation

SmartCer requires **Python 3.12 or newer**. The runtime dependencies are `gymnasium`, `numpy`,
`pandas` and `holidays`.

## From PyPI

```bash
pip install smartcer
# with Stable-Baselines3, for the training example and the results callback
pip install "smartcer[rl]"
```

## From source

A source checkout also gives access to the synthetic example scenario used in the
[quickstart](quickstart.md).

```bash
git clone https://github.com/unisi-lab305/SmartCer
cd SmartCer
pip install -e .          # plain editable installation
pip install -e .[all]     # tests, lint tools and Stable-Baselines3
make test
```

To install a tagged release directly from GitHub:

```bash
pip install git+https://github.com/unisi-lab305/SmartCer@v1.0.1
```

## Environment registration

The environment ID is `SmartCer-v1`. It is registered when `smartcer` is imported:

```python
import gymnasium as gym
import smartcer  # registers SmartCer-v1

env = gym.make('SmartCer-v1', scenario='path/to/scenario')
```

With Gymnasium 1.0 or newer, import `smartcer` before `gym.make`, or use
`gym.make('smartcer:SmartCer-v1', ...)`.

## Building these docs

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.
