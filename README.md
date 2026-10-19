# SmartCer

> [!WARNING]
> **SmartCer 1.0 is under development on `main`.** The 0.6.2 API described below (`SmartCer-v0`,
> `smartcer.cer_env`, the 0.6.2 rewards and run configuration) has been removed; the environment
> is now `SmartCer-v1` (`docs/CONFIG.md`, `docs/FORMULATION.md`). To use the old version, check out
> the tag `v0.6.2` (`git checkout v0.6.2`) and follow the README of that version. This README will
> be rewritten for the 1.0 release.

**SmartCer** is a Gymnasium environment for the simulation of Renewable Energy Communities (CERs). SmartCer allows the
development and testing of reinforcement learning algorithms for the control of energy storage systems in a CER. The
objective of these RL-based controllers is to minimize energy costs, maximize incentives, and maintain grid balance.  
The environment is based on the [Gymnasium](https://gymnasium.farama.org/index.html) framework, which provides a
standardized interface for reinforcement learning environments.  

For detailed information on how to set up and use the environment, refer to the sections below.

---

## Requirements

- Python ≥ 3.10
- Virtual Python environment
- Python package manager (pip)

---

## Installation

1. Clone the repository locally:
    ```bash
    git clone https://github.com/unisi-lab305/SmartCer
    cd SmartCer
    ```
2. Create and activate a Python virtual environment:
    - Using `venv`:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    - Using `conda`:
        ```bash
        conda create --name smartcer python=3.10
        conda activate smartcer
        ```
3. Install the required dependencies:
    - For standard installation:
        ```bash
        pip install .
        ```
    - For editable installation (for development):
        ```bash
        pip install -e .
        ```

---

## Usage

Here’s a basic example of how to interact with **SmartCer**:

```python
from smartcer.cer_env import SmartCer

# to reproduce the results
seed = 42

# path to the CER data folder
path = './path_to_cer'

# configuration of the CER simulation
cer_config = {
    'simulation': {
      'start': '2022-01-01 00:00:00',
      'end': '2022-01-01 23:00:00',
      'delta_input': '1h',
      'delta_output': 'YE',
      'seed': seed
    },
    'reward': {
      'type': 'cost',
      'single_agent': True,
      'constraints': {
        'penalty': True,
        'coefficient': 0.3,
        'max_ratio': 0.99
      },
      'scaler': {
         'apply': True,
         'clip_min': None,
         'clip_max': None,
         'reset_interval': None
      }
    }
  }

# create the SmartCer environment
cer = SmartCer(cer_path=path, cer_config=cer_config)

# compute the baselines and save results to "baseline_no_storage" and "baseline_rbc" folders
cer.compute_baselines(no_storage=True, rbc=True)

# loop for RL training
results = None
done = False
obs, info = cer.reset(seed)
episodes = 1
for i in range(cer.sim_hours * episodes):
    # Note: replace actions with your RL agent choose_actions method and use rl_step function
    actions = cer.action_space.sample()    
    new_obs, reward, terminated, truncated, new_info, _ = cer.step(actions)

    done = terminated or truncated
    if done:
        # terminate the episode
        results = cer.get_sim_results()
        cer.reset()
    else:
        # update the observation and info
        obs = new_obs
        info = new_info
# save the results to CER data folder
cer.save_results(results, output_dir='baseline_random', save_mode='csv')
```

---

## Citation

If you use **SmartCer** in your research, please cite this repository as follows:

```bibtex
@misc{smartcer,
    author = {Guiducci, Leonardo},
    title = {SmartCer: A Gymnasium Environment for Renewable Energy Communities},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/unisi-lab305/SmartCer}},
}
```

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./LICENSE) file.

