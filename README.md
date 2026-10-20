# SmartCer

**SmartCer** is a [Gymnasium](https://gymnasium.farama.org/) environment for Italian Renewable
Energy Communities (CER, *comunità energetiche rinnovabili*). A community is a set of producers,
consumers and prosumers, some of them with a battery. In every hour it earns an incentive on the
shared energy (*energia condivisa*), the minimum between the energy its members feed into the grid
and the energy they take from it. A reinforcement learning agent controls the batteries to
maximise the social welfare of the community: sales, minus purchases and storage costs, plus the
incentive.

- One continuous action per battery in [-1, 1], with dynamic bounds derived from the battery
  dynamics and a penalty on the part of an action that exceeds them.
- Reward R_t = W_t − σ K_t (welfare minus the weighted constraint penalty); `profit` and
  `energy` rewards, collective or per node.
- Uniform or TIP incentive (the Italian *tariffa incentivante premio*, with the PNRR correction).
- Observation presets, explicit information timing, min-max normalisation fitted on a training
  window.
- Forecasts of generation, load and sale price: perfect, or with simulated errors (MMFE).
- A MILP oracle: optimal actions for an asymmetric actor-critic, replay check and Fit Score.
- Baseline controllers (idle batteries, rule-based), results tables and KPIs, a Stable-Baselines3
  callback.
- One year of hourly data with 10 nodes runs in about 3-4 s.

The model is defined in [docs/FORMULATION.md](docs/FORMULATION.md); every scenario and run option
is in [docs/CONFIG.md](docs/CONFIG.md).

---

## Installation

Python 3.12 or newer. SmartCer is installed from a GitHub tag (it is not on PyPI):

```bash
pip install git+https://github.com/unisi-lab305/SmartCer@v1.0.0
# with Stable-Baselines3, for the training example and the results callback
pip install "SmartCer[rl] @ git+https://github.com/unisi-lab305/SmartCer@v1.0.0"
```

For development:

```bash
git clone https://github.com/unisi-lab305/SmartCer
cd SmartCer
pip install -e .[all]       # tests, lint tools and Stable-Baselines3
make test
```

---

## Quick start

```python
import gymnasium as gym
import smartcer   # registers SmartCer-v1

env = gym.make('SmartCer-v1', scenario='path/to/scenario', episode_length=168, random_start=True)
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()      # one action per battery, positive charges
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info['welfare'], info['penalty'])     # welfare breakdown and K_t of the last step
```

The environment can also be built directly, e.g. `SmartCerEnv('path/to/scenario', ...)` from
`smartcer.gym`. `env.unwrapped.obs_layout` maps each feature name to its slice of the flat
observation vector.

### Scenario

A scenario is a folder with a `config.json` and one CSV file per node:

```json
{
  "format_version": "1.0",
  "name": "molise",
  "delta_t": 1.0,
  "units": "kWh",
  "incentive": {"model": "uniform", "unit_value": 0.11},
  "nodes": {
    "pv_1": {
      "type": "producer",
      "csv": "data/pv_1.csv",
      "storage": {
        "capacity": 100, "power_charge_max": 30, "power_discharge_max": 30,
        "efficiency_charge": 0.95, "efficiency_discharge": 0.95, "soc_min": 0.1,
        "usage_cost": 0.01
      }
    },
    "aggregated_load": {"type": "consumer", "csv": "data/consumers.csv"}
  }
}
```

The CSV columns are `datetime`, `production` and `consumption` (kWh per step, or kW with
`units: kW`), `sell_price` and `buy_price` (€/kWh). Scenarios in the 0.6.2 format are still read,
with a `DeprecationWarning`. The full format is in [docs/CONFIG.md](docs/CONFIG.md), sections 1-4.

### Run options

```python
from smartcer.gym import SmartCerEnv

env = SmartCerEnv(
    'path/to/scenario',
    start='2022-01-01 00:00', end='2022-09-30 23:00',  # period
    episode_length=168, random_start=True,             # one-week episodes, random start
    observation='extended', lookback=24,               # preset 'meco' ([M24]) or 'extended'
    observe_current_step=True,                         # False: causal timing (g, l of t - 1)
    reward='welfare', penalty_coefficient=10.0,        # R_t = W_t - σ K_t
    seed=0,
)
env.normalizer.save('normalizer.json')   # reuse the training bounds in an evaluation env
test = SmartCerEnv('path/to/scenario', start='2022-10-01 00:00', normalizer='normalizer.json')
```

---

## Forecasts

With `forecast` and `forecast_horizon` H, the observation gets a `forecast` section for the next H
steps:

- the sale price;
- the generation and load of every battery node;
- the totals of the nodes without a battery (e.g. aggregated consumers).

```python
# true future values
env = SmartCerEnv('path/to/scenario', forecast='perfect', forecast_horizon=12)
# simulated forecast errors, preset 'mmfe_perfect', 'mmfe_standard' or 'mmfe_noisy'
env = SmartCerEnv('path/to/scenario', forecast='mmfe_standard', forecast_horizon=12)
```

`MMFEForecast` follows the martingale model of forecast evolution:

- The innovations of each target hour are drawn once per episode and revealed one per step. The
  error variance therefore grows linearly with the lead time, and the forecasts of the same hour
  stay consistent as it approaches.
- The errors of different nodes are correlated through a common factor, which is AR(1) over the
  target hours.
- PV errors scale with a monthly reference profile and are zero at night; load errors are
  multiplicative.
- The noise follows the reset seed.

The preset values are provisional, pending calibration. Custom parameters, with a smaller σ for an
aggregated load:

```python
from smartcer.data import load_scenario
from smartcer.forecasts import MMFEForecast, MMFEParameters

community = load_scenario('path/to/scenario')
forecast = MMFEForecast(community, MMFEParameters(
    sigma_generation=0.07, sigma_load=0.03, rho=0.5, phi=0.8,
    sigma_overrides={'community/load': 0.01}))
env = SmartCerEnv(community, forecast=forecast, forecast_horizon=12)
```

`info['privileged']['true_future']` holds the true values of the forecast steps, for a critic.
Details: [docs/CONFIG.md](docs/CONFIG.md), section 6, Forecasts.

---

## MILP oracle

The optimal battery trajectories of a MILP (provisional file format) go in an oracle folder:

```text
path/to/scenario/oracle/
├── trajectory.csv   # datetime, node_id, p_cha_kw, p_dis_kw, soc_kwh: one row per step and battery
└── objective.csv    # optional: date, objective (welfare of the day, €)
```

```python
from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import check_replay

env = SmartCerEnv('path/to/scenario', oracle='path/to/scenario/oracle')
obs, info = env.reset(seed=0)
info['privileged']        # {'t', 'oracle_action': a*_t, 'oracle_soc': s*_t}, for the critic only

result = check_replay(env, 'results/oracle_replay')   # plays a* over the whole period
result.passed, result.relative_gap, result.fit_score
```

- The optimal actions are a\* = p^cha / P^cha − p^dis / P^dis.
- `check_replay` checks that no step is penalised and that the cumulative welfare lies within
  1e-3 of the MILP objective. It writes a per-day comparison.
- The MILP's initial state of charge, inferred from the first row of the file, must match the
  scenario's `soc_initial`.

---

## Baselines and results

```python
from smartcer.gym.controllers import compute_baselines

kpis = compute_baselines(env, 'results/baselines', controllers=('naive', 'rbc_community'), seed=0)
```

- The controllers are `naive` (idle batteries), `rbc_node` and `rbc_community` (the rule-based
  controller of [M24] in two modes), and `replay` (the oracle actions).
- Each run writes a community table, one table per node, the KPIs and the run configuration to an
  output folder, never into the scenario folder.
- The `TrackResults` wrapper records any run in the same format.
- The KPI formulas are in [smartcer/gym/metrics/KPI.md](smartcer/gym/metrics/KPI.md).

### Training with Stable-Baselines3

```python
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from smartcer.gym import SmartCerEnv
from smartcer.gym.tracking import ResultsCallback, TrackResults

env = SmartCerEnv('path/to/scenario', end='2022-09-30 23:00', episode_length=168,
                  random_start=True, forecast='mmfe_standard', forecast_horizon=12, seed=0)
vec_env = DummyVecEnv([lambda: TrackResults(env)])
callback = ResultsCallback('results/sac', save_freq=10, run_config=env.run_config())
model = SAC('MlpPolicy', vec_env, seed=0)
model.learn(total_timesteps=200_000, callback=callback)   # KPIs logged under smartcer/
```

---

## Migrating from 0.6.2

The 0.6.2 code is at the tag `v0.6.2`. Version 1.0.0 is a rewrite, and its results are not
numerically comparable with those of 0.6.2. The defects S1-S18 are listed in
[docs/refactor/REFACTOR_v1.md](docs/refactor/REFACTOR_v1.md), section 2.

### API

| 0.6.2 | 1.0.0 |
|---|---|
| `from smartcer.cer_env import SmartCer`; `SmartCer(cer_path, cer_config)` | `gym.make('SmartCer-v1', scenario=..., ...)` or `smartcer.gym.SmartCerEnv(scenario, ...)`; `SmartCerEnv.from_v062_config(cer_path, cer_config)` maps a 0.6.2 run configuration |
| environment ID `SmartCer-v0` | `SmartCer-v1` |
| `step()` returns 6 items, the reward is a list `[reward, penalty, signal]` | `step()` returns `(obs, reward, terminated, truncated, info)` with a float reward; the components are in `info['reward']` (S3) |
| `rl_step()` flattens the observation | the observation is a flat `Box`; `env.obs_layout` names its parts |
| `cer.sim_hours` | `env.episode_length` |
| `cer.compute_baselines(no_storage=True, rbc=True)` | `compute_baselines(env, output_dir, controllers=('naive', 'rbc_community'))` |
| `cer.get_sim_results()`, `cer.save_results(...)` into `cer_data/<scenario>/simulations/` | the `TrackResults` wrapper and `tracker.save(output_dir)`; output folders inside the scenario or `cer_data` are refused (S18) |
| `smartcer.cer_metrics`, `smartcer.storage_controllers` | `smartcer.gym.metrics`, `smartcer.gym.controllers` |

### Configuration keys

| 0.6.2 | 1.0.0 |
|---|---|
| `config.json` with `Producers`, `Consumers`, `Prosumers` | `format_version: "1.0"`, `nodes` with `type`; the old format is read with a `DeprecationWarning` |
| storage `rated_power` | `power_charge_max`, `power_discharge_max` (kW, node side) |
| storage `efficiency` | `efficiency_charge`, `efficiency_discharge` |
| storage `alfa` | `alpha` (per hour) |
| storage `beta` | ignored: degradation is not modelled |
| CSV `price_exp`, `price_imp` | `sell_price`, `buy_price` (the old names are still accepted) |
| (implicit hourly data) | `delta_t`, `units` |
| `simulation.start`, `simulation.end`, `simulation.seed` | `start`, `end`, `seed` |
| `reward.type` `'cost'` / `'profit'` | `reward='welfare'` (collective) or `'profit'` with `reward_per_node=True` (`single_agent`) |
| `reward.constraints.coefficient` | `penalty_coefficient` (default 10) |
| `reward.scaler.*` | removed: no reward scaling inside the environment (S9) |
| `simulation.delta_input`, `delta_output`, `reward.constraints.max_ratio` | removed (never used) |

### Behaviour

| Change | Defect |
|---|---|
| The state of charge never leaves [S^min, S^max]: the action bounds are derived from the battery dynamics, with both efficiencies | S4 |
| Explicit time step Δt: energy = power × Δt, so sub-hourly data work | S5 |
| The rule-based controller computes energies within the bounds and divides by the battery power, not by the capacity | S6 |
| The information timing is an option, `observe_current_step` (True observes g and l of step t, as in 0.6.2; False is causal) | S7 |
| Consumers can have a battery | S8 |
| The terminal observation is that of the step after the episode, not step 0 | S10 |
| KPIs are named after what they measure: `self_consumption` is now `sharing_rate`; `net_metering` and `energy_balance` are removed | S12 |
| About 12 times faster (one year of Leather: about 3.5 s instead of 43 s) | S13 |
| `reset(seed)` also seeds the action space | S17 |
| Errors raise exceptions instead of calling `sys.exit` | S11 |

---

## Documentation

- [docs/FORMULATION.md](docs/FORMULATION.md): physics, economics, reward and information timing.
  It prevails over the code.
- [docs/CONFIG.md](docs/CONFIG.md): scenario format, run arguments, observation, forecasts,
  oracle files and outputs.
- [smartcer/gym/metrics/KPI.md](smartcer/gym/metrics/KPI.md): KPI formulas.
- [CHANGELOG.md](CHANGELOG.md): changes per version.

The model follows two papers:

- [S22] M. Stentati, S. Paoletti, A. Vicino, *Optimization of energy communities in the Italian
  incentive system*, IEEE PES ISGT-Europe 2022.
- [M24] L. Guiducci, G. Palma, M. Stentati, A. Rizzo, S. Paoletti, *A Reinforcement Learning
  approach to the management of Renewable Energy Communities*, MECO 2024.

---

## Citation

If you use **SmartCer** in your research, please cite this repository:

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
