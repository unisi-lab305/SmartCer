# SmartCer configuration reference

SmartCer has two configuration levels:

- the **scenario**: a folder with `config.json` and one CSV file per node, the physical and
  economic description of the community (this document, sections 1-5);
- the **run**: the arguments of the environment (period, episodes, observation, reward, ...),
  section 6.

Symbols and equations are in [FORMULATION.md](FORMULATION.md). Typical use:

```python
import gymnasium as gym
import smartcer  # registers SmartCer-v1

env = gym.make('SmartCer-v1', scenario='path/to/scenario', start='2022-01-01 00:00',
               end='2022-12-31 23:00', episode_length=24 * 7, random_start=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Without the environment, `smartcer.data.load_scenario(path, start, end)` returns the
`Community` (nodes, incentive model, timestamps) and `community.step(t, actions)` simulates one
step.

## 1. Scenario format 1.0

```json
{
  "format_version": "1.0",
  "name": "molise",
  "description": "optional free text",
  "delta_t": 1.0,
  "units": "kWh",
  "incentive": {"model": "uniform", "unit_value": 0.11},
  "nodes": {
    "pv_1": {
      "type": "producer",
      "category": "plant",
      "csv": "data/pv_1.csv",
      "photovoltaic": {"tip_base": 0.08, "tip_threshold": 0.12, "fi_zone": 0.004, "pnrr": 0.0},
      "storage": {
        "capacity": 100, "power_charge_max": 30, "power_discharge_max": 30,
        "efficiency_charge": 0.95, "efficiency_discharge": 0.95, "soc_min": 0.1,
        "usage_cost": 0.01, "charge_limit": "production"
      }
    },
    "aggregated_load": {"type": "consumer", "csv": "data/consumers.csv"}
  }
}
```

### Top-level keys

| Key | Required | Default | Meaning |
|---|---|---|---|
| `format_version` | yes | – | must be `"1.0"`; without it the file is read as 0.6.2 (section 4) |
| `name` | no | folder name | scenario name |
| `description` | no | – | free text, not used |
| `delta_t` | no | `1.0` | step Δt in hours; every CSV must have exactly this spacing |
| `units` | no | `"kWh"` | `"kWh"`: CSV energies per step; `"kW"`: average power over the step (multiplied by Δt) |
| `incentive` | yes for `uniform` | `{"model": "uniform"}` | incentive model, section 1.1 |
| `nodes` | yes | – | object `{node_id: node}`; its order is the node order, and the order of the actions is that of the nodes with a battery |

Unknown keys raise `ValueError`.

### 1.1 Incentive

| `model` | Other keys | Incentive I_t (FORMULATION section 9) |
|---|---|---|
| `uniform` (default) | `unit_value`: a number (€/kWh), or `{"csv": "<path>", "column": "<name>"}` | π^inc_t · E^sh_t |
| `tip` | none | E^sh_t / Σ_u e^gri_u · Σ_u c_u,t e^gri_u, with the per-plant coefficient c of each node's `photovoltaic` section and its sell price as zonal price; nodes without a plant (consumers) have c = 0 |

A `unit_value` CSV has a `datetime` column and the named column; its timestamps in the period must
be those of the nodes. With `tip`, every producer and prosumer needs the four TIP keys.

### 1.2 Node keys

| Key | Required | Meaning |
|---|---|---|
| `type` | yes | `producer` (generation only), `consumer` (load only) or `prosumer` |
| `csv` | no | CSV path relative to the scenario folder; default `nodes/<type>s/<base>.csv`, where `<base>` is the node id without a `_N` suffix (`house_1` → `house.csv`, the 0.6.2 rule) |
| `category` | no | free text (e.g. `factory`); not used by the model |
| `photovoltaic` | with `tip` | `tip_base`, `tip_threshold`, `fi_zone` (€/kWh), `pnrr` (%, 0-40); `pv_power` and `pz` are accepted and not used |
| `storage` | no | battery, section 1.3; without it the node has no action |

### 1.3 Storage keys

| Key | Required | Default | Unit | Meaning |
|---|---|---|---|---|
| `capacity` | yes | – | kWh | S^max |
| `power_charge_max` | yes | – | kW | P^cha, node side |
| `power_discharge_max` | yes | – | kW | P^dis, node side |
| `efficiency_charge` | yes | – | – | η^cha in (0, 1] |
| `efficiency_discharge` | yes | – | – | η^dis in (0, 1] |
| `soc_min` | no | 0 | fraction of capacity | S^min |
| `soc_initial` | no | `soc_min` | fraction of capacity | S^ini |
| `alpha` | no | 0 | per hour | self-discharge; over a step the battery keeps (1 - α)^Δt of its charge |
| `usage_cost` | no | 0 | €/kWh of cell throughput | π^sto |
| `charge_limit` | no | `production` | – | charge cap c: `production` (c = g) or `surplus` (c = max(0, g - l)); a consumer's battery can never charge under `production` |

The 0.6.2 keys are also accepted, with a `DeprecationWarning`: `rated_power` → both power limits,
`efficiency` → both efficiencies, `alfa` → `alpha`, `beta` ignored (no degradation). Combining a
0.6.2 key with the 1.0 key it maps to, unknown keys and missing required keys raise `ValueError`.

## 2. CSV files

One file per node (several nodes may share one through `csv` or the `_N` rule; it is read once).

| Column | Producer | Consumer | Prosumer | Unit |
|---|---|---|---|---|
| `datetime` | required | required | required | ISO 8601, e.g. `2022-01-01 00:00:00`, start of the step |
| `production` | required | ignored | required | kWh per step, or kW with `units: kW` |
| `consumption` | ignored | required | required | kWh per step, or kW with `units: kW` |
| `sell_price` (alias `price_exp`) | required | optional | required | €/kWh |
| `buy_price` (alias `price_imp`) | ignored | required | required | €/kWh |

Other columns are ignored. A file with both an alias and its 1.0 name raises `ValueError`. A
consumer's `sell_price` is read when the column exists (and then validated like a required column):
it pays the energy the consumer's battery exports (discharge beyond the load); without the column
that energy earns nothing. A producer's buy price is 0: it never imports under the charge rule.

## 3. Validation

Errors (`ValueError`) name the node, the file, the column and the first offending row as a CSV line
number (header = line 1) with its timestamp:

- missing `datetime` or required columns; unparsable dates;
- missing or non-numeric values in a required column, negative `production` or `consumption`
  (checked in the period);
- unsorted or duplicated timestamps, or a spacing different from `delta_t` (checked on the whole
  file);
- nodes whose timestamps in the period differ (format 1.0), or whose numbers of rows in the period
  differ (both formats);
- an empty period, invalid keys or values in `config.json`.

## 4. Scenario format 0.6.2

A `config.json` without `format_version` and with the groups `Producers`, `Consumers`,
`Prosumers` is read with a `DeprecationWarning`, which also lists the keys that are not used:

| 0.6.2 | 1.0 |
|---|---|
| group `Producers` / `Consumers` / `Prosumers` | `type` `producer` / `consumer` / `prosumer`; node order: producers, consumers, prosumers |
| `Type` (e.g. `plant`, `factory`) | `category` |
| `Photovoltaic` | `photovoltaic` |
| `Storage` | `storage` with the 0.6.2 keys of section 1.3; `soc_initial` = `soc_min` |
| CSV in `nodes/<type>s/<base>.csv` | default `csv` |
| (implicit) | `delta_t: 1.0`, `units: "kWh"`, `incentive: {"model": "tip"}` |

**Timestamps.** As in 0.6.2, the rows of the nodes are aligned by position and the community uses
the timestamps of the first node. Every file must have the same number of rows (`ValueError`
otherwise); the period `start`–`end` is selected on the first node's timestamps and every node is
cut at the same rows. Unsorted, duplicated or wrongly spaced timestamps, and nodes whose
timestamps differ from the first node's, are reported in one `TimestampWarning`
(`smartcer.data.TimestampWarning`) instead of raising. The local Leather scenario needs this:
three of its files have timestamps shifted by one hour between 27/03 and 31/10/2022, and a period
may start or end inside that block. Converting a scenario to format 1.0 requires regular
timestamps.

## 5. Units inside SmartCer

Energies are kWh per step, powers kW, prices €/kWh, Δt hours (FORMULATION section 2). With
`units: kW` the loader multiplies `production` and `consumption` by Δt.

## 6. Run configuration

The run is configured by the arguments of `SmartCerEnv` (`smartcer.gym`), also accepted by
`gymnasium.make('SmartCer-v1', ...)`:

| Argument | Default | Meaning |
|---|---|---|
| `scenario` | – | scenario folder (sections 1-4) or a `smartcer.core.Community` |
| `start`, `end` | first / last row | period, both inclusive (with a folder only) |
| `episode_length` | whole period | steps per episode |
| `random_start` | `False` | first step of each episode drawn with `env.np_random`; otherwise the first step of the period |
| `observe_current_step` | `True` | information timing, FORMULATION section 12; with `False` the generation and load of the first row of the data are observed as 0 |
| `observation` | `'meco'` | observation preset: `'meco'` ([M24] Tab. II) or `'extended'`; see below and `env.obs_layout` |
| `lookback` | `0` | past steps of generation and load per battery node (`'extended'` only) |
| `normalizer` | `None` | fitted normaliser of generation and load: a `MinMaxNormalizer` or the path of its JSON file |
| `normalizer_window` | whole period | `(first, last)` timestamps, both inclusive (either may be `None`), of the training window the normaliser is fitted on; not with `normalizer` |
| `reward` | `'welfare'` | reward type: `'welfare'` (R_t = W_t - σ K_t), `'profit'` or `'energy'` (FORMULATION section 11) |
| `reward_per_node` | `False` | per-node variant of `'profit'` or `'energy'`: r_u = m_u - σ k_u per battery node in `info['reward']['nodes']`, scalar reward Σ_u r_u |
| `penalty_coefficient` | `10.0` | σ |
| `forecast` | `None` | forecasts of the observation's `forecast` section: `'perfect'` (true future values, `smartcer.forecasts.PerfectForecast`) or a provider with `reset(rng)` and `get(t, horizon)` (`smartcer.forecasts.ForecastProvider`) |
| `forecast_horizon` | `0` | forecast horizon H in steps, ≥ 1 with `forecast`; episodes then use the steps 0 … n - H - 1 of the period only, so that every forecast lies in the data |
| `oracle` | `None` | MILP oracle: an oracle folder (below) or a `smartcer.oracle.OracleProvider` of the same community; its data go to `info['privileged']` and to the `ReplayController` |
| `seed` | `None` | seed used by the first `reset()` called without a seed |

Actions are clipped to [-1, 1] without penalty before the nodes clip them to their dynamic bounds
(penalised, FORMULATION section 7); non-finite actions raise `ValueError`. Episodes end on their last
step with `truncated=True`.

### Observation

The observation is a flat `Box`; `env.obs_layout` maps each feature name to its slice. Sections,
in order:

| Section | Features | Presets |
|---|---|---|
| time | `time/month`, `time/hour` (sin, cos), `time/day_of_week` (one-hot, Monday first) | both |
| time | `time/holiday`: 1 on Italian national holidays (`holidays` package), else 0; regional and local holidays (e.g. the patron saint of a municipality) are not included | `extended` |
| price | `price/sell`, `price/buy`, €/kWh, not normalised, when all nodes share the same series; otherwise per battery node (below) | both |
| community | `community/generation`, `community/load`: totals of the nodes without a battery | `extended` |
| nodes | per battery node, in action order: [`<node>/sell_price`, `<node>/buy_price`], `<node>/generation`, `<node>/load`, [`<node>/generation_lookback`, `<node>/load_lookback`: `lookback` values, most recent first], `<node>/soc` (scaled to [S^min, S^max]), `<node>/a_lower`, `<node>/a_upper` | both (lookback: `extended`) |
| forecast | with `forecast`, per battery node in action order: `forecast/<node>/generation`, `forecast/<node>/load`, H values each, ordered by lead time: steps t + 1 … t + H with `observe_current_step=True`, t … t + H - 1 with `False` (FORMULATION section 12); empty without `forecast` | both |

Generation and load (also the community totals, the lookback and the forecasts, which use the
bounds of `<node>/generation` and `<node>/load`) are min-max scaled with the
bounds of the training window `normalizer_window` and clipped to [0, 1], so that values beyond the
training range (e.g. in a test period) stay inside the space; `info['obs_clipped']` counts them.
The forecast provider is reset with `env.np_random` at every `reset()`, so that a stochastic
provider is reproducible from the seed.
With `observe_current_step=False`
they are those of step t - 1; values before the first row of the data are 0. Save the normaliser of
a training environment and pass it to the evaluation one:

```python
train = SmartCerEnv('path/to/scenario', end='2022-09-30 23:00')
train.normalizer.save('normalizer.json')    # bounds, window and scenario name
test = SmartCerEnv('path/to/scenario', start='2022-10-01 00:00', normalizer='normalizer.json')
```

### Info

`info` of `reset()`: `t`, `timestamp`, `privileged`, `episode_start`, `episode_end`,
`obs_clipped`. `info` of `step()`: `t`, `timestamp`, `privileged`, `obs_clipped`, `reward` (`type`,
`main`, `penalty` = weighted penalty,
`total` = R_t and, with `reward_per_node`, `nodes`: node → `main`, `penalty`, `total`), `welfare`
(revenue, cost, storage cost, incentive, shared, exported and imported energy, total), `penalty`
(K_t), `action` (after the [-1, 1] clip) and `nodes` (per-node flows, economics, state of charge,
requested and applied action, bounds). `privileged` holds data for the critic only, about the step
of the observation it comes with (the step of the next action; the last step again with the
terminal observation). It is empty without `oracle` and `forecast`; otherwise it holds `t` (that
step), with `oracle` `oracle_action` (a\*_t, one value per battery in action order) and
`oracle_soc` (the oracle state of charge at the end of step t, kWh), and with `forecast`
`true_future` (`'<node>/<quantity>'` → the true values of the observed forecast steps, kWh per
step). `obs_clipped` is the number of values of the returned
observation that were clipped to [0, 1] because they lie outside the normaliser's training range
(the zeros before the first row of the data are not counted); it is 0 when the normaliser is fitted
on the whole period.

### 0.6.2 run keys

`SmartCerEnv.from_v062_config(cer_path, cer_config)` maps a 0.6.2 `cer_config` with a
`DeprecationWarning` that lists the keys it does not use:

| 0.6.2 key | 1.0 argument |
|---|---|
| `simulation.start`, `simulation.end` | `start`, `end` |
| `simulation.seed` | `seed` |
| `reward.type` `'profit'` or `'cost'` with `reward.single_agent: false` | `reward='welfare'` |
| `reward.type` `'profit'` or `'cost'` with `reward.single_agent: true` | `reward='profit'`, `reward_per_node=True` |
| `reward.type` `'energy'` | `reward='energy'`, with `reward_per_node=True` if `reward.single_agent` is true |
| `reward.constraints.coefficient` | `penalty_coefficient` (0 if `reward.constraints.penalty` is false) |
| `simulation.delta_input`, `simulation.delta_output`, `reward.constraints.max_ratio`, `reward.scaler.*` | not used (no reward scaling in the environment) |

With `single_agent: true` the per-node rewards are in `info['reward']['nodes']` and the scalar
reward is their sum (0.6.2 returned a list). Other reward types raise `ValueError`.

### Output directory

Runs are written to an output directory chosen by the caller (e.g. `results/`, gitignored), never
into the scenario folder or a `cer_data` folder (`ValueError`; 0.6.2 wrote into
`cer_data/<scenario>/simulations/`):

```python
from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import compute_baselines

env = SmartCerEnv('path/to/scenario', start='2022-01-01', end='2022-12-31 23:00')
kpis = compute_baselines(env, 'results/baselines/my_run', seed=0)   # 'naive', 'rbc_community'
```

| File | Content |
|---|---|
| `<controller>/community.csv` | one row per step: `datetime`, `episode`, `t`, `reward`, `reward_main`, `reward_penalty`, `welfare`, `revenue`, `cost`, `storage_cost`, `incentive`, `shared_energy_kwh`, `export_kwh`, `import_kwh`, `penalty` (K_t), `obs_clipped` |
| `<controller>/nodes/<node>.csv` | one row per step: `datetime`, `episode`, `t` and the node's flows, economics, state of charge, requested and applied action, bounds, k (and the node reward with `reward_per_node`) |
| `<controller>/kpis.json` | KPIs of the episode (`smartcer/gym/metrics/KPI.md`) |
| `<controller>/run_config.json` | `env.run_config()`: version, scenario, period, episode, observation and reward settings, controller |
| `kpis.json` | KPIs of every controller |

The same tables come from any run through the `TrackResults` wrapper
(`env.get_wrapper_attr('tracker').save(folder, env.unwrapped.run_config())`) and, during
Stable-Baselines3 training, from `ResultsCallback(output_dir, save_freq)`, which also logs the KPIs
of each episode under `smartcer/`.

### Oracle files

The MILP outputs of the DIISM are read by `smartcer.oracle.read_oracle`, the only function that
knows their format. The format below is provisional, pending the DIISM answers (brief section 8).
An oracle folder holds:

| File | Columns | Content |
|---|---|---|
| `trajectory.csv` | `datetime`, `node_id`, `p_cha_kw`, `p_dis_kw`, `soc_kwh` | one row per step and battery node, in any order. `datetime`: start of the step, without time zone, as in the scenario CSV files. `p_cha_kw`, `p_dis_kw`: average charge and discharge power of the step on the node side (FORMULATION section 4), ≥ 0 (values down to -1e-9 are read as 0). `soc_kwh`: state of charge at the end of the step |
| `objective.csv` (optional) | `date`, `objective` | one row per calendar day: the MILP objective of the day as a welfare in € (FORMULATION section 10, larger is better) |

Other columns are ignored. The node ids must be the battery nodes of the scenario, every step of
the period needs a row for every battery node (rows outside the period are ignored), and every day
of the period needs an objective. `OracleProvider` computes a\* = p^cha / P^cha - p^dis / P^dis per
step and rejects |a\*| > 1 + 1e-9.

`check_replay` plays a\* over the whole period with the `ReplayController` (create the environment
without `episode_length` and forecasts) and checks that K_t ≤ 1e-9 at every step and, with an
objective, that the cumulative welfare lies within `rtol` (default 1e-3) of the cumulative
objective:

```python
from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import check_replay

env = SmartCerEnv('path/to/scenario', oracle='path/to/scenario/oracle')
result = check_replay(env, 'results/oracle_replay', rtol=1e-3)   # ReplayCheck; result.passed
```

| File | Content |
|---|---|
| `oracle_daily.csv` | one row per day: `date`, `welfare` of the replay and, with an objective, `objective` and `difference` = welfare - objective |
| `oracle_replay.json` | the `ReplayCheck` fields (steps, penalty maximum and steps, largest state-of-charge difference from the oracle, welfare, objective, relative gap, rtol, Fit Score of the daily welfare against the objective, passed) and the run configuration |

`compute_baselines(env, folder, controllers=['naive', 'rbc_community', 'replay'])` also writes the
tables of the replay. The replay test of the test suite looks for oracle folders in
`smartcer/cer_data/<scenario>/oracle/` and is skipped when there is none.
