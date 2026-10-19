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
the timestamps of the first node. Unsorted, duplicated or wrongly spaced timestamps, and nodes
whose timestamps differ from the first node's, are reported in one `TimestampWarning`
(`smartcer.data.TimestampWarning`) instead of raising; the nodes must still have the same number of
rows in the period. The local Leather scenario needs this: three of its files have timestamps
shifted by one hour between 27/03 and 31/10/2022. Converting a scenario to format 1.0 requires
regular timestamps.

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
| `observation` | `'meco'` | observation preset ([M24] Tab. II); see `env.obs_layout` |
| `reward` | `'welfare'` | R_t = W_t - σ K_t (other types from phase 4) |
| `penalty_coefficient` | `10.0` | σ |
| `seed` | `None` | seed used by the first `reset()` called without a seed |

Actions are clipped to [-1, 1] without penalty before the nodes clip them to their dynamic bounds
(penalised, FORMULATION section 7); non-finite actions raise `ValueError`. Episodes end on their last
step with `truncated=True`.

### 0.6.2 run keys

`SmartCerEnv.from_v062_config(cer_path, cer_config)` maps a 0.6.2 `cer_config` with a
`DeprecationWarning` that lists the keys it does not use:

| 0.6.2 key | 1.0 argument |
|---|---|
| `simulation.start`, `simulation.end` | `start`, `end` |
| `simulation.seed` | `seed` |
| `reward.type` `'profit'` or `'cost'` with `reward.single_agent: false` | `reward='welfare'` |
| `reward.constraints.coefficient` | `penalty_coefficient` (0 if `reward.constraints.penalty` is false) |
| `simulation.delta_input`, `simulation.delta_output`, `reward.constraints.max_ratio`, `reward.scaler.*` | not used (no reward scaling in the environment) |

`reward.type: 'energy'` and `reward.single_agent: true` raise `ValueError`.

Forecast and oracle providers and the output directory are added in later phases.
