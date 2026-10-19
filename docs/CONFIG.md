# SmartCer configuration reference

SmartCer has two configuration levels:

- the **scenario**: a folder with `config.json` and one CSV file per node, the physical and
  economic description of the community (this document, sections 1-5);
- the **run**: the arguments of a simulation (period, episodes, reward, observation, ...). Only the
  period exists so far, as arguments of the loader (section 6); the environment arguments are added
  in phase 3 of the 1.0 refactoring.

Symbols and equations are in [FORMULATION.md](FORMULATION.md). A scenario is loaded with:

```python
from smartcer.data import load_scenario

community = load_scenario('path/to/scenario', start='2022-01-01 00:00', end='2022-12-31 23:00')
result = community.step(0, actions)   # one action per battery, in community.storage_node_ids order
```

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
| `sell_price` (alias `price_exp`) | required | ignored | required | €/kWh |
| `buy_price` (alias `price_imp`) | ignored | required | required | €/kWh |

Other columns are ignored. A file with both an alias and its 1.0 name raises `ValueError`. Prices a
node type does not require are set to 0: a producer never imports under the charge rule, and the
energy a consumer's battery exports (discharge beyond the load) earns nothing.

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

| Argument of `load_scenario` | Default | Meaning |
|---|---|---|
| `start` | first row | first timestamp of the period, inclusive |
| `end` | last row | last timestamp of the period, inclusive |

The environment arguments (episodes, reward, observation, `observe_current_step`, forecast and
oracle providers, output directory) and the mapping of the 0.6.2 run keys (`simulation.start`,
`simulation.end`, `simulation.seed`, `reward.type`, `reward.constraints.coefficient`) are added in
phase 3.
