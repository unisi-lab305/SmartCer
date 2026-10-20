---
name: new-scenario
description: Create a new SmartCer scenario in the 1.0 format, smartcer/cer_data/<name>/config.json, from the user's CSV files; check columns, units and time step of the CSV files, validate the scenario with the loader and run one episode with the baseline controllers. Use when the user asks to add, create or set up a scenario, a CER or a community for SmartCer.
---

# New scenario

Create `smartcer/cer_data/<name>/` in the 1.0 format of `docs/CONFIG.md` (sections 1-3), check
its CSV files, and run one episode. The argument, if any, is the scenario name: `$ARGUMENTS`.

## Rules

- `smartcer/cer_data/` is gitignored and may hold confidential company data. Never commit
  anything from it. Never modify or delete an existing scenario folder: if
  `smartcer/cer_data/<name>/` already exists, stop and ask for another name.
- Copy the user's CSV files into the new folder only when the user agrees; otherwise reference
  them with `csv` paths relative to the scenario folder.
- Never write run outputs into `cer_data`: use `results/<name>/` (gitignored).
- Do not guess battery or incentive parameters. Ask for every value the user has not given,
  except the documented defaults (`soc_min` 0, `soc_initial` = `soc_min`, `alpha` 0,
  `usage_cost` 0, `charge_limit` `production`).

## Step 1: Collect the scenario

Ask for, or read from the user's message:

1. the scenario name (folder name; letters, digits, `_` and `-`);
2. the CSV file of every node, and each node's type: `producer`, `consumer` or `prosumer`;
3. for every battery: `capacity` (kWh), `power_charge_max` and `power_discharge_max` (kW, node
   side), `efficiency_charge` and `efficiency_discharge`, and optionally `soc_min`,
   `soc_initial`, `alpha`, `usage_cost` and `charge_limit`;
4. the incentive: `uniform` with `unit_value` (€/kWh, or a CSV column), or `tip` with the
   `photovoltaic` section of every producer and prosumer (`tip_base`, `tip_threshold`,
   `fi_zone`, `pnrr`);
5. the period of interest, if the files cover more than needed.

## Step 2: Check the CSV files

For every file, with pandas, report:

- **Columns.** `datetime` is required. `production` is required for producers and prosumers,
  `consumption` for consumers and prosumers. `sell_price` is required for producers and
  prosumers, optional for consumers. `buy_price` is required for consumers and prosumers. The
  0.6.2 aliases `price_exp` and `price_imp` are accepted. List missing columns and the columns
  that will be ignored.
- **Time step.** The timestamps must be sorted, unique and evenly spaced. Report the spacing
  (it becomes `delta_t` in hours), the first and last timestamp, and the first irregular row if
  any. Files of different nodes must cover the same steps of the period.
- **Units.** Values must be energies per step (`units: "kWh"`, the default) or average powers
  (`units: "kW"`). Ask the user which, if the file does not say. A typical hint: with Δt = 0.25 h
  and kW values, the yearly sum of `consumption` is four times the yearly energy.
- **Values.** Report missing or non-numeric values and negative `production` or `consumption`,
  with the first offending row.

Stop and report if a file fails these checks; propose the fix, but do not edit the user's files.

## Step 3: Write config.json

Write `smartcer/cer_data/<name>/config.json` with `format_version: "1.0"`, `name`, `delta_t`,
`units`, `incentive` and `nodes`. Use the example of `docs/CONFIG.md` section 1 as the template.
Node order is the order in `nodes`; the order of the actions is that of the nodes with a battery.

## Step 4: Validate and run one episode

Run, with the scenario path and the period:

```python
from smartcer.data import load_scenario
from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import compute_baselines

community = load_scenario('smartcer/cer_data/<name>', start=None, end=None)
print(community)   # nodes, batteries, steps
env = SmartCerEnv('smartcer/cer_data/<name>')
kpis = compute_baselines(env, 'results/<name>/baselines', controllers=('naive', 'rbc_community'),
                         seed=0)
```

`load_scenario` raises `ValueError` naming the node, file, column and row of any problem; fix
`config.json` (not the data) and repeat.

## Step 5: Report

Report to the user:

- the scenario folder and the nodes: type, battery yes or no, CSV file;
- Δt, units, period and number of steps;
- for `naive` and `rbc_community`: welfare, incentive, shared energy, sharing rate, battery
  throughput, and penalised steps (`rbc_community` must have 0);
- any warning from the loader;
- that `smartcer/cer_data/` is gitignored and the new folder must not be committed.
