# Quickstart

This page runs a first simulation. SmartCer needs a *scenario*: a folder with a `config.json` and
one CSV file per node.

```{optional-figure} /_static/figures/pipeline.png
:alt: A SmartCer study in four steps: input data, configuration, simulation, results.
:width: 90%

From data to results.
```

```{important}
The PyPI package does not ship any scenario. The example below uses the **synthetic** scenario
`test/fixtures/mini_cer_v1` of the repository: 4 nodes, 2 of them with a battery, 48 hours of
made-up hourly data, used by the test suite. It is available from a **source checkout**
(`git clone` and `pip install -e .`, see [Installation](installation.md)); it is not installed by
`pip install smartcer`. The numbers it produces illustrate the API and have no meaning for a real
community. To use your own data, see the scenario format below.
```

## The example

The code is in `examples/first_simulation.py` and resolves the scenario path relative to the
repository root, so it runs from any directory:

```bash
python examples/first_simulation.py
```

```{literalinclude} ../examples/first_simulation.py
:language: python
:start-at: from pathlib import Path
```

What it does:

1. `gym.make('SmartCer-v1', ...)` builds the environment on the scenario, with episodes of 24
   steps.
2. `env.reset(seed=0)` returns the first observation, a flat vector, and an `info` dictionary.
3. The loop applies random actions: one value in [-1, 1] per battery, positive to charge. The
   episode ends with `truncated=True` after the last step.
4. `info['welfare']` is the breakdown of the community welfare of the last step and
   `info['penalty']` the constraint penalty K_t.
5. `compute_baselines` runs the `naive` (idle batteries) and `rbc_community` (rule-based)
   controllers over the same episode, returns their KPIs and writes tables to
   `results/first_simulation/`. See [KPIs and outputs](kpis.md).

## Your own scenario

A scenario folder looks like this:

```json
{
  "format_version": "1.0",
  "name": "my_cer",
  "delta_t": 1.0,
  "units": "kWh",
  "incentive": {"model": "uniform", "unit_value": 0.11},
  "nodes": {
    "pv_1": {
      "type": "producer",
      "csv": "data/pv_1.csv",
      "storage": {
        "capacity": 100, "power_charge_max": 30, "power_discharge_max": 30,
        "efficiency_charge": 0.95, "efficiency_discharge": 0.95, "soc_min": 0.1
      }
    },
    "aggregated_load": {"type": "consumer", "csv": "data/consumers.csv"}
  }
}
```

The CSV columns are `datetime`, `production` and `consumption` (kWh per step, or kW with
`units: kW`), `sell_price` and `buy_price` (€/kWh); which columns a node needs depends on its
type. The full format is described in the [configuration reference](configuration.md).

## Next steps

- [Concepts](concepts.md): what the observation, the action and the reward contain, and how to
  add forecasts.
- [API reference](api.md): `SmartCerEnv` and its arguments.
