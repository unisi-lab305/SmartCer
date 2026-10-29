# KPIs and outputs

## KPIs

{py:func}`smartcer.gym.metrics.compute_kpis` computes the indicators of one run from the tables of
a `ResultsTracker`. It returns a dictionary with these keys (formulas and units:
[KPI.md](https://github.com/unisi-lab305/SmartCer/blob/main/smartcer/gym/metrics/KPI.md)):

| Group | Keys |
|---|---|
| Run | `steps` |
| Economics | `reward`, `welfare`, `revenue`, `cost`, `storage_cost`, `incentive` |
| Energy | `shared_energy_kwh`, `export_kwh`, `import_kwh`, `sharing_rate` (shared / exported), `coverage_rate` (shared / imported) |
| Batteries (community) | `throughput_kwh`, `equivalent_full_cycles`, `efc_usable` |
| Constraint penalty | `penalty_mean`, `penalty_max`, `penalty_steps` |
| Per battery, under `batteries` | `charge_kwh`, `discharge_kwh`, `throughput_kwh`, `equivalent_full_cycles`, `efc_usable`, `penalty_mean`, `penalty_max`, `penalty_steps` |
| With an oracle, per battery | `soc_gap_mean_kwh`, `soc_gap_max_kwh`: absolute gap between the agent's and the oracle's state of charge |

Rates are fractions. Equivalent full cycles are computed on the maximum capacity
(`equivalent_full_cycles`) and on the usable capacity (`efc_usable`).

Two functions compare runs:

- `smartcer.gym.metrics.fit_score(reference, candidate)`: the Fit Score F, in percent.
- `smartcer.gym.metrics.daily_welfare(community_table)`: the welfare of each day.

## Recording a run

Two ways to record results:

- `smartcer.gym.tracking.TrackResults(env)` wraps an environment; the last completed episode is
  available as `env.get_wrapper_attr('tracker')`, a `ResultsTracker` with `community_table()`,
  `node_tables()`, `kpis()` and `save(output_dir, run_config)`.
- `smartcer.gym.controllers.compute_baselines(env, output_dir)` runs controllers and saves each
  of them.

Results are never written into a scenario folder or a `cer_data` folder: such an output folder
raises `ValueError`.

## Files written

`tracker.save(output_dir)` writes:

```text
output_dir/
├── community.csv        # one row per step
├── nodes/<node_id>.csv  # one row per step, one file per node
├── kpis.json
└── run_config.json      # only when a run configuration is passed
```

`compute_baselines` writes one such folder per controller, `output_dir/<controller>/`, and
`output_dir/kpis.json` with the KPIs of all of them. `run_config.json` adds the controller name.

`community.csv` columns: `datetime`, `episode`, `t`, `reward`, `reward_main`, `reward_penalty`,
`welfare`, `revenue`, `cost`, `storage_cost`, `incentive`, `shared_energy_kwh`, `export_kwh`,
`import_kwh`, `penalty`, `obs_clipped`.

Node tables contain the fields of the step record of the node (generation, load, charge and
discharge power and energy, net, export and import energy, revenue, cost, throughput, storage
cost, penalty, state of charge, requested and applied action, action bounds) and, for the
battery nodes with an oracle, `oracle_soc_kwh` and `soc_gap_kwh`.

`check_replay` writes `oracle_daily.csv` and `oracle_replay.json`.

## Stable-Baselines3

`smartcer.gym.tracking.ResultsCallback(output_dir, save_freq=1, run_config=None)` is a
Stable-Baselines3 callback for a `DummyVecEnv` of `TrackResults` environments. At every completed
episode it logs the KPIs under `smartcer/<kpi>` and every `save_freq` episodes it saves the run
to `output_dir/episode_<n>/`. It requires the `rl` extra.
