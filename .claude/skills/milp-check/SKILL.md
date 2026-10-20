---
name: milp-check
description: Check a MILP oracle against SmartCer by replaying its optimal battery actions (check_replay) and report the welfare gap per day, penalised steps, state-of-charge differences and the Fit Score. Use when the user asks to check, validate or replay the MILP or DIISM oracle files of a scenario, or to compare the environment with the MILP.
---

# MILP check

Replay the optimal actions of a MILP oracle over a scenario period with
`smartcer.gym.controllers.check_replay` and report the comparison. The argument, if any, is the
scenario folder or the oracle folder: `$ARGUMENTS`.

## Rules

- Read-only on `smartcer/cer_data/`: never modify the scenario or the oracle files, and never
  commit them or files derived from them at hourly resolution.
- Write outputs to `results/milp_check/<scenario>/` (gitignored), never into `cer_data`.
- The oracle file format is provisional (`docs/CONFIG.md`, section 6, Oracle files). It is
  parsed only by `smartcer.oracle.read_oracle`. If the files do not match it, report the
  mismatch and stop: do not change the adapter unless the user asks.

## Step 1: Locate the files

- The scenario folder holds `config.json`. The oracle folder is `<scenario>/oracle/` unless the
  user names another one.
- The oracle folder holds `trajectory.csv` (`datetime`, `node_id`, `p_cha_kw`, `p_dis_kw`,
  `soc_kwh`) and optionally `objective.csv` (`date`, `objective`).
- Read the first and last `datetime` of `trajectory.csv`. The replay period must start at the
  first step of the file, because the replay starts from `soc_initial`, which is the MILP's
  state only there. The period must also consist of whole days.

## Step 2: Replay

```python
from pathlib import Path

from smartcer.gym import SmartCerEnv
from smartcer.gym.controllers import check_replay

scenario = Path('<scenario>')
env = SmartCerEnv(scenario, start='<first step of the file>', end='<last step of a whole day>',
                  oracle=scenario / 'oracle')
result = check_replay(env, f'results/milp_check/{scenario.name}', rtol=1e-3)
```

- If building the environment raises because the inferred initial state of charge differs from
  `soc_initial`, report both values. The scenario's `soc_initial` must be the MILP's S^ini.
  `OracleProvider.from_folder(folder, community, soc_initial_tolerance_kwh=...)` accepts a
  different tolerance (default 1e-3 kWh), but only if the user asks.
- An `OracleWarning` reports rows that charge and discharge in the same step. Include its count
  and first timestamp in the report.

## Step 3: Report

From the `ReplayCheck` fields and from `results/milp_check/<scenario>/oracle_daily.csv`:

- **verdict:** `passed`: no penalised step and a relative welfare gap ≤ `rtol`;
- **totals:** steps replayed, welfare of the replay, MILP objective, relative gap, Fit Score;
- **feasibility:** `penalty_steps` and `penalty_max` (real bound violations),
  `clipped_rows` and `clipped_max_kwh` (violations ≤ 1e-6 kWh moved onto the bound),
  `simultaneous_rows`, and `soc_error_max_kwh` (largest difference from the oracle's state of
  charge);
- **daily gap:** a table of the days with the largest |welfare − objective|, at most 10, with the
  date, welfare, objective and difference; plus the number of days with |difference| > rtol ×
  |objective of the day|.

If the check fails, say which condition failed. Give the likely causes in this order:

1. a different incentive model or value (`uniform` against `tip`, `unit_value`);
2. a different storage usage cost π^sto;
3. powers written in kWh per step instead of kW, with Δt ≠ 1 h;
4. a different charge rule (`charge_limit`);
5. objective terms beyond the welfare of FORMULATION section 10.

These are the open points of `docs/FORMULATION.md` section 15.
