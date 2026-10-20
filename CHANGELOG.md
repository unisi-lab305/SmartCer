# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `smartcer.__version__` attribute
- Optional dependency groups `dev` (test and lint tools), `rl` (Stable-Baselines3) and `all`
- Test suite with a synthetic 4-node scenario, characterisation tests of the 0.6.2 behaviour and
  strict expected-failure tests documenting the known defects
- Makefile, pytest configuration, pre-commit hooks, CI workflow and tag-triggered GitHub Release
  workflow
- New battery model `smartcer.core.Battery` following `docs/FORMULATION.md`: separate charge and
  discharge power limits and efficiencies on the node side, explicit time step, dynamic action
  bounds derived from the dynamics, clipping with a constraint penalty, self-discharge (defined
  per hour) and storage usage cost; the state of charge never leaves its limits (0.6.2 defect S4)
- `Battery.from_config` reads storage sections in the 1.0 and the 0.6.2 format
- Scenario format 1.0 (`format_version`, `delta_t`, `units` in kWh per step or kW, `uniform` or
  `tip` incentive, explicit CSV paths, 1.0 storage keys) documented in `docs/CONFIG.md`
- Scenario loader `smartcer.data.load_scenario` for the 1.0 and the 0.6.2 formats, reading every CSV
  once; validation errors name the node, the file, the column and the first offending row
- Community model `smartcer.core.Community` with producer, consumer and prosumer nodes, shared
  energy, `uniform` and `tip` incentives and the welfare breakdown of `docs/FORMULATION.md`; on the
  0.6.2 scenarios with idle batteries it reproduces the 0.6.2 community results
- Nodes support an explicit time step and consumers with a battery (0.6.2 defects S5 and S8); the
  charge rule `production` or `surplus` is set per battery with `charge_limit`; a consumer's optional
  `sell_price` column pays the energy its battery exports
- Gymnasium environment `SmartCerEnv`, registered as `SmartCer-v1`: one continuous action per
  battery (`Box(-1, 1)`), reward welfare minus σ times the mean constraint penalty, flat
  observation with a named layout (`obs_layout`, preset `meco`), episodes of configurable length
  with optional random start, option `observe_current_step` for the information timing
- `SmartCerEnv.from_v062_config` to run 0.6.2 run configurations (`simulation`, `reward` keys)
- Reward types `welfare` (default), `profit` and `energy` (0.6.2 legacy) in
  `smartcer.gym.rewards`, selected with `reward`; `reward_per_node=True` reports one reward per
  battery node in `info['reward']['nodes']` and returns their sum
- Observation preset `extended`: Italian holiday flag, totals of generation and load of the nodes
  without a battery and optional `lookback` of generation and load per battery node
- `MinMaxNormalizer`: generation and load are scaled with bounds fitted on a declared training
  window (`normalizer_window`, default the whole period), saved to and loaded from JSON
  (`env.normalizer.save(path)`, `normalizer=path`)
- `info['privileged']` in the reset and step info, with data for the critic only about the step of
  the returned observation: the oracle action and state of charge at the start of the step
  (`oracle`) and the true values of the forecast steps (`forecast`); empty without them
- `info['obs_clipped']` in the reset and step info: the number of observation values clipped to
  [0, 1] because they lie outside the normaliser's training range
- Baseline controllers in `smartcer.gym.controllers`: `NaiveController` (idle batteries) and
  `RBCController` ([M24] Algorithm 1, computed in energy within the dynamic bounds, so never
  penalised) with two modes: `node` (each battery follows its own node; a producer's battery
  discharges when its generation is 0) and `community` (the batteries follow the community balance,
  pro rata to the energy each can take or deliver; the reference baseline); `compute_baselines`
  runs them (default `naive` and `rbc_community`) and writes the community table, one table per
  node, the KPIs and the run configuration to an output directory
- Results tracking in `smartcer.gym.tracking`: `TrackResults` wrapper with a `ResultsTracker`
  (community and per-node tables, CSV export) and the Stable-Baselines3 `ResultsCallback`
- KPIs in `smartcer.gym.metrics` (welfare and its components, shared energy, sharing and coverage
  rates, imports, exports, battery throughput, equivalent full cycles on the nominal and on the
  usable capacity, penalty statistics) and the
  Fit Score, documented in `smartcer/gym/metrics/KPI.md`
- `SmartCerEnv.run_config()`, `SmartCerEnv.current_step` and `SmartCerEnv.scenario_path`
- Step benchmark `benchmarks/bench_step.py` (`make bench`): whole-period episodes with random
  actions, time per episode and per step
- Forecast providers in `smartcer.forecasts` (`ForecastProvider` protocol, `PerfectForecast`):
  with `forecast='perfect'` (or a provider) and `forecast_horizon` H the observation gets the
  forecasts of generation and load of every battery node for the next H steps
  (`forecast/<node>/<quantity>`), scaled like the current values; episodes stop early enough for
  the forecasts of every step and of the step after the episode to lie in the data, and the
  provider is reset with the environment's random generator
- MILP oracle in `smartcer.oracle`: `OracleProvider` reads the optimal battery trajectories of an
  oracle folder (`trajectory.csv`, optional daily `objective.csv`; provisional format, parsed by
  the single adapter `read_oracle`) and computes the optimal actions; environment argument
  `oracle`; rows with simultaneous charge and discharge are counted and reported with an
  `OracleWarning`
- `ReplayController` (`'replay'`) plays the oracle actions; `check_replay` replays them over a
  period of whole days, checks that no step is penalised and that the welfare matches the MILP
  objective within a relative tolerance (default 1e-3), reports the rows with simultaneous charge
  and discharge, and writes a per-day comparison
- `MMFEForecast`: simulated forecast errors following the martingale model of forecast evolution
  (error variance linear in the lead time, consistent forecasts across issues, errors correlated
  between nodes and over adjacent target steps; photovoltaic errors scaled on a monthly reference
  profile, zero at night; multiplicative load errors; optional bias), reproducible from the reset
  seed; presets `mmfe_perfect`, `mmfe_standard` and `mmfe_noisy` (provisional values) or custom
  `MMFEParameters`, with optional per-series σ (`sigma_overrides`, e.g. a smaller σ for an
  aggregated load)
- The forecast section of the observation also holds the sale price over the horizon
  (`forecast/price_sell`, or per battery node) and the forecasts of the community totals of the
  nodes without a battery (`forecast/community/generation`, `forecast/community/load`)
- The oracle's initial state of charge is inferred from the first row of its file and must match
  the scenario's `soc_initial` within a configurable tolerance (default 1e-3 kWh); the replay
  moves oracle actions that exceed a bound by at most 1e-6 kWh (solver round-off) onto the bound
  and reports them, and must start at the first step of the oracle file
- With an oracle, the battery tables of a run hold the oracle state of charge and the gap from it,
  and the KPIs its mean and maximum per battery (`soc_gap_mean_kwh`, `soc_gap_max_kwh`)
- `run_config()` records the settings of the forecast provider (`forecast_config`); the step
  benchmark takes `--forecast` and `--horizon`

### Changed

- Python 3.12 or newer is now required
- `holidays` is now a runtime dependency
- `import smartcer` registers `SmartCer-v1`; the `gymnasium.envs` entry point registers it with
  Gymnasium < 1.0
- `tqdm` is no longer a dependency
- `info['reward']` holds `type`, `main`, `penalty` and `total` (instead of `welfare`, `penalty`,
  `total`)
- Scaled generation and load are clipped to [0, 1], so values beyond the training range stay inside
  the observation space
- `SmartCerEnv.from_v062_config` maps the 0.6.2 reward type `energy` and `single_agent: true`
  (per-node rewards) instead of raising

### Deprecated

- 0.6.2 storage keys `rated_power`, `efficiency` and `alfa` (mapped to their 1.0 equivalents) and
  `beta` (ignored: battery degradation is not modelled)
- 0.6.2 scenario format (no `format_version`): still read, with a `DeprecationWarning`; its
  timestamp irregularities are reported in a `TimestampWarning` and rows are aligned by position
  as in 0.6.2

### Removed

- Environment `SmartCer-v0` and the 0.6.2 modules `smartcer.cer_env`, `smartcer.cer_utils`,
  `smartcer.cer_models`, `smartcer.cer_reward` (with the running reward scaler) and
  `smartcer.cer_preprocessing`
- 0.6.2 modules `smartcer.cer_metrics` (replaced by `smartcer.gym.metrics`) and
  `smartcer.storage_controllers` (replaced by `smartcer.gym.controllers`)

### Fixed

- `pip install .` now installs all subpackages (previously `smartcer.cer_env` failed to import
  with `ModuleNotFoundError: smartcer.storage_controllers`)
- The environment is Gymnasium-compliant: five-item `step()`, float reward, boolean `truncated`,
  observation inside its space, `check_env` of Gymnasium and Stable-Baselines3 pass (0.6.2 defect
  S3)
- The 0.6.2 reward type `cost` is accepted when mapping a 0.6.2 run configuration (S2)
- The current step's generation is observed only with `observe_current_step=True` (S7); identical
  episodes give identical rewards (S9); the terminal observation describes the step after the
  episode, or its last step at the end of the data, instead of wrapping to step 0 (S10);
  `reset(seed)` also seeds the action space (S17)
- The rule-based controller executes the intended energy: its action is the energy divided by the
  battery power, not by the capacity (S6)
- KPIs are named after what they measure: shared / exported is the sharing rate (0.6.2 called it
  `self_consumption`); `net_metering` and `energy_balance` are gone (S12)
- Run results are written to an output directory chosen by the caller, never into the scenario
  folder (S18)
- 0.6.2 scenarios whose files have shifted timestamps (Leather) can be split with `start` and
  `end` anywhere: the period is selected on the first node's timestamps and every node is cut at
  the same rows; files with different numbers of rows raise `ValueError`
- Speed (S13): one year of Leather (10 nodes, 4 batteries) with random actions takes 3.2 s
  (364 µs per step) instead of 51-52 s with 0.6.2 on the same machine, 16 times faster (43 s in
  the 0.6.2 measurement of the refactoring brief): CSV files are read once, a step reads
  preloaded values without pandas, and the observation is built once per step from precomputed
  static features

---

## [0.6.2] - 2025-03-17

First tagged version; the tag was created retrospectively at the start of the 1.0 refactoring.

### Added

- Gymnasium environment `SmartCer` (ID `SmartCer-v0`) for Italian Renewable Energy Communities
  (CER) with producers, consumers and prosumers, each optionally equipped with a battery
- Scenario loader for `config.json` (`Producers`, `Consumers`, `Prosumers`) and per-node hourly
  CSV files, with the `_N` suffix to reuse the CSV of a base node
- Battery model with rated power, efficiency, minimum state of charge, self-discharge and capacity
  degradation
- Hourly shared energy and per-plant TIP incentive with the PNRR capital-grant correction
- Continuous action per battery in [-1, 1] with dynamic upper and lower bounds in the observation
- Rewards `energy` and `profit`, per node or collective, with a constraint-violation penalty and an
  optional running reward scaler
- Observation encoders (one-hot, periodic, min-max, scaling) and a flattening `rl_step()` helper
- Rule-based storage controller and baselines without storage and with the rule-based controller
- Results export (community and node CSV tables, KPI JSON, run configuration JSON) and ten KPIs
- Installable package (`pyproject.toml`) and citation in the README

### Fixed

- Self-consumption and constraint-violation computations (October 2024)
- Miscellaneous bug fixes (March 2025)
