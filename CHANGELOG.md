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
  without a battery, optional `lookback` of generation and load per battery node, and a section
  reserved for forecasts
- `MinMaxNormalizer`: generation and load are scaled with bounds fitted on a declared training
  window (`normalizer_window`, default the whole period), saved to and loaded from JSON
  (`env.normalizer.save(path)`, `normalizer=path`)
- `info['privileged']` (empty) in the reset and step info, reserved for the critic's data

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

### Fixed

- `pip install .` now installs all subpackages (previously `smartcer.cer_env` failed to import
  with `ModuleNotFoundError: smartcer.storage_controllers`)
- The environment is Gymnasium-compliant: five-item `step()`, float reward, boolean `truncated`,
  observation inside its space, `check_env` of Gymnasium and Stable-Baselines3 pass (0.6.2 defect
  S3)
- The 0.6.2 reward type `cost` is accepted when mapping a 0.6.2 run configuration (S2)
- The current step's generation is observed only with `observe_current_step=True` (S7); identical
  episodes give identical rewards (S9); the terminal observation describes the last step (S10);
  `reset(seed)` also seeds the action space (S17)
- 0.6.2 scenarios whose files have shifted timestamps (Leather) can be split with `start` and
  `end` anywhere: the period is selected on the first node's timestamps and every node is cut at
  the same rows; files with different numbers of rows raise `ValueError`

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
