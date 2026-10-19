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
  per hour) and storage usage cost; the state of charge never leaves its limits (0.6.2 defect S4). Not yet used by the
  0.6.2 environment
- `Battery.from_config` reads storage sections in the 1.0 and the 0.6.2 format
- Scenario format 1.0 (`format_version`, `delta_t`, `units` in kWh per step or kW, `uniform` or
  `tip` incentive, explicit CSV paths, 1.0 storage keys) documented in `docs/CONFIG.md`
- Scenario loader `smartcer.data.load_scenario` for the 1.0 and the 0.6.2 formats, reading every CSV
  once; validation errors name the node, the file, the column and the first offending row
- Community model `smartcer.core.Community` with producer, consumer and prosumer nodes, shared
  energy, `uniform` and `tip` incentives and the welfare breakdown of `docs/FORMULATION.md`; on the
  0.6.2 scenarios with idle batteries it reproduces the 0.6.2 community results
- Nodes support an explicit time step and consumers with a battery (0.6.2 defects S5 and S8); the
  charge rule `production` or `surplus` is set per battery with `charge_limit`

### Changed

- Python 3.12 or newer is now required
- `holidays` is now a runtime dependency

### Deprecated

- 0.6.2 storage keys `rated_power`, `efficiency` and `alfa` (mapped to their 1.0 equivalents) and
  `beta` (ignored: battery degradation is not modelled)
- 0.6.2 scenario format (no `format_version`): still read, with a `DeprecationWarning`; its
  timestamp irregularities are reported in a `TimestampWarning` and rows are aligned by position
  as in 0.6.2

### Fixed

- `pip install .` now installs all subpackages (previously `smartcer.cer_env` failed to import
  with `ModuleNotFoundError: smartcer.storage_controllers`)

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
