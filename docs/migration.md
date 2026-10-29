# Migrating from 0.x

The 0.6.2 code is at the tag `v0.6.2`. Version 1.0.0 is a rewrite, and its results are not
numerically comparable with those of 0.6.2. The labels S3, S4, ... in the tables are the numbers of
the defects of 0.6.2 that each change fixes; the regression tests of the new code are named
`test_s<n>_*`.

## API

| 0.6.2 | 1.0.0 |
|---|---|
| `from smartcer.cer_env import SmartCer`; `SmartCer(cer_path, cer_config)` | `gym.make('SmartCer-v1', scenario=..., ...)` or `smartcer.gym.SmartCerEnv(scenario, ...)`; `SmartCerEnv.from_v062_config(cer_path, cer_config)` maps a 0.6.2 run configuration |
| environment ID `SmartCer-v0` | `SmartCer-v1` |
| `step()` returns 6 items, the reward is a list `[reward, penalty, signal]` | `step()` returns `(obs, reward, terminated, truncated, info)` with a float reward; the components are in `info['reward']` (S3) |
| `rl_step()` flattens the observation | the observation is a flat `Box`; `env.obs_layout` names its parts |
| `cer.sim_hours` | `env.episode_length` |
| `cer.compute_baselines(no_storage=True, rbc=True)` | `compute_baselines(env, output_dir, controllers=('naive', 'rbc_community'))` |
| `cer.get_sim_results()`, `cer.save_results(...)` into `cer_data/<scenario>/simulations/` | the `TrackResults` wrapper and `tracker.save(output_dir)`; output folders inside the scenario or `cer_data` are refused (S18) |
| `smartcer.cer_metrics`, `smartcer.storage_controllers` | `smartcer.gym.metrics`, `smartcer.gym.controllers` |

## Configuration keys

| 0.6.2 | 1.0.0 |
|---|---|
| `config.json` with `Producers`, `Consumers`, `Prosumers` | `format_version: "1.0"`, `nodes` with `type`; the old format is read with a `DeprecationWarning` |
| storage `rated_power` | `power_charge_max`, `power_discharge_max` (kW, node side) |
| storage `efficiency` | `efficiency_charge`, `efficiency_discharge` |
| storage `alfa` | `alpha` (per hour) |
| storage `beta` | ignored: degradation is not modelled |
| CSV `price_exp`, `price_imp` | `sell_price`, `buy_price` (the old names are still accepted) |
| (implicit hourly data) | `delta_t`, `units` |
| `simulation.start`, `simulation.end`, `simulation.seed` | `start`, `end`, `seed` |
| `reward.type` `'cost'` / `'profit'` | `reward='welfare'` (collective) or `'profit'` with `reward_per_node=True` (`single_agent`) |
| `reward.constraints.coefficient` | `penalty_coefficient` (default 10) |
| `reward.scaler.*` | removed: no reward scaling inside the environment (S9) |
| `simulation.delta_input`, `delta_output`, `reward.constraints.max_ratio` | removed (never used) |

## Behaviour

| Change | Defect |
|---|---|
| The state of charge never leaves [S^min, S^max]: the action bounds are derived from the battery dynamics, with both efficiencies | S4 |
| Explicit time step Δt: energy = power × Δt, so sub-hourly data work | S5 |
| The rule-based controller computes energies within the bounds and divides by the battery power, not by the capacity | S6 |
| The information timing is an option, `observe_current_step` (True observes g and l of step t, as in 0.6.2; False is causal) | S7 |
| Consumers can have a battery | S8 |
| The terminal observation is that of the step after the episode, not step 0 | S10 |
| KPIs are named after what they measure: `self_consumption` is now `sharing_rate`; `net_metering` and `energy_balance` are removed | S12 |
| About 12 times faster (one year of hourly data with 10 nodes: about 3.5 s instead of 43 s) | S13 |
| `reset(seed)` also seeds the action space | S17 |
| Errors raise exceptions instead of calling `sys.exit` | S11 |
