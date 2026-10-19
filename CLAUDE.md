# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartCer is a Gymnasium environment for Italian Renewable Energy Communities (CER, *comunità
energetiche rinnovabili*). A community is a set of producers, consumers and prosumers, some with a
battery; it receives an hourly incentive on the shared energy (*energia condivisa*), the minimum
between the total energy exported and the total energy imported by its members. An RL agent
controls the batteries.

**Status:** version 0.6.2 is being refactored into 1.0.0 following
[docs/refactor/REFACTOR_v1.md](docs/refactor/REFACTOR_v1.md); progress, deviations and open
questions are in [docs/refactor/PROGRESS.md](docs/refactor/PROGRESS.md). Phases 0-6 are complete:
the Gymnasium environment `SmartCer-v1` (`smartcer/gym/`, with its rewards, observation presets,
baseline controllers, results tracking and KPIs) wraps the 1.0 physics and economics
(`smartcer/core/`) and the scenario loader (`smartcer/data/`); all 0.6.2 modules were removed.
One year of Leather takes about 3 s (`make bench`). Scenario and run arguments:
[docs/CONFIG.md](docs/CONFIG.md).

**Authoritative formulation:** [docs/FORMULATION.md](docs/FORMULATION.md) defines the physics, the
economics and the reward of SmartCer 1.0. It prevails over the code and over the papers. Do not
change the meaning of anything in it without asking Leonardo; any change to those equations must
update the file in the same commit.

**Repository**: <https://github.com/unisi-lab305/SmartCer>
**License**: MIT

## Environment Registration and Installation

**Installation:**

```bash
pip install -e .[dev]     # development (tests and lint tools)
pip install -e .[all]     # development + Stable-Baselines3
pip install .             # plain installation
```

Python 3.12 or newer is required. Runtime dependencies: `gymnasium`, `numpy`, `pandas`,
`holidays`.

**Environment ID**: `SmartCer-v1`, registered on `import smartcer` (`smartcer/__init__.py` calls
`smartcer.gym.register_envs()`, idempotent). The `gymnasium.envs` entry point (`__root__` in
`pyproject.toml`) registers it automatically only with Gymnasium < 1.0, which still loads entry
points; with Gymnasium >= 1.0 import `smartcer` first or use `gym.make('smartcer:SmartCer-v1', ...)`.

```python
import gymnasium as gym
import smartcer

env = gym.make('SmartCer-v1', scenario='path/to/scenario', episode_length=168, random_start=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# direct use: from smartcer.gym import SmartCerEnv; SmartCerEnv('path/to/scenario', ...)
# 0.6.2 run configuration: SmartCerEnv.from_v062_config(cer_path, cer_config)
```

Arguments: `docs/CONFIG.md` section 6.

## Development Commands

```bash
make help                   # list all targets (from the `## ` comments in the Makefile)
make test                   # all tests (unit + integration)
make test-unit              # test/unit/ only
make test-integration       # test/integration/ only
make test-coverage          # coverage report (terminal, htmlcov/, coverage.xml)
make bench                  # step benchmark: one year of the local Leather scenario, 3 episodes
python benchmarks/bench_step.py --scenario <folder> --episodes 5 --observation extended
make version                # check that pyproject.toml and smartcer/__init__.py agree
make clean                  # remove caches, coverage and build files

ruff check --fix .          # linter with auto-fix
ruff format .               # formatter
pylint smartcer/            # additional linting
pre-commit install          # setup hooks (one-time)
pre-commit run --all-files  # manual run
```

Pre-commit hooks fix trailing whitespace and end-of-file newlines, validate YAML, JSON and TOML,
reject large files and private keys, enforce LF line endings and sort imports with Ruff.

**Reference values of 0.6.2:** `test/fixtures/local/` (gitignored) holds the hourly community series
and the annual KPIs of 0.6.2 on the local Leather 2022 scenario with idle batteries. They were
written in phase 0 by `scripts/make_reference_v062.py`, removed in phase 3 with the 0.6.2 modules
it needs (run it from commit `d878969` to regenerate the files).

## Claude Code Commands

Five slash commands are available in `.claude/commands/`:

| Command | Description |
|---------|-------------|
| `/bump-version [patch/minor/major]` | Bump version in `pyproject.toml` + `smartcer/__init__.py`, rotate CHANGELOG, commit |
| `/add-and-commit [verbose]` | Update CHANGELOG, stage all changes, write a conventional commit |
| `/pr [title]` | Push branch and open a GitHub Pull Request into `main` |
| `/release` | Create and push the `vX.Y.Z` git tag (triggers the GitHub Release workflow) |
| `/write-test <module>` | Write tests for a `smartcer` module in `test/unit/` and `test/integration/` |

**Typical release workflow:**

1. Make code changes and `/add-and-commit` — updates CHANGELOG and commits
2. `/bump-version patch` (or `minor`) — bumps both version files, rotates CHANGELOG, commits
3. `/release` — tags `vX.Y.Z` and pushes; CI validates and creates the GitHub Release

### Claude Code Behavior

**IMPORTANT**: When working with this codebase:

- **DO NOT** run the linter (`ruff check`) or the formatter (`ruff format`) automatically; run
  them only when explicitly requested
- Pre-commit hooks handle formatting during commits
- Commit messages follow conventional commits (`<type>(<scope>): <description>`), one line unless
  asked otherwise, with no `Co-Authored-By` lines and no references to AI tools
- Open a pull request only when asked; ask before pushing tags

**Refactoring protocol** (until 1.0.0, details in `docs/refactor/REFACTOR_v1.md` section 0):

- At the start of a session read `docs/refactor/REFACTOR_v1.md`, `docs/FORMULATION.md`,
  `docs/refactor/PROGRESS.md` and this file
- One phase per request, on a branch `refactor/v1-phase-<N>-<slug>` created from `main`; do not
  start the next phase without an explicit go-ahead
- Strangler approach: new modules are built next to the 0.6.2 ones, which stay untouched until
  their replacement passes its tests
- When the brief conflicts with the code or is ambiguous, stop and ask
- `smartcer/cer_data/` (local scenarios, gitignored, partly confidential) is read-only; never
  commit files derived from it at hourly resolution
- The storage-node-environment repository (`../storage-node-environment`) is the reference for code
  to port: copy and adapt, never import it, never modify it

**Documentation Maintenance**:

- **AUTOMATICALLY** update CLAUDE.md when the package structure, commands or conventions change
- Updates MUST be in the same commit as the code changes
- See [Maintaining CLAUDE.md](#maintaining-claudemd)

## CI/CD Pipeline

### GitHub Actions Workflows

**CI Tests** (`.github/workflows/ci.yml`):
- **Triggers**: push to `main` and to `refactor/**` branches, pull request to `main`
- **Runner**: Ubuntu latest, Python 3.12, `pip install -e .[all]`
- **Actions**:
  1. Unit tests (`make test-unit`)
  2. Integration tests (`make test-integration`)
  3. Coverage report (`make test-coverage`), uploaded as an artifact (HTML + XML)
  4. Builds the wheel, installs it in a clean virtual environment and imports `smartcer`,
     `smartcer.core`, `smartcer.data`, `smartcer.gym`, `smartcer.gym.controllers`,
     `smartcer.gym.metrics` and `smartcer.gym.tracking` from outside the repository (guards
     defect S1, missing subpackages)
- **View results**: `https://github.com/unisi-lab305/SmartCer/actions`

**GitHub Release** (`.github/workflows/release.yml`):
- **Trigger**: push of a git tag `vX.Y.Z`
- **Validations**: tag == `pyproject.toml` == `smartcer/__init__.py`; `CHANGELOG.md` has a
  `## [X.Y.Z]` entry; unit and integration tests pass
- **Actions**: builds sdist and wheel (`python -m build`), `twine check`, creates the GitHub Release
  with the CHANGELOG section as notes and the built files attached
- **No PyPI publication** until 11/11/2026 (decision D11): the step is a commented block in the
  workflow. Users install from the tag:
  `pip install git+https://github.com/unisi-lab305/SmartCer@vX.Y.Z`

Local test data (`smartcer/cer_data/`, `test/fixtures/local/`) are absent in CI: tests that need
them are skipped.

## Version Management

### Semantic Versioning

This project follows [Semantic Versioning](https://semver.org/): **MAJOR.MINOR.PATCH**.

**CRITICAL**: Update versions in BOTH files:
1. `pyproject.toml` (line 7: `version = "X.Y.Z"`)
2. `smartcer/__init__.py` (line 1: `__version__ = 'X.Y.Z'`)

**Version numbers in both files MUST ALWAYS be identical** (`make version` checks them).

The version stays at `0.6.2` during the refactoring; it becomes `1.0.0` in phase 8.

### Increment Rules

- **PATCH**: bug fixes, refactoring, docs, performance → automatic
- **MINOR**: new features, API additions, new node, reward or controller types → automatic +
  update CLAUDE.md
- **MAJOR**: breaking changes, architectural redesigns → **ASK USER** before incrementing

### Update Procedure

1. **CHANGELOG.md**: add user-facing entries under `[Unreleased]` (Added, Changed, Deprecated,
   Removed, Fixed, Security), or use `/add-and-commit`
2. **CLAUDE.md**: update affected sections for MINOR and MAJOR changes
3. **Bump**: `/bump-version patch|minor|major`, or edit both files and run `make version`
4. **Release**: `/release` (tag `vX.Y.Z`, GitHub Release); ask before pushing tags

Git tags mark released versions (`vX.Y.Z`). `v0.6.2` marks the code before the refactoring.

## Package Structure

```
SmartCer/
├── smartcer/                       # package
│   ├── __init__.py                 # __version__, registration of SmartCer-v1
│   ├── core/                       # 1.0 physics and economics, no Gymnasium import
│   │   ├── storage/                # base.py (StorageSystem), battery.py (Battery, BatteryStep)
│   │   ├── nodes/                  # node.py (Node, NodeStep, NODE_TYPES)
│   │   └── community/              # shared_energy.py, incentives.py (tip_coefficient,
│   │                               # UniformIncentive, TipIncentive), welfare.py
│   │                               # (WelfareBreakdown), community.py (Community, CommunityStep)
│   ├── data/                       # schema.py (parse_config, ScenarioConfig, TimestampWarning),
│   │                               # loader.py (load_scenario, read_config)
│   ├── gym/                        # __init__.py (register_envs, ENV_ID), cer_env.py (SmartCerEnv),
│   │   │                           # run_config.py (map_v062_run_config)
│   │   ├── observation/            # encoders.py, normalizers.py (MinMaxNormalizer), schema.py
│   │   │                           # (ObservationBuilder, presets meco and extended, fit_normalizer)
│   │   ├── rewards/                # base.py (RewardCalculator, RewardBreakdown), calculators.py
│   │   │                           # (welfare, profit, energy), config.py (REWARD_REGISTRY,
│   │   │                           # make_reward)
│   │   ├── controllers/            # base.py (Controller), naive.py (NaiveController), rbc.py
│   │   │                           # (RBCController), baselines.py (CONTROLLERS, run_episode,
│   │   │                           # compute_baselines)
│   │   ├── tracking/               # tracker.py (ResultsTracker, TrackResults wrapper),
│   │   │                           # callback.py (ResultsCallback, Stable-Baselines3)
│   │   └── metrics/                # kpi.py (compute_kpis), fit_score.py (fit_score,
│   │                               # daily_welfare), KPI.md (formulas and units)
│   └── cer_data/                   # local scenarios (gitignored, read-only)
├── test/
│   ├── conftest.py                 # shared fixtures, description of the synthetic scenario
│   ├── fixtures/mini_cer/          # synthetic scenario, 0.6.2 format, 4 nodes, 48 h (committed)
│   ├── fixtures/mini_cer_v1/       # the same scenario in the 1.0 format (committed)
│   ├── fixtures/local/             # 0.6.2 reference values on Leather (gitignored)
│   ├── unit/core/                  # layer rules of smartcer/core/; storage/, nodes/, community/
│   ├── unit/data/                  # schema and loader on small scenarios written to tmp_path
│   ├── unit/gym/                   # observation features, normaliser, rewards, run configuration,
│   │                               # actions and arguments, controllers, tracker, KPIs, Fit Score
│   ├── integration/data/           # mini_cer in both formats; Leather and the 0.6.2 reference
│   └── integration/gym/            # check_env (Gymnasium, SB3), episodes, rewards, timing,
│                                   # normaliser window and one year in the space, baselines,
│                                   # SB3 results callback, performance rules and benchmark,
│                                   # SAC (slow)
├── benchmarks/                     # bench_step.py (time per step and per episode)
├── docs/                           # FORMULATION.md, CONFIG.md, refactor/REFACTOR_v1.md,
│                                   # refactor/PROGRESS.md
├── .claude/commands/               # slash commands
└── .github/workflows/              # ci.yml, release.yml
```

Run outputs go to a folder chosen by the caller, e.g. `results/` (gitignored). The rest of the
target 1.0 structure (forecasts, oracle) is in `docs/refactor/REFACTOR_v1.md` section
4; add it here only as it is created.

## Code Architecture

### Layer rule

`smartcer/core/` holds the physics and economics of `docs/FORMULATION.md` and imports neither
`gymnasium` nor `smartcer.gym`; it has no `print`, `sys.exit` or `assert` (checked by
`test/unit/core/test_layer_rules.py`).

### Storage (1.0), `smartcer/core/storage/`

```
StorageSystem (ABC): get_bounds(delta_t, charge_cap_kwh), step(action, delta_t, charge_cap_kwh), reset()
└── Battery(capacity, power_charge_max, power_discharge_max, efficiency_charge, efficiency_discharge,
            soc_min=0.0, soc_initial=None, alpha=0.0, usage_cost=0.0, charge_limit='production')
    - get_bounds(delta_t, charge_cap_kwh) → (a_upper, a_lower), both in [0, 1]; does not change the state
    - step(action, delta_t, charge_cap_kwh) → BatteryStep (frozen dataclass)
    - reset() → state of charge back to soc_initial
    - from_config(config) → Battery from a 1.0 or 0.6.2 `storage` section
```

- Formulation sections 4-7: self-discharge s̃ = max(S^min, (1 - α)^Δt s) first (`alpha` is per
  hour), then bounds from s̃,
  clipping of the action to [-a_lower, a_upper] with penalty k, then
  s = s̃ + η^cha p^cha Δt - p^dis Δt / η^dis. Both power limits are on the node side.
- `charge_cap_kwh` is the cap c of section 6, computed by the node from `charge_limit`
  (`production`: c = g; `surplus`: c = max(0, g - l)); `None` means no cap. The battery stores
  `charge_limit` but does not use it.
- `soc_min` and `soc_initial` are fractions of capacity (`soc_initial` defaults to `soc_min`);
  the state is `soc_kwh`, with `soc_min_kwh` and `soc_initial_kwh`.
- `BatteryStep` fields: `action_requested`, `action_applied`, `a_upper`, `a_lower`, `p_cha_kw`,
  `p_dis_kw`, `charge_kwh` (node → battery), `discharge_kwh` (battery → node), `throughput_kwh`
  (cell side), `storage_cost` (`usage_cost` × throughput, section 10), `penalty` (k), `soc_kwh`.
- Invalid parameters, actions outside [-1, 1] or non-finite, `delta_t ≤ 0` and negative caps raise
  `ValueError`; a state of charge outside [S^min, S^max] by more than 1e-9 kWh raises
  `RuntimeError` (implementation error); within 1e-9 it is clamped to the limit.
- `from_config` maps the 0.6.2 keys with a `DeprecationWarning` (`rated_power` → both power limits,
  `efficiency` → both efficiencies, `alfa` → `alpha`, `beta` ignored); unknown keys, a 0.6.2 key
  combined with its 1.0 target and missing required keys raise `ValueError`.

### Nodes and community (1.0), `smartcer/core/nodes/`, `smartcer/core/community/`

```
load_scenario(path, start=None, end=None) → Community          (smartcer/data/loader.py)
Community(nodes, incentive, timestamps, delta_t, name)
    - node_ids, storage_node_ids (order of the actions), n_steps, node(node_id), reset()
    - step(t, actions | None) → CommunityStep(t, nodes: tuple[NodeStep], welfare, penalty K_t)
        1. Node.step(t, action) per node → NodeStep (battery called with the node's charge cap)
        2. shared_energy(export, import) → E^sh
        3. incentive.compute(t, E^sh, export) → I (UniformIncentive or TipIncentive)
        4. welfare_breakdown(...) → WelfareBreakdown(revenue, cost, storage_cost, incentive,
           shared_energy_kwh, export_kwh, import_kwh, total)
Node(node_id, node_type, delta_t, generation_kwh, load_kwh, sell_price, buy_price, battery=None,
     photovoltaic=None, category=None)
    - charge_cap(t): `production` c = g, `surplus` c = max(0, g - l) (decision D6)
    - get_bounds(t) → (a_upper, a_lower); step(t, action=None) → NodeStep; reset()
```

- One `Node` class for the three types (`NODE_TYPES`); the data are per-step arrays in kWh and
  €/kWh, converted once by the loader; `step()` reads Python floats, no pandas.
- `NodeStep` fields: `node_id`, `t`, `generation_kwh`, `load_kwh`, `p_cha_kw`, `p_dis_kw`,
  `charge_kwh`, `discharge_kwh`, `net_kwh` (n), `export_kwh`, `import_kwh`, `revenue`, `cost`,
  `throughput_kwh`, `storage_cost`, `penalty`, `soc_kwh` (None without battery),
  `action_requested`, `action_applied`, `a_upper`, `a_lower`.
- `TipIncentive` holds the per-step, per-node coefficients (0 for nodes without a plant) and
  `potential(t, export)`; its incentive divides by the total export of all nodes.
- Loader: 1.0 scenarios are validated strictly; 0.6.2 scenarios are aligned by position (the
  period is selected on the first node's timestamps and every node is cut at the same rows) and
  their timestamp irregularities raise a `TimestampWarning` (docs/CONFIG.md section 4).

### Environment (1.0), `smartcer/gym/`

```
SmartCerEnv(scenario, start=None, end=None, episode_length=None, random_start=False,
            observe_current_step=True, observation='meco', lookback=0, normalizer=None,
            normalizer_window=None, reward='welfare', reward_per_node=False,
            penalty_coefficient=10.0, seed=None)
    - community (Community), storage_node_ids (action order), obs_layout (name → slice),
      normalizer (MinMaxNormalizer, `.save(path)`; pass the path as `normalizer` to reuse it),
      scenario_path, current_step (step of the next action), run_config() (JSON-serialisable)
    - action_space Box(-1, 1, (N,), float32); observation_space flat Box (ObservationBuilder)
    - reset(seed, options) → (obs, info): batteries to S^ini, first step 0 or random (np_random),
      seeds the action space
    - step(action) → (obs, reward, terminated=False, truncated, info)
        clip action to [-1, 1] (no penalty; non-finite → ValueError) → Community.step(t, action)
        → reward calculator (R_t = W_t - σ K_t by default); truncated on the last step of the
        episode, whose observation is
        the terminal one (same step, new state); step() then needs a reset
    - from_v062_config(cer_path, cer_config) → SmartCerEnv (0.6.2 run keys, DeprecationWarning;
      single_agent True → reward_per_node, 'energy' → reward 'energy')
```

- `info` of a step: `t`, `timestamp`, `privileged` (`{}` until phase 7, also in the reset info),
  `obs_clipped` (values of the returned observation clipped to [0, 1], also in the reset info),
  `reward` (`type`, `main`, `penalty` = weighted penalty, `total` = R_t, and `nodes` with
  `reward_per_node`), `welfare` (breakdown), `penalty` (K_t), `action` (after the [-1, 1] clip),
  `nodes` (node id → NodeStep fields).
- Rewards (`smartcer/gym/rewards/`, FORMULATION section 11): template method in
  `RewardCalculator.compute(step) → RewardBreakdown(main, penalty, total, nodes)`: collective
  M_t - σ K_t, or per node r_u = m_u - σ k_u over the battery nodes with R_t = Σ_u r_u. Types
  `welfare` (W_t, collective only), `profit` (revenue - cost - storage cost per node; collective
  adds I_t, equal to `welfare`), `energy` (-i_u², 0.6.2 legacy). `make_reward` reads the read-only
  `REWARD_REGISTRY` (`MappingProxyType`); no reward scaling.
- Observation (`smartcer/gym/observation/schema.py`), sections `time`, `price`, `community`,
  `nodes`, `forecast` (`ObservationBuilder.sections`). Preset `meco`: `time/month`, `time/hour`
  (sin, cos), `time/day_of_week` (one-hot); prices as `price/sell`, `price/buy` if all nodes share
  them, else `<node>/sell_price`, `<node>/buy_price` per battery node; per battery node
  `<node>/generation`, `<node>/load`, `<node>/soc` (scaled to [S^min, S^max]), `<node>/a_lower`,
  `<node>/a_upper`. Preset `extended` adds `time/holiday` (Italian national holidays only),
  `community/generation`, `community/load` (totals of the nodes without a battery) and, with
  `lookback` L, `<node>/generation_lookback`, `<node>/load_lookback` (L values, most recent first);
  the `forecast` section is empty until phase 7. Static features are precomputed for every step;
  only SoC and bounds are computed in `step()`. Bounds of constant features are widened by 1 so
  that low < high.
- Generation and load are min-max scaled with a `MinMaxNormalizer` fitted by `fit_normalizer` on
  the training window `normalizer_window` only, then clipped to [0, 1]
  (`ObservationBuilder.clipped` counts the clipped values per step); JSON via `save`/`load`
  (bounds, window, scenario name).
- `observe_current_step=False`: g and l of step t - 1 (0 at the first row of the data) and A^upper
  without the charge cap; the clipping still uses the full bounds.

**Action convention:** one action per battery in [-1, 1]; positive charges, negative discharges.

**Performance rules** (phase 6, `test/integration/gym/test_env_performance.py`): no pandas code
runs inside `step()` (nodes read preloaded Python floats, timestamps are a numpy array) and the
observation is built once per step; measure changes with `benchmarks/bench_step.py`.

### Controllers, tracking and metrics, `smartcer/gym/controllers/`, `tracking/`, `metrics/`

```
Controller(env)                     bound to the unwrapped SmartCerEnv; reads env.community
    - reset(); act(observation, info) → actions for env.current_step, in [-1, 1]
├── NaiveController ('naive')       zero actions
└── RBCController(env, mode='community')   FORMULATION section 13 ([M24] Algorithm 1);
                                    names 'rbc_node', 'rbc_community'
CONTROLLERS                         read-only registry: name → factory taking the env
run_episode(env, controller, seed) → ResultsTracker
compute_baselines(env, output_dir, controllers=('naive', 'rbc_community'), seed=None)
    → {name: KPIs}
    writes <output_dir>/<name>/ (community.csv, nodes/<id>.csv, kpis.json, run_config.json) and
    <output_dir>/kpis.json; refuses the scenario folder and any cer_data folder (S18)
TrackResults(env) wrapper → env.get_wrapper_attr('tracker'): ResultsTracker
    - community_table(), node_tables(), kpis(), save(output_dir, run_config) of the last
      completed episode (only the current and the last completed episode are kept)
ResultsCallback(output_dir, save_freq=1, run_config=None)   SB3, DummyVecEnv of TrackResults:
    logs KPIs under 'smartcer/<kpi>' at every completed episode, saves every save_freq episodes
compute_kpis(community_table, node_tables, capacities, soc_min) → dict   (formulas: KPI.md)
fit_score(reference, candidate) → F (%), daily_welfare(community_table) → W per day
```

- The RBC computes energies (node side) within the dynamic bounds (`node.get_bounds(t)`), so its
  actions are never clipped or penalised; it returns float64 actions, applied exactly (a float32
  equal to a bound may exceed it by 1e-8). Mode `node`: d = g - l, charge min(d, E^cha) or
  discharge min(-d, E^dis); a producer with g = 0 discharges E^dis. Mode `community` (reference
  baseline): balance B_t = Σ g - Σ l over all nodes, every battery charges (discharges) the same
  fraction min(1, |B_t| / Σ E) of its E^cha (E^dis). It reads g and l of step t from the
  scenario, as in [M24].
- KPIs: welfare and its components, shared, exported and imported energy, sharing rate
  (shared / exported), coverage rate (shared / imported), battery charge, discharge, cell
  throughput, equivalent full cycles on S^max (`equivalent_full_cycles`) and on S^max - S^min
  (`efc_usable`), penalty mean, max and steps (> 1e-9). Rates are fractions.

**Known defects** S1-S18 of 0.6.2 are listed in `docs/refactor/REFACTOR_v1.md` section 2; their
regression tests on the new API are named `test_s<n>_*`. All 0.6.2 modules and their
`xfail(strict=True)` tests are removed; S6 is `test_s6_rbc_executes_intended_energy`.

## Testing

- Layout: `test/unit/` (one component) and `test/integration/` (several components, full
  episodes); each test module sets `pytestmark = pytest.mark.unit` or `pytest.mark.integration`
- Markers (strict): `unit`, `integration`, `slow`
- Warnings are errors except `DeprecationWarning` and `PendingDeprecationWarning`; run the
  Gymnasium checker on `gym.make(ENV_ID, ...).unwrapped` (an env without a spec makes it warn)
- `--import-mode=importlib`: no `__init__.py` in test folders; same-named test files in different
  folders do not clash
- Synthetic scenario `test/fixtures/mini_cer/` (0.6.2 format) and `test/fixtures/mini_cer_v1/`
  (the same in the 1.0 format): round numbers, results computable by hand (see the docstring of
  `test/conftest.py`); `write_scenario` (conftest) writes small 1.0 scenarios to `tmp_path`
- Loading the local Leather scenario raises a `TimestampWarning`: expect it with `pytest.warns`
- Tests on local data (`test/fixtures/local/`, `smartcer/cer_data/`) skip when it is missing; tests
  on optional dependencies use `pytest.importorskip`
- Known defects: `@pytest.mark.xfail(strict=True, raises=<expected exception>, reason='S<n>: ...')`;
  an unexpected pass fails the suite. When a defect is fixed in a new module, add a regression
  test on the new API named `test_s<n>_...`; the xfail test goes with the module it tests

## Code Style Conventions

### MANDATORY CODING STANDARDS

**CRITICAL**: All code written in this repository MUST follow these standards:

- **Language**: English for code, comments, docstrings and docs
- **Type hints**: mandatory for all functions, methods and class attributes
- **Docstrings**: Google style on every class, method and function
- **Quotes**: single quotes `'...'`
- **Naming**: `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE_CASE`
  constants
- **Line length**: 100 characters; indentation 4 spaces; target Python 3.12
- **Inline comments**: lowercase first letter, used sparingly, only for non-obvious logic
- **No `print`** in library code (use `logging` or `warnings`); **no `sys.exit`**; no `assert`
  for input validation: raise exceptions with diagnostic messages (node, column, row, value)
- **Imports**: absolute; stdlib → third-party → project, sorted (Ruff isort settings in
  `pyproject.toml`)

### Documentation Format Template

```python
def example_function(param1: int, param2: str, optional_param: float = 1.0) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Brief description of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2.
        optional_param: Description of optional parameter. Defaults to 1.0.

    Returns:
        Tuple containing:
            - np.ndarray: Description of array returned.
            - dict[str, Any]: Description of dictionary returned.

    Raises:
        ValueError: When invalid parameters are provided.
    """
```

## Physical Units

| Quantity | Unit | Notes |
|----------|------|-------|
| Energy, capacity | kWh | CSV values are energies per step (0.6.2: per hour) |
| Power | kW | 1.0 (`smartcer.core`): energy = power × Δt. 0.6.2 has no time step: `rated_power` is used as kWh per step, correct only for hourly data (defect S5) |
| Time step | hours | `delta_t` from 1.0 (`docs/FORMULATION.md` section 2) |
| Prices, incentives | €/kWh | `price_exp` sell, `price_imp` buy; TIP parameters in €/kWh |
| State of charge | kWh | `soc_min` is configured as a fraction of capacity and stored in kWh |
| Efficiency, self-discharge | dimensionless | [0, 1] |
| PNRR | percentage | [0, 40] |

Symbols, units and configuration keys of 1.0 are in `docs/FORMULATION.md` section 3.

## Domain Glossary

| Term | Meaning |
|------|---------|
| CER | *Comunità energetica rinnovabile*, Renewable Energy Community: members (producers, consumers, prosumers) under the same primary substation that share the energy of renewable plants. Sharing is virtual: each member keeps its own grid connection and contracts |
| Energia condivisa | Shared energy: in every hour, the minimum between the energy fed into the grid by the community plants and the energy withdrawn by the members (`docs/FORMULATION.md` section 9) |
| TIP | *Tariffa incentivante premio*: incentive paid per kWh of shared energy, specific to each plant: a base value (`tip_base`) plus a part that grows as the zonal price falls below 0.180 €/kWh, capped (`tip_threshold`), plus a geographic correction (`fi_zone`) |
| PNRR | *Piano nazionale di ripresa e resilienza*: capital grant up to 40 % of a plant's cost; the TIP of a funded plant is reduced by a factor up to 50 % (`1 - 0.5 · pnrr / 40`) |
| Zonal price | *Prezzo zonale* P^z: hourly day-ahead wholesale price of the market zone of the plant; SmartCer approximates it with the node's sell price. The 0.6.2 key `pz` holds the zone name (e.g. `CNOR`), not the price |
| Ritiro dedicato | Scheme under which the GSE buys the energy fed into the grid, at the zonal price; the sell price π^egr of the scenarios typically represents it |

## Maintaining CLAUDE.md

CLAUDE.md must describe the code as it is: no paths that do not exist, no APIs that are not
implemented. Update it in the same commit as the change.

| Code Change | CLAUDE.md Sections to Update |
|-------------|------------------------------|
| New module or package | Package Structure, Code Architecture |
| New or changed command or Make target | Development Commands, Claude Code Commands |
| New config option or environment argument | Environment Registration and Installation |
| New convention or test rule | Code Style Conventions, Testing |
| Refactoring phase completed | Project Overview (status), and every section the phase touched |
| Breaking API change | All affected sections + migration notes |

Before committing an update, check that code examples use current signatures, file paths exist,
import statements match the package structure and version numbers are consistent. Files to keep in
sync: `README.md`, `CHANGELOG.md`, `docs/FORMULATION.md`, `docs/refactor/PROGRESS.md`,
`pyproject.toml`.
