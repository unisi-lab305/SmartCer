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
questions are in [docs/refactor/PROGRESS.md](docs/refactor/PROGRESS.md). Phases 0 (infrastructure
and safety net), 1 (battery) and 2 (nodes, community and loader) are complete: the environment is
still the 0.6.2 code, covered by characterisation tests and by strict expected-failure tests of its
known defects; the new 1.0 physics and economics (`smartcer/core/`) and scenario loader
(`smartcer/data/`) are not used by the 0.6.2 environment yet. Scenario keys:
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
`holidays`, and `tqdm` (used only by the 0.6.2 environment, removed in phase 3).

**Environment ID**: `SmartCer-v0`, registered on `import smartcer` (`smartcer/__init__.py`).

```python
from smartcer.cer_env import SmartCer

env = SmartCer(cer_path='path/to/scenario', cer_config=run_config)
obs, info = env.reset(seed=0)
obs, reward_list, terminated, truncated, info, done = env.step(actions)   # 0.6.2 signature
```

The 0.6.2 `step()` is not Gymnasium-compliant (defect S3): it returns 6 items, the reward is a list
`[reward, penalty, signal]` and `truncated` is `None`. See `test/conftest.py` for a complete
`cer_config` and `README.md` for the 0.6.2 usage example.

## Development Commands

```bash
make help                   # list all targets (from the `## ` comments in the Makefile)
make test                   # all tests (unit + integration)
make test-unit              # test/unit/ only
make test-integration       # test/integration/ only
make test-coverage          # coverage report (terminal, htmlcov/, coverage.xml)
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

**Reference values of 0.6.2:** `python scripts/make_reference_v062.py` runs 0.6.2 on the local
Leather 2022 scenario with idle batteries (about 1 minute) and writes the hourly community series
and the annual KPIs to `test/fixtures/local/` (gitignored). The script is deleted in phase 3; its
outputs stay.

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
     `smartcer.cer_env`, `smartcer.core` and `smartcer.data` from outside the repository (guards
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
├── smartcer/                       # package (0.6.2 code)
│   ├── __init__.py                 # __version__, registration of SmartCer-v0
│   ├── core/                       # 1.0 physics and economics, no Gymnasium import
│   │   ├── storage/                # base.py (StorageSystem), battery.py (Battery, BatteryStep)
│   │   ├── nodes/                  # node.py (Node, NodeStep, NODE_TYPES)
│   │   └── community/              # shared_energy.py, incentives.py (tip_coefficient,
│   │                               # UniformIncentive, TipIncentive), welfare.py
│   │                               # (WelfareBreakdown), community.py (Community, CommunityStep)
│   ├── data/                       # schema.py (parse_config, ScenarioConfig, TimestampWarning),
│   │                               # loader.py (load_scenario, read_config)
│   ├── cer_env.py                  # SmartCer environment (gym.Env)
│   ├── cer_utils.py                # node loader, OHESpace, energy trade, TIP, CSV and JSON export
│   ├── cer_models/                 # node_models.py (Node, Producer, Consumer, Prosumer),
│   │                               # energy_models.py (StorageSystem, Battery)
│   ├── cer_reward/                 # reward.py (Reward), scaler.py (RewardScaler)
│   ├── cer_preprocessing/          # preprocessing.py (encoders by operator overloading)
│   ├── cer_metrics/                # scores.py (compute_kpis), KPI.md
│   ├── storage_controllers/        # abc_controller.py, rbc_controller.py (StandardRBC)
│   └── cer_data/                   # local scenarios (gitignored, read-only)
├── test/
│   ├── conftest.py                 # shared fixtures, description of the synthetic scenario
│   ├── fixtures/mini_cer/          # synthetic scenario, 0.6.2 format, 4 nodes, 48 h (committed)
│   ├── fixtures/mini_cer_v1/       # the same scenario in the 1.0 format (committed)
│   ├── fixtures/local/             # 0.6.2 reference values on Leather (gitignored)
│   ├── unit/core/                  # layer rules of smartcer/core/; storage/, nodes/, community/
│   ├── unit/data/                  # schema and loader on small scenarios written to tmp_path
│   ├── unit/v062/                  # TIP coefficient; known defects S2, S4, S5, S6
│   ├── integration/data/           # mini_cer in both formats; Leather vs the 0.6.2 reference
│   └── integration/v062/           # characterisation; known defects S3, S7-S10; reference values
├── scripts/make_reference_v062.py  # writes test/fixtures/local/ (deleted in phase 3)
├── docs/                           # FORMULATION.md, CONFIG.md, refactor/REFACTOR_v1.md,
│                                   # refactor/PROGRESS.md
├── .claude/commands/               # slash commands
└── .github/workflows/              # ci.yml, release.yml
```

The target 1.0 structure (`smartcer/core/`, `smartcer/data/`, `smartcer/gym/`, ...) is in
`docs/refactor/REFACTOR_v1.md` section 4; add it here only as it is created.

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
- Loader: 1.0 scenarios are validated strictly; 0.6.2 scenarios are aligned by position and their
  timestamp irregularities raise a `TimestampWarning` (docs/CONFIG.md section 4).

### 0.6.2 data flow

```
SmartCer (gym.Env), smartcer/cer_env.py
├── node_loader() reads <scenario>/config.json and one CSV per node
│   └── Producer / Consumer / Prosumer (Node), each with an optional Battery (StorageSystem)
├── step(actions: OrderedDict[node_id, np.ndarray (1,)])
│   1. Node.set_storage(action): energy = action × rated_power → Battery.charge(energy, production)
│   2. get_energy_trade(): node export and import; set_tip_and_costs(): TIP coefficient, revenue,
│      expense per node
│   3. community: exported, imported, shared = min(exported, imported),
│      potential TIP = Σ coefficient × export, actual TIP = shared / exported × potential TIP
│   4. Reward: 'energy' or 'profit', per node (single_agent=True) or collective, minus penalty,
│      optional RewardScaler
├── get_sim_results() / save_results(): community table, one table per node, KPI JSON, run JSON
└── compute_baselines(): idle batteries and StandardRBC
```

**Scenario format (0.6.2):** `config.json` with `Producers`, `Consumers`, `Prosumers`; each node has
`Type`, optional `Photovoltaic` (`pv_power`, `pnrr`, `tip_base`, `tip_threshold`, `fi_zone`, `pz`)
and optional `Storage` (`capacity`, `rated_power`, `soc_min` as a fraction, `efficiency`, `alfa`,
`beta`). CSV files in `nodes/<type>s/<base>.csv` with a `datetime` column
(`%Y-%m-%d %H:%M:%S`) and `production`, `consumption`, `price_exp`, `price_imp` as required by the
node type; a node id with a `_N` suffix reuses the CSV of its base name (`house_1` → `house.csv`).

**Action convention:** one action per battery in [-1, 1]; positive charges, negative discharges.

**Known defects** S1-S18 of 0.6.2 are listed in `docs/refactor/REFACTOR_v1.md` section 2; S2-S10
are reproduced by tests marked `xfail(strict=True)` in `test/*/v062/test_known_defects.py`.

## Testing

- Layout: `test/unit/` (one component) and `test/integration/` (several components, full
  episodes); each test module sets `pytestmark = pytest.mark.unit` or `pytest.mark.integration`
- Markers (strict): `unit`, `integration`, `slow`
- Warnings are errors except `DeprecationWarning` and `PendingDeprecationWarning`; the only other
  ignored warning is the `SyntaxWarning` of defect S14, until `node_models.py` is removed
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
  test on the new API named `test_s<n>_...`; the 0.6.2 xfail test stays until its module is
  removed in phase 3

## Code Style Conventions

### MANDATORY CODING STANDARDS

**CRITICAL**: All code written in this repository MUST follow these standards (the 0.6.2 modules
predate them and are not reformatted):

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
