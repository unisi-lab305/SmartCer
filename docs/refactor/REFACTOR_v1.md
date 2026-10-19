# SmartCer 1.0 – refactoring brief for Claude Code

This brief is the complete specification for turning SmartCer 0.6.2 into SmartCer 1.0. Read it in full, together with `docs/FORMULATION.md`, before changing anything. The two documents are written for you; they assume no other context.

---

## 0. Working protocol

1. **Session start.** Read this file, `docs/FORMULATION.md`, `docs/refactor/PROGRESS.md` (created in phase 0) and `CLAUDE.md` (from phase 0 on).
2. **One phase per request.** Phases are in section 7. Work on a branch `refactor/v1-phase-<N>-<slug>` created from `main`. Do not start the next phase without an explicit go-ahead from Leonardo.
3. **End of every phase:**
   - run `make test` and every acceptance check of the phase;
   - update `docs/refactor/PROGRESS.md`: what was done, acceptance results with the actual numbers, deviations from this brief and why, open questions;
   - add user-facing entries under `[Unreleased]` in `CHANGELOG.md`;
   - update `CLAUDE.md` if the package structure, commands or conventions changed;
   - commit (conventional commits) and stop with the report of section 9. Open a pull request only when asked.
4. **Ask before:** changing the meaning of anything in `docs/FORMULATION.md`; adding a runtime dependency not listed in this brief; bumping the MAJOR version; removing files that are tracked in git and not listed for removal here; pushing tags.
5. **Read-only material.**
   - `smartcer/cer_data/` holds local scenarios (gitignored, partly confidential company data). Read it, never modify it, never commit files derived from it at hourly resolution.
   - The storage-node-environment repository at `SNE_PATH` (default `../storage-node-environment`; if it is not there, stop and ask for the path) is the reference for code to port. Copy and adapt; never import it, never modify it.
6. **Code standards** (also in `CLAUDE.md`): English for code, comments, docstrings and docs; type hints everywhere; Google-style docstrings; single quotes; `snake_case`, `PascalCase`, `UPPER_SNAKE_CASE`; line length 100; lowercase inline comments, used sparingly; no `print` in library code (use `logging` or `warnings`); no `sys.exit`; raise exceptions with diagnostic messages.
7. **Commit messages:** conventional commits (`<type>(<scope>): <description>`), one line unless asked otherwise; no `Co-Authored-By` lines and no references to AI tools, as in the storage-node-environment commands.
8. **When this brief conflicts with the code or is ambiguous, stop and ask.** Do not resolve it silently.
9. **Strangler approach.** Build the new modules next to the old ones. The old modules (`smartcer/cer_env.py`, `cer_utils.py`, `cer_models/`, `cer_reward/`, `cer_preprocessing/`, `cer_metrics/`, `storage_controllers/`) stay untouched until their replacement passes its tests, and are removed in the phase indicated in section 7.

---

## 1. Context

- **SmartCer** is a Gymnasium environment for Italian Renewable Energy Communities (CER, *comunità energetiche rinnovabili*). Nodes are producers, consumers and prosumers, some with a battery. The community receives an hourly incentive on shared energy (*energia condivisa*), the minimum between total exported and total imported energy.
- **Use in the SUMMA project (class A).** A centralised agent controls all batteries of a CER with Soft Actor-Critic and an asymmetric actor-critic: the critic also receives the optimal actions a\* computed by a MILP oracle (method of MECO 2024, see `docs/FORMULATION.md`). Data: a CER in Molise, about 10 prosumers with a battery plus about 100 consumers aggregated into one load, one year at hourly resolution. Later the agent state will include forecasts (12 h horizon, simulated); that work starts after this refactoring and only needs the hooks of phase 7.
- **storage-node-environment** (`SNE_PATH`, version 0.14.0) is a single-node battery environment by the same author, maintained as a pip package with 438 tests, CI and an automated release. It is the reference for the battery code and for the software engineering. The two projects keep different scopes: SmartCer models communities, storage-node-environment a single node.
- **Deadline.** The project closes on 11/11/2026. Phases 0-4 and 6 are the minimum needed for class A.

---

## 2. Current state of SmartCer 0.6.2

Last commit 17/03/2025, no tests, about 2,200 lines of Python. Main files: `smartcer/cer_env.py` (661 lines, environment), `smartcer/cer_utils.py` (428, loader, spaces, energy trade, TIP, CSV export), `smartcer/cer_models/node_models.py` (433), `smartcer/cer_models/energy_models.py` (189, battery), `smartcer/cer_reward/` (reward and running scaler), `smartcer/cer_preprocessing/preprocessing.py` (encoders via operator overloading), `smartcer/cer_metrics/scores.py` (10 KPIs), `smartcer/storage_controllers/` (rule-based controller).

Local scenarios in `smartcer/cer_data/`: `Leather` (10 nodes: producers `pv`, `pv_storage`; consumers `cartotecnica`, `utente_singolo_1`, `utente_singolo_2`, `utente_coppia`; prosumers `conceria_dolmen`, `toscana_fridge`, `utente_famiglia`, `utente_famiglia_storage`; 4 batteries; hourly 2022), `cer_azienda`, `cer_coffee`, `cer_retail`, `cer_utente`. CSV values are energies per hour (kWh); price columns are `price_imp` and `price_exp`; node ids with a `_N` suffix reuse the CSV of their base name.

### Verified defects

Each defect needs a regression test in the phase that fixes it.

| ID | Severity | Location | Defect |
|---|---|---|---|
| S1 | critical | `pyproject.toml`, `packages = ["smartcer"]` | `pip install .` does not install the subpackages: `ModuleNotFoundError: smartcer.storage_controllers` |
| S2 | critical | `cer_reward/reward.py:40`, README | the README and the saved `sim_config.json` files use `reward.type = 'cost'`, which the code rejects (`energy`, `profit` only) |
| S3 | critical | `cer_env.py:255-280`, `:377-467` | not Gymnasium-compliant: `step()` returns 6 items, the reward is a list `[reward, penalty, signal]`, `truncated` is `None`; the observation has `cer_production` and `cer_consumption` that the space does not declare; `flatdim` fails on the custom `OHESpace`; `check_env` and `gym.make('SmartCer-v0')` fail |
| S4 | high | `cer_models/energy_models.py:143-149` | the discharge limit is applied to delivered energy, not to the energy taken from the cell: the state of charge drops below `soc_min` (in about half of the steps of a random year on Leather, by up to 0.6% of capacity with efficiency 0.98); with `soc_min = 0` the state of charge becomes negative, an assertion fails and `Node.set_storage` calls `sys.exit(1)` |
| S5 | high | `cer_models/node_models.py:163` | no time step: energy = action × rated power, correct only for hourly data |
| S6 | high | `storage_controllers/rbc_controller.py:41, 46, 52, 57` | the rule-based controller normalises by capacity instead of rated power (surplus 40 kWh, 30 kW, 100 kWh: action 0.30, executed 9 kWh instead of 30); the saved `baseline_rbc` results are invalid |
| S7 | high | `cer_env.py:200-249` | the observation used to choose the action of step t contains the generation and load of step t, and the upper bound uses the generation of step t (implicit perfect nowcast); this becomes the explicit option `observe_current_step` |
| S8 | medium | `node_models.py:166`, `cer_env.py:408-412` | a consumer with a battery raises `KeyError: 'production'`; `step()` ignores consumer batteries |
| S9 | medium | `cer_reward/scaler.py`, `cer_env.py:309-329` | running reward standardisation inside the environment, never reset between episodes, active in evaluation; first step has infinite variance and a scaled reward of 0 |
| S10 | medium | `cer_env.py:211` | the terminal observation wraps to t = 0 (`time_step % sim_hours`) |
| S11 | medium | `node_models.py:169-174` | `sys.exit(1)` inside the library |
| S12 | medium | `cer_metrics/scores.py`, `cer_metrics/KPI.md` | KPI `self_consumption` is the sharing rate (shared / exported); `KPI.md` describes the TIP as an "optimisation potential"; `net_metering` and `energy_balance` have no clear physical meaning |
| S13 | low | `node_models.py:139-151`, `Node.reset()` | 4.9 ms per step with 10 nodes (one year in 43 s); 86% of the time in `get_current_data()` (pandas access per value); the observation is computed twice per step; CSV files are re-read at every `reset()` |
| S14 | low | `node_models.py:90` | `SyntaxWarning: invalid escape sequence '\d'` on Python 3.12 |
| S15 | low | `energy_models.py:173-177` | ad hoc degradation formula; state of charge not clipped to the reduced capacity |
| S16 | low | `cer_reward/reward.py`, `scaler_loader()` | `single_agent=True` returns one reward per node (misleading name) |
| S17 | low | `cer_env.py:478-517` | `reset(seed)` does not seed the action space; no environment random generator |
| S18 | low | `.gitignore`, `save_results()` | results are written into `cer_data/<scenario>/simulations/` |

Also: `delta_input`, `delta_output` and `max_ratio` are never read; `.idea/` is tracked in git.

### What must be preserved

- Hourly shared energy, the per-plant TIP coefficient with the PNRR correction and its proportional allocation (`cer_env.py:427-452`, `cer_utils.py:276-293`): this becomes the `tip` incentive model of `docs/FORMULATION.md` section 9.
- The scenario layout (`config.json` with `Producers`, `Consumers`, `Prosumers`; `nodes/<type>s/<base>.csv`; `_N` suffix reuse), readable with a `DeprecationWarning`.
- Day-of-week one-hot encoding in the state (missing in storage-node-environment, relevant for industrial loads).
- The output structure used in reports: a community table, one table per node, a KPI JSON, the run configuration JSON.

---

## 3. Decisions (fixed on 18/10/2026)

| ID | Topic | Decision |
|---|---|---|
| D1 | Reuse of the physics | Port the storage-node-environment code into SmartCer, adapted; no parity test and no dependency on storage-node-environment, not even for development |
| D2 | Action space | Continuous `Box(-1, 1, (N,))`, one action per battery; no discrete wrapper, no action masking |
| D3 | Information timing | Explicit option `observe_current_step` (default `True`, as in MECO 2024 and 0.6.2); semantics in `docs/FORMULATION.md` section 12 |
| D4 | Battery convention | The formulation of the papers (`docs/FORMULATION.md`): [S22] dynamics with both power limits on the node side, bounds derived from the dynamics |
| D5 | Units | Internal power in kW with an explicit time step; scenario CSV in kWh per step by default (`units: kWh`), `units: kW` accepted |
| D6 | Charge source | No charging from the grid: charge energy ≤ node generation in the step (`charge_limit: production`); `surplus` is an option pending the DIISM answer |
| D7 | Battery degradation | Not modelled; legacy `beta` read and ignored with a `DeprecationWarning` |
| D8 | Version and ID | 1.0.0, environment ID `SmartCer-v1`, `SmartCer-v0` removed |
| D9 | Python | `>= 3.12` |
| D10 | Compatibility | The loader reads both the 0.6.2 and the 1.0 scenario formats, with a `DeprecationWarning` for 0.6.2; no migration of existing files |
| D11 | Distribution | Install from a GitHub tag (`pip install git+https://github.com/unisi-lab305/SmartCer@v1.0.0`); PyPI publication enabled only after 11/11/2026 |

**Project infrastructure as in storage-node-environment:** `CLAUDE.md`, Claude Code commands, semantic versioning with the version kept in two synchronised files, `CHANGELOG.md` in Keep a Changelog format, Makefile, pre-commit, CI and a release workflow (phase 0).

---

## 4. Target architecture

```text
SmartCer/
├── CLAUDE.md, CHANGELOG.md, README.md, Makefile, pyproject.toml, pytest.ini, .pre-commit-config.yaml
├── .claude/commands/     add-and-commit.md, bump-version.md, pr.md, release.md, write-test.md
├── .github/workflows/    ci.yml, release.yml
├── docs/
│   ├── FORMULATION.md    # provided, authoritative
│   ├── CONFIG.md         # scenario and run configuration reference (phase 2)
│   └── refactor/         REFACTOR_v1.md (this file), PROGRESS.md
├── smartcer/
│   ├── __init__.py       # __version__, registration
│   ├── core/             # pure physics and economics, no Gymnasium import
│   │   ├── storage/      base.py, battery.py
│   │   ├── nodes/        base.py, producer.py, consumer.py, prosumer.py (or one class with a type)
│   │   └── community/    shared_energy.py, incentives.py, welfare.py
│   ├── data/             loader.py, schema.py
│   ├── forecasts/        base.py, perfect.py
│   ├── oracle/           provider.py
│   └── gym/
│       ├── __init__.py   # register_envs(), SmartCer-v1
│       ├── cer_env.py    # SmartCerEnv
│       ├── observation/  schema.py, encoders.py, normalizers.py
│       ├── rewards/      base.py, calculators.py, config.py
│       ├── controllers/  base.py, naive.py, rbc.py, replay.py
│       ├── tracking/     tracker.py, callback.py
│       └── metrics/      kpi.py, fit_score.py, KPI.md
├── test/                 unit/, integration/, fixtures/, conftest.py
├── scripts/              make_reference_v062.py
└── benchmarks/           bench_step.py
```

Layer rule: `core/` does not import `gym/`, `gymnasium` or `pandas` at step time. The environment sends actions to the nodes, the nodes call their battery and return their flows, the community module computes shared energy, incentive and welfare, the reward module turns welfare and penalty into the reward. Forecasts enter only the observation; oracle data enter only `info['privileged']`.

---

## 5. Porting from storage-node-environment

### Port and adapt

| Source in `SNE_PATH` | Destination | Adaptation |
|---|---|---|
| `storage_node_env/core/storage/base.py`, `battery.py` | `smartcer/core/storage/` | change the convention to `docs/FORMULATION.md` sections 4-7: discharge limit on the node side (in storage-node-environment it is on the cell side), bounds with η and S^min, charge cap passed by the node, action-based API, `usage_cost`, raise instead of assert |
| `storage_node_env/core/nodes/base.py` | `smartcer/core/nodes/base.py` | keep the CSV time-step validation; make the battery optional; preload numpy arrays; no `print` |
| `storage_node_env/gym/rewards/base.py`, `calculators.py`, `config.py` | `smartcer/gym/rewards/` | keep the template method (main term minus weighted penalties); the factory must use the registry in `config.py` (in storage-node-environment it uses `if/elif`); no `normalize` flag (it is a no-op there) |
| `storage_node_env/gym/preprocessing/encoders.py`, `normalizers.py` | `smartcer/gym/observation/` | per-instance schema instead of the module-level `FEATURE_REGISTRY` and `OBSERVATION_SCHEMA`; no hard-coded `year - 2024`; fit on a declared training window and serialise to JSON |
| `storage_node_env/gym/tracking/` | `smartcer/gym/tracking/` | one community table and one table per node; CSV export; SB3 callback |
| `storage_node_env/gym/metrics/` | `smartcer/gym/metrics/` | only what the KPI list of phase 5 needs; visualisations optional |
| `CLAUDE.md` | `CLAUDE.md` | same section structure, rewritten for SmartCer, only existing paths (the source mentions `core/base/`, `examples/` and two notebooks that do not exist) |
| `.claude/commands/*.md` | `.claude/commands/` | see phase 0 |
| `Makefile`, `pytest.ini`, `.pre-commit-config.yaml`, ruff settings in `pyproject.toml` | same | rename the package; same targets |
| `.github/workflows/ci.yml`, `release.yml` | same | release without the PyPI steps (D11) |
| `test/conftest.py` patterns | `test/conftest.py` | fixtures for the synthetic scenario |

### Do not port

- `spaces.Discrete` actions, `action_masks()` and the action-to-power table (D2).
- `reset()` that executes a neutral action on the first row and the extra terminal step with reward 0: in SmartCer the first step is controlled and the episode ends on its last valid step (phase 3).
- Module-level mutable registries.
- The normaliser fitted on the whole CSV.
- Tests that fail instead of being skipped when optional dependencies are missing: use `pytest.importorskip`.

---

## 6. Configuration

Two levels, documented in `docs/CONFIG.md` in phase 2:

- **Scenario** (`<scenario>/config.json` plus CSV files): physical and economic description of the community. Format 1.0 adds at least `format_version: "1.0"`, `delta_t` (hours), `units` (`kWh` default or `kW`), `incentive` (`model`: `uniform` with `unit_value` as a number or a CSV column, or `tip`), and per node: `type`, optional `csv` path (default: the 0.6.2 folder rule), optional `photovoltaic` (TIP parameters), optional `storage` with the keys of `docs/FORMULATION.md` section 3 plus `charge_limit`. Column requirements: producer `production`, `sell_price`; consumer `consumption`, `buy_price`; prosumer all four. Extra columns are ignored.
- **Run** (environment arguments): period (`start`, `end`), episode settings, reward, observation, `observe_current_step`, forecast and oracle providers, output directory.
- **0.6.2 format** (no `format_version`): read with a `DeprecationWarning`; incentive model `tip`; battery keys mapped as `rated_power` → both power limits, `efficiency` → both efficiencies, `soc_min` unchanged (fraction), `alfa` → `alpha`, `beta` ignored; `soc_initial` defaults to `soc_min` as in 0.6.2. The 0.6.2 run keys `simulation.start`, `simulation.end`, `simulation.seed`, `reward.type`, `reward.constraints.coefficient` are mapped, with `type: 'profit'` or the older `'cost'` (S2) and `single_agent: False` → reward `welfare`; unknown or unused keys are reported in the warning.
- Validation errors name the node, the column and the first offending row: missing columns, NaN in required columns, unsorted or duplicated timestamps, CSV frequency different from `delta_t`, nodes with different timestamps.

---

## 7. Phases

### Phase 0 · Project infrastructure and safety net

1. Create an annotated tag `v0.6.2` on the current `main` locally; ask before pushing it.
2. `pyproject.toml`: `requires-python = '>=3.12'`; `[tool.setuptools.packages.find] include = ['smartcer*']`; runtime dependencies `gymnasium`, `numpy`, `pandas`, `holidays`; extras `dev` (pytest, pytest-cov, ruff, pylint, pre-commit), `rl` (stable-baselines3), `all`; ruff settings as in storage-node-environment. Keep the version at `0.6.2` until phase 8. Add `__version__ = '0.6.2'` to `smartcer/__init__.py`.
3. `Makefile` with `test`, `test-unit`, `test-integration`, `test-coverage`, `version` (checks that `pyproject.toml` and `smartcer/__init__.py` agree), `clean`, `help` (from `## ` comments); `pytest.ini` with strict markers (`unit`, `integration`, `slow`) and warnings as errors except deprecations; `.pre-commit-config.yaml` as in storage-node-environment.
4. `.github/workflows/ci.yml`: on push and pull request to `main`, Python 3.12, install `.[all]`, unit tests, integration tests, coverage report, plus a step that builds the wheel, installs it in a clean virtual environment and imports `smartcer` (guards S1). `release.yml` on tags `vX.Y.Z`: version consistency between tag, `pyproject.toml` and `__init__.py`; CHANGELOG entry present; tests; build; `twine check`; GitHub Release with the built files. No PyPI step: leave a commented block marked "enable after 11/11/2026 (D11)".
5. `CHANGELOG.md` in Keep a Changelog format with an empty `[Unreleased]` and a `[0.6.2] - 2025-03-17` entry summarising the git history.
6. `CLAUDE.md` with the section structure of `SNE_PATH/CLAUDE.md`: project overview, installation and registration, development commands, Claude Code commands, CI/CD, version management (semantic versioning, two version files, PATCH/MINOR automatic, MAJOR only after asking), package structure, code architecture, code style, physical units, maintaining `CLAUDE.md`. Add a domain glossary (CER, energia condivisa, TIP, PNRR, zonal price, ritiro dedicato) and the rule that `docs/FORMULATION.md` is authoritative. Describe the code as it is at the end of each phase; no paths that do not exist.
7. `.claude/commands/`: copy the 5 commands of `SNE_PATH/.claude/commands/` and adapt them:
   - `bump-version`, `release`: version file `smartcer/__init__.py`; `release` creates and pushes the tag, which triggers the GitHub Release only (no PyPI wording);
   - `write-test`: target the `test/unit/` and `test/integration/` layout and the Makefile `## ` help mechanism (the source command creates `test/<module>/` and uses `$(GREEN)` colour variables that its Makefile does not define);
   - `add-and-commit`, `pr`: project name only.
8. `docs/refactor/PROGRESS.md` created; `docs/FORMULATION.md` is already in the repository: check that it is there and leave it unchanged.
9. Propose removing `.idea/` from version control and adding it to `.gitignore`; do it only after confirmation.
10. Test scaffolding: `test/conftest.py`; a synthetic scenario `test/fixtures/mini_cer/` in the 0.6.2 format with 4 nodes (a producer with a battery, a consumer, a prosumer with a battery, a prosumer without), 48 hourly rows, round numbers so that results can be computed by hand. It is committed; Leather data are not.
11. Characterisation tests on the current 0.6.2 code (they must pass now):
    - `get_tip_coefficient` on hand-computed cases (PNRR 0, 30 and 40; zonal price above and below 0.180 €/kWh; threshold binding and not binding);
    - one step of the synthetic scenario with zero actions: exported and imported energy per node, shared energy = min of the totals, actual TIP = shared / exported × Σ coefficient × exported;
    - energy balance identity per node over a random 48 h run: production + discharge + import = consumption + charge + export.
12. Reference values for later phases: `scripts/make_reference_v062.py` runs 0.6.2 on Leather 2022 with zero actions and writes the annual community KPIs and hourly community series to `test/fixtures/local/` (gitignored). Tests that use them are skipped when the folder is missing. Run the script in this phase: it depends on the 0.6.2 modules and is deleted in phase 3, while its outputs stay.
13. Tests marked `xfail(strict=True)` that reproduce S2-S10 where a unit test can show them (S1 is guarded by the CI install step). Before the old modules are removed in phase 3, each of them is replaced by a regression test on the new API.

*Acceptance:* `pip install .` and `import smartcer` work in a clean Python 3.12 environment; `make version` passes; `make test` passes; CI is green on the branch; `/add-and-commit` and `/bump-version patch` work on a throwaway branch (then discard it).

### Phase 1 · Battery

1. `smartcer/core/storage/base.py`: abstract `StorageSystem` with `get_bounds`, `step`, `reset`.
2. `smartcer/core/storage/battery.py`, `Battery` as in `docs/FORMULATION.md` sections 4-7. Suggested API (adjust if needed, and document the final one):

   ```python
   class Battery(StorageSystem):
       def __init__(self, capacity: float, power_charge_max: float, power_discharge_max: float,
                    efficiency_charge: float, efficiency_discharge: float, soc_min: float = 0.0,
                    soc_initial: float | None = None, alpha: float = 0.0,
                    usage_cost: float = 0.0, charge_limit: str = 'production') -> None: ...
       def get_bounds(self, delta_t: float, charge_cap_kwh: float | None) -> tuple[float, float]: ...
           # (a_upper, a_lower), both in [0, 1]
       def step(self, action: float, delta_t: float, charge_cap_kwh: float | None) -> BatteryStep: ...
           # clips, updates the state of charge, returns a dataclass with requested and applied
           # action, p_cha_kw, p_dis_kw, node-side energies, cell throughput, penalty k, soc_kwh
       @classmethod
       def from_config(cls, config: dict) -> 'Battery': ...  # 1.0 and 0.6.2 keys (section 6)
   ```

   `soc_min` and `soc_initial` are fractions of capacity; `soc_initial` defaults to `soc_min`. Invalid parameters raise `ValueError`.

*Acceptance:*
- 100,000 random actions with Δt ∈ {0.25, 1.0} h and random parameters: the state of charge stays in [S^min, S^max] within 1e-9 and the penalty is zero whenever the action lies in the bounds.
- Hand cases with 10 kWh, 5 kW, η = 0.95, `soc_min = 0`, Δt = 1 h and no charge cap (`charge_cap_kwh=None` means no cap): from 5 kWh, full charge → state of charge +4.75 kWh, 5 kWh from the node; from 8 kWh, full discharge → state of charge −5.263 kWh, 5 kWh to the node; from 5 kWh, full discharge is clipped to a = −0.95 (4.75 kWh to the node, state of charge 0) with penalty 0.05.
- Actions equal to A^upper and −A^lower reach the limits without crossing them.
- Legacy configuration mapping tested, including the `DeprecationWarning`.
- No `sys.exit` and no `print` in `smartcer/core/`.

### Phase 2 · Nodes, community and loader

1. `smartcer/core/nodes/`: node with optional battery, data preloaded as numpy arrays (generation, load, sell price, buy price, all converted to kWh per step according to `units`), timestamps parsed once, CSV frequency checked against Δt. `step(t, action)` returns a dataclass with g, l, p_cha, p_dis, n, e^gri, i^gri, revenue, cost, storage cost, penalty k, state of charge, requested and applied action, bounds. Charge cap per `charge_limit` (D6).
2. `smartcer/core/community/`: pure functions for shared energy, the `uniform` and `tip` incentive models and the welfare breakdown (revenue, cost, storage cost, incentive, shared energy, total), as in `docs/FORMULATION.md` sections 8-10.
3. `smartcer/data/`: loader and schema for both formats (section 6), `docs/CONFIG.md`.

*Acceptance:*
- The phase 0 characterisation tests, re-pointed to the new functions, pass with the `tip` model.
- With idle batteries, the annual community KPIs on Leather match the 0.6.2 reference values within 1e-6 (skipped if the local reference is missing).
- The welfare of the synthetic scenario matches a hand calculation, for both incentive models.
- The synthetic scenario written in the 1.0 format gives the same results as in the 0.6.2 format.
- A consumer with a battery loads and steps without error (its battery cannot charge under D6).

### Phase 3 · Gymnasium environment

1. `smartcer/gym/cer_env.py`, `SmartCerEnv`: `reset(seed=None, options=None) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)` with a `float` reward R_t = W_t − σK_t.
2. Action space `Box(-1, 1, (N,), float32)` in the fixed order of `storage_node_ids` (public attribute). Clipping and penalty as in `docs/FORMULATION.md` section 7.
3. `observe_current_step` (D3, default `True`) as in `docs/FORMULATION.md` section 12.
4. Observation: flat `Box` with attribute `obs_layout` (feature name → slice) so that per-entity encoders can read it. Preset `meco` ([M24] Tab. II): sell and buy prices without normalisation, as one global pair when all nodes share the same series and otherwise inside each battery node's block (a documented deviation, needed for scenarios such as Leather), month and hour cyclical, day of week one-hot; per battery node l and g min-max, state of charge scaled to [S^min, S^max], A^lower and A^upper. The `extended` preset follows in phase 4.
5. `info`: welfare breakdown, K_t, per-node results (reward components, state of charge, applied action, bounds), timestamp; key `privileged` reserved for phase 7.
6. Episodes: `episode_length` (steps, default the whole period) and `random_start` using `self.np_random`; the episode ends on its last valid step with `truncated=True`; the terminal observation comes from that step; `reset(seed)` also seeds the action space.
7. Registration through the entry point `gymnasium.envs` with ID `SmartCer-v1`; remove `SmartCer-v0`, `smartcer/cer_env.py`, `cer_utils.py`, `cer_models/`, `cer_reward/`, `cer_preprocessing/`.

*Acceptance:*
- `gymnasium.utils.env_checker.check_env` and the Stable-Baselines3 `check_env` pass.
- `gymnasium.make('SmartCer-v1', ...)` works.
- SAC from Stable-Baselines3 runs 2,000 steps on the synthetic scenario (test marked `slow`, with `importorskip`).
- Two runs with the same seed and actions give identical trajectories.
- A test checks the observed generation, load and A^upper with `observe_current_step` true and false.

### Phase 4 · Rewards, observation and critic channel

1. `smartcer/gym/rewards/` with the template method: type `welfare` (default, R_t = W_t − σK_t); types `profit` (per node, reported in `info`) and `energy` (0.6.2 legacy). No scaler inside the environment.
2. `smartcer/gym/observation/`: per-instance schema built from the run configuration. Preset `extended`: time features (hour and month cyclical, day of week one-hot, Italian holiday flag), prices, community aggregates of the nodes without a battery, per battery node l, g, state of charge, A^lower, A^upper, optional lookback; an empty section reserved for forecasts. Normaliser fitted on a declared training window (default: the run period), saved to and loaded from JSON.
3. `info['privileged']` key created (empty dictionary until phase 7).

*Acceptance:* one unit test per feature; over one year every observation lies in the observation space; a test proves that the normaliser fitted on a training window does not use data of the test window.

### Phase 5 · Controllers, tracking and metrics

1. `smartcer/gym/controllers/`: `NaiveController` (zero action) and `RBCController` ([M24] Algorithm 1 as in `docs/FORMULATION.md` section 13); `compute_baselines()` writes to the run output directory, never into `cer_data/`.
2. `smartcer/gym/tracking/`: results tracker with a community table and a per-node table, CSV export, SB3 callback.
3. `smartcer/gym/metrics/`: KPIs (welfare and its components, shared energy, sharing rate = shared / exported, coverage rate = shared / imported, imports, exports, battery throughput, equivalent full cycles, penalty statistics) and the Fit Score; `KPI.md` rewritten with formulas and units. Remove `cer_metrics/` and `storage_controllers/`.
4. Regenerate the Leather baselines in a local output directory (not committed).

*Acceptance:* the RBC executes the intended energy in hand cases; Fit Score = 100 on identical series and equal to a hand calculation on a small example; KPIs of the synthetic scenario match hand calculations.

### Phase 6 · Performance

1. `benchmarks/bench_step.py`: one year of Leather with random actions, reporting time per step and per episode.
2. No pandas access inside `step()`; the observation is computed once per step.

*Acceptance:* at least 5 times faster than 0.6.2 (43 s per year on Leather); numbers recorded in `PROGRESS.md` and `CHANGELOG.md`.

### Phase 7 · Hooks for forecasts and the oracle

1. `smartcer/forecasts/`: protocol `ForecastProvider` with `reset(rng: np.random.Generator)` and `get(t: int, horizon: int) -> dict[str, np.ndarray]` keyed `'<node_id>/<quantity>'`; implementation `PerfectForecast`. The forecast section of the observation uses `obs_layout` keys `forecast/<node_id>/<quantity>` and the same normalisation as the current quantity. Episode windows end H steps before the end of the data. The generator passed to `reset` is `self.np_random`.
2. `smartcer/oracle/`: `OracleProvider` that reads MILP outputs in the CSV format `datetime, node_id, p_cha_kw, p_dis_kw, soc_kwh` (plus an optional daily objective file) and returns a\* = p_cha / P^cha − p_dis / P^dis and the oracle state of charge per step. The format may change after the DIISM answers: keep the parsing in one adapter function.
3. `info['privileged']` holds `oracle_action`, `oracle_soc` and, when forecasts are enabled, `true_future`.
4. `ReplayController` that plays the oracle actions.
5. Replay test, skipped when the oracle files are absent: playing a\* over the whole period gives K_t = 0 at every step (within 1e-9) and a cumulative welfare within a relative tolerance (default 1e-3, configurable) of the MILP objective; a per-day comparison is written to the output directory.

*Acceptance:* with `PerfectForecast` and horizon H the observation contains the true future values in the documented positions; the replay test passes on a synthetic oracle trajectory generated in the test (and on the DIISM files when available).

### Phase 8 · Documentation and release

1. README: what SmartCer is, installation from the tag, minimal example with `gymnasium.make('SmartCer-v1', ...)`, a short SAC example, link to `docs/FORMULATION.md` and `docs/CONFIG.md`, citation; migration table from 0.6.2 to 1.0.0 (API, configuration keys, behaviour changes such as S4, S6, S7).
2. Final update of `CLAUDE.md` and `CHANGELOG.md`.
3. `/bump-version major` to 1.0.0 after confirmation; `/release` after confirmation; check that `pip install git+https://github.com/unisi-lab305/SmartCer@v1.0.0` works in a clean environment.
4. Optional, only if Leonardo confirms: project skills in `.claude/skills/`: `new-scenario` (create `cer_data/<name>/config.json` in the 1.0 format, check columns, units and frequency of the CSV files, run one episode) and `milp-check` (run the replay of phase 7 and report the welfare gap per day).

---

## 8. Pending inputs from the DIISM

Questions on the MILP are pending with the DIISM (the MILP authors). Until the answers arrive, use the defaults of `docs/FORMULATION.md` section 15 and keep each item configurable: charge rule, incentive value and model, storage usage cost, initial state of charge, oracle file format. Do not block any phase on these answers.

---

## 9. End-of-phase report

Reply with this structure and copy it into `docs/refactor/PROGRESS.md`:

```markdown
## Phase N · <title> – <date>
- Branch and commits:
- Done:
- Acceptance checks: <command> → <result, with numbers>
- Deviations from the brief and why:
- Open questions for Leonardo:
- Ready for phase N+1: yes / no (why)
```
