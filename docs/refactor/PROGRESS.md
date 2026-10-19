# SmartCer 1.0 refactoring – progress

Progress log of the refactoring specified in [REFACTOR_v1.md](REFACTOR_v1.md). One section per
phase, in the format of section 9 of the brief, newest last.

## Phase 0 · Project infrastructure and safety net – 2026-10-19

- Branch and commits: `refactor/v1-phase-0-infrastructure` (from `main` at `dd1c26c`), pushed to
  `origin` for CI; no pull request.
  - `c6d1c96` build: require Python 3.12, package all subpackages and add version attribute
  - `15b0c82` ci: add Makefile, pytest config, pre-commit hooks and GitHub workflows
  - `ba19422` test: add synthetic scenario, characterisation and known-defect tests of 0.6.2
  - `ba9c565` docs: add CLAUDE.md, CHANGELOG and Claude Code commands
  - `6d2a949` ci(release): drop separator lines from the release notes
  - `c99f0c7` docs(refactor): add phase 0 progress report (this file), plus a follow-up commit
    completing this list
  - local annotated tag `v0.6.2` on `dd1c26c` (not pushed)
- Done:
  1. Tag `v0.6.2` created locally on the current `main`.
  2. `pyproject.toml`: `requires-python >= 3.12`, `packages.find include = ['smartcer*']` (fixes S1),
     runtime dependencies `gymnasium`, `numpy`, `pandas`, `holidays` (+ `tqdm`, see deviations),
     extras `dev`, `rl`, `all`, Ruff settings as in storage-node-environment; version `0.6.2`;
     `__version__ = '0.6.2'` in `smartcer/__init__.py`.
  3. `Makefile` (`test`, `test-unit`, `test-integration`, `test-coverage`, `version`, `clean`,
     `help` from `## ` comments); `pytest.ini` (strict markers `unit`, `integration`, `slow`,
     warnings as errors except deprecations); `.pre-commit-config.yaml` as in
     storage-node-environment.
  4. `.github/workflows/ci.yml` (Python 3.12, `pip install -e .[all]`, unit, integration, coverage,
     wheel installed in a clean virtual environment and imported outside the repository) and
     `release.yml` (tag `vX.Y.Z`: version consistency, CHANGELOG entry, tests, build,
     `twine check`, GitHub Release with the CHANGELOG section and the built files; PyPI step
     commented out, marked "enable after 11/11/2026 (D11)").
  5. `CHANGELOG.md` (Keep a Changelog): `[Unreleased]` with the phase 0 entries and
     `[0.6.2] - 2025-03-17` summarising the git history.
  6. `CLAUDE.md` with the section structure of storage-node-environment, rewritten for the current
     code, plus a testing section, the refactoring protocol, the domain glossary and the rule that
     `docs/FORMULATION.md` is authoritative.
  7. `.claude/commands/`: `add-and-commit` and `pr` copied (no project-specific text);
     `bump-version` on `smartcer/__init__.py`; `release` without PyPI wording (tag → GitHub
     Release); `write-test` rewritten for the `test/unit/` and `test/integration/` layout and the
     Makefile `## ` help mechanism.
  8. `docs/refactor/PROGRESS.md` created; `docs/FORMULATION.md` present and unchanged.
  9. `.idea/`: already removed from version control and ignored by Leonardo's commits `99fbee0`
     and `dd1c26c` before the phase started; nothing left to do.
  10. `test/conftest.py` and the synthetic scenario `test/fixtures/mini_cer/` (0.6.2 format, 48
      hourly rows from 2022-01-03, round numbers): `pv_bat` producer with battery, `house_1`
      consumer (reads `house.csv`, exercising the `_N` rule), `factory_bat` prosumer with battery,
      `shop` prosumer without battery. Full description in the `conftest.py` docstring.
  11. Characterisation tests (pass on 0.6.2): `get_tip_coefficient` on 5 hand cases (PNRR 0, 30,
      40; zonal price below, equal to and above 0.180; threshold binding and not binding) plus the
      PNRR > 40 rejection; zero-action steps at t = 0, 10, 16, 34 with hand-computed export and
      import per node, shared energy and actual TIP; shared = min of the totals and actual TIP =
      shared / exported × Σ coefficient × exported over the whole idle run; per-node energy
      balance over a random 48 h run (seed 42).
  12. `scripts/make_reference_v062.py` run on Leather 2022 with zero actions; outputs in
      `test/fixtures/local/` (gitignored): `leather_2022_community.csv` (8760 hourly rows, full
      precision) and `leather_2022_kpis.json` (metadata, annual sums, 0.6.2 KPIs unrounded).
      `test/integration/v062/test_reference_v062.py` checks their consistency and skips when the
      folder is missing.
  13. 16 tests `xfail(strict=True, raises=<expected exception>)` for S2-S10:
      S2 (1), S3 (7: 6-item step, list reward, `truncated` None, observation outside its space,
      `flatdim`, `check_env`, `gym.make` + reset), S4 (2: below S^min, negative with
      `soc_min = 0`), S5 (1: 15-minute data), S6 (1), S7 (1), S8 (1), S9 (1), S10 (1). Each fails on
      0.6.2 for the documented reason (checked with `--runxfail`). S1 is guarded by the CI wheel
      step.
- Acceptance checks:
  - `pip install .` in a new venv (Python 3.12.13) → OK; from outside the repository
    `import smartcer, smartcer.cer_env, smartcer.storage_controllers.rbc_controller` → OK,
    `smartcer.__version__ == '0.6.2'`, `SmartCer-v0` registered. Wheel
    `smartcer-0.6.2-py3-none-any.whl`: all 16 modules of the 6 subpackages, 0 `cer_data` entries.
  - `make version` → `Version OK: 0.6.2`; with `__init__.py` set to 0.6.3 → `ERROR: version
    mismatch`, exit 2.
  - `make test` → 15 passed, 16 xfailed in 3.9 s (with the local reference); without
    `test/fixtures/local/` → 12 passed, 3 skipped, 16 xfailed. `make test-unit` → 6 passed,
    5 xfailed; `make test-integration` → 9 passed, 11 xfailed; `make test-coverage` → 74 % of 918
    statements.
  - `python scripts/make_reference_v062.py` → 8760 steps in 64 s. Leather 2022, idle batteries:
    imported 1,508,025.313 kWh, exported 834,147.422 kWh, shared 582,962.759 kWh, potential TIP
    41,594.044 €, actual TIP 28,345.971 €, expense 428,738.533 €, revenue 261,919.385 €.
  - CI on the branch → green, run 39215079533 on `ba9c565` (1 min 54 s): unit 6 passed,
    5 xfailed; integration 6 passed, 3 skipped (no local reference in CI), 11 xfailed; coverage
    74 %; wheel installed in `/tmp/wheel-env` and imported → `Installed smartcer 0.6.2`. GitHub
    annotations only warn that `actions/checkout@v4`, `setup-python@v5` and `upload-artifact@v4`
    run on Node.js 24 instead of 20, and that `ubuntu-latest` is moving to Ubuntu 26.
  - `/add-and-commit` on the throwaway branch `tmp/phase-0-commands-check` → CHANGELOG entry under
    `[Unreleased]`, files staged by name, one-line commit `style(test): ...` (`e728f6e`).
    `/bump-version patch` → 0.6.3 in both files, CHANGELOG rotated to `## [0.6.3] - 2026-10-19`,
    `make version` OK, commit `chore: bump version to 0.6.3` (`ca1296d`) with exactly the three
    files. The release workflow's CHANGELOG check and notes extraction were also run on the rotated
    file. Branch deleted, never pushed.
- Deviations from the brief and why:
  - `tqdm` stays a runtime dependency: `smartcer/cer_env.py` imports it at module level, so
    without it `from smartcer.cer_env import SmartCer` fails after `pip install .` (and the S1
    import check could not run). Removed in phase 3 with that module.
  - CI also runs on push to `refactor/**`: with triggers only on `main` and pull requests, "CI is
    green on the branch" could not be checked without opening a pull request.
  - `pytest.ini` ignores the S14 `SyntaxWarning: invalid escape sequence`: with warnings as errors
    the 0.6.2 `node_models.py` would not import on a fresh interpreter, and the old modules must
    stay untouched. The filter goes with that module in phase 3.
  - `pytest.ini` uses `--import-mode=importlib` and the test folders have no `__init__.py`, so that
    `test/unit/v062/test_known_defects.py` and `test/integration/v062/test_known_defects.py` (and
    later same-named files) do not clash. `write-test` documents it.
  - The reference JSON stores annual sums and the 0.6.2 KPIs at full precision
    (`compute_kpis(rounding=15)`) instead of the default 2 decimals, because phase 2 compares
    them within 1e-6.
  - The `v0.6.2` tag points to `dd1c26c`, the current `main` as the brief says; its package code is
    identical to `6ba645e` (17/03/2025), the later commits only add the two docs and remove
    `.idea/`.
- Open questions for Leonardo:
  1. Push the `v0.6.2` tag to `origin`? (Pushing it does not trigger the release workflow: the
     tagged commit `dd1c26c` contains no workflow files.)
  2. Local `main` is one commit ahead of `origin/main` (`dd1c26c remove .idea`); the phase 0 branch
     contains it, so a pull request will include it. Push `main` first, or let the PR carry it?
  3. Is keeping `tqdm` until phase 3 acceptable?
  4. `README.md` still says Python ≥ 3.10 and shows the `cost` reward: leave it for phase 8 as
     planned, or fix the Python requirement now?
- Ready for phase 1: yes, once CI is confirmed green (see above); the open questions do not block
  it.
- Answers (2026-10-19): 1) push the `v0.6.2` tag – done; 2) push `main` first – done
  (`99fbee0..dd1c26c`); 3) `tqdm` until phase 3 – ok; 4) README left to phase 8; 5) actions
  bumped in a separate commit – `aef0d49` ci: bump checkout, setup-python and upload-artifact to
  v7 (Node.js 24), on the phase 0 branch, CI run 39216505716 green without the Node.js 20 warning.

## Phase 1 · Battery – 2026-10-19

- Branch and commits: `refactor/v1-phase-1-battery`, created from the tip of
  `refactor/v1-phase-0-infrastructure` (`aef0d49`, see deviations), pushed to `origin` for CI; no
  pull request.
  - `8302ddc` feat(core): add battery model with dynamic action bounds and legacy config mapping
  - `33e7a9a` test(core): add battery unit, configuration, invariant and layer-rule tests
  - `45a43a5` docs(refactor): add phase 1 progress report (this section)
  - after the merge of phase 0 (pull request #2, merge commit `251db89`) the branch was rebased on
    `main`: the three commits above became `f4d20c2`, `fb924e2`, `2b43d58`, followed by
    `52e18cd` fix(core): define battery self-discharge per hour (answer 2 below) and the commit
    that updates this report
- Done:
  1. `smartcer/core/storage/base.py`: abstract `StorageSystem` with `get_bounds`, `step`, `reset`.
  2. `smartcer/core/storage/battery.py`: `Battery` as in `docs/FORMULATION.md` sections 4-7 and the
     storage term of section 10, ported from storage-node-environment and adapted (node-side
     discharge limit, bounds with η and S^min, charge cap passed by the node, action-based API,
     `usage_cost`, exceptions instead of assertions, no rounding of the state). Final API:
     - `Battery(capacity, power_charge_max, power_discharge_max, efficiency_charge,
       efficiency_discharge, soc_min=0.0, soc_initial=None, alpha=0.0, usage_cost=0.0,
       charge_limit='production')`, as suggested in the brief; `soc_min`, `soc_initial` are
       fractions, `soc_initial` defaults to `soc_min`; state `soc_kwh`, plus `soc_min_kwh`,
       `soc_initial_kwh`.
     - `get_bounds(delta_t, charge_cap_kwh) -> (a_upper, a_lower)`, both in [0, 1], from the state
       after self-discharge; it does not change the state.
     - `step(action, delta_t, charge_cap_kwh) -> BatteryStep`, a frozen dataclass with
       `action_requested`, `action_applied`, `a_upper`, `a_lower`, `p_cha_kw`, `p_dis_kw`,
       `charge_kwh` (node → battery), `discharge_kwh` (battery → node), `throughput_kwh` (cell
       side), `storage_cost`, `penalty`, `soc_kwh`.
     - `reset()`; `Battery.from_config(config)` for the 1.0 keys of FORMULATION section 3 and the
       0.6.2 keys of brief section 6, with a `DeprecationWarning` naming each mapped key.
     - `smartcer.core` and `smartcer.core.storage` export `Battery`, `BatteryStep`,
       `StorageSystem`.
  3. Regression tests on the new API for S4 (two cases, same numbers as the 0.6.2 xfail tests) and
     for S5 at battery level (Δt = 0.25 h); S15: `beta` is ignored, the capacity never changes.
     The 0.6.2 xfail tests stay until their modules are removed in phase 3.
  4. `test/unit/core/test_layer_rules.py`: AST check that `smartcer/core/` has no `print`,
     `sys.exit`, `assert` and no import of `gymnasium` or `smartcer.gym`.
  5. `CLAUDE.md` (status, package structure, layer rule, Battery API, testing rule, units) and
     `CHANGELOG.md` (`Added`, `Deprecated`) updated in the feature commit.
- Acceptance checks:
  - 100,000 random actions (`test_battery_invariants.py`, seed 20261019): 1,000 batteries with
    random capacity, power limits, efficiencies in [0.7, 1], `soc_min` 0 or in [0, 0.5], random
    initial state, α 0 or in [0, 0.05], Δt ∈ {0.25, 1.0} h, charge caps None / 0 / random; actions
    uniform, exactly ±1 or 0, and exactly A^upper or -A^lower. State of charge always in
    [S^min, S^max] within 1e-9, applied action within the bounds, penalty 0 for every action inside
    the bounds and equal to section 7 otherwise (within 1e-12), charge ≤ cap, Δs = η^cha · charge −
    discharge / η^dis (within 1e-9) → passed, 5.9 s locally, under 1 s in CI.
  - Hand cases (10 kWh, 5 kW, η = 0.95, `soc_min = 0`, Δt = 1 h, no cap) → from 5 kWh full charge:
    SoC 9.75 kWh (+4.75), 5 kWh from the node; from 8 kWh full discharge: SoC 2.7368 kWh
    (−5.2632), 5 kWh to the node; from 5 kWh full discharge: a = −0.95, 4.75 kWh to the node,
    SoC 0 (within 1e-12), penalty 0.05 → passed.
  - Actions equal to A^upper and −A^lower (Δt 0.25 and 1 h; no cap, binding cap 0.3 kWh,
    non-binding cap 0.6 kWh) reach S^max, S^min or the cap-limited value within 1e-9, never beyond
    → passed.
  - Legacy mapping (Leather storage section; `beta` = 0.5 has no effect; conflicts, unknown,
    missing and invalid keys raise `ValueError`), `DeprecationWarning` checked with `pytest.warns`
    → passed.
  - No `sys.exit` and no `print` in `smartcer/core/` → AST test passed; `grep` finds none.
  - `make test` → 63 passed, 16 xfailed in 10.1 s (48 new tests: 34 `test_battery.py`,
    8 `test_battery_config.py`, 1 invariant test, 5 layer-rule tests). `make test-coverage` → 77 %
    overall, `smartcer/core/storage/battery.py` 98 % (129 statements, 2 missed).
  - CI on the branch → green, run 39217718289 on `33e7a9a`: unit 54 passed, 5 xfailed; integration
    6 passed, 3 skipped, 11 xfailed; wheel import OK; no Node.js deprecation warning.
  - Speed (informative, for phase 6): `step()` ≈ 14 µs, `get_bounds()` ≈ 3.7 µs on the development
    laptop; Leather (4 batteries, 8760 h) ≈ 0.5 s per year in the battery model.
- Deviations from the brief and why:
  - The branch starts from the phase 0 branch, not from `main`: phase 0 is approved but not merged,
    and a branch from `main` would lack the tests, the Makefile and CI. It needs no change once
    phase 0 is merged (the branch then contains `main`).
  - `BatteryStep` also carries `a_upper`, `a_lower` and `storage_cost` (needed by the node record of
    phase 2 and by the welfare of section 10); the node-side energies are named `charge_kwh` and
    `discharge_kwh`.
  - Actions outside [−1, 1] or not finite raise `ValueError` instead of being clipped: section 7
    defines clipping and penalty for a ∈ [−1, 1], and the phase 3 action space is `Box(−1, 1)`.
  - A state of charge beyond its limits by at most 1e-9 kWh (floating-point rounding at the limits)
    is clamped to the limit; a larger excess raises `RuntimeError`. storage-node-environment rounded
    the state to 6 decimals and asserted with a 1e-6 tolerance.
  - `from_config` rejects unknown keys and a 0.6.2 key combined with its 1.0 target (e.g.
    `rated_power` with `power_charge_max`); the brief does not say how to treat them.
  - The layer-rule test also forbids `assert` in `smartcer/core/` (CLAUDE.md: no assertions for
    validation).
- Open questions for Leonardo:
  1. Phase 0 pull request: shall I open it (`/pr`) so that `main` gets the infrastructure before
     phase 2? Phase 1 would then follow as its own pull request.
  2. α is "self-discharge per step" in FORMULATION section 3, so the same scenario with Δt = 0.25 h
     loses four times as much per hour as with Δt = 1 h. Keep it per step (as written, default 0),
     or should the formulation define it per hour (α_step = 1 − (1 − α_h)^Δt)? I have not changed
     the formulation.
  3. Out-of-range actions raise `ValueError` (see deviations): fine, or clip them to [−1, 1]
     silently and penalise as section 7?
- Ready for phase 2: yes; the open questions do not block it (question 1 only decides where the
  phase 2 branch starts).
- Answers (2026-10-19): 1) phase 0 pull request opened and merged into `main` (#2, merge commit
  `251db89`); this branch rebased on `main` and opened as its own pull request; phase 2 starts
  from `main` after both merges. 2) α is now defined per hour, α_step = 1 − (1 − α)^Δt:
  `docs/FORMULATION.md` sections 3 and 5 and the `Battery` docstrings updated in `52e18cd`, with a
  test that four 0.25 h idle steps lose as much as one 1 h step. 3) `Battery.step` keeps raising
  on actions outside [−1, 1] or non-finite; in phase 3 the environment clips actions to the
  `Box(−1, 1)` space before the nodes, without penalty for that clip, and raises on non-finite
  actions.
- Phase 1 pull request #3 merged into `main` (merge commit `4c2e0be`), CI of the pull request
  green.

## Phase 2 · Nodes, community and loader – 2026-10-19

- Branch and commits: `refactor/v1-phase-2-nodes-community-loader`, created from `main` at
  `4c2e0be` (phases 0 and 1 merged), pushed to `origin` for CI; no pull request.
  - `8af6bf1` feat(core): add nodes, community shared energy, incentives and welfare
  - `6d3d792` feat(data): add scenario schema and loader for the 1.0 and 0.6.2 formats
  - `c0d0742` test: add node, community and loader tests and the 1.0-format synthetic scenario
  - `2516dcb` ci: import the core and data subpackages from the installed wheel
  - `9bef16a` docs(refactor): add phase 2 progress report (this section)
- Done:
  1. `smartcer/core/nodes/node.py`: one `Node` class for producer, consumer and prosumer, optional
     battery, per-step arrays (generation, load, sell and buy price, in kWh and €/kWh) validated
     once and read as Python floats in `step()`; `charge_cap(t)` by `charge_limit` (D6:
     `production` c = g, `surplus` c = max(0, g − l)); `get_bounds(t)`; `step(t, action)` →
     `NodeStep` with g, l, p^cha, p^dis, node-side energies, n, e^gri, i^gri, revenue, cost,
     throughput, storage cost, k, SoC, requested and applied action, bounds; `reset()` restores the
     battery only (no file is re-read, S13).
  2. `smartcer/core/community/`: pure functions `shared_energy`, `tip_coefficient` (vectorised,
     ValueError for PNRR outside [0, 40]), `uniform_incentive`, `tip_incentive`,
     `welfare_breakdown`; models `UniformIncentive` (constant or per-step π^inc) and `TipIncentive`
     (per-step, per-node coefficients, `potential()`); `Community` with `step(t, actions)` →
     `CommunityStep` (node records, `WelfareBreakdown`, K_t), the object the phase 3 environment
     will wrap.
  3. `smartcer/data/schema.py` and `loader.py`: `load_scenario(path, start, end)` → `Community` for
     the 1.0 format (section 6 of the brief) and the 0.6.2 format (DeprecationWarning listing unused
     keys; incentive `tip`; storage keys mapped by `Battery.from_config`; `_N` rule). Every CSV is
     read once; aliases `price_exp`/`price_imp`; `units: kW` multiplied by Δt. Errors name node,
     file, column and first offending CSV line with its timestamp: missing columns, missing or
     non-numeric values, negative energies, unsorted, duplicated or wrongly spaced timestamps,
     nodes with different timestamps (1.0), different row counts (both formats).
  4. 0.6.2 timestamps (answer of 2026-10-19 to the question raised during the phase): rows aligned
     by position as in 0.6.2, community timestamps of the first node, irregularities reported in
     one `TimestampWarning`. Needed by Leather: `cartotecnica`, `conceria_dolmen`,
     `toscana_fridge` (and `cer_azienda/azienda`) have labels shifted by +1 h for 5,209 rows (gap
     at 2022-03-27 23:00, duplicate 2022-10-31 00:00), while the other nodes are regular.
  5. `test/fixtures/mini_cer_v1/`: the synthetic scenario in the 1.0 format (explicit `csv` paths,
     1.0 column names and storage keys, `tip` model).
  6. `docs/CONFIG.md` (scenario reference; run arguments limited to the period until phase 3);
     `CLAUDE.md` and `CHANGELOG.md` updated; the CI wheel check also imports `smartcer.core` and
     `smartcer.data`.
  7. Regression tests: S5 (`test_s5_*` in `test/unit/core/nodes/test_node.py` and
     `test/unit/data/test_loader.py`: Δt = 0.25 h, `units: kW`), S8
     (`test_s8_consumer_battery_cannot_charge`, `test_consumer_with_battery_loads_and_steps`),
     S11 and S14 (layer-rule test over the new modules: no `sys.exit`, no `print`; raw-string
     regex for the `_N` rule).
- Acceptance checks:
  - Phase 0 characterisation tests re-pointed to the new functions, `tip` model
    (`test/integration/data/test_mini_cer_scenario.py`,
    `test/unit/core/community/test_incentives.py`): idle steps t = 0, 10, 16, 34 (flows per node,
    shared 0/3/0/3 kWh, TIP 0/0.2925/0/0.27 €), shared and TIP identities over 48 h, per-node
    energy balance over a random run, the 5 TIP coefficient hand cases → passed.
  - Leather 2022, idle batteries, against the 0.6.2 reference (`test_leather_reference.py`) →
    passed: all seven hourly community series (imported, exported, shared, expense, revenue,
    potential and actual TIP) within 1.2e-13 (tolerance 1e-6), annual sums identical (difference
    0.0), timestamps identical. Annual idle welfare −138,473.18 €.
  - Welfare of the synthetic scenario by hand, both models (10:00 of day 1): idle W = 1.5925 €
    (`tip`) and 1.6 € (`uniform`, 0.1 €/kWh); with actions (pv_bat −0.5 at S^min, factory_bat
    +0.5) W = 1.125 € and 1.1 €, K_t = 0.25 → passed.
  - 1.0 format vs 0.6.2 format: identical `CommunityStep` records at every step of a random 48 h
    run → passed.
  - Consumer with a battery loads and steps; a full charge is clipped to 0 with k = 1, discharge
    works → passed.
  - `make test` → 139 passed, 16 xfailed in 19 s (75 new tests); `make test-coverage` → 82 %
    overall, new modules 90-100 %.
  - CI on the branch → green, run 39222440322 on `c0d0742`: unit 119 passed, 5 xfailed;
    integration 16 passed, 4 skipped (no local data in CI), 11 xfailed; wheel import OK.
  - Informative: every local scenario (Leather, cer_azienda, cer_coffee, cer_retail, cer_utente)
    loads; Leather loads in 0.5 s and an idle year takes 3.5 s (405 µs per step; 0.6.2: 43 s).
- Deviations from the brief and why:
  - One `Node` class with a `node_type` in `smartcer/core/nodes/node.py` (the brief allows "one
    class with a type"; the file name follows the content).
  - `smartcer/core/community/community.py` (`Community`, `CommunityStep`) is not in the brief's
    file list: it steps the nodes and combines shared energy, incentive, welfare and K_t, so that
    phase 2 can be tested on full runs and phase 3 only wraps it.
  - CSV reading and its time-step validation are in `smartcer/data/loader.py`, not in the node
    (storage-node-environment reads the CSV in the node): `smartcer/core/` stays free of pandas.
  - Lenient timestamps for the 0.6.2 format, as decided during the phase (see item 4).
  - Additional validation: negative `production` or `consumption` and non-numeric values are
    errors; unknown keys are errors in the 1.0 format and are listed in the DeprecationWarning in
    the 0.6.2 format.
  - The mapping of the 0.6.2 run keys (`simulation.*`, `reward.*`) is left to phase 3, where the run
    configuration (environment arguments) is created; the loader takes `start` and `end` only.
- Open questions for Leonardo:
  1. Prices a node type does not require are set to 0: the energy a consumer's battery exports
     (discharge beyond its load) earns nothing, and a producer's buy price is 0 (it never imports
     under the charge rule). Accept, or read `sell_price` for consumers when the column exists?
  2. `tip` model: the shared fraction E^sh / Σ e^gri uses the export of all nodes, and nodes without
     a plant have c = 0 (as 0.6.2, where only producers and prosumers exported). With the other
     reading (sum over plant nodes only) a consumer battery's export would raise the incentive of
     the plants. Keep the current reading? FORMULATION section 9 could state it explicitly.
  3. The shifted Leather timestamps (item 4): do you want to correct the CSVs (I never modify
     `cer_data/`)? A 1.0 conversion of Leather needs regular timestamps.
- Ready for phase 3: yes; the open questions do not block it.
- Answers (2026-10-19): 1) a consumer's `sell_price` is read when the column exists, 0 otherwise
  (loader, `docs/CONFIG.md`, test). 2) the 0.6.2 reading of the `tip` model is kept and stated in
  `docs/FORMULATION.md` section 9: both sums run over all nodes, nodes without a plant have c = 0,
  so battery exports of nodes without a plant lower the average coefficient (test
  `test_tip_export_without_plant_lowers_the_average_coefficient`). 3) the Leather CSVs are not
  modified. Finding on the shifted rows: in `cartotecnica`, `conceria_dolmen` and `toscana_fridge`
  the 5,209 rows whose labels are 1 h later than those of the regular nodes form one contiguous
  block, regular labels 2022-03-27 23:00 → 2022-10-30 23:00 (file labels 2022-03-28 00:00 →
  2022-10-31 00:00), every row shifted by exactly +1 h. The 2022 Italian daylight-saving period
  (Europe/Rome) runs from 2022-03-27 02:00 CET (→ 03:00 CEST) to 2022-10-30 03:00 CEST (→ 02:00
  CET), 5,208 hours. The block covers the same two dates and has the same length within one hour,
  but its edges are not at the clock changes: it starts 21 h after the spring change and ends 20 h
  after the autumn change, at 23:00 of both days. It looks like a summer-time correction applied at
  day granularity rather than at the actual change times; whether the values or the labels are the
  local-time ones cannot be told from the timestamps alone. The same block is in
  `cer_azienda/azienda.csv`.
- Phase 2 pull request merged into `main` on Leonardo's request (see the phase 3 section).

