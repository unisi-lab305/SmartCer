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

## Phase 3 · Gymnasium environment – 2026-10-19

- Branch and commits: phase 2 follow-ups (`b499e92` feat(data): read the optional sell price of
  consumers, `3084b3e` docs(formulation): state the 0.6.2 reading of the tip incentive, `48b4498`
  docs(refactor): record phase 2 answers and the Leather timestamp finding) were merged with phase
  2 (pull request #4, merge commit `d878969`). `refactor/v1-phase-3-gym-env`, created from `main`
  at `d878969`, pushed to `origin`; pull request opened, not merged (Leonardo reviews and merges
  phases 3 and 7).
  - `39efa84` feat(gym): add the SmartCer-v1 Gymnasium environment with the meco observation
  - `5c3aa69` test(gym): add compliance, episode, timing, action and SAC tests of SmartCer-v1
  - `44aaf48` refactor!: remove the 0.6.2 environment, SmartCer-v0 and the 0.6.2 models
  - `7decf34` docs(refactor): add phase 3 progress report (this section)
- Done:
  1. `smartcer/gym/cer_env.py`, `SmartCerEnv`: `reset(seed=None, options=None) -> (obs, info)`,
     `step(action) -> (obs, reward, terminated, truncated, info)` with a float reward
     R_t = W_t − σK_t (σ = `penalty_coefficient`, default 10). It wraps the phase 2 `Community`.
  2. Action space `Box(-1, 1, (N,), float32)` in `storage_node_ids` order (public). As answered on
     2026-10-19: actions are clipped to [−1, 1] before the nodes without penalty, non-finite
     actions or a wrong shape raise `ValueError`; the nodes then clip to the dynamic bounds with
     the section 7 penalty.
  3. `observe_current_step` (default True) as in FORMULATION section 12; with False the observed g
     and l are those of step t − 1, 0 at the first row of the data (answer of 2026-10-19), and the
     observed A^upper is computed without the charge cap.
  4. Observation: flat `Box` with `obs_layout` (feature name → slice), preset `meco`
     (`smartcer/gym/observation/`): `time/month`, `time/hour` (sin, cos), `time/day_of_week`
     (one-hot); prices not normalised, as `price/sell`, `price/buy` when every seller and every
     buyer share one series, otherwise `<node>/sell_price`, `<node>/buy_price` in each battery
     node's block (synthetic scenario and Leather: per node); per battery node `<node>/generation`
     and `<node>/load` min-max over the period, `<node>/soc` scaled to [S^min, S^max],
     `<node>/a_lower`, `<node>/a_upper`. Step-independent features are precomputed.
  5. `info`: `t`, `timestamp`, reward components (`welfare`, `penalty` = σK, `total`), welfare
     breakdown, K_t, the clipped action and, per node, the `NodeStep` fields (reward components
     revenue, cost, storage cost and k, state of charge, requested and applied action, bounds).
     `info['privileged']` is left to phases 4 and 7.
  6. Episodes: `episode_length` (default the whole period), `random_start` with `self.np_random`;
     every step is controlled and the episode ends on its last valid step with `truncated=True`
     (`terminated` is always False); the terminal observation is built for that step with the new
     state; `reset(seed)` also seeds the action space; `seed` argument for the first reset.
  7. Registration of `SmartCer-v1` on `import smartcer` (idempotent `register_envs()`) and through
     the `gymnasium.envs` entry point (`__root__`); `SmartCer-v0` removed. Removed:
     `smartcer/cer_env.py`, `cer_utils.py`, `cer_models/`, `cer_reward/`, `cer_preprocessing/`,
     `scripts/make_reference_v062.py` (its outputs in `test/fixtures/local/` stay), the 0.6.2
     characterisation and known-defect tests (replaced by regression tests on the new API) and
     `tqdm`. `smartcer/storage_controllers/abc_controller.py`: the `cer_models` import moved under
     `TYPE_CHECKING` (answer of 2026-10-19), so that module and the S6 xfail test live until
     phase 5. The S14 warning filter is gone with `node_models.py`.
  8. `SmartCerEnv.from_v062_config(cer_path, cer_config)` maps the 0.6.2 run keys of brief section
     6 (period, seed, `reward.type` 'profit' or 'cost' with `single_agent` False → welfare,
     `reward.constraints.coefficient` → σ, 0 if the penalty is off) with a DeprecationWarning
     listing the unused keys; 'energy' and `single_agent` True raise `ValueError` until the
     rewards of phase 4.
  9. Regression tests replacing the 0.6.2 xfail tests: S2 (`test_s2_reward_type_cost_is_mapped_to_
     welfare`), S3 (`test_s3_gym_make_reset_and_step`, `test_s3_spaces_are_flattenable`, both
     `check_env`), S7 (`test_s7_causal_observation_hides_current_generation`), S9
     (`test_s9_identical_episodes_give_identical_rewards`), S10
     (`test_s10_terminal_observation_comes_from_last_step`), S17 (`test_reset_seeds_the_action_
     space`); S4, S5, S8 since phases 1-2; S6 stays an xfail until phase 5.
  10. `docs/CONFIG.md` section 6 (environment arguments and 0.6.2 run keys), `CLAUDE.md`,
      `CHANGELOG.md`; the CI wheel check imports `smartcer.core`, `smartcer.data`, `smartcer.gym`
      and `smartcer.storage_controllers`.
- Acceptance checks:
  - `gymnasium.utils.env_checker.check_env` (on `gym.make(...).unwrapped`, fixed start and random
    start with `observe_current_step=False`) and the Stable-Baselines3 `check_env(warn=True)` →
    passed with warnings as errors.
  - `gymnasium.make('SmartCer-v1', scenario=...)` with the passive checker, reset and step → passed.
  - SAC (Stable-Baselines3 2.8, CPU) 2,000 steps on the synthetic scenario, test marked `slow` with
    `importorskip` → passed, 67 s locally, in CI within the 2 min 39 s job.
  - Two runs with the same seed and sampled actions (random start, 12-step episodes) give identical
    trajectories; a different seed does not → passed.
  - `observe_current_step` True and False: observed generation, load and A^upper of pv_bat at
    t = 0, 8, 9 (True: g_t, A^upper with cap 0 then 1; False: g_{t−1} with 0 at the first row,
    A^upper 1 without cap, applied bound still with the cap, k = 1 for a night charge) → passed.
  - Reward at 10:00 of day 1 with actions (−0.5, 0.5): W = 1.125 €, K = 0.25, R = −1.375 (phase 2
    hand values) → passed.
  - `make test` → 168 passed, 1 xfailed (S6) in 106 s (SAC 67 s); `make test-coverage` → 92 %,
    `smartcer/gym/` 96-100 %. CI → green, run 39228214100 on `44aaf48`: unit 137 passed,
    1 xfailed; integration 27 passed, 4 skipped (local data); wheel import OK.
  - Clean virtual environment (Python 3.12.13, Gymnasium 1.4.0): wheel installed without `tqdm`,
    `import smartcer` registers `SmartCer-v1`, `SmartCer-v0` absent.
  - Informative: Leather, one year of random actions through the environment, 5.1 s (0.6.2: 43 s);
    observation of 39 features.
- Deviations from the brief and why:
  - Gymnasium ≥ 1.0 (1.4.0 here) no longer loads `gymnasium.envs` entry points, so the entry point
    alone would not register the environment: it is declared (named `__root__`, for Gymnasium
    < 1.0) and `import smartcer` registers `SmartCer-v1`; `gym.make('smartcer:SmartCer-v1')` imports
    the package itself.
  - Observation bounds of constant features (e.g. a producer's buy price of 0, a fixed tariff) are
    widened to [v, v + 1], because Gymnasium's checker rejects low == high.
  - Every battery node's price block holds both prices, also the one its type does not use (0 for
    a producer's buy price), so that every battery block has the same layout.
  - The min-max scaling of g and l uses the environment's whole period; fitting on a declared
    training window is phase 4.
  - `info['privileged']` is not created yet: phase 4 creates it (empty) and phase 7 fills it.
  - The 0.6.2 run-key mapping (brief section 6) is implemented here, with the environment
    arguments it maps to; 0.6.2 rewards other than the collective profit wait for phase 4.
  - `smartcer/storage_controllers/abc_controller.py` changed (import guard), as answered.
  - `test/integration/v062/test_reference_v062.py` (consistency of the local reference files) was
    kept and moved to `test/integration/data/test_reference_files.py`.
- Open questions for Leonardo:
  1. Per-node price blocks: keep both prices for every battery node (uniform layout, one constant
     feature for producers), or drop the price a node type does not use?
  2. When the rewards of phase 4 exist, should `from_v062_config` map the 0.6.2 types 'energy' and
     `single_agent` True to them, or keep raising (the brief maps only the collective profit)?
  3. `README.md` now describes the removed 0.6.2 API (`SmartCer-v0`, `smartcer.cer_env`); it is
     rewritten in phase 8 as agreed — tell me if it should get a short notice earlier.
- Ready for phase 4: yes, after the review and merge of the phase 3 pull request.
- Pull request #5 reviewed and merged by Leonardo (merge commit `cef6f9f`).
- Answers (2026-10-19): 1) both prices stay in every battery node's block. 2) in phase 4 the 0.6.2
  `energy` type maps to the energy reward and `single_agent: true` to per-node rewards in `info`,
  with the scalar reward equal to their sum and a DeprecationWarning (done, see phase 4). 3) a short
  notice in `README.md` now (done in phase 4); the full rewrite stays in phase 8.

## Phase 4 · Rewards, observation and critic channel – 2026-10-19

- Branch and commits: `refactor/v1-phase-4-rewards-observation`, created from `main` at `cef6f9f`
  (phase 3 merged), pushed to `origin`; pull request opened, not merged.
  - `5fc81b4` feat(gym): add reward types, the extended observation preset and a training-window
    normaliser
  - `5b1cc35` test(gym): add reward, normaliser, observation feature and one-year observation tests
  - `f3e822c` docs: add a 1.0 development notice to the README
  - docs(refactor): add phase 4 progress report (this section, also completing the phase 3 commit
    list)
- Done:
  1. `smartcer/gym/rewards/`, ported from storage-node-environment and adapted:
     - `base.py`: template method `RewardCalculator.compute(step) → RewardBreakdown(main, penalty,
       total, nodes)`, main term minus σ times the constraint penalty. Collective variant
       R_t = M_t − σK_t; per-node variant r_u = m_u − σk_u for every battery node, with
       R_t = Σ_u r_u. No `normalize` flag and no scaler.
     - `calculators.py`: `welfare` (M = W_t, default, collective only); `profit` (m_u = revenue −
       cost − storage cost; the collective variant adds I_t and equals `welfare`); `energy`
       (0.6.2 legacy, m_u = −(i^gri_u)², collective sum over all nodes).
     - `config.py`: `RewardConfig` and a read-only `REWARD_REGISTRY` (`MappingProxyType`); the
       factory `make_reward` reads the registry, with no if/elif chain.
     - Environment arguments `reward` and `reward_per_node`. `info['reward']` holds `type`, `main`,
       `penalty` (weighted), `total` and, per node, `nodes` (node → main, penalty, total).
  2. `docs/FORMULATION.md` section 11 states the two variants (collective with M_t summed over all
     nodes, per node over U_b with R_t = Σ r_u), as answered on 2026-10-19 (see open question 2).
  3. `from_v062_config` (answer 2 of phase 3): `profit`/`cost` with `single_agent: true` →
     `reward='profit', reward_per_node=True`; `energy` → `reward='energy'`, per node if
     `single_agent` is true. The DeprecationWarning says that the scalar reward is the sum of the
     per-node rewards in `info`.
  4. `smartcer/gym/observation/`: each environment builds its own schema from its arguments
     (preset, timing, `lookback`, normaliser), in sections `time`, `price`, `community`, `nodes`,
     `forecast` (`ObservationBuilder.sections`).
     - Preset `extended`: the time features of `meco`, plus `time/holiday` (Italian national
       holidays from the `holidays` package); the prices as in `meco`; `community/generation` and
       `community/load` (totals of the nodes without a battery); per battery node g, l, optional
       `lookback` L (`<node>/generation_lookback`, `<node>/load_lookback`, most recent first),
       SoC, A^lower, A^upper; an empty `forecast` section.
     - `normalizers.py`: `MinMaxNormalizer`, fitted by `fit_normalizer(community, window)` on the
       steps of the training window only. It scales and clips to [0, 1], and is saved to and
       loaded from JSON (format, type, scenario, window, bounds).
     - Environment arguments `normalizer_window` (default: the run period) and `normalizer` (a
       `MinMaxNormalizer` or a JSON path), plus the attribute `env.normalizer`. The `meco` preset
       uses the same normaliser, which closes the scaling deviation of phase 3.
  5. `info['privileged'] = {}` in the reset and step info.
  6. `README.md`: notice that 1.0 is under development and the 0.6.2 API removed, with the `v0.6.2`
     tag for the old version (answer 3 of phase 3).
  7. `docs/CONFIG.md` section 6 (new arguments, observation sections, normaliser workflow, info
     keys, 0.6.2 reward mapping), `CLAUDE.md` and `CHANGELOG.md`.
- Acceptance checks:
  - One unit test per feature (`test/unit/gym/test_observation.py`):
    - layout and sections of `extended`: contiguous, sections in vector order, empty `forecast`;
    - month, hour and day of week;
    - holiday flag: 6 January across midnight → 0, 0, 1, 1, and 2 June;
    - prices per node: 0.10 sell, 0 buy for a producer, 0.30 buy;
    - community aggregates: generation 0 → 0 and 5 → 1; constant load 8 kWh → 0;
    - per-node g and l;
    - lookback: [1, 1, 0] at t = 10, [1, 0, 0] with `observe_current_step=False`, zeros before
      the first row;
    - SoC 0.125, A^lower 0.405 and A^upper 1 after a 5 kWh charge;
    - reserved prefix `community`, and a normaliser without the community's series.

    All passed. Normaliser (`test_normalizers.py`): fit, transform with clipping, constant series,
    JSON round trip, validation → passed.
  - Every observation of a year lies in the observation space
    (`test/integration/gym/test_observation_window.py`):
    - synthetic year, 8,760 steps, random actions, normaliser fitted on January–March (summer
      generation above the training maximum): `meco` with timing True, and `extended` with
      lookback 4 and timing False → 0 observations outside the space;
    - Leather 2022 (local): `extended`, lookback 3, timing False, normaliser on January–June → 0
      outside the space.

    All passed.
  - Normaliser fitted on a training window ignores the test window: two 48 h scenarios, identical
    on day 1 and with generation and load × 100 on day 2, normaliser fitted on day 1.
    - identical bounds, equal to the day-1 minimum and maximum;
    - identical observations over the training window;
    - day-2 observations inside the space (clipped to 1).

    Passed. A saved normaliser reloaded by an evaluation environment on day 2 keeps the training
    bounds and window → passed.
  - Rewards:
    - hand values at 10:00 of day 1 (`test_rewards.py`):
      - `welfare` R = −1.375;
      - per-node `profit`: pv_bat 1.0 − 5.0 = −4.0, factory_bat 0.5 − 0 = 0.5, sum −3.5;
      - collective `energy` −5 − 2.5 = −7.5; per-node `energy` −5, 0;
      - the storage cost is subtracted from the profit;
      - the template works with a minimal subclass;
      - the registry is read-only;
    - over full episodes (`test_env_rewards.py`): R = W − σK at every step; collective `profit` =
      `welfare`; per-node totals sum to the reward and match each node's info.

    All passed. `check_env` (Gymnasium) also passes with `extended`, lookback 3 and per-node
    `profit`.
  - `make test` → 208 passed, 1 xfailed (S6) in 200 s (40 new tests). `make test-coverage` → 93 %
    overall, `smartcer/gym/` 96–100 % (`rewards/` 98–100 %, `normalizers.py` 98 %,
    `schema.py` 99 %).
  - CI run 39240436895 on `f3e822c` → green: unit 167 passed, 1 xfailed; integration 36 passed, 5
    skipped (local data); coverage 93 %; wheel import OK.
  - Informative: Leather `extended` with lookback 6 has 90 features (`meco`: 39); a normaliser
    saved from a January–20 March environment is reused by a 21 March–December one. One Leather
    year took 10.6–11.9 s on this branch against 10.4 s for `main` measured at the same time (the
    machine was slower than in phase 3, 5.1 s), so no regression.
- Deviations from the brief and why:
  - `profit` and `energy` have a collective and a per-node variant, selected with `reward_per_node`
    (the 0.6.2 `single_agent`). The brief lists `profit` as "per node"; FORMULATION section 11
    also defines its collective variant (with the incentive), which equals `welfare`. `welfare` has
    no per-node variant: its incentive is collective.
  - One penalty term (σK, or σk_u per node) instead of the weighted violation and storage-usage
    penalties of storage-node-environment: the storage cost is already in the welfare (section 10).
  - The keys of `info['reward']` changed from `welfare`, `penalty`, `total` (phase 3) to `type`,
    `main`, `penalty`, `total`, because the main term is no longer always the welfare.
  - Scaled values are clipped to [0, 1]: values beyond the training range (e.g. a sunnier test
    period) saturate instead of leaving the space.
  - Prices stay unnormalised in `extended`, as in `meco`. Their space bounds come from the
    environment's own period; they only define the `Box`, they are not a scaling.
  - Lookback only for g and l of the battery nodes and only in `extended` (`lookback > 0` with
    `meco` raises); `meco` stays [M24] Tab. II.
  - The holiday flag uses the national calendar of `holidays.country_holidays('IT')`: 13 days in
    2022, one of which, the first Sunday of November (Giornata dell'unità nazionale), is a Sunday.
    No regional or patron-saint days.
  - The community aggregates are added only when the nodes without a battery have that quantity,
    and `community` is a reserved prefix for node ids.
- Open questions for Leonardo:
  1. Leather cannot be split with `start`/`end` inside its shifted block (2022-03-27 23:00 →
     2022-10-30 23:00). The lenient 0.6.2 loader aligns rows by position but selects the period
     on each node's own labels, so a boundary inside the block gives 6,551 rows against 6,552 and a
     `ValueError`. `normalizer_window` is not affected, and splits outside the block work. Shall I
     select the period on the first node's timestamps and slice every node by the same positions
     (a small loader fix, consistent with the lenient reading)? In this phase or in phase 5?
  2. FORMULATION section 11 now defines the two variants. Per node, the penalty is σk_u (as in
     0.6.2) and R_t = Σ r_u, so the total penalty is σ|U_b|K_t rather than σK_t. Confirm the
     wording, or prefer the collective penalty σK_t split over the nodes?
  3. Clipping scaled g and l to [0, 1] outside the training range (current), or leave them
     unclipped with an unbounded space?
  4. For the Molise scenario, should the holiday flag also include a regional or local holiday
     (e.g. the patron saint of the municipality)? `holidays` supports subdivisions; it would be one
     more argument.
- Ready for phase 5: yes; the open questions do not block it. The phase 4 pull request is open and
  waits for your go-ahead to merge.

- Pull request #6 merged by Leonardo after review (merge commit `ff00575`).
- Answers (2026-10-19), all implemented in phase 5 (see below):
  1. Leather split: fix now, as the first separate commit of phase 5. Select the period on the
     first node's timestamps and cut every node at the same rows; raise a clear error if the node
     lengths still differ, never realign silently; add a regression test with a split inside the
     summer-time block.
  2. FORMULATION section 11 wording confirmed. Add one sentence: with N batteries the per-node
     penalty is about N times the collective one, so the two variants (and their training curves)
     are not numerically comparable.
  3. Keep clipping scaled values to [0, 1]. Add `info['obs_clipped']`, the number of observation
     values clipped at that step, and document it in CONFIG.md section 6.
  4. National holidays only; document the limit in CONFIG.md.

## Phase 5 · Controllers, tracking and metrics – 2026-10-19

- Branch and commits: `refactor/v1-phase-5-controllers-metrics`, created from `main` at `ff00575`
  (phase 4 merged), pushed to `origin`; pull request opened, not merged.
  - `3e067cd` fix(data): select the 0.6.2 period on the first node and cut every node at the same
    rows (answer 1)
  - `a43be7d` feat(gym): count clipped observation values in info and state the per-node penalty
    scale (answers 2-4)
  - `b3974ee` feat(gym): add baseline controllers, results tracking and KPIs
  - `f609046` refactor!: remove the 0.6.2 metrics and storage controllers
  - docs(refactor): add phase 5 progress report (this section)
- Done:
  1. Phase 4 answers:
     - Loader, 0.6.2 format (`smartcer/data/loader.py`): the period is selected on the first
       node's timestamps and every node is cut at the same row positions. Every file must have the
       same number of rows; otherwise `ValueError` naming the node, its rows and the first node's.
       The whole-year rows of each node are unchanged; `docs/CONFIG.md` section 4.
     - FORMULATION section 11: with N = |U_b| batteries the per-node penalty is N σ K_t, N times
       the collective one, and the main terms differ too, so collective and per-node rewards,
       including their training curves, are not numerically comparable.
     - `info['obs_clipped']` (reset and step info): the number of values of the returned
       observation that lie outside the normaliser's training range and were clipped to [0, 1].
       It is precomputed per step (`ObservationBuilder.clipped`,
       `MinMaxNormalizer.outside_range`). The zeros before the first row of the data are padding
       and are not counted.
     - Holiday flag: national holidays only, documented in CONFIG.md section 6 and in the schema
       docstring.
  2. `smartcer/gym/controllers/` (ported from storage-node-environment and adapted to continuous
     actions):
     - `Controller` (`base.py`) is bound to the unwrapped `SmartCerEnv`; `act(observation, info)`
       returns the actions for `env.current_step`.
     - `NaiveController` (`naive.py`) returns zero actions.
     - `RBCController` (`rbc.py`) follows FORMULATION section 13: d = g − l, then charge
       min(d, S^max − s) or discharge min(−d, s − S^min), divided by P Δt and limited to [−1, 1].
       The environment then clips the action to the dynamic bounds, with the usual penalty.
     - `baselines.py`:
       - `CONTROLLERS`, a read-only registry;
       - `run_episode(env, controller, seed)`, which returns a tracker;
       - `compute_baselines(env, output_dir, controllers=('naive', 'rbc'), seed)`. For each
         controller it writes `<output_dir>/<name>/` with `community.csv`, `nodes/<node>.csv`,
         `kpis.json` and `run_config.json`, plus a summary `<output_dir>/kpis.json`. It refuses
         the scenario folder and any `cer_data` folder (S18).
  3. `smartcer/gym/tracking/`:
     - `ResultsTracker` builds the community table (reward, welfare and its components, K_t,
       `obs_clipped`) and one table per node (`NodeStep` fields, plus the node reward with
       `reward_per_node`). It provides the KPIs and `save(output_dir, run_config)` (CSV and JSON),
       and keeps only the current and the last completed episode.
     - The `TrackResults` wrapper feeds the tracker (`env.get_wrapper_attr('tracker')`).
     - The Stable-Baselines3 `ResultsCallback(output_dir, save_freq, run_config)` logs the KPIs of
       each completed episode under `smartcer/` and saves every `save_freq` episodes. Its
       Stable-Baselines3 import is guarded.
  4. `smartcer/gym/metrics/`:
     - `compute_kpis`: welfare, revenue, cost, storage cost, incentive and reward; shared, exported
       and imported energy; sharing rate (shared / exported) and coverage rate (shared /
       imported); battery charge, discharge, cell throughput and equivalent full cycles
       (throughput / 2 S^max); penalty mean, maximum and penalised steps (k > 1e-9). Totals cover
       the community, with a block per battery.
     - `fit_score` ([M24] eq. 11) and `daily_welfare`.
     - `KPI.md` rewritten with formulas and units; it replaces the 0.6.2 KPIs (S12).
  5. Environment additions: `run_config()` (JSON for the output files), `current_step`,
     `scenario_path`.
  6. Removed `smartcer/cer_metrics/`, `smartcer/storage_controllers/` and the S6 xfail test
     (`test/unit/v062/`). The regression tests on the new API are
     `test_s6_rbc_executes_intended_energy`, `test_s12_rates_are_named_after_what_they_measure`
     and `test_s18_baselines_refuse_scenario_folders`. No 0.6.2 module and no xfail test is left.
     The CI wheel check now imports `smartcer.gym.controllers`, `smartcer.gym.metrics` and
     `smartcer.gym.tracking`.
  7. Leather 2022 baselines regenerated locally in `results/baselines/leather_2022/` (`results/`
     added to `.gitignore`, not committed).
  8. `docs/CONFIG.md` (section 4; section 6: `obs_clipped`, holidays, output directory and files),
     `CLAUDE.md`, `CHANGELOG.md`.
- Acceptance checks:
  - The RBC executes the intended energy in hand cases (`test/unit/gym/test_controllers.py`):
    - S6 case: surplus 40 kWh, 30 kW, 100 kWh, empty battery → action 1.0, 30 kWh charged,
      k = 0 (0.6.2: 0.3, 9 kWh);
    - prosumer 10 kWh / 5 kW / η = 0.9:
      - surplus 3 → action 0.6, 3 kWh charged, SoC 2.7;
      - deficit 2 → action −0.4, 2 kWh to the node, no import;
      - battery at 9 kWh with surplus 3 → action 0.2, 1 kWh;
      - Δt = 0.25 h with surplus 1 kWh → action 0.8, 1 kWh;
    - a producer stays idle at night; a consumer discharges 1 kWh at action −0.2;
    - near S^min the rule asks 0.478 kWh while the cell can deliver 0.430: the environment clips
      it to −A^lower with k = 0.0096.

    All passed.
  - Fit Score: 100 on identical series; hand example (1, 2, 3) vs (1, 2, 4) → 50; mean
    candidate → 0; reversed series → −300; errors for different lengths, empty, non-finite or
    constant reference → passed.
  - KPIs of the synthetic scenario match hand calculations (`test_baselines.py`):
    - naive, 48 h: exported 320 kWh, imported 456 kWh, shared 48 kWh, sharing rate 0.15,
      coverage rate 0.105;
    - naive economics: revenue 40 €, cost 111.2 €, TIP incentive 4.5 €, welfare −66.7 €;
    - RBC: pv_bat charges 20 kWh (18 kWh in the cell) until full, never discharges, 0.45 cycles,
      no penalty.

    KPIs on hand tables (sums, rates, cycles 0.5 and 0.1, penalty mean, maximum and steps) →
    passed.
  - Tracker:
    - its tables equal the step info;
    - per-node rewards are recorded;
    - an unfinished episode is dropped;
    - CSV files read back equal;
    - KPI and run configuration JSON are written.

    All passed. `ResultsCallback` with SAC, 3 episodes of 8 steps and `save_freq=2` → saves
    `episode_00002` and logs `smartcer/welfare` → passed.
  - Loader regression:
    - a 0.6.2 period starting inside a shifted block cuts both nodes at the same rows;
    - files with different row counts raise;
    - Leather split on 2022-06-01: 3,624 + 5,136 = 8,760 steps, with timestamps and every node's
      data equal to the whole-year load.

    All passed. The three tests fail on the previous loader.
  - `obs_clipped`: the training-window test counts 0 over the training day, 4 clipped loads at
    06:00 and 8 values at noon of the ×100 day (`test_observation_window.py`); 0 with a normaliser
    fitted on the whole period → passed.
  - `make test` → 237 passed in 97 s (29 new tests; S6 is no longer an xfail).
    `make test-coverage` → 95 % overall (93 % in phase 4); `controllers/` 94-100 %, `metrics/` 100 %,
    `tracking/` 93-95 %.
  - Wheel built and installed in the clean virtual environment: `smartcer.gym.controllers`,
    `metrics` and `tracking` import without Stable-Baselines3; `smartcer.storage_controllers` and
    `smartcer.cer_metrics` are gone.
  - CI run 39250885494 on `f609046` → green: unit 190 passed; integration 41 passed, 6 skipped
    (local data); coverage 95 %; wheel import of the new subpackages OK.
  - Leather 2022 baselines (local, 8,760 steps, both controllers in 9.7 s):

    | KPI | naive | rbc |
    |---|---|---|
    | welfare (€) | −138,473.18 | −140,523.12 |
    | revenue / cost (€) | 261,919.39 / 428,738.53 | 255,368.35 / 423,151.70 |
    | incentive (€) | 28,345.97 | 27,260.24 |
    | shared / exported / imported (kWh) | 582,963 / 834,147 / 1,508,025 | 569,131 / 811,840 / 1,486,695 |
    | sharing / coverage rate | 0.699 / 0.387 | 0.701 / 0.383 |
    | equivalent full cycles (total) | 0 | 75.2 |
    | penalised steps, max k | 0 | 274, 0.0196 |

    The RBC lowers the welfare by 2,050 € (1.5 %). Its batteries store surpluses that the
    community would otherwise share and be paid for, and lose energy in the cell (−13,832 kWh
    shared, −1,086 € of incentive). Its penalties come from discharges near S^min
    (`conceria_dolmen` 195 steps, `toscana_fridge` 79). The producer battery `pv_storage` charges
    91.8 kWh on 1-2 January and then stays full all year (0.45 cycles).
- Deviations from the brief and why:
  - Loader: in the 0.6.2 format every file must now have the same number of rows (before, only
    the rows of the period had to match). Cutting files of different lengths at the same
    positions could pair rows of different dates. Leather has 8,760 rows in every file.
  - `obs_clipped` does not count the zeros before the first row of the data (lookback and causal
    timing): they are padding, not data outside the training range.
  - Controllers read the true state from `env.community` (not the scaled observation), and the
    RBC uses g and l of step t whatever `observe_current_step` is, as in [M24] and section 13.
  - The RBC does not clip to the dynamic bounds itself: section 13 says the environment clips it
    "as any other action", so it can be penalised near S^min (see the open questions).
  - `compute_baselines` takes a configured environment (period, episode, reward) and a required
    output directory, rather than a run configuration file. It writes the 0.6.2 report structure
    (community table, one table per node, KPI JSON, run configuration JSON).
  - Tracking is a Gymnasium wrapper (`TrackResults`), not a flag inside the environment as in
    storage-node-environment. The tracker keeps only the current and the last completed episode.
    The callback saves every N completed episodes rather than every N steps, logs the KPIs to the
    Stable-Baselines3 logger, and needs a `DummyVecEnv` (the tracker lives in the training
    process).
  - KPI rates are fractions, not percentages as in 0.6.2. Equivalent full cycles use the nominal
    capacity S^max. The storage-node-environment visualisations are not ported (optional in the
    brief).
  - The Fit Score is implemented but not used by `compute_baselines`: its reference, the MILP
    oracle, comes in phase 7.
  - New public API on `SmartCerEnv`: `run_config()`, `current_step`, `scenario_path`.
- Open questions for Leonardo:
  1. RBC for producers: with section 13 a producer has d = g ≥ 0, so its battery fills up and
     never discharges (Leather `pv_storage`: full after 91.8 kWh, idle for the rest of 2022).
     0.6.2 discharged a producer's battery whenever its production was 0. Keep section 13 as
     written, or add a producer rule (changes FORMULATION section 13)?
  2. RBC near S^min: the rule discharges min(−d, s − S^min) on the node side, but the cell
     delivers at most η^dis (s̃ − S^min), so the environment clips and penalises the action (274
     steps on Leather, max k 0.0196, welfare effect negligible). Keep the literal rule, or let
     the rule use the cell limits (η^dis (s̃ − S^min) and (S^max − s̃) / η^cha), so that it never
     triggers the penalty?
  3. Equivalent full cycles = throughput / (2 S^max), with the nominal capacity. Prefer the usable
     capacity S^max − S^min?
  4. On Leather the RBC baseline is 2,050 € per year below idle batteries, because individual
     self-consumption competes with sharing. Is that the expected baseline for the class A
     comparison, or should the RBC target the community balance instead of the node's own?
     (Section 13 is [M24] Algorithm 1, per node.)
- Ready for phase 6: yes; the open questions do not block it (phase 6 is performance). The phase 5
  pull request is open and waits for your go-ahead to merge.

- Pull request #7 merged by Leonardo after review (merge commit `4a72b38`).
- Answers (2026-10-19), implemented in their own commit at the start of phase 6 (`02d0c4d`):
  1. and 4. `RBCController` gets a `mode`:
     - `node`: the section 13 rule plus a producer rule, under which a producer's battery
       discharges when g = 0 (as in 0.6.2); for single-node use;
     - `community`: balance B_t = Σ g − Σ l over all nodes. A surplus charges the batteries pro
       rata to their charge headroom; a deficit discharges them pro rata to their available
       discharge energy.

     `community` is the class A reference baseline and the default of `compute_baselines`.
     Produce both modes for Leather 2022 and extend FORMULATION section 13 with both. Do not
     require the RBC to beat the naive controller: report welfare, shared energy and incentive
     for the three controllers as they are.
  2. The RBC uses the environment's dynamic bounds (cell limits with efficiencies and S^min), the
     same ones applied to actions. Acceptance: zero penalised steps on Leather in both modes.
  3. Keep equivalent full cycles on the nominal capacity; add `efc_usable` on S^max − S^min and
     document both in KPI.md.

## Phase 6 · Performance – 2026-10-19

- Branch and commits: `refactor/v1-phase-6-performance`, created from `main` at `4a72b38`.
  - `02d0c4d` feat(gym): add node and community RBC modes within the dynamic bounds and cycles on
    usable capacity (the phase 5 answers, in their own commit)
  - `9fe7b61` perf(gym): add the step benchmark and tests guarding the step performance rules
  - this report
- Done:
  1. Phase 5 answers (`smartcer/gym/controllers/rbc.py`, `baselines.py`, `metrics/kpi.py`):
     - `RBCController(env, mode='community')`, modes `node` and `community` (names `rbc_node`
       and `rbc_community`). `CONTROLLERS` now maps a name to a factory taking the environment.
       `compute_baselines` runs `naive` and `rbc_community` by default.
     - Both modes work in energy (node side) within the dynamic bounds of section 6. Each
       battery can take E^cha = A^upper P^cha Δt and deliver E^dis = A^lower P^dis Δt, read from
       `node.get_bounds(t)`. The actions are float64, so an action equal to a bound is applied
       exactly.
     - `node`: d = g − l; charge min(d, E^cha), discharge min(−d, E^dis); a producer with g = 0
       discharges E^dis.
     - `community`: B_t over all nodes; every battery charges f · E^cha with
       f = min(1, B_t / Σ E^cha), or discharges f · E^dis with f = min(1, −B_t / Σ E^dis).
     - FORMULATION section 13 rewritten with both modes, the energies E^cha and E^dis, and the
       statement that the RBC is not required to beat idle batteries.
     - `efc_usable` = throughput / (2 (S^max − S^min)), per battery and in total, next to
       `equivalent_full_cycles`. `compute_kpis` takes the S^min of each battery and the tracker
       passes it. KPI.md explains both.
     - Leather 2022 baselines for `naive`, `rbc_node` and `rbc_community` regenerated in
       `results/baselines/leather_2022/` (gitignored, not committed).
  2. `benchmarks/bench_step.py` (`make bench`): whole-period episodes of a scenario (default:
     local Leather) with random actions drawn before the timed loop. It reports construction
     time, reset time, time per episode and per step, and the speed-up against the 43 s of 0.6.2.
  3. Step audit against item 2 of the brief. `step()` already met both rules, from the design of
     phases 2-4, so no code change was needed:
     - nodes read preloaded Python floats;
     - `community.timestamps` is a numpy `datetime64` array;
     - tracking runs in a wrapper, outside the environment;
     - the observation is built once per step from precomputed static features.

     `test/integration/gym/test_env_performance.py` now guards these rules:
     - no Python-level pandas call during 24 steps with the extended preset, lookback, causal
       timing and per-node rewards;
     - exactly one `build` call per reset and per step;
     - a smoke test of the benchmark script.
  4. CLAUDE.md (status, commands, structure, performance rules, RBC and KPIs), CHANGELOG
     (benchmark, RBC modes, `efc_usable`, the S13 speed figures) and the Makefile (`bench`).
- Acceptance checks:
  - Speed, measured on this laptop, Leather 2022: 10 nodes, 4 batteries, 8,760 steps, preset
    `meco`, random actions. The two runs were back to back:
    - 0.6.2 (tag `v0.6.2`): 51.0 s and 52.3 s per year (5.8–6.0 ms per step);
    - 1.0, `python benchmarks/bench_step.py --episodes 5`: median 3.19 s per year (best 3.07 s),
      364 µs per step.

    | Comparison | Speed-up |
    |---|---|
    | 0.6.2 timed in the same session | **16×** |
    | 43 s, the 0.6.2 figure in the brief | **13.5×** |

    Required: 5×. The laptop's timings vary by up to a factor of 2 with its load. Across six
    benchmark runs in this session, the median year ranged from 2.17 s to 4.30 s (248–491 µs per
    step), 10× to 20× faster than 43 s. Other figures:
    - preset `extended`: 3.84 s and 4.03 s;
    - construction: 0.2–0.35 s (`meco`) and 1.0 s (`extended`, holiday calendar);
    - reset: under 1 ms (0.6.2: 0.4–0.65 s, because it re-reads the CSV files).
  - `test_env_performance.py`: no pandas code inside `step()`, one observation build per step,
    benchmark smoke test → passed.
  - RBC hand cases (`test_controllers.py`):
    - S6: action 1.0, 30 kWh;
    - headroom case (9 of 10 kWh, η = 0.9, surplus 3 kWh): 1.111 kWh, action 0.2222, battery
      full;
    - near S^min: action −A^lower exactly, no penalty;
    - Δt = 0.25 h: action 0.8;
    - producer with g = 0: discharges 1.62 kWh (action −0.324), exported;
    - consumer: discharges 1 kWh;
    - community, five steps on two batteries and a consumer: B = 6 → actions (0.4, 0.2);
      B = −3 → (−0.2, −0.1); B = −10 → both empty; B = 25 → both at full power; B = 0 → idle.

    No penalty in any case → passed.
  - Synthetic scenario (`test_baselines.py`):
    - `rbc_node`, hand values:
      - pv_bat: charge 40 kWh, discharge 32.4 kWh, throughput 72 kWh, 1.8 cycles on S^max,
        2.0 on S^max − S^min;
      - factory_bat: charge 80 kWh, discharge 64.8 kWh, throughput 144 kWh;
      - community: revenue 31.48 €, cost 91.76 €, shared energy 80.4 kWh.
    - `rbc_community`: in every step, Σ charge = min(B⁺, Σ E^cha) and
      Σ discharge = min(B⁻, Σ E^dis), with no penalty.

    → passed.
  - Zero penalised steps on Leather in both modes: `test_rbc_is_never_penalised_on_leather`
    (local) gives `penalty_steps` 0 and `penalty_max` 0 for `node` and `community` → passed.
  - Leather 2022, whole year:

    | Controller | Welfare (€) | Δ vs naive | Shared energy (kWh) | Incentive (€) | Throughput (kWh) | Cycles (S^max / usable) | Penalised steps |
    |---|---|---|---|---|---|---|---|
    | naive | −138,473.18 | – | 582,962.8 | 28,345.97 | 0 | 0 / 0 | 0 |
    | rbc_node | −139,770.92 | −1,297.74 (−0.94 %) | 577,158.4 | 27,633.98 | 109,237.4 | 188.3 / 209.3 | 0 |
    | rbc_community | −135,233.69 | +3,239.48 (+2.34 %) | 627,270.2 | 30,464.00 | 102,492.8 | 176.7 / 196.3 | 0 |

    - `rbc_community` shares 44,307 kWh more than naive (+7.6 %) and earns 2,118 € more
      incentive.
    - `rbc_node` shares 5,804 kWh less and loses 712 € of incentive. Its loss against naive
      shrank from 2,050 € (phase 5) to 1,298 €, because producer batteries now discharge at
      night and no action is clipped. `pv_storage` makes 365.0 usable cycles: it fills by day
      and empties every night.
  - `make test` → 245 passed in 200 s (8 new tests); `make test-coverage` → 96 %.
  - CI run 39256404477 on `9fe7b61` → success:
    - unit: 192 passed;
    - integration: 45 passed, 8 skipped (local data absent);
    - coverage: 96 %;
    - the wheel imports every subpackage.
- Deviations from the brief and why:
  - Item 2 of phase 6 needed no change to `step()`; the work is the benchmark and the regression
    tests. I tried one micro-optimisation: reading the `NodeStep` fields with one `attrgetter` in
    the step info. It gave no measurable gain, so I reverted it.
  - The benchmark draws the random actions before the timed loop. `action_space.sample()` takes
    about half of the time of a naive loop (4.7 s against 2.2 s per year), so leaving it in would
    not measure the environment. The benchmark reports the median of the episodes because of the
    timing noise.
  - To compare on the same machine, I ran 0.6.2 again from a temporary `v0.6.2` worktree in the
    scratchpad, since removed. Only `reset` and `step` were called; its `save_results`, which
    writes into `cer_data`, was not called, and nothing in `cer_data` changed.
  - Phase 5 answers:
    - "Available charge headroom" is read as E^cha, the energy the battery can take in the step:
      power limit, charge cap, η^cha and S^max. With this reading the pro-rata split never
      exceeds a battery's bounds, and the whole surplus is placed whenever Σ E^cha ≥ B_t. The
      same applies to discharge with E^dis.
    - The RBC returns float64 actions. A float32 action equal to a bound can exceed it by about
      1e-8, which counts as a penalised step (tolerance 1e-9). The environment converts actions
      to float64 anyway. `NaiveController` keeps float32.
    - The registry name `rbc` became `rbc_node` and `rbc_community`. It was never released, so
      there is no deprecation.
    - The producer rule applies only to mode `node`: mode `community` does not look at node
      types.
    - Two phase 5 hand tests changed with answer 2:
      - the headroom case now charges 1.111 kWh (the cell limit), not 1 kWh;
      - the near-S^min case is no longer penalised.
- Open questions for Leonardo:
  1. With `charge_limit: production`, mode `community` can charge the battery of a prosumer
     that is itself importing, from its own generation, when the community has a surplus. This
     follows D6, but the node then imports more from the grid while its battery charges. With
     `surplus` the battery could not charge. Is that acceptable for the class A baseline, or
     should mode `community` charge only batteries whose node has a surplus?
- Ready for phase 7: yes. The pull request is open and waits for your go-ahead to merge.
