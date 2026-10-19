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
  - docs(refactor): add phase 1 progress report (this section)
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
