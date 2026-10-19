Write tests for the module `$ARGUMENTS` following the established conventions of this project.

## Step 1: Explore the module

Before writing any code, read the following files to understand what needs to be tested:

1. Read `smartcer/$ARGUMENTS/__init__.py` (or `smartcer/$ARGUMENTS.py` for a single-file module)
   to identify the public API (exported classes, functions, constants).
   - Note: if `$ARGUMENTS` is in dot notation (e.g. `smartcer.core.storage`), convert it to path
     notation (`smartcer/core/storage`)
2. Read each source file referenced in `__init__.py` to understand function signatures, parameters,
   and expected behavior
3. Read `docs/FORMULATION.md` for the equations the module implements: expected values in the tests
   come from the formulation, not from the code under test
4. Read `test/conftest.py` and existing files inside `test/unit/` and `test/integration/` as style
   references (pick the two most relevant ones)

## Step 2: Choose the test files

Tests live in two folders, mirroring the package path below `smartcer/`:

```
test/
├── unit/<package path>/test_<submodule>.py         (one component in isolation)
└── integration/<package path>/test_<submodule>.py  (several components together, e.g. a full episode)
```

- `<package path>` is the module path without the leading `smartcer/`
  (e.g. `smartcer.core.storage` → `test/unit/core/storage/`)
- Do not create `__init__.py` files: pytest runs with `--import-mode=importlib` (see `pytest.ini`),
  so test files with the same name in different folders do not clash; shared helpers go in
  `test/conftest.py` as fixtures
- Create one test file per logical group (e.g. `test_battery.py`, `test_bounds.py`)
- Mark every test module with its folder marker at module level:
  `pytestmark = pytest.mark.unit` or `pytestmark = pytest.mark.integration`; add
  `pytest.mark.slow` to tests that take more than a few seconds
- If the module has few exports, a single `test_<module>.py` file is acceptable

## Step 3: Write the tests

### Mandatory coding standards (enforced — do not deviate)

- **Language**: ALL code, comments, docstrings, variable names in English
- **Type hints**: mandatory on every function and fixture parameter
- **Return type**: all test functions must be annotated `-> None`
- **Quotes**: single quotes `'...'` for all strings
- **Naming**: `test_<what_is_being_tested>` for test functions, `snake_case` for variables
- **Imports**: absolute only (e.g. `from smartcer.core.storage import Battery`), never relative
- **Import order**: stdlib → third-party → project modules, sorted alphabetically within each group
- **Docstrings**: Google style, mandatory on every test function and fixture
- **Optional dependencies**: use `pytest.importorskip('<module>')` (e.g. `stable_baselines3`), never
  a hard import that fails when the extra is not installed
- **Local data**: tests that need files under `smartcer/cer_data/` or `test/fixtures/local/` must be
  skipped when those folders are missing (they are gitignored and absent in CI)

### Docstring template

```python
def test_something(fixture_param: Path) -> None:
    """
    Test that <subject> <does what>.

    Args:
        fixture_param: Description of the fixture.
    """
```

### Test philosophy — essential tests only

- 3–9 tests per file: focus on happy paths and the most important behaviors
- Test public API only — do not test private methods or internal implementation details
- Each test must be short (5–15 lines of test code), focused on a single behavior
- Prefer hand-computed expected values on the synthetic scenario `test/fixtures/mini_cer/`
- Target execution time: the full new test suite must run in under 10 seconds
- Do NOT test: exhaustive parameter combinations, library internals, visualization, performance benchmarks

### Test categories to consider (apply only those relevant to this module)

1. **Initialization**: instantiate main classes with valid arguments → verify type and key attributes
2. **Validation**: invalid parameters → `pytest.raises(ValueError)` with a check of the message
3. **Factory functions / loaders**: call factory functions → verify return types
4. **Core operations**: call the main operation of each class/function → verify output type, shape, or value range
5. **I/O error handling**: if the module reads files, test `FileNotFoundError` with a missing path using `pytest.raises`
6. **Constants / mappings**: verify bidirectional consistency for any mappings or enumerations

### Fixtures

- Add new shared fixtures to `test/conftest.py` if they will be reused across multiple test files
- Define local fixtures inside the test file if used only there
- Follow the fixture pattern from `test/conftest.py`:

```python
@pytest.fixture
def fixture_name(parent_fixture: Path) -> ReturnType:
    """
    Brief description.

    Args:
        parent_fixture: Description.

    Returns:
        Description of what is returned.
    """
    return value
```

## Step 4: Update the Makefile

Add a focused make target so the module's tests can be run in isolation. The `help` target lists
every target that has a `## ` comment, so no other change is needed:

1. Add `test-<module>` to the `.PHONY` line
2. Add the target next to the other test targets:
   ```makefile
   test-<module>: ## Run <module> tests only
   	python -m pytest test/unit/<package path>/ test/integration/<package path>/ -v
   ```
   (list only the folders that exist; the recipe line is indented with a tab)
3. Run `make help` and check that the new target appears

## Step 5: Run and verify

After writing all files:

1. Run `make test-<module>` and fix any failures before finishing
2. Run `make test` to verify existing tests still pass
3. Report the final test count and execution time
