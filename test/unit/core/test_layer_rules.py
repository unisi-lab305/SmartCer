"""
Static checks of the layer rules of `smartcer/core/`.

No `print`, no `sys.exit`, no `assert` statements and no imports of `gymnasium` or `smartcer.gym`.
"""

import ast
from pathlib import Path

import pytest

import smartcer.core


pytestmark = pytest.mark.unit

CORE_PATH = Path(smartcer.core.__file__).parent
FORBIDDEN_IMPORTS = ('gymnasium', 'gym', 'smartcer.gym')
CORE_FILES = sorted(CORE_PATH.rglob('*.py'))


def _violations(path: Path) -> list[str]:
    """
    Layer-rule violations found in one source file.

    Args:
        path: Python source file.

    Returns:
        list[str]: One description per violation, with the line number.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == 'print':
                found.append(f'print at line {node.lineno}')
            if (isinstance(func, ast.Attribute) and func.attr == 'exit'
                    and isinstance(func.value, ast.Name) and func.value.id == 'sys'):
                found.append(f'sys.exit at line {node.lineno}')
        elif isinstance(node, ast.Assert):
            found.append(f'assert at line {node.lineno}')
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [
                node.module or '']
            for name in names:
                if any(name == f or name.startswith(f'{f}.') for f in FORBIDDEN_IMPORTS):
                    found.append(f'import of {name} at line {node.lineno}')
    return found


def test_core_package_has_source_files() -> None:
    """
    Test that the scan below sees the core sources (guards against an empty glob).
    """
    assert CORE_PATH / 'storage' / 'battery.py' in CORE_FILES


@pytest.mark.parametrize('path', CORE_FILES, ids=lambda p: p.relative_to(CORE_PATH).as_posix())
def test_core_module_respects_layer_rules(path: Path) -> None:
    """
    Test that a core module has no print, sys.exit, assert or Gymnasium import.

    Args:
        path: Python source file under smartcer/core/.
    """
    assert _violations(path) == []
