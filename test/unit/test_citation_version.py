"""
Consistency of the citation metadata: the version and the year of the BibTeX of `README.md` and
`docs/citing.md` and of `CITATION.cff` against `smartcer.__version__`.

Only the `@software` entry is checked. When bumping the version, update these three files.
"""

import re
from pathlib import Path

import pytest

import smartcer


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
BIBTEX_FILES = ('README.md', 'docs/citing.md')
CFF_FILE = 'CITATION.cff'
SOFTWARE_ENTRY = re.compile(r'@software\{smartcer,(.*?)\n\}', re.DOTALL)


def _read(relative: str) -> str:
    """
    Text of a repository file.

    Args:
        relative: Path relative to the repository root.

    Returns:
        str: Content of the file.
    """
    path = REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f'{relative} not found (not a source checkout)')
    return path.read_text(encoding='utf-8')


def _bibtex_field(relative: str, field: str) -> str:
    """
    Value of a field of the `@software` entry of a file.

    Args:
        relative: Path relative to the repository root.
        field: BibTeX field name.

    Returns:
        str: Value between the braces.
    """
    entry = SOFTWARE_ENTRY.search(_read(relative))
    assert entry, f'{relative}: no @software{{smartcer, ...}} entry'
    match = re.search(rf'^\s*{field}\s*=\s*\{{([^}}]*)\}}', entry.group(1), re.MULTILINE)
    assert match, f'{relative}: the @software entry has no {field} field'
    return match.group(1).strip()


def _cff_field(field: str) -> str:
    """
    Value of a top-level scalar key of `CITATION.cff`.

    Args:
        field: Key name.

    Returns:
        str: Value without quotes.
    """
    match = re.search(rf'^{field}:\s*[\'"]?([^\'"\n]+)[\'"]?\s*$', _read(CFF_FILE), re.MULTILINE)
    assert match, f'{CFF_FILE}: no top-level {field} key'
    return match.group(1).strip()


@pytest.mark.parametrize('relative', BIBTEX_FILES)
def test_bibtex_version_matches_the_package(relative: str) -> None:
    """
    Test that the BibTeX version equals `smartcer.__version__`.

    Args:
        relative: File with the BibTeX entry.
    """
    assert _bibtex_field(relative, 'version') == smartcer.__version__, relative


def test_cff_version_matches_the_package() -> None:
    """
    Test that the version of `CITATION.cff` equals `smartcer.__version__`.
    """
    assert _cff_field('version') == smartcer.__version__


@pytest.mark.parametrize('relative', BIBTEX_FILES)
def test_bibtex_year_is_not_before_the_release_year(relative: str) -> None:
    """
    Test that the BibTeX year is not earlier than the year of `date-released`.

    Args:
        relative: File with the BibTeX entry.
    """
    released = int(_cff_field('date-released')[:4])
    assert int(_bibtex_field(relative, 'year')) >= released, relative
