"""Sphinx configuration of the SmartCer documentation."""

from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst.directives.images import Figure
from sphinx.application import Sphinx

import smartcer


DOCS_DIR = Path(__file__).resolve().parent

project = 'SmartCer'
author = 'Leonardo Guiducci'
copyright = '2024, unisi-lab305'
release = smartcer.__version__
version = release

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

# CONFIG.md only points to configuration.md
exclude_patterns = [
    '_build',
    'CONFIG.md',
]

source_suffix = {'.md': 'markdown'}
myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath']
myst_heading_anchors = 3

# autodoc and napoleon (Google style)
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {'members': True, 'show-inheritance': True}
# stable-baselines3 is optional; mocking it keeps the build independent of torch
autodoc_mock_imports = ['stable_baselines3', 'torch']
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

# theme
html_theme = 'furo'
html_title = f'SmartCer {release}'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'source_repository': 'https://github.com/unisi-lab305/SmartCer/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

# logos and favicon are exported separately: use them only when the files exist
if (DOCS_DIR / '_static' / 'logo-light.svg').exists():
    html_theme_options['light_logo'] = 'logo-light.svg'
if (DOCS_DIR / '_static' / 'logo-dark.svg').exists():
    html_theme_options['dark_logo'] = 'logo-dark.svg'
if (DOCS_DIR / '_static' / 'favicon.png').exists():
    html_favicon = '_static/favicon.png'


class OptionalFigure(Figure):
    """Figure directive that shows a placeholder note when the image file is missing."""

    def run(self) -> list[nodes.Node]:
        """Return the figure, or a note naming the missing file."""
        relative = self.arguments[0].lstrip('/')
        if (DOCS_DIR / relative).exists():
            return super().run()
        note = nodes.note()
        note += nodes.paragraph(text=f'Figure to be added: {relative}')
        return [note]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the optional-figure directive."""
    app.add_directive('optional-figure', OptionalFigure)
    return {'parallel_read_safe': True}
