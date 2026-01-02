# Configuration file for the Sphinx documentation builder.
# QUSIM - NV Center Quantum Simulator Documentation

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../sim'))

# -- Project information -----------------------------------------------------
project = 'QUSIM'
copyright = '2024-2025, Leon Kaiser, MSQC Goethe University Frankfurt'
author = 'Leon Kaiser'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# MyST Parser settings for Markdown support
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# Napoleon settings for Google/NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# MathJax configuration for LaTeX rendering
mathjax3_config = {
    'tex': {
        'macros': {
            'ket': [r'\left| #1 \right\rangle', 1],
            'bra': [r'\left\langle #1 \right|', 1],
            'braket': [r'\left\langle #1 | #2 \right\rangle', 2],
            'expect': [r'\left\langle #1 \right\rangle', 1],
            'comm': [r'\left[ #1, #2 \right]', 2],
            'anticomm': [r'\left\{ #1, #2 \right\}', 2],
            'tr': r'\mathrm{Tr}',
            'Tr': r'\mathrm{Tr}',
            'rho': r'\rho',
            'hbar': r'\hbar',
            'dagger': r'\dagger',
        }
    }
}

# Templates and static files
templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_title = 'QUSIM Documentation'
html_short_title = 'QUSIM'

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2962FF",
        "color-brand-content": "#2962FF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82B1FF",
        "color-brand-content": "#82B1FF",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_logo = None
html_favicon = None

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'preamble': r'''
\usepackage{braket}
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# Suppress warnings
suppress_warnings = ['myst.header']
