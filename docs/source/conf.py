# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime

# -- Path setup --------------------------------------------------------------
# Add the project root directory to the path so that autodoc can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'scCellFie'
copyright = f'{datetime.datetime.now().year}, Wellcome Sanger Institute'
author = 'Erick Armingol'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'nbsphinx',  # for Jupyter notebooks
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo' #'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'https://raw.githubusercontent.com/earmingol/scCellFie/refs/heads/main/docs/source/_static/scCellFie-Logo.png' #'_static/scCellFie-logo.png'  # Add your logo here
# html_favicon = '_static/favicon.ico'  # Add your favicon here
html_css_files = [
    'custom.css',
]

html_context = {
  'display_github': True,
  'github_user': 'earmingol',
  'github_repo': 'scCellFie',
  'github_version': 'master/docs/',
}

# -- nbsphinx configuration -------------------------------------------------
nbsphinx_execute = 'never'  # Options: 'always', 'never', 'auto'
nbsphinx_allow_errors = True  # Set to False once tutorials are stable

# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable/', None),
    'cobra': ('https://cobrapy.readthedocs.io/en/latest/', None),
}

# -- Napoleon settings -----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- autodoc settings ------------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'imported-members': True,
}
