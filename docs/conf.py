# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import sphinx_autosummary_accessors
import warnings
from matplotlib import MatplotlibDeprecationWarning

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ext'))
sys.path.insert(0, os.path.abspath(".."))

import shoot


# %% Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Shom Ocean Objects Tracker'
copyright = '2024, The Shom team'
author = 'The Shom team'
version = shoot.__version__
release = version

# %% General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    #    "sphinx.ext.linkcode",
    'sphinx.ext.intersphinx',
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    'sphinxarg.ext',
    'genlogos',
]

templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# %% Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "image_light": "_static/shoot-logo-light.png",
        "image_dark": "_static/shoot-logo-dark.png",
    },
    "repository_url": "https://gitlab.com/GitShom/STM/shoot",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "path_to_docs": "docs/",
}

# %% Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/fr/3/', None),
    'cmocean': ('https://matplotlib.org/cmocean/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numba:': ('https://numba.readthedocs.io/en/stable/', None),
    'numpy': ("https://numpy.org/doc/stable/", None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
}

# %% Sphinx gallery
warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
}

# %% IPython
ipython_warning_is_error = False

# %% Autosumarry
autosummary_generate = True
