import os
import sys
from pengwann.version import __version__ as VERSION

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pengWann"
copyright = "2025, Patrick J. Taylor"
author = "Patrick J. Taylor"
release = VERSION

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

autodoc_typehints = "none"

bibtex_bibfiles = ["refs.bib"]
mathjax3_config = {
    "loader": {"load": ["[tex]/braket"]},
    "tex": {"packages": {"[+]": ["braket"]}},
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
