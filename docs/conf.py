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
copyright = "2024-2025, Patrick J. Taylor"
author = "Patrick J. Taylor"
release = VERSION

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "numpydoc",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pymatgen": ("https://pymatgen.org", "https://pymatgen.org/objects.inv"),
}

# autodoc

autodoc_typehints = "none"

# numpydoc

numpydoc_show_inherited_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = "all"
numpydoc_xref_aliases = {
    "complex": ":class:`python:complex`",
    "DescriptorCalculator": "pengwann.descriptors.DescriptorCalculator",
    "AtomicInteractionContainer": "pengwann.interactions.AtomicInteractionContainer",
    "AtomicInteraction": "pengwann.interactions.AtomicInteraction",
    "WannierInteraction": "pengwann.interactions.WannierInteraction",
    "np.dtype": "numpy.dtype",
    "np.int_": "numpy.int_",
    "Structure": "pymatgen.core.structure.Structure",
    "SharedMemory": "multiprocessing.shared_memory.SharedMemory",
}

# bibtex

bibtex_bibfiles = ["refs.bib"]
mathjax3_config = {
    "loader": {"load": ["[tex]/braket"]},
    "tex": {"packages": {"[+]": ["braket"]}},
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_logo = "_static/logo.svg"
