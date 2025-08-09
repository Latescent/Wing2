# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))

project = "Honey Bee Wing Analysis"
copyright = "2025, Saman Khodarahmi, Parsa Mobini Dehkordi, Mohammad Mehdi Ghorbani"
author = "Saman Khodarahmi, Parsa Mobini Dehkordi, Mohammad Mehdi Ghorbani"
release = "2025"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core extension for pulling in docstrings
    "sphinx.ext.napoleon",  # To understand Google-style docstrings
    "sphinx.ext.viewcode",  # Adds links to highlighted source code
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
