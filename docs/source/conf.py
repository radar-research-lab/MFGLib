# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MFGLib"
copyright = "2025, RADAR Research Lab"
author = "RADAR Research Lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "nbsphinx",
]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosectionlabel_prefix_document = True

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "super"

# In the algorithm docstrings, the footnote references aren't explicitly referenced
suppress_warnings = ["ref.footnote"]

# We want the '$' symbol to get stripped when someone copies a code snippet.
copybutton_prompt_text = "$ "

# Prevent autodoc from using fully qualified object names
add_module_names = False

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
