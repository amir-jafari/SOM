"""Sphinx configuration for NNSOM Toolbox."""
import os
import sys

# Make the NNSOM package importable (src/ must be in path so 'import NNSOM' works)
sys.path.insert(0, os.path.abspath(".."))

os.environ.setdefault("MPLBACKEND", "Agg")

# ── Project information ───────────────────────────────────────────────────────
project   = "NNSOM"
author    = "Martin Hagan, Amir Jafari, Lakshmi Sravya Chalapati, Ei Tanaka"
copyright = "2024, Martin Hagan, Amir Jafari, Lakshmi Sravya Chalapati, Ei Tanaka"
release   = "1.8.2"
version   = "1.8"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings automatically
    "sphinx.ext.napoleon",      # Google / NumPy docstring styles
    "sphinx.ext.viewcode",      # [source] links on every page
    "sphinx.ext.intersphinx",   # cross-link numpy, scipy, matplotlib docs
    "sphinx.ext.mathjax",       # render LaTeX math
    "sphinx.ext.autosummary",   # summary tables + per-item pages
    "sphinx.ext.todo",
    "sphinx_copybutton",        # copy button on code blocks
    "sphinx_design",            # grid / card directives
    "myst_parser",              # include .md files as pages
    "nbsphinx",                 # include Jupyter notebooks
    "nbsphinx_link",
]

# ── autodoc ───────────────────────────────────────────────────────────────────
autosummary_generate  = True
autodoc_member_order  = "bysource"
autodoc_default_options = {
    "members":          True,
    "undoc-members":    True,
    "show-inheritance": True,
}

# ── napoleon ──────────────────────────────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True

# ── intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3",        None),
    "numpy":      ("https://numpy.org/doc/stable",     None),
    "scipy":      ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable",    None),
}

# ── nbsphinx ──────────────────────────────────────────────────────────────────
nbsphinx_allow_errors     = True
nbsphinx_execute          = "never"
nbsphinx_timeout          = 300
nbsphinx_kernel_name      = "python3"
nbsphinx_link_target_root = "../../examples/Tabular/Iris/notebook"

nbsphinx_prolog = r"""
.. raw:: html

    <a href="https://colab.research.google.com/github/amir-jafari/SOM/blob/main/examples/Tabular/Iris/notebook/{{ env.doc2path(env.docname, base=None).replace('src/docs/', 'examples/Tabular/Iris/notebook/').replace('.nblink', '.ipynb') }}" target="_blank" rel="noopener noreferrer">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align:text-bottom"/>
    </a>
"""

# ── General ───────────────────────────────────────────────────────────────────
templates_path   = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Old files replaced by guide/ and api/ structure
    "NNSOM.rst",
    "modules.rst",
    "usage.rst",
    "installation.rst",
    "dependencies.rst",
    "quick_use.rst",
    "getting_help.rst",
]
source_suffix    = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = ["dollarmath"]

# ── HTML theme ────────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only":                  False,
    "prev_next_buttons_location": "bottom",
    "style_external_links":       True,
    "collapse_navigation":        False,
    "sticky_navigation":          True,
    "navigation_depth":           4,
    "includehidden":              True,
    "titles_only":                False,
}

html_context = {
    "github_user":    "amir-jafari",
    "github_repo":    "SOM",
    "github_version": "main",
    "doc_path":       "src/docs/",
}

html_static_path = ["_static"]
html_css_files   = ["custom.css"]
html_title       = "NNSOM Toolbox"
html_short_title = "NNSOM"

# ── MathJax ──────────────────────────────────────────────────────────────────
mathjax3_config = {
    "tex": {
        "inlineMath":  [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# ── Copy-button — skip prompts in code blocks ─────────────────────────────────
copybutton_prompt_text      = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True