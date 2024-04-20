# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath("../NNSOM"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NNSOM'
copyright = '2024, Lakshmi Sravya Chalapati, Ei Tanaka'
author = 'Lakshmi Sravya Chalapati, Ei Tanaka, Amir Jafari, Martin Hagan'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    #"sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",

]

sphinx_gallery_conf = {
     # 'examples_dirs': '../../examples/Tabular/Iris/py',   # path to your example scripts
     # 'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     # 'filename_pattern': '.*\\.py$',  # Changed to include all Python files
     # 'ignore_pattern': '__init__\\.py',
     #
        'examples_dirs': '../../examples/Tabular/Iris/py',  # path to your example scripts
        'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
        #'capture_stdout': True,  # Captures output from stdout
        'image_scrapers': ('matplotlib',),
        'filename_pattern': '.*\\.ipynb$',  # Changed to include all Python files
        'ignore_pattern': '__init__\\.py'

}

# inside conf.py
html_context = {
    "github_user": "amir-jafari",
    "github_repo": "SOM",
    "github_version": "main",  # e.g., 'main' or 'master'
    "doc_path": "",  # e.g., 'source/' if your .rst files are in the 'source' folder
}


html_theme_options = {
    # ...
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/amir-jafari/SOM",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/NNSOM/",
            "icon": "fas fa-box-open",
        },

        # Add other icons and links as needed
    ],
    "use_edit_page_button": True,  # If you want an edit button for source files
    "navigation_with_keys": True,
    # ...
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
