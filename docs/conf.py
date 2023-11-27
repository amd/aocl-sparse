# ##############################################################################
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'AOCL-Sparse'
copyright = '2020-2024, Advanced Micro Devices, Inc'
author = 'Advanced Micro Devices, Inc'
version = ''
release ='4.2.0.0'

# -- General configuration ---------------------------------------------------
extensions = ['sphinxcontrib.bibtex', 'breathe', 'sphinx_collapse']
bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = 'author_year'
breathe_default_project = 'aocl-sparse'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_template']

# -- MathJax -----------------------------------------------------------------
mathjax3_config = {
    'chtml' : {
        'mtextInheritFont' : 'true',
    }
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'rocm_docs_theme'
html_theme_options = {
    "link_main_doc": False,
    "flavor": "local",
    "repository_provider" : None,
    "navigation_with_keys" : False,
}
