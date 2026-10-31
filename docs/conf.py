# Configuration file for the Sphinx documentation builder.


# Define common settings here
project = "LlamaFactory"
copyright = "2024, LlamaFactory Team"
author = "LlamaFactory Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_js_files = [
    "js/switcher.js",
]

html_css_files = [
    "css/lang-switcher.css",
    "css/tables.css",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3
