# -- Path setup --------------------------------------------------------------
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # repo root

# -- Project information -----------------------------------------------------
project = 'IcContextImitationLearning'
author = 'David & Riccardo'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "autoapi.extension",     # parses source code directly (no imports)
    "sphinx.ext.napoleon",   # Google/NumPy docstrings
    "sphinx.ext.viewcode",
]

# AutoAPI: point to the folder you want documented
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "reworked_diffusion_policy")]
autoapi_add_toctree_entry = True

# Optional (nice-to-have)
autoapi_root = "api"  # put generated pages under /api/
# autoapi_ignore = ["*/tests/*", "*/scripts/*"]
# autoapi_python_use_implicit_namespaces = True  # if you have namespace pkgs (no __init__.py)

# Napoleon (docstring parsing) options
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML --------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
