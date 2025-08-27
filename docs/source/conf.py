# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import inspect
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
os.environ["MATPLOTLIBRC"] = os.path.join(project_root, "docs", "source")
os.chdir(project_root)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tri3d"
copyright = "2024, Nicolas Granger"
author = "Nicolas Granger"

pkg_version = importlib.metadata.version("tri3d")
if len(pkg_version.split("-")) > 1:
    tag, revision, commit = pkg_version.split("-")
    revision = revision[1:]  # remove 'r' prefix
    release = "v{}-{}-{}".format(tag, revision, commit)
    version = "latest ({})".format(commit)
else:
    tag, revision, commit = pkg_version, None, None
    release = tag
    version = "stable ({})".format(release)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinxrun",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for extensions --------------------------------------------------

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

autodoc_default_options = {
    'member-order': 'bysource',
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
}

nbsphinx_execute = "always"

plot_working_directory = project_root
plot_html_show_formats = False
plot_html_show_source_link = False
plot_formats = [(".jpeg", 200)]

# -- Options for Linkcode extension -------------------------------------------

if revision is not None:
    linkcode_revision = f"v{tag}-{revision}-{commit}"
else:
    linkcode_revision = f"v{tag}"
linkcode_url = (
    "https://github.com/CEA-LIST/tri3d/blob/"
    + linkcode_revision
    + "/{filepath}#L{linestart}-L{linestop}"
)


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    mod = importlib.import_module(info["module"])

    obj = mod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        filepath = inspect.getsourcefile(obj)
        if filepath is None:
            return
        filepath = os.path.relpath(filepath, project_root)
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop
    )


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
