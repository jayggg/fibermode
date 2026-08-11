"""Setup script for fibermode.

The repository root doubles as the ``fibermode`` package (this file's
directory contains ``__init__.py`` directly), so ``package_dir`` maps
the ``fibermode`` package onto ".". Install in editable mode from the
repository root::

    pip install -e .

``pyeigfeast`` is NOT listed in install_requires: it has no PyPI
release and no setup.py of its own, so pip cannot fetch or build it.
Clone it and put it on PYTHONPATH (or give it its own editable install
once it has a setup.py) before importing fibermode. See README.md.

``ngsolve`` and ``netgen_mesher`` (its mesher) are also NOT listed:
many users build/install NGSolve outside of pip (conda, source build,
MPI/MKL-enabled builds), and a plain ``pip install ngsolve`` would
pull in a second, unrelated copy that can shadow or conflict with it.
Manage that dependency yourself; see README.md.
"""

from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fibermode",
    version="0.1.0",
    description=(
        "Computing guided and leaky optical fiber modes with NGSolve "
        "and the FEAST contour-integral eigensolver"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jay Gopalakrishnan",
    url="https://github.com/jayggg/fibermode",
    license="MIT",
    packages=[
        "fibermode",
        "fibermode.arf",
        "fibermode.bragg",
        "fibermode.nanf",
        "fibermode.pbg",
        "fibermode.solvers",
        "fibermode.stepindex",
        "fibermode.utilities",
    ],
    package_dir={"fibermode": "."},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
        "cxroots",
        # celluloid: only used by the (currently broken) PBG wavelength
        # animation demo — omitted so it's not a required dependency
        # ngsolve, netgen_mesher: manage separately (see README) so
        # this doesn't clobber a non-pip NGSolve install
        # pyeigfeast: not on PyPI, install separately (see README)
    ],
)
