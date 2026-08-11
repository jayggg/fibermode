
# Computing Optical Fiber Modes

**Using the `fibermode` package**


The Python package `fibermode` contains facilities for 
computing guided and leaky modes of optical fibers.  
They provide both semi-analytical closed-form solutions and numerical
solutions based on finite element discretizations of the scalar
Helmholtz and vector Maxwell curl-curl eigenproblems.  The numerical
facilities are based on finite elements and are built atop
[NGSolve](https://ngsolve.org). The eigensolver is the FEAST contour
integral eigensolver.


## Capabilities

**Semi-analytical methods**

- LP modes of step-index fibers via characteristic equations and Bessel
  functions — `StepIndexExact`
- Scalar and vector leaky modes of Bragg fibers via the transfer matrix
  method — `BraggExactScalar`, `BraggExactVector`

**Numerical methods (FEM + FEAST)**

- Guided and leaky modes (scalar and vector) of arbitrary fiber geometries, including 
  - step-index fibers — `StepIndex`,
  - Bragg fibers — `Bragg`,
  - antiresonant fibers — `ARF`, `NANF`,
  - photonic bandgap fibers — `PBG`.


Key algorithmic features include the  FEAST polynomial eigensolver with contour integration in the complex plane,  a frequency-dependent PML for accurate leaky mode eigenvalues, nondimensional *Z*-plane formulation to avoid numerical round-off in large  eigenvalue problems, confinement loss (dB/m) computed directly from complex propagation constants, and mode visualization tools.

## Installation

```
git clone git@github.com:jayggg/fibermode.git
```

Add the directory containing `fibermode/` to your `PYTHONPATH`.

Install the dependencies in doc/requirements.txt. The two primary dependencies are [NGSolve](https://ngsolve.org), a
high-performance finite element library, and
[pyeigfeast](https://bitbucket.org/jayggg/pyeigfeast/src/master/),
which implements the
FEAST polynomial eigensolver for NGSolve. Additional dependencies are
[numpy](https://numpy.org), [scipy](https://scipy.org) (Bessel functions and
root finding), [matplotlib](https://matplotlib.org), and
[cxroots](https://github.com/RParini/cxroots) (complex-plane root finding for
leaky modes).

## Quick start

```python
from fibermode import StepIndexExact, StepIndex, Bragg, BraggExactScalar

# Semi-analytical LP modes of a named fiber
f = StepIndexExact('Nufern_Yb')
betas = f.XtoBeta(f.propagation_constants(ell=0))

# Numerical guided modes (FEM + FEAST)
fiber = StepIndex(fibername='Nufern_Yb')
betas, zsqrs, modes = fiber.guidedmodes(p=3)
```

Further simple demos are available at the `demos` folder.


## Documentation & tutorial notebooks

See `docs` folder.


## Cite

If you use `fibermode` in your research or teaching, please cite it:

```bibtex
@misc{fibermode,
  author       = {Gopalakrishnan, Jay},
  title        = {Computing Optical Fiber Modes Using \texttt{fibermode}},
  howpublished = {\url{https://github.com/jayggg/fibermode}},
  year         = {2026}
}
```

## License

Write to  [Jay Gopalakrishnan](https://web.pdx.edu/~gjay/)
for consultations, bug reports and suggestions. Pull requests  are welcome at
<https://github.com/jayggg/fibermode>.
Code and documentation 
are released under the [MIT License](LICENSE).

