"""
Numerically compute guided LP modes of a step-index fiber
by finite element solution of a Helmholtz eigenproblem
and compare them with exact semi-analytical solutions.
"""

from fibermode import StepIndex
import warnings

fb = StepIndex(fibername='Nufern_Yb')

# Finite element solution of Helmholtz eigenproblem for LP modes:
betas, zsqrs, Y = fb.guidedmodes(p=3)

# Solve semianalytically and also name the numerical betas by LP convention
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    n2i, exactbetas = fb.name2indices(betas)

# Report
print('\nRESULTS:', '#' * 55)
print('Computed non-dimensional Z-squared values:\n', zsqrs)
print('LP names:\n', n2i)
print('Computed approximation of physical propagation constants:\n', betas)
print('Exact physical propagation constants:\n', exactbetas)
print('#' * 64)

# save results into a temporary file (all saved files are in "outputs" folder.)
#
#   (See loadmodes.py on how to load modes saved in this file.)
fb.savemodes('my_tmp_output', betas, Y)
