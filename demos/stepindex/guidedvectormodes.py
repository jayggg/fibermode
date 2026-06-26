"""
Compute guided vector (Maxwell) modes of a step-index fiber
using Nedelec+Lagrange mixed finite elements.

[Reference: DOI of SISC paper with the method 10.1137/24M1651605]
"""

from fiberamp import FiberMode

fbm = FiberMode(fibername='Nufern_Yb')
betas, zsqrs, E, phi, _ = fbm.guidedvecmodes(ctr=-10, rad=1,
                                             p=4, niterations=100, nrestarts=0,
                                             stop_tol=1e-9)
print('\n' + '\nRESULTS:', '#'*55)
print('Computed non-dimensional Z-squared values:\n', zsqrs)
print('Computed approximation of physical propagation constants:\n', betas)
print('(Compare with exact values by running guidedvectormodesexact.py.)')
print('#'*64)

E.draw(name='E')
phi.draw(name='phi')
