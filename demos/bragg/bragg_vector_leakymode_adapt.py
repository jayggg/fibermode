"""
Adaptive computation of vector leaky modes for a Bragg fiber.

Uses the fiber geometry and run parameters from Notebook 2.3.  The exact
Z² value (computed via BraggExactVector) is used as a reference to track
eigenvalue error across adaptive iterations.

Adaptive loop: Solve (non-selfadjoint FEAST) → Estimate (DWR) → Mark → Refine

Run on command line "netgen bragg_vector_leakymode_adapt.py" to
visualize each adaptive iteration in Netgen GUI.
"""

import numpy as np
import ngsolve as ng
from scipy.optimize import newton

from fibermode.bragg import Bragg, BraggExactVector

# Geometry: same fiber as Notebook 2.3 (λ = 2.45 µm, air core + glass ring)

ts = [4.0775e-05, 1e-5, 1e-5]
mats = ['air', 'glass', 'air']
ns = [1.00027717, 1.4388164768221814, 1.00027717]
wl = 2.45e-6
scale = 15e-6
maxhs = [.1, .1, .1]

ts_pml = ts + [5e-5]
mats_pml = mats + ['Outer']
ns_pml = ns + [ns[0]]
maxhs_pml = maxhs + [.1]

fb = Bragg(ts=ts_pml,
           scale=scale,
           mats=mats_pml,
           maxhs=maxhs_pml,
           ns=ns_pml,
           wl=wl)

ng.Draw(fb.mesh)

# Exact Z² via semi-analytical BraggExactVector (HE₁₁ mode, ν = 1)

fb_exact = BraggExactVector(ts=ts, scale=scale, mats=mats, ns=ns, wl=wl)

nu = 1
outer = 'h1'
k_low = fb_exact.k0 * fb_exact.ns[0] * fb_exact.scale
beta_exact = newton(fb_exact.determinant,
                    np.array(.9999 * k_low),
                    args=(nu, outer),
                    tol=1e-15)
exact_z2 = fb.sqrZfrom(beta_exact / fb.L)
print(f'Exact Z² = {exact_z2}')

# Run the adaptive loop

order = 3
visualize = True  # pause at each iteration to display estimator and mode

stepper = fb.leakyvecmodes_adapt_gen(
    order,
    radius=0.1,
    center=0.78,  # from Notebook 2.3
    alpha=2,
    maxndofs=200000,
    nspan=4,
    npts=4,
    niterations=200,
    nrestarts=0,
    trustme=False,
    seed=10,
    verbose=True)
try:
    while True:
        state = next(stepper)
        if visualize:
            ng.Draw(ng.Norm(state['uR'].gridfun(i=0).components[0])**2,
                    fb.mesh,
                    name='Transverse E Intensity')
            ng.Draw(state['eevis'])
            input('* Pausing for visualization. Enter any key to continue')
except StopIteration as done:
    Zsqrs, errestimates, ndofs, ER, EL, phiR, phiL, beta, _ = done.value

ER.draw('Transverse E field')

# Error history

print('\nConvergence:')
print(f'{"ndofs":>10}  {"Z² error":>14}  {"DWR estimator":>14}')
for n, z, (eta, _) in zip(ndofs[1:], Zsqrs, errestimates):
    err = abs(z[0] - exact_z2)
    print(f'{n:10d}  {err:14.6e}  {eta:14.6e}')

print(f'\nFinal β = {beta}')
