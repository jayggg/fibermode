"""
Adaptive computation of vector leaky modes for a Bragg fiber.

Compares the adaptive FEM solution to the exact analytic solution
obtained via transfer matrices (Yeh et al.).

Adaptive loop: Solve (non-selfadjoint FEAST) → Estimate (DWR) → Mark → Refine
"""

import numpy as np
import ngsolve as ng
from scipy.optimize import newton

from fibermode.bragg import BraggExact, Bragg


# Geometry: air core / glass ring / air cladding / PML

ts = [4.0775e-05, 1e-5, 1e-5]       # layer thicknesses (m)
ts_pml = ts + [5e-5]
mats = ['air', 'glass', 'air']      # material names
mats_pml = mats + ['Outer']
n_air = 1.00027717                  # refractive indices
n_glass = 1.4388164768221814
ns = [n_air, n_glass, n_air]
ns_pml = ns + [n_air]
wl = 2.45e-6                        # wavelength (m)
scale = 15e-6                       # characteristic length L (m)
maxhs = [.1, .1, .1]                # mesh sizes (dimensionless)
maxhs_pml = maxhs + [.1]

# Exact Bragg fiber: transfer matrix + Newton solve

bragg_e = BraggExact(ts=ts, scale=scale, mats=mats, ns=ns, maxhs=maxhs, wl=wl)

nu = 1
outer = 'h1'
k_low = bragg_e.k0 * bragg_e.ns[0] * bragg_e.scale
beta = newton(bragg_e.determinant, np.array(.9999 * k_low),
              args=(nu, outer), tol=1e-15)
print(f'Exact β (scaled) = {beta}')
print(f'|det| residual   = {abs(bragg_e.determinant(beta, nu, outer)):.2e}')

# Numerical Bragg fiber

bragg_n = Bragg(ts=ts_pml, scale=scale, mats=mats_pml,
                maxhs=maxhs_pml, ns=ns_pml, wl=wl)

ng.Draw(bragg_n.index, bragg_n.mesh, 'Refractive index')

# FEAST search region: convert β to Z², conjugate to second sheet

exact_z2 = bragg_n.sqrZfrom(beta / bragg_n.L)

center   = exact_z2

print(f'\nExact  Z² = {exact_z2}')

# Adaptive solve

zsqr, errestimates, ndofs, e_r, e_l, phi_r, phi_l, betas, _ = \
    bragg_n.leakyvecmodes_adapt(
        p=3,
        radius=0.1,
        center=center,
        alpha=2,
        maxndofs=200000,
        npts=4,
        nspan=4,
        niterations=100,
        nrestarts=0,
        autoupdate=True,
        verbose=True)

# Visualize final mode (if script invoked from command line with netgen)

ng.Draw(e_r.gridfun(),   bragg_n.mesh, 'E_r')
ng.Draw(phi_r.gridfun(), bragg_n.mesh, 'phi_r')

# Error history vs exact

print('\nConvergence:')
print(f'{"ndofs":>10}  {"Z² error":>14}  {"DWR estimator":>14}')
for k, (n, z, (eta, _)) in enumerate(zip(ndofs[1:], zsqr, errestimates)):
    err = abs(z[0] - exact_z2)
    print(f'{n:10d}  {err:14.6e}  {eta:14.6e}')

beta_num = betas[0]
print(f'\nNumerical β = {beta_num:.6e} m⁻¹')
print(f'Exact     β = {beta / bragg_e.scale:.6e} m⁻¹')
print(f'|Δβ| / |β|  = {abs(beta_num - beta / bragg_e.scale) / abs(beta / bragg_e.scale):.2e}')
