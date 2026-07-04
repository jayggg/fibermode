"""
Adaptive computation of scalar leaky modes for a Bragg fiber.

Uses a known exact Z value as reference to track eigenvalue error
across adaptive iterations.

Adaptive loop: Solve (non-selfadjoint FEAST) → Estimate (DWR) → Mark → Refine
"""

import numpy as np
import ngsolve as ng

from fibermode.bragg import Bragg

# Geometry: simple 3-layer Bragg fiber with an outer PML layer

fb = Bragg(
    scale=5e-5,
    ts=[5e-5, 1e-5, 2e-5, 5e-5],  # last layer wide enough for smooth PML
    mats=['air', 'glass', 'air', 'Outer'],
    ns=[1, 1.44, 1, 1],
    bcs=['r0', 'r1', 'R', 'OuterCircle'],
    maxhs=[.10, .15, .15, .15],
    wl=1.2e-6)

ng.Draw(fb.mesh)

# Known exact Z (not Z²): used only for error reporting

exact_z = 2.4126736594918357 - 0.000142991376098823j
exact_z2 = exact_z**2
print(f'Using known exact Z² = {exact_z2} for error computations')

radius = 0.05  # search radius in Z² plane
order = 4
center = exact_z2

# Run the adaptive loop

zsqrs, ndofs, yr, yl, beta, _ = fb.leakymode_adapt(order,
                                                   radiusZ2=radius,
                                                   centerZ2=center,
                                                   maxndofs=100000,
                                                   nspan=4,
                                                   npts=4,
                                                   alpha=5,
                                                   niterations=100,
                                                   nrestarts=0,
                                                   verbose=True)

yr.draw('Right eigen function')

# Error history

print('\nConvergence:')
print(f'{"ndofs":>10}  {"Z error":>14}  {"Imag(Z error)":>16}')
for n, z in zip(ndofs[1:], zsqrs):
    err = np.sqrt(z[0]) - exact_z
    print(f'{n:10d}  {abs(err):14.7f}  {err.imag:+16.7f}')

print(f'\nFinal β = {beta}')
