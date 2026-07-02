"""Demo: Bragg fiber leaky scalar mode via polynomial PML FEAST.

A case where a core-localized leaky mode is found.
"""

from fibermode.bragg import Bragg

p = 4

B = Bragg(
    wl=1.7e-6,
    ts=[4e-5, 1e-5, 1e-5, 1e-4],  # core | glass | air clad | PML
    mats=['air', 'glass', 'air', 'Outer'],
    ns=[1, 1.44, 1, 1],
    maxhs=[0.10, 0.05, 0.2, 0.2],
    bcs=['r0', 'r1', 'R', 'OuterCircle'],
    curveorder=p,
    scale=1e-6,
)

Z, y, yl, beta, P, info = B.leakymode(
    p,
    ctr=0.06,
    rad=0.01,
    alpha=5,
    nspan=4,
    npts=4,
    niterations=200,
    nrestarts=0,
)

y.draw()  # visualizes when called using netgen <thisfilename>

# Error relative to the exact transfer-matrix value from BraggExactScalar
exact_z = 0.06069691615738331 - 1.9987492290870344e-05j
exact_beta = B.betafrom(exact_z**2)
relbetaerr = abs(beta[0] - exact_beta) / abs(exact_beta)
print(f'\nNumerical Z  = {Z[0]}')
print(f'Exact Z      = {exact_z}')
print(f'|error|  in nondim  Z = {abs(Z[0] - exact_z):.3e}')
print(f'Relative |error| in β = {relbetaerr:.3e}  (rad/m)')
print(f'Converged    = {info["converged"]}')
