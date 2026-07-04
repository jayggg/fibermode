"""Demo: Bragg fiber leaky scalar mode via polynomial PML FEAST.

A not very useful leaky mode with reduced core-localization, extending
to the cladding with easily visible oscillations.

"""

from fibermode.bragg import Bragg

p = 4
B = Bragg(
    wl=1.2e-6,
    ts=[5e-5, 1e-5, 2e-5, 1e-4],  # core | glass | air clad | PML
    mats=['air', 'glass', 'air', 'Outer'],
    ns=[1, 1.44, 1, 1],
    bcs=['r0', 'r1', 'R', 'OuterCircle'],
    maxhs=[0.10, 0.02, 0.2, 0.2],
    curveorder=p,
    scale=5e-5,
)

Z, y, yl, beta, P, info = B.leakymode(
    p,
    ctr=2.4,
    rad=0.02,
    alpha=5,
    nspan=4,
    npts=4,
    niterations=200,
    nrestarts=0,
)

y.draw()  # visualizes when called using netgen <thisfilename>

# Error relative to the exact transfer-matrix value from BraggExactScalar
exact_z = 2.4126736594918357 - 0.000142991376098823j
exact_beta = B.betafrom(exact_z**2)
relbetaerr = abs(beta[0] - exact_beta) / abs(exact_beta)
print(f'\nNumerical Z  = {Z[0]}')
print(f'Exact Z      = {exact_z}')
print(f'|error|  in nondim  Z = {abs(Z[0] - exact_z):.3e}')
print(f'Relative |error| in β = {relbetaerr:.3e}')
print(f'Converged    = {info["converged"]}')
