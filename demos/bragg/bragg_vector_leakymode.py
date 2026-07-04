"""Demo: Bragg fiber leaky vector mode via full Maxwell curl-curl FEAST.
"""

from fibermode.bragg import Bragg

p = 3

C = Bragg(
    wl=1.2e-6,
    ts=[5e-5, 1e-5, 2e-5, 1e-4],  # core | glass | air clad | PML
    mats=['air', 'glass', 'air', 'Outer'],
    ns=[1, 1.44, 1, 1],
    bcs=['r0', 'r1', 'R', 'OuterCircle'],
    maxhs=[0.10, 0.02, 0.2, 0.2],
    curveorder=p,
    scale=1.e-6,
)

# leakyvecmodes searches in the Z^2-plane directly (not Z, unlike leakymode)
betas, Zsqrs, Es, phis, R = C.leakyvecmodes(
    p=p,
    ctr=0.0023,
    rad=0.0001,
    alpha=5,
    nspan=4,
    npts=4,
    expected_dim=2,
    check_contour=10,
    niterations=200,
    nrestarts=0,
    seed=1,
)

phis.draw('phi')  # visualizes when called using netgen <thisfilename>
Es.draw('E')

# Error relative to the exact transfer-matrix value from BraggExactVector
exact_z2 = 0.002336593147489907 - 7.216918042895482e-07j
exact_beta = C.betafrom(exact_z2)
relbetaerr = abs(betas[0] - exact_beta) / abs(exact_beta)
print(f'\nNumerical Z²  = {Zsqrs[0]}')
print(f'Exact Z²      = {exact_z2}')
print(f'|error| in Z² = {abs(Zsqrs[0] - exact_z2):.3e}')
print(f'Relative |error| in β = {relbetaerr:.3e}')
