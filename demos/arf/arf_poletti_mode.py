"""
Computing some modes of a 6-tube ARF structure in Poletti's paper
"""

from fibermode import ARF

outermaterials = 'air'
freecapil = False
a = ARF(name='poletti', outermaterials=outermaterials, freecapil=freecapil)
# a.refine()  # Refine mesh if needed

# Solver parameters.

p = 2  # finite element degree
ctr = 2.24  # FEAST contour center
rad = 0.02  # FEAST contour radius
nspan = 10  # number of vectors in initial span
alpha = 5  # PML decay strength

solverparams = dict(p=p, ctr=ctr, rad=rad, nspan=nspan, alpha=alpha)

# Solve for leaky mode using freq-dependent PML & polynomial eig
print('POLY PML' + '-' * 60)
z, y, yl, beta, P, _ = a.leakymode(**solverparams)
y.draw()

# Try other solvers, like the standard PML & linear eig:
print('SMOOTH PML' + '-' * 60)
z2, yy, yyl, beta2, P2, _ = \
    a.leakymode_smooth(p, centerZ2=ctr**2, radiusZ2=5*rad, alpha=5)
print('AUTO PML' + '-' * 60)
z3, y3, yl3, beta3, P3 = \
    a.leakymode_auto(p, centerZ2=ctr**2, radiusZ2=5*rad, alpha=5)

#
# NOTE ------------------------------------------------
#
# For wavelength = 1800nm, we found these work:
#
#     Table 1
#     'OuterAir' Material: 'air', 'polymer', 'silica'
#     'Outer' Material:    'air'
#     +------+--------+--------+
#     | Mode | Center | Radius |
#     +------+--------+--------+
#     | LP01 | 2.24   | 0.02   |
#     | LP11 | 3.57   | 0.02   |
#     | LP21 | 4.75   | 0.02   |
#     | LP02 | 5.09   | 0.02   |
#     +------+--------+--------+
#
#     Table 2
#     'OuterAir' Material: 'polymer'
#     'Outer' Material:    'polymer'
#     +------+--------+--------+
#     | Mode | Center | Radius |
#     +------+--------+--------+
#     | LP01 | 48.553 | 0.005  |
#     | LP11 | 48.632 | 0.005  |
#     | LP21 | 48.734 | 0.005  |
#     | LP02 | 48.768 | 0.005  |
#     +------+--------+--------+
#
#     Table 3
#     'OuterAir' Material: 'silica'
#     'Outer' Material:    'silica'
#     +------+--------+--------+
#     | Mode | Center | Radius |
#     +------+--------+--------+
#     | LP01 | 54.198 | 0.005  |
#     | LP11 | 54.270 | 0.005  |
#     | LP21 | 54.361 | 0.005  |
#     | LP02 | 54.392 | 0.005  |
#     +------+--------+--------+
