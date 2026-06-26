"""
Compute leaky modes of a step-index fiber.
Usage of different types of PML are shown.
"""

from fibermode import StepIndex

fiber = StepIndex(fibername='Nufern_Yb', R=2, refine=1)

center = 1.96 - 0.19j  # center of circle to search for Z-resonance values
radius = 0.3  # search radius
p = 3  # polynomial degree

################################################################
# This uses Nannen+Wess frequency-dependent PML and a polynomial
# eigensolver as described in ["Computing leaky modes of optical
# fibers using a FEAST algorithm for polynomial eigenproblems, by
# Gopalakrishnan, Parker and Vandenberge,
# https://doi.org/10.1016/j.wavemoti.2021.102826]

print('POLY:' + 30 * '-')
z, y, yl, beta, P, _ = fiber.leakymode(p,
                                       rad=radius,
                                       ctr=center,
                                       alpha=5,
                                       verbose=False)
y.draw('poly')

exact_z = 1.957793326920255 - 0.18543240054910448j  # see docs/1.3
print('Error in computed non-dimensional Z-resonance values:\n',
      abs(exact_z - z))

################################################################
# This uses a standard PML with smooth hand-made coefficients
# (The complex transformation can be seen in the modesolver code.)

print('SMOOTH:' + 30 * '-')

z2s, y2, yl2, beta, P2, _ = fiber.leakymode_smooth(p,
                                                   radiusZ2=radius**2,
                                                   centerZ2=center**2,
                                                   alpha=5,
                                                   verbose=False)
y2.draw('smooth')
print('Square of error in computed non-dimensional Z-resonance values:\n',
      abs(exact_z**2 - z2s))

################################################################
# This uses NGSolve automatic PML by complex mesh transformation

print('AUTO:' + 30 * '-')

z2a, y2, yl2, beta, P2 = fiber.leakymode_auto(p,
                                              radiusZ2=radius**2,
                                              centerZ2=center**2,
                                              alpha=5,
                                              verbose=False)
y2.draw('auto')
print('Square of error in computed non-dimensional Z-resonance values:\n',
      abs(exact_z**2 - z2a))
