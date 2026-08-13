"""Guided modes of a bent hollow air duct, via guidedhelicalmodes.

The simplest possible waveguide: a circular duct filled with a single
medium (index n=1 everywhere, so no core/cladding index contrast at
all), confined only by a perfectly reflecting (Dirichlet) outer wall.
Its unbent guided modes are exactly the Bessel-function modes of the
scalar Helmholtz equation on a disk,

    u_mn(r, theta) = J_m(j_{m,n} r / Rout) exp(i m theta),
    beta_mn = sqrt(k**2 - (j_{m,n} / Rout)**2),

where j_{m,n} is the n-th positive zero of J_m. But on bending the duct,
the modes shift in a way that is not analytically tractable.

See ModeSolver.guidedhelicalmodes; theory from
[Gopalakrishnan & Neunteufel, Guided modes of helical waveguides,
Wave Motion, 2025. https://doi.org/10.1016/j.wavemoti.2025.103621]
"""

import numpy as np
from scipy.special import jn_zeros

from fibermode import StepIndex, StepIndexExact

# Plain air duct: core and cladding share the same index (NA=0), so
# nothing but the hard outer wall at r=Rout confines the mode.
Rout = 2.0  # nondimensional duct radius (core radius fixed at 1)
k = 3.0  # nondimensional wavenumber

fiber = StepIndexExact(rcore=1, rclad=Rout, nclad=1, ncore=1, ks=k)
duct = StepIndex(fiber=fiber, Rout=Rout, h=.3, hcore=.3)

# Exact ground truth for the unbent duct's fundamental (m=0, n=1) mode.
m, n = 0, 1
jmn = jn_zeros(m, n)[-1]
beta_exact = np.sqrt(k**2 - (jmn / Rout)**2)

if __name__ == '__main__':

    print('Exact straight-duct beta (Bessel):', beta_exact)

    # guidedhelicalmodes searches (and reports) the nondimensional
    # Z² = L²(k²n₀² - β²) of this class, so convert the exact beta
    # above into a Z² search center. Here L = rcore = 1, so the two
    # planes differ only by Z² = k² - β².
    ctr = duct.sqrZfrom(beta_exact)
    rad = 2 * (duct.L**2) * beta_exact * .05  # |dZ²/dβ| = 2L²β

    # Sanity check: a numerically-straight duct (very large bend
    # radius) should reproduce the Bessel value above.
    Z2, y, yl, P = duct.guidedhelicalmodes(a=1e4,
                                           b=0,
                                           p=3,
                                           center=ctr,
                                           radius=rad,
                                           nspan=2)
    print('Nearly-straight duct beta:', duct.betafrom(Z2))
    y.draw(name='unbent')

    # Now actually bend the duct and see the mode shift.
    a_bent = 8.0
    Z2, y, yl, P = duct.guidedhelicalmodes(a=a_bent,
                                           b=0,
                                           p=3,
                                           center=ctr,
                                           radius=rad,
                                           nspan=2)
    print('Bent duct (a=%g) beta:' % a_bent, duct.betafrom(Z2))
    y.draw(name='bent')
