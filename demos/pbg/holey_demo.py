"""Fundamental mode of an air-hole ("holey") fiber, using the PBG class.

Unlike the other demos in this folder, `fiber_dicts.holey` is not a
photonic bandgap fiber: its 'tube' sites are air (n_tube=1 < n_clad),
so the solid core guides by an ordinary total-internal-reflection-like
(index-guiding) mechanism. The surrounding air-hole lattice merely
lowers the cladding's average index, the same principle as a standard
index-guided photonic crystal fiber (PCF). This demo  shows that `PBG` class's
geometry machinery is general enough to build either kind of fiber.

The search centers below were located by a preliminary broad search
(center=1.5, rad=1, p=4), which turns up Z values near 1.69 and 2.67
(fundamental and next higher-order mode respectively); the fundamental
mode center used here narrows in on the first of those.
"""

from fibermode import PBG
from fibermode.pbg.fiber_dicts.holey import params

if __name__ == '__main__':

    A = PBG(params)

    # Scalar search, fundamental mode.
    center = 1.46180577 - 1.68583967e-09j
    radius = .001
    p = 4

    z, y, yl, beta, P, _ = A.leakymode(p,
                                       rad=radius,
                                       ctr=center,
                                       alpha=A.alpha,
                                       niterations=5,
                                       npts=4,
                                       nspan=2,
                                       nrestarts=0)
    y.draw()

    # Vector search, same mode (center squared, per leakyvecmodes'
    # Z^2-plane convention).
    betas, zsqrs, E, phi, R = \
        A.leakyvecmodes(rad=radius, ctr=center**2,
                        alpha=A.alpha, p=1,
                        niterations=5, npts=4,
                        nspan=2, nrestarts=0)
    E.draw(name='E')
    phi.draw(name='phi')
