"""PBG leaky-mode test."""

import numpy as np

from fibermode import PBG
from fibermode.pbg.fiber_dicts.rod import params


def test_pbg_rod_litchinitser_fundamental_mode():
    """Solve the fundamental mode of the 6-rod PBG fiber from
    fiber_dicts/rod.py, both scalar and vector, and check against
    reference values.

    This fiber -- a single ring of 6 high-index dielectric rods
    (n_tube=1.8) around a solid core of the SAME index as the
    cladding (n_core=n_clad=1.44) -- is the design from

        N. M. Litchinitser et al., "Resonances in Microstructured
        Optical Waveguides."

    It's also the fiber studied in section 5.3 of

        J. Gopalakrishnan, J. Grosek, G. Pinochet-Soto, and P.
        Vandenberge, "Adaptive resolution of fine scales in modes of
        microstructured optical fibers," SIAM J. Sci. Comput.,
        47(1):B108-B130, 2025. DOI: 10.1137/24M1651605

    We expect a fundamental-mode search with

        scalar: ctr = 1.73806037-0.01388821j, rad=.01, p=3

    to be successful from prior numerical experience, which also
    provided the pinned reference values used for this test.

    Scalar leakymode() solves the *scalar* Helmholtz problem, which is
    only a weakly-guiding approximation (not really valid here since
    Δn = n_tube - n_clad = 0.36 is large). So although Z^2 from the vector
    solve does NOT match (Z_scalar)^2, the vector search below is
    centered at (Z_scalar)^2 anyway, since that's close enough
    for FEAST to converge onto the true vector eigenvalues.

    The vector eigenvalue is a near-degenerate pair (the two
    polarization states of the fundamental hybrid mode, split by the
    hexagonal lattice's imperfect rotational symmetry: Re differs by
    ~3e-5, Im by ~3e-4). This pair requires nspan/npts with real slack
    above the true multiplicity of 2. (Often with nspan=2,
    FEAST's SVD-based kernel-cleaning step would sometimes collapse the
    subspace from 2 vectors to 1, after which the reduced iteration
    couldn't converge to either eigenvalue and just drifts, so
    we use nspan=4 for robust convergence.)
    """

    b = PBG(params)

    # --- Scalar (Z-plane) ---
    Z, y, yl, beta, P, _ = b.leakymode(2,
                                       rad=.01,
                                       ctr=1.73806037 - 0.01388821j,
                                       alpha=5,
                                       npts=2,
                                       nspan=2,
                                       niterations=100,
                                       nrestarts=0)

    Zref = 1.73806038 - 0.01388821j
    assert abs(Z[0] - Zref) < 1e-2 * abs(Zref), \
        "leakymode did not converge to the expected fundamental mode"

    # --- Vector (Z^2-plane); see docstring for why nspan/npts=4 ---
    center2 = (1.73806037 - 0.01388821j)**2
    betas, Zsqrs, E, phi, R = b.leakyvecmodes(p=1,
                                              rad=.1,
                                              ctr=center2,
                                              alpha=b.alpha,
                                              npts=2,
                                              nspan=4,
                                              niterations=100,
                                              nrestarts=0)

    assert len(Zsqrs) == 2, \
        "expected the near-degenerate polarization pair (2 eigenvalues)"

    Z2refs = [3.1394368 - 0.08614398j, 3.13943344 - 0.08581423j]

    # Match by nearest reference; FEAST doesn't guarantee ordering.
    unmatched = list(Z2refs)
    for z2 in Zsqrs:
        j = int(np.argmin([abs(z2 - r) for r in unmatched]))
        ref = unmatched.pop(j)
        assert abs(z2 - ref) < 1e-2 * abs(ref), \
            "leakyvecmodes did not converge to the expected mode pair"
    assert not unmatched


if __name__ == "__main__":
    test_pbg_rod_litchinitser_fundamental_mode()
