"""Guided modes of a coiled step-index fiber, via guidedhelicalmodes.

Reproduces demo_2d_model.py of the Zenodo supplement
[https://doi.org/10.5281/zenodo.15530385] to the paper
[Gopalakrishnan & Neunteufel, Guided modes of helical waveguides,
Wave Motion, 2025. https://doi.org/10.1016/j.wavemoti.2025.103621],
whose reduced 2D model (reduced2dmodel.py there) is what
ModeSolver.guidedhelicalsystem implements.

The reference example is already nondimensional since core radius is
set to 1, so L = rcore = 1. We reproduce that first in this demo.

The second half of the demo re-runs the same physical waveguide expressed in
different units. Namely, every cross section length multiplied by s and
the wavenumber divided by s, for which Z² = L²(k²n₀² - β²) is invariant
while β scales as 1/s.

We also show usage of StepIndex.corefraction (not available for all fibers).
It is reported for each computed mode.

Agreement of the computed Z² sets (to discretization error) in this
demo is also incorporated into a test function (test_helical.py).
"""

from math import sqrt, pi

from fibermode import StepIndex, StepIndexExact

# First we reproduce a reference run from the openly archived
# code at https://zenodo.org/records/15530385 :
#    Waveguide parameters of demo_2d_model.py from that code, where all
#    lengths are already nondimensionalized by the core radius rcore = 1
#    are as follows:
rcore = 1.
rclad = 2.2
nclad = 1.
ncore = 4.  # = sqrt(-(V0 - nclad**2)) for V0 = -15
k = 1.  # wavenumber ("wavelen" in the reference code)
a = 3.  # helix radius
b = 5 / (2 * pi)  # helix pitch
# Reference results, quoted there as beta² of the modes found:
betasqr_ref = [13.736, 7.554, 6.382, 1.214, .6]
search_radius = .03  # reference search radius, in the beta plane
p = 4


def build(scale=1.):
    """StepIndex model of the above fiber, with all lengths scaled by
    `scale` and the wavenumber inversely scaled, i.e., the same physical
    waveguide in different units, so mode betas scale as 1/scale."""

    fib = StepIndexExact(rcore=rcore * scale,
                         rclad=rclad * scale,
                         nclad=nclad,
                         ncore=ncore,
                         ks=k / scale)
    return StepIndex(fiber=fib,
                     Rout=rclad / rcore,
                     curveorder=p,
                     h=.4,
                     hcore=.2)


def findmodes(fbm, scale=1., betasqrs=None):
    """Search near each reference beta of `betasqrs` (default: all of
    them) in the model `fbm` built at the given `scale`.

    Returns one (bsqr, rescaled_betasqrs, Z2, corefractions) tuple per
    search, where `rescaled_betasqrs` are beta² values scaled back to
    the reference units, so they are comparable to `bsqr` (and to each
    other) whatever `scale` was used. See `report` for printing these.
    """

    results = []
    for bsqr in betasqr_ref if betasqrs is None else betasqrs:
        beta = sqrt(bsqr) / scale  # physical beta at this scale
        # Convert the reference beta-plane search disk to the Z² plane
        # that guidedhelicalmodes searches: Z² = (L k n₀)² - (L β)², so
        # |dZ²/dβ| = 2 L² β.
        ctr = fbm.sqrZfrom(beta)
        rad = 2 * (fbm.L**2) * beta * (search_radius / scale)

        Z2, y, yl, P = fbm.guidedhelicalmodes(a=a * scale,
                                              b=b * scale,
                                              center=ctr,
                                              radius=rad,
                                              p=p,
                                              nspan=5,
                                              npts=1,
                                              niterations=100,
                                              stop_tol=1e-7,
                                              verbose=False)
        cfs = fbm.corefraction(y.gridfun())
        betas = fbm.betafrom(Z2)
        results.append((bsqr, (betas * scale)**2, Z2, cfs))
    return results


def report(results):
    """Print what findmodes returned."""

    for bsqr, betasqrs, Z2, cfs in results:
        print('  near beta² = %g (reference):' % bsqr)
        for bb, zz, cf in zip(betasqrs, Z2, cfs):
            print('    beta² = %12.9f   Z² = %12.9f   core energy %2d%%' %
                  (bb, zz, 100 * cf))


if __name__ == '__main__':

    print('\nDEMO BEGINS:' + '=' * 52)
    print('Reference beta² values:', betasqr_ref)
    print('=' * 64)

    print('At the reference scale (L = rcore = 1):')
    report(findmodes(build()))
    print('=' * 64)

    # Same waveguide, different units: Z² should be unchanged and
    # (beta * scale)² should again match the reference beta² values.
    scale = 12.5
    print('\nSame waveguide with all lengths x%g, k /%g (L = %g):' %
          (scale, scale, rcore * scale))
    report(findmodes(build(scale), scale))
    print('=' * 64)
