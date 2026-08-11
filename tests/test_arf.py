"""ARF leaky-mode test."""

from fibermode import ARF


def test_arf_poletti_pml_consistency():
    """Solve the same near-lossless LP01-like mode of the 6-tube Poletti
    ARF (wavelength 1800nm) via three independently implemented PML
    formulations, following demos/arf/arf_poletti_mode.py:

    - leakymode: Nannen-Wess frequency-dependent PML, a nonlinear
      polynomial eigenproblem in Z (contour search around Z=2.24).
    - leakymode_smooth: handmade C^2 smooth PML, a linear
      eigenproblem in Z^2 (contour search around Z^2=2.24^2).
    - leakymode_auto: NGSolve's automatic PML, also linear in Z^2,
      same contour as leakymode_smooth.

    ARF has no closed-form ("exact") solution to check against, unlike
    StepIndex/Bragg. Instead, this checks that three differently
    discretized formulations of the same physical eigenproblem agree
    with each other -- a stronger correctness signal than comparing

    A loose regression check against a hand-verified reference is kept
    too, to catch FEAST converging on the wrong mode entirely
    (something the cross-formulation check alone wouldn't catch if the
    same wrong mode happened to be nearest the contour in all three
    cases).

    Reference (p=2, verified by running the demo on 2026-08-13):
        Z   (poly PML)        = 2.24773208 - 6.85e-09j
        Z^2 (smooth/auto PML) = 5.05233313 - ~3e-08j

    """
    p = 2
    ctr, rad, nspan, alpha = 2.24, 0.02, 10, 5

    a = ARF(name='poletti')

    Z, y, yl, beta, P, _ = a.leakymode(p=p,
                                       ctr=ctr,
                                       rad=rad,
                                       nspan=nspan,
                                       alpha=alpha)

    Z2_smooth, *_ = a.leakymode_smooth(p,
                                       centerZ2=ctr**2,
                                       radiusZ2=5 * rad,
                                       alpha=alpha)

    Z2_auto, *_ = a.leakymode_auto(p,
                                   centerZ2=ctr**2,
                                   radiusZ2=5 * rad,
                                   alpha=alpha)

    Z2_poly = Z[0]**2

    # Cross-formulation consistency (see docstring): loose rtol since
    # the three formulations discretize the PML differently.
    assert abs(Z2_poly - Z2_smooth[0]) < 1e-4 * abs(Z2_smooth[0]), \
        "poly-PML and smooth-PML formulations disagree on Z^2"
    assert abs(Z2_poly - Z2_auto[0]) < 1e-4 * abs(Z2_auto[0]), \
        "poly-PML and auto-PML formulations disagree on Z^2"

    # Regression check against the hand-verified reference value.
    Zref = 2.24773208 - 6.85e-09j
    assert abs(Z[0] - Zref) < 1e-2 * abs(Zref), \
        "leakymode did not converge to the expected LP01-like mode"

    # This mode is expected to be very low loss (near band center).
    assert abs(Z[0].imag) < 1e-6


if __name__ == "__main__":
    test_arf_poletti_pml_consistency()
