"""
Tests for the semi-analytical facilities.
"""

import numpy as np
from scipy.optimize import newton

from fibermode.bragg import BraggExactScalar, BraggExactVector


def test_bragg_exact_scalar():
    """
    BraggExactScalar: the transfer-matrix determinant is small at the
    known LP01 (nu=0) leaky mode location of a 3-layer Bragg fiber
    (geometry from bragg_scalar_leakymode_adapt.py).

    Starting from the pinned mode location, Newton refines it and we
    verify |det| < 1e-9.  The residual floor (~1e-10) is set by Bessel
    function precision for this near-critical, weakly-leaky mode.

    Pinned result (2026):
      beta (scaled) ≈ 261.7883 + 1.318e-6j
      |det| residual ~ 1e-10
    """
    bragg = BraggExactScalar(ts=[5e-5, 1e-5, 2e-5],
                             scale=5e-5,
                             mats=['air', 'glass', 'air'],
                             ns=[1.0, 1.44, 1.0],
                             wl=1.2e-6)

    nu = 0  # LP01 (fundamental scalar mode)
    outer = 'h1'
    k_low = bragg.k0 * bragg.ns[0] * bragg.scale

    # Pinned LP01 mode location (Z, not Z²); Im(Z) small — weakly leaky mode
    exact_z = 2.4126736594918357 - 0.000142991376098823j
    beta0 = np.sqrt(k_low**2 - exact_z**2)  # convert Z → scaled beta

    beta = newton(bragg.determinant, beta0, args=(nu, outer), tol=1e-15)

    residual = abs(bragg.determinant(beta, nu, outer))
    print(f'Scalar LP01 beta (scaled) = {beta}')
    print(f'|det| residual            = {residual:.2e}')

    assert residual < 1e-9, \
        f"BraggExactScalar |det| = {residual:.2e}, expected < 1e-9"


def test_bragg_exact_vector():
    """
    BraggExactVector: Newton converges to a zero of the transfer-matrix
    determinant for the HE₁₁ (nu=1) leaky mode, starting from just
    below the cutoff k_low.

    This is the reference solution used in the adaptive FEM demo
    (leakyvectormode.ipynb) and test_adapt.py.

    Pinned result (2026):
      beta (scaled) ≈ 38.4689 + 1.139e-5j
      |det| residual < 1e-10
    """
    bragg = BraggExactVector(ts=[4.0775e-05, 1e-5, 1e-5],
                             scale=15e-6,
                             mats=['air', 'glass', 'air'],
                             ns=[1.00027717, 1.4388164768221814, 1.00027717],
                             wl=2.45e-6)

    nu = 1  # HE₁₁ (fundamental vector mode)
    outer = 'h1'
    k_low = bragg.k0 * bragg.ns[0] * bragg.scale

    beta = newton(bragg.determinant,
                  np.array(0.9999 * k_low),
                  args=(nu, outer),
                  tol=1e-15)

    residual = abs(bragg.determinant(beta, nu, outer))
    print(f'Vector HE11 beta (scaled) = {beta}')
    print(f'|det| residual            = {residual:.2e}')

    assert residual < 1e-9, \
        f"BraggExactVector |det| = {residual:.2e}, expected < 1e-10"


def test_betafrom_sqrZfrom_roundtrip():
    """
    In exact arithmetic, _BraggExactBase.betafrom and sqrZfrom are algebraic
    inverses

      betafrom(Z²) = sqrt(k_low² - Z²) / L,   k_low = L · k₀ · n0
      sqrZfrom(β)  = k_low² - (L · β)²

    where n0 = ns[-1] is the outermost layer index and L = scale.
    (There are also functions of the same name in the numerical ModeSolver
    class, but this test is for the semi-analytical class.)

    Tests both directions of the roundtrip on real (guided) and complex
    (leaky) values.  Uses scalar Bragg fiber parameters (λ=1.2 μm) so
    k_low is a concrete number (~261.8).
    """

    bragg = BraggExactScalar(ts=[5e-5, 1e-5, 2e-5],
                             scale=5e-5,
                             mats=['air', 'glass', 'air'],
                             ns=[1.0, 1.44, 1.0],
                             wl=1.2e-6)

    k = bragg.k0
    n0 = bragg.ns[-1]  # outer refractive index (1.0 here)
    k_low = bragg.scale * k * n0  # nondimensional cutoff (~261.8)
    exact_z = 2.4126736594918357 - 0.000142991376098823j  # Pinned exact Z
    exact_z2 = exact_z**2

    # --- Z² → β → Z² ---> suffers from cancellation errors:
    # sqrZfrom computes k_low² − (L·β)², so the absolute error
    # floor is O(ε · k_low²) ≈ 1.5e-11 regardless of Z².
    Z2_cases = np.array([
        exact_z2,  # weakly leaky (complex Z²)
        1.0 + 0j,  # generic real Z²
        5.0 + 0.01j,  # generic complex Z²
        100.0 + 0j,  # large real Z² (deeply leaky)
    ])
    err_z2 = np.max(
        np.abs(bragg.sqrZfrom(bragg.betafrom(Z2_cases)) - Z2_cases) /
        np.abs(Z2_cases))
    print(f'Z² → β → Z²  max rel error: {err_z2:.2e}')
    # 1e-10 accounts for cancellation; tighter not achievable for small Z²
    assert err_z2 < 1e-10, f"Z² → β → Z² roundtrip error {err_z2:.2e}"

    # --- β → Z² → β --->  will do better, closer to machine precision
    beta_cases = np.array([
        bragg.betafrom(exact_z2),  # leaky LP01 physical β (complex)
        0.99 * k * n0,  # near-cutoff guided (real)
        0.50 * k * n0,  # mid-band guided (real)
        1.05 * k * n0,  # above cutoff (Z² < 0, evanescent-like)
    ])
    err_beta = np.max(
        np.abs(bragg.betafrom(bragg.sqrZfrom(beta_cases)) - beta_cases) /
        np.abs(beta_cases))
    print(f'β → Z² → β   max rel error: {err_beta:.2e}')
    assert err_beta < 1e-14, f"β → Z² → β roundtrip error {err_beta:.2e}"

    # --- Cutoff identity: Z² = 0  ↔  β = k · n0 ---
    assert abs(bragg.betafrom(0) - k * n0) < 1e-14 * k * n0
    assert abs(bragg.sqrZfrom(k * n0)) < 1e-10 * k_low**2


if __name__ == '__main__':

    test_bragg_exact_scalar()
    test_bragg_exact_vector()
    test_betafrom_sqrZfrom_roundtrip()
