import numpy as np
from scipy.optimize import newton

from fibermode.bragg import BraggExactScalar, BraggExactVector, Bragg


def test_leakyvec_adapt():
    """
    Test the adaptive leaky vector mode solver for a Bragg fiber
    by reproducing expected results for the leaky fundamental mode
    (ν=1, h1) at λ=2.45 μm.   This test is extracted from an example
    demo notebook presented at the 2026 NGSolve workshop in Zurich.
    """

    # Very coarse mesh (for testing purposes only)
    ts = [4.0775e-05, 1e-5, 1e-5]  # layer thicknesses (m): core | glass | air
    mats = ['air', 'glass', 'air']
    ns = [1.00027717, 1.4388164768221814, 1.00027717]  # refractive indices
    wl = 2.45e-6  # wavelength (m)
    scale = 15e-6  # characteristic length L (m) for nondimensionalization
    maxhs = [.5, .5, .5]  # initial mesh sizes (fraction of layer radius)

    # PML additions appending an absorbing outer layer
    ts_pml = ts + [5e-5]
    mats_pml = mats + ['Outer']
    ns_pml = ns + [ns[0]]
    maxhs_pml = maxhs + [.5]

    bragg_n = Bragg(ts=ts_pml,
                    scale=scale,
                    mats=mats_pml,
                    maxhs=maxhs_pml,
                    ns=ns_pml,
                    wl=wl)

    # Run adaptivity solver
    stepper = bragg_n.leakyvecmodes_adapt_gen(p=3,
                                              radius=0.1,
                                              center=0.78,
                                              alpha=2,
                                              maxndofs=200000,
                                              autoupdate=True,
                                              verbose=True,
                                              npts=4,
                                              nspan=4,
                                              niterations=100,
                                              nrestarts=0)
    state = next(stepper)

    # Compute exact semi-analytical solution for comparison
    bragg_e = BraggExactVector(ts=ts, scale=scale, mats=mats, ns=ns, wl=wl)

    nu = 1  # azimuthal mode number for vector fundamental mode (HE₁₁)
    outer = 'h1'  # outgoing solution is the Hankel function of the first kind
    k_low = bragg_e.k0 * bragg_e.ns[0] * bragg_e.scale
    beta_exact = newton(bragg_e.determinant,
                        np.array(.9999 * k_low),
                        args=(nu, outer),
                        tol=1e-15)

    print(f'Exact β (scaled) = {beta_exact}')
    print(f'Exact β physical = {beta_exact/bragg_e.scale}')

    exact_z2 = bragg_n.sqrZfrom(beta_exact / bragg_n.L)

    Zsqrs = state['Zsqrs']
    errestimates = state['errestimates']
    ndofs = state['ndofs']
    for n, z, (eta, _) in zip(ndofs[1:], Zsqrs, errestimates):
        err = abs(z[0] - exact_z2)
        print(f'Computed Z² = {z[0]}')
        print(f'Exact Z²    = {exact_z2}')
    print(f'{"ndofs":>10}  {"Z² error":>14}  {"DWR estimator":>14}')
    for n, z, (eta, _) in zip(ndofs[1:], Zsqrs, errestimates):
        err = abs(z[0] - exact_z2)
        print(f'{n:10d}  {err:14.6e}  {eta:14.6e}')

    # Output as of 2026 test creation:
    #   (for comparison with future code revisions)
    #
    # Computed Z² = (0.7909752789479554-3.8503398109857235e-05j)
    # Exact Z²    = (0.78653268033122-0.000876656714587343j)
    #      ndofs        Z² error   DWR estimator
    #       3289    4.520972e-03    1.596768e-01
    # -----------------------------------------------------------

    # Check if expected error sizes are obtained (as above)
    eta, _ = errestimates[0]
    z2 = Zsqrs[0]
    err = max(abs(z2 - exact_z2))
    assert err < 1e-2, \
        "Bragg vector leaky mode error not of expected accuracy!"
    assert eta < 2e-1, \
        "Bragg vector leaky mode estimator not of expected accuracy!"


def test_leakyscalar_adapt():
    """
    Test the adaptive scalar leaky mode solver for a Bragg fiber
    by reproducing expected results for the leaky LP01 (nu=0) mode
    at λ=1.2 μm.  Geometry from bragg_scalar_leakymode_adapt.py.
    """

    # Very coarse mesh (for testing purposes only)
    bragg_n = Bragg(ts=[5e-5, 1e-5, 2e-5, 5e-5],
                    scale=5e-5,
                    mats=['air', 'glass', 'air', 'Outer'],
                    ns=[1.0, 1.44, 1.0, 1.0],
                    maxhs=[.5, .5, .5, .5],
                    wl=1.2e-6)

    # Known LP01 mode location (from BraggExactScalar; see test_bragg_exact.py)
    exact_z = 2.4126736594918357 - 0.000142991376098823j
    exact_z2 = exact_z**2

    stepper = bragg_n.leakymode_adapt_gen(p=2,
                                          radiusZ2=0.05,
                                          centerZ2=exact_z2,
                                          alpha=5,
                                          maxndofs=200000,
                                          verbose=True,
                                          npts=4,
                                          nspan=3,
                                          niterations=100,
                                          nrestarts=0)

    states = [next(stepper), next(stepper)]  # run exactly 2 iterations

    print(f'{"ndofs":>10}  {"Z² error":>14}  {"DWR estimator":>14}')
    for s in states:
        err = abs(s['zsqr'][0] - exact_z2)
        eta = np.sum(s['ee'])
        print(f'{s["ndof"]:10d}  {err:14.6e}  {eta:14.6e}')

    # Output as of 2026 test creation:
    #   (for comparison with future code revisions)
    #
    #   ndofs        Z² error   DWR estimator
    #    227    1.859600e-02    9.989210e-01
    #    519    6.370828e-03    2.501187e-01
    #
    # -----------------------------------------------------------

    s2 = states[-1]
    err = abs(s2['zsqr'][0] - exact_z2)
    eta = np.sum(s2['ee'])
    assert err < 1e-2, \
        "Bragg scalar leaky mode error not of expected accuracy!"


if __name__ == '__main__':

    test_leakyvec_adapt()
    test_leakyscalar_adapt()
