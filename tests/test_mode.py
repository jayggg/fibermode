from fibermode import StepIndex, StepIndexExact
import ngsolve as ng
from ngsolve import dx, grad
import numpy as np
import warnings
from scipy.spatial.distance import directed_hausdorff


def test_guided_residual():
    """
    Does StepIndex.guidedmodes give modes with small residuals?
    """

    p = 2
    fb = StepIndex(fibername='Nufern_Yb', curveorder=p, R=2)
    betas, zsqrs, Y = fb.guidedmodes(p=p,
                                     stop_tol=1e-14,
                                     niterations=200,
                                     verbose=False)
    Z2 = ng.Vector(zsqrs)

    X = ng.H1(fb.mesh, order=p, dirichlet='OuterCircle', complex=True)
    u, v = X.TnT()
    A = ng.BilinearForm(X)
    A += grad(u) * grad(v) * dx + fb.V * u * v * dx
    B = ng.BilinearForm(X)
    B += u * v * dx
    with ng.TaskManager():
        A.Assemble()
        B.Assemble()

    t = ng.MultiVector(Y._mv[0], len(Y._mv))
    t[:] = A.mat * Y._mv - (B.mat * Y._mv).Scale(Z2)
    residuals = np.diag(abs(ng.InnerProduct(t, t).NumPy()))

    assert max(residuals) < 1e-11, \
        "Step-index guided modes are not accurate."
    print("Test passed: Guided modes have small residuals:\n", residuals)
    print('#' * 70)


def test_guided_vec_residual():
    """
    Does StepIndex.guidedvecmodes give guided vector modes with small
    mixed-system residuals (see ResolventVectorMode.block_residuals in
    solvers/modesolver.py)?  Are the computed eigenvalues close to the
    semi-analytical exact values (StepIndexExact.vec_propagation_constants)?
    We check these across search disks that isolate eigenspaces of
    different dimension.

    Both residual & error checks are kept since they catch different bugs:
    the residual check verifies (E, phi, Z²) are self-consistent with whatever
    system got assembled, but would stay small even if the discretization
    encoded the wrong physics (e.g. a sign error in V or n² that's still
    internally consistent); the exact-value check catches that, but says
    nothing about whether E, phi solve the discrete system.

    Search disks below (ctr, rad) are informed by the semi-analytical
    roots from demos/stepindex/guidedvectormodesexact.py:

    * (-10, 1): TE01 (Z²=-10.187), TM01 (Z²=-10.184), and the exactly
      degenerate m=2 hybrid pair (Z²=-10.184, x2) -- dim 4.
    * (-1.657778, 0.5): the exactly degenerate m=1 hybrid pair at
      Z²=-1.657778 -- dim 2.

    """

    p = 2
    fb = StepIndex(fibername='Nufern_Yb', curveorder=p, R=2)
    fexact = StepIndexExact('Nufern_Yb')

    # All exact guided vector-mode Z² roots needed below: TE0/TM0
    # (multiplicity 1) and HYBRID m=1,2 (multiplicity 2, since each
    # hybrid root is a degenerate pair).
    exact_zsqrs = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for m0 in ('TE', 'TM'):
            for ys, *_ in fexact.vec_propagation_constants(0, m0name=m0):
                exact_zsqrs += [-y**2 for y in ys]
        for m in (1, 2):
            for ys, *_ in fexact.vec_propagation_constants(m):
                exact_zsqrs += [-y**2 for y in ys for _ in range(2)]
    exact_zsqrs = np.array(exact_zsqrs)

    search_disks = [
        (-10, 1),
        (-1.657778, 0.5),
    ]

    for ctr, rad in search_disks:
        betas, zsqrs, E, phi, R = fb.guidedvecmodes(ctr=ctr,
                                                    rad=rad,
                                                    p=p,
                                                    niterations=100,
                                                    nrestarts=0,
                                                    stop_tol=1e-9,
                                                    verbose=False)
        assert len(zsqrs) > 0, \
            f"guidedvecmodes found no modes in (ctr={ctr}, rad={rad})"

        residuals_phi, residuals_X = R.block_residuals(E, phi, zsqrs)

        assert max(residuals_phi) < 1e-7, \
            f"(ctr={ctr}, rad={rad}): phi = -Dinv B E residual too " \
            f"large: {max(residuals_phi):.2e}"
        assert max(residuals_X) < 1e-7, \
            f"(ctr={ctr}, rad={rad}): guidedvecmodes invariant-subspace " \
            f"residual too large: {max(residuals_X):.2e}"

        exact = exact_zsqrs[np.abs(exact_zsqrs - ctr) < rad]
        assert len(exact) == len(zsqrs), \
            f"(ctr={ctr}, rad={rad}): found {len(zsqrs)} mode(s) but " \
            f"{len(exact)} exact root(s) fall in this search disk"

        # Hausdorff distance between the computed and exact Z² point sets:
        computed_pts = zsqrs.real.reshape(-1, 1)
        exact_pts = exact.reshape(-1, 1)
        eig_error = max(
            directed_hausdorff(computed_pts, exact_pts)[0],
            directed_hausdorff(exact_pts, computed_pts)[0])
        assert eig_error < 1e-3, \
            f"(ctr={ctr}, rad={rad}): computed Z² too far from exact " \
            f"values (Hausdorff distance): {eig_error:.2e}"

        print(
            f'\nCASE: Search disk (ctr={ctr}, rad={rad}): found {len(zsqrs)} '
            f'guided vector mode(s), Z² = {zsqrs}')
        print(" Exact Z²:\n ", exact)
        print(" Hausdorff distance (computed vs exact Z²):\n ", eig_error)
        print(" Y-residual: D phi + B E:\n ", residuals_phi)
        print(" X-residual: A E + C phi - Z^2 M E:\n ", residuals_X)

    print("Test passed: Guided vector modes have small residuals and "
          "match exact eigenvalues across two search disks.")
    print('#' * 70)


def test_leaky_residual():
    """
    Does StepIndex.leakymodes give small residuals, AND an eigenvalue close
    to the exact value?  exact_z below is the same trusted reference value
    (see docs/1.3) already used in demos/stepindex/leakymodes.py for this
    same (center, radius, alpha) search window -- reused here rather than
    recomputed.
    """

    p = 3
    fb = StepIndex(fibername='Nufern_Yb', curveorder=p, R=2)
    center = 1.96 - 0.19j  # center of circle to search for Z-resonance values
    radius = 0.3  # search radius

    zsqrs, Y, Yl, beta, _ = fb.leakymode_auto(p,
                                              radiusZ2=radius**2,
                                              centerZ2=center**2,
                                              alpha=5,
                                              verbose=True)
    Z2 = ng.Vector(zsqrs)
    A, B, X = fb.autopmlsystem(p, alpha=5)
    t = ng.MultiVector(Y._mv[0], len(Y._mv))
    t[:] = A.mat * Y._mv - (B.mat * Y._mv).Scale(Z2)
    residuals = np.diag(abs(ng.InnerProduct(t, t).NumPy()))
    print("Leaky mode residuals:", residuals)
    assert max(residuals) < 1e-11, \
        "Step-index leaky modes are not accurate."

    exact_z = 1.957793326920255 - 0.18543240054910448j  # see docs/1.3
    eig_errors = np.abs(exact_z**2 - zsqrs)
    eig_error = eig_errors.max()
    print("Error in computed Z² vs exact:\n ", eig_errors)
    assert eig_error < 1e-1, \
        f"computed leaky Z² too far from exact value: {eig_error:.2e}"

    print("Test passed: Leaky modes have small residuals:\n", residuals)
    print('#' * 70)


if __name__ == '__main__':

    test_guided_residual()
    test_guided_vec_residual()
    test_leaky_residual()
