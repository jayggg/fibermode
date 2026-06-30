"""
Definition of ModeSolver class and its methods for computing
modes of various fibers.
"""
from warnings import warn
from ngsolve import curl, div, grad, dx, Conj, Integrate, InnerProduct, CF
from ngsolve import sqrt, sin, cos
from numpy import conj
from pyeigfeast import NGvecs, SpectralProjNG
from pyeigfeast import SpectralProjNGR, SpectralProjNGPoly
import ngsolve as ng
import numpy as np
import sympy as sm


class ModeSolver:
    """Provides algorithms to compute modes of various fibers,
    (including microstructured fibers with or without radial
    symmetry).  The key inputs are a cross section mesh, a
    characteristic length L, the constant refractive index n0 in the
    unbounded complement, the refractive index and the nondimensional
    index well V (all described below in more detail). The latter two
    (index & V) are expected to be provided as attributes of derived
    classes containing configuration details of specific fibers.

    HOW SCALAR MODES ARE COMPUTED:

    The Helmholtz mode in physical coordinates is given by

         Δu + k² n² u = β² u.

    The transverse refractive index n is a function implemented
    by a derived class, and it's assumed that it takes a
    constant value n₀ outside a fixed radius R₀ which bounds
    all inhomogeneities.

    What's implemented are algorithms for a non-dimensional version
    of the above, obtained after fixing a characteristic length scale L,
    and transforming the above to the following

         -Δu + V u = Z² u

    where Z² = L² (k² n₀² - β²) and the mode u (not to be confused with the
    physical mode) is defined on a non-dimensional (unit sized) domain.
    The nondimensional function V is an index well, akin  to a
    Schrödinger potential well, and is given in terms of the
    physical material properties by

          V = L²k² (n₀² - n²)   if r < R₀,
          V = 0                 if r > R₀.

    Here R₀ is the nondimensional radius such that n is constant
    beyond LR₀ in the  physical domain.

    HOW VECTOR MODES ARE COMPUTED:

    See [Gopalakrishnan, Grosek, Pinochet-Soto, Vandenberge. Adaptive
    resolution of fine scales in modes of microstructured optical
    fibers, SISC 2025, https://doi.org/10.1137/24M1651605] for details.

    CLASS ATTRIBUTES:

    * L: the characteristic transverse length scale, described above.
    * n0: constant refractive index in the unbounded r > R₀ region.
    * mesh: input mesh of non-dimensionalized transverse domain.
    * Further attributes assumed to be set by derived classes and used
      by ModeSolver can be listed, together with descriptions, by calling
      needs(printrequirements=True).
    """

    def __init__(self, mesh, L, n0):

        self.mesh = mesh
        self.L = L
        self.n0 = n0

        self.ngspmlset = False  # changes to True when NGSolve pml set
        self.gamma = None  # set in set_vecpml_coeff if using smooth vec pml

        print('ModeSolver: Checking if mesh has required regions')
        print('Mesh has ', mesh.ne, ' elements, ', mesh.nv, ' points, '
              ' and ', mesh.nedge, ' edges.')

        # When PML may be  used, it is put in the region R < r < Rout.
        # The PML region R < r < Rout is assumed to be called 'Outer'
        # in the given mesh.

        if sum(self.mesh.Materials('Outer').Mask()) == 0:
            raise ValueError('Input mesh must have a region called Outer')

        # The final outer radius of a circle terminating the
        # computational domain is assumed to be a boundary region
        # named 'OuterCircle' of the given mesh. It is the circle of
        # radius r = self.Rout, an attribute assumed to be set by a
        # derived class. The presence of 'OuterCircle' and self.Rout
        # are checked below.

        if sum(self.mesh.Boundaries('OuterCircle').Mask()) == 0:
            raise ValueError('Input mesh must have a terminating boundary ' +
                             'called OuterCircle')

        # Check if remaining required attributes are set
        self.needs()

    def needs(self, printrequirements=False):
        """
        Lists the attributes that must be set by a derived class
        before calling any of the implemented algorithms.
        """

        allneeds = ['Rout', 'R', 'V', 'index', 'k', 'curveorder']
        absent = []
        for need in allneeds:
            if not hasattr(self, need):
                absent.append(need)
        if len(absent) or printrequirements:
            print('Derived class must set these attributes:')
            print('  Rout = terminating radius of computational domain')
            print('  R = radius where PML may start (R < Rout)')
            print('  V = nondimensional index well function')
            print('  index = physical refractive index function')
            print('  k = wavenumber k')
            print('  curveorder = order of geometry approximation')
        for need in absent:
            print('*** Attribute', need, 'not set yet!')
        if len(absent):
            raise ValueError('Ensure all expected attributes are set')

    def betafrom(self, Z2):
        """
        Returns physical propagation constants (β), given
        nondimensional Z² values, input in Z2, per the formula
        β = sqrt(L²k²n₀² - Z²) / L . """
        return np.sqrt((self.L * self.k * self.n0)**2 - Z2) / self.L

    def sqrZfrom(self, betas):
        """ Return values of nondimensional Z squared, given physical
        propagation constants betas, ie, return Z² = L² (k²n₀² - β²). """
        return (self.L * self.k * self.n0)**2 - (self.L * betas)**2

    def boundarynorm(self, y):
        """
        Returns  L² norm of all functions in the span y restricted to
        the outermost boundary r= Rout.
        """

        def outint(u):
            dl = dx(definedon=self.mesh.Boundaries('OuterCircle'))
            s = abs(ng.Integrate(u * ng.Conj(u) * dl, self.mesh))
            return np.sqrt(s)

        bdrnrms = y.applyfnl(outint)
        print('Mode boundary L² norm = %.1e' % np.max(bdrnrms))
        return bdrnrms

    def estimatepolypmldecay(self, Z, alpha):
        """
        Returns an estimate of mode boundary norm, per predicted decay of
        the frequency dependent PML for given Z and alpha.
        """

        decayrate = alpha * (self.Rout - self.R) + \
            self.R * Z.imag
        bdryval = np.exp(-decayrate) / np.sqrt(np.abs(Z) * np.pi / 2)
        bdrnrm0 = bdryval * 2 * np.pi * self.Rout
        print('PML decay estimates boundary norm ~ %.1e' % max(bdrnrm0))
        return bdrnrm0

    def power(self, Etv, phi, beta):
        """
        Find power of mode with transverse electric and magnetic fields E, H.
        If E and H are from different modes, this method finds 'inner product'
        of the two modes according to the orthogonality type relationship
        found in Marcuse, Light Transmission Optics 2nd edition, eq 8.5.12 (and
        also in Snyder's Optical Waveguide Theory equation 11-13).
        """
        Sz = self.S(Etv, phi, beta)[1]
        p = ng.Integrate(Sz, self.mesh)
        return p

    def maxwell_product(self, E1, H2):
        """
        Find maxwell type product of two modes with transverse electric and
        magnetic fields E1, H2 respectively.

        This method finds 'inner product' of the two modes according to the
        orthogonality type relationship found in Marcuse, Light Transmission
        Optics 2nd edition, eq 8.5.12 (and also in Snyder's Optical Waveguide
        Theory equation 11-13).

        Parameters
        ----------
        E : ngsolve coefficient function
            Transverse electric field E = (Ex, Ey) (or (Er, Ey) for bent modes)
        H : same type as E
            Magnetic field of second mode (organized in same way).

        Returns
        -------
        p : float or complex
            Returns power if E and H belong to same mode.  Returns 'dot
            product type' value if E and H belong to distinct modes.

        """
        Sz = E1[0] * Conj(H2[1]) - E1[1] * Conj(H2[0])
        p = 1 / 2 * ng.Integrate(Sz, self.mesh)
        return p

    def H_field_from_E(self, Etv, phi, beta):
        """Return the H field (Htv, Hz) from transverse E field and
        phi = i beta Ez (found using vector modesolver).

        We assume the beta is provided unscaled, and we then non-dimensionalize
        to get beta_s = beta * self.L and k_s = self.k * self.L.  This is then
        used to form the H field.  These give the appropriate values for the
        non-dimensionalized mesh we are using.

        In order to avoid numeric difficulties, we scale the H field by the
        negative imaginary unit times the vacuum impedence 𝜂0 defined by
        𝜂0 := (𝜇0/𝜀0)^(1/2).

        This transforms Maxwell's equations to give

                    curl E = k0 H
                    curl H = k0 e_r E

        where e_r is the relative permittivity.
        """
        beta_s = beta * self.L
        k_s = self.k * self.L
        dphi_dx, dphi_dy = grad(phi)

        J_Etv = ng.CF((Etv[1], -Etv[0]))
        rot_phi = ng.CF((dphi_dy, -dphi_dx))

        Htv = -1j / (k_s * beta_s) * (rot_phi + beta_s**2 * J_Etv)
        Hz = 1 / k_s * curl(Etv)

        return Htv, Hz

    def S(self, Etv, phi, beta):
        """Return time averaged Poynting vector S = 1/2 E x H*.

        Here we again scale the H field by -1j * 𝜂0 with 𝜂0 defined by
        𝜂0 := (𝜇0/𝜀0)^(1/2).

        This transforms Maxwell's equations to give

                    curl E = k0 H
                    curl H = k0 e_r E

        where e_r is the relative permittivity.
        """
        beta_s = beta * self.L
        k_s = self.k * self.L

        J_Etv = ng.CF((Etv[1], -Etv[0]))

        # Stv = J_Etv * Conj(curl(Etv)) + phi / (k_s *
        # beta_s * conj(beta_s)) * \
        #     (Conj(grad(phi)) + conj(beta_s)**2 * Conj(Etv))

        Stv = -1j * (J_Etv * Conj(curl(Etv)) + np.abs(beta_s)**-2 * phi *
                     Conj(grad(phi)) + conj(beta_s) / beta_s * phi * Conj(Etv))

        Sz = 1 / (k_s * conj(beta_s)) * \
            (Etv * Conj(grad(phi)) + conj(beta_s)**2 * Etv.Norm()**2)

        return 1 / 2 * Stv, 1 / 2 * Sz

    # ###################################################################
    # FREQ-DEPENDENT PML BY POLYNOMIAL EIGENPROBLEM #####################

    def polypmlsystem(self, p, alpha=1):
        """
        Returns AA, X

          AA is a list of 4 cubic matrix polynomial coefficients on FE space X
        """

        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with poly.')

        dx_pml = dx(definedon=self.mesh.Materials('Outer'))
        dx_int = dx(definedon=~self.mesh.Materials('Outer'))
        R = self.R
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y) + 0j
        X = ng.H1(self.mesh, order=p, complex=True)
        u, v = X.TnT()
        ux, uy = grad(u)
        vx, vy = grad(v)

        AA = [ng.BilinearForm(X, check_unused=False)]
        AA[0] += (s * r / R) * grad(u) * grad(v) * dx_pml
        AA[0] += s * (r - R) / (R * r * r) * (x * ux + y * uy) * v * dx_pml
        AA[0] += s * (R - 2 * r) / r**3 * (x * ux + y * uy) * (x * vx +
                                                               y * vy) * dx_pml
        AA[0] += -s**3 * (r - R)**2 / (R * r) * u * v * dx_pml
        AA[0] += 0 * u * v * dx

        # I always thought sparsity in ngsolve is based on the FE space only,
        # not the integrators, but that seems to be no longer accurate today.
        # To force AA[i]'s to be of the same sparsity, I am having to add
        # this 0 * u * v * dx.

        AA += [ng.BilinearForm(X)]
        AA[1] += grad(u) * grad(v) * dx_int
        AA[1] += self.V * u * v * dx_int
        AA[1] += 2 * (r - R) / r**3 * (x * ux + y * uy) * (x * vx +
                                                           y * vy) * dx_pml
        AA[1] += 1 / r**2 * (x * ux + y * uy) * v * dx_pml
        AA[1] += -2 * s * s * (r - R) / r * u * v * dx_pml
        AA[1] += 0 * u * v * dx

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[2] += R / s / r**3 * (x * ux + y * uy) * (x * vx + y * vy) * dx_pml
        AA[2] += -R * s / r * u * v * dx_pml
        AA[2] += 0 * u * v * dx

        AA += [ng.BilinearForm(X, check_unused=False)]
        AA[3] += -u * v * dx_int
        AA[3] += 0 * u * v * dx

        with ng.TaskManager():
            for i in range(len(AA)):
                try:
                    AA[i].Assemble()
                except Exception:
                    print('*** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
                    AA[i].Assemble()
        return AA, X

    def leakymode(self,
                  p,
                  ctr=2,
                  rad=0.1,
                  alpha=1,
                  npts=8,
                  within=None,
                  rhoinv=0.0,
                  quadrule='circ_trapez_shift',
                  nspan=5,
                  seed=1,
                  inverse=None,
                  verbose=True,
                  **feastkwargs):
        """
        Solve the polynomial PML eigenproblem to compute leaky modes with
        losses [Nannen+Wess]. A custom polynomial feast uses the given
        centers and radii to search for the modes.

        PARAMETERS:

        p:        Polynomial degree of finite elements.
        alpha:    PML strength.
        nspan:    Dimension of random initial eigenspace iterate.
        seed:     Fix seed for reproducing random initial iterate.
        npts, ctrs, radi, within, rhoinv, quadrule:
                  These paramaters are passed to SpectralProjNGPoly
                  constructor. See documentation there.
        feastkwargs: Further keyword arguments passed to the feast(...)
                  method of the spectral projector. See documentation there.

        OUTPUTS:  Z, y, yl, beta, P, moreoutputs

        Z: nondimensional polynomial eigenvalue
        y: right eigenspan
        yl: left eigenspan
        beta: physical propagation constant
        P: the SpectralProjNGPoly object used to compute Z
        moreoutputs: dictionary of more outputs

        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)
        self.p = p
        AA, X = self.polypmlsystem(p=p, alpha=alpha)
        X3 = ng.FESpace([X, X, X])
        print('Set freq-dependent PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout - self.R))

        Y = NGvecs(X3, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)

        P = SpectralProjNGPoly(AA,
                               X,
                               radius=rad,
                               center=ctr,
                               npts=npts,
                               within=within,
                               rhoinv=rhoinv,
                               quadrule=quadrule,
                               verbose=verbose,
                               inverse=inverse)

        Z, Y, hist, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)
        ews, cgd = hist[-2], hist[-1]
        if not cgd:
            print('*** Iterations did not converge')

        y = P.first(Y)
        yl = P.last(Yl)
        y.centernormalize(self.mesh(0, 0))
        yl.centernormalize(self.mesh(0, 0))

        print('Results:\n Z:', Z)
        beta = self.betafrom(Z**2)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        bdrnrm = self.boundarynorm(y)
        if np.max(bdrnrm) > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')
            self.estimatepolypmldecay(Z, alpha)

        moreoutputs = {
            'longY': Y,
            'longYl': Yl,
            'ewshistory': ews,
            'bdrnorm': bdrnrm,
            'converged': cgd
        }

        return Z, y, yl, beta, P, moreoutputs

    def leakymode_poly(self,
                       p,
                       ctr=2,
                       rad=0.1,
                       alpha=1,
                       npts=8,
                       within=None,
                       rhoinv=0.0,
                       quadrule='circ_trapez_shift',
                       nspan=5,
                       seed=1,
                       inverse=None,
                       **feastkwargs):
        """
        This method is an alternate implementation of the polynomial
        eigensolver using NGSolve bilinear forms in a product finite
        element space. It has been useful sometimes in testing and
        debugging. It should give the same results as leakymode(...),
        and its arguments are as documented in leakymode(...).
        It's more expensive than leakymode(...).
        """

        print('ModeSolver.leakymode_poly called on this object:\n', self)
        print('Set freq-dependent PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout - self.R))
        self.p = p
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with poly.')

        X = ng.H1(self.mesh, order=p, complex=True)

        # This implementation of [Nannen+Wess]'s frequency-dependent PML is
        # makes a cubic eigenproblem using 3 copies of X:

        X3 = ng.FESpace([X, X, X])

        u0, u1, u2 = X3.TrialFunction()
        v0, v1, v2 = X3.TestFunction()
        u0x, u0y = grad(u0)
        u1x, u1y = grad(u1)
        u2x, u2y = grad(u2)
        v2x, v2y = grad(v2)

        pmlbegin = self.R
        dx_pml = dx(definedon=self.mesh.Materials('Outer'))
        dx_int = dx(definedon=self.mesh.Materials('core|clad'))

        R = pmlbegin
        s = 1 + 1j * alpha
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y) + 0j

        A = ng.BilinearForm(X3)
        B = ng.BilinearForm(X3)

        A += u1 * v0 * dx
        A += u2 * v1 * dx

        A += (s * r / R) * grad(u0) * grad(v2) * dx_pml
        A += s * (r - R) / (R * r * r) * (x * u0x + y * u0y) * v2 * dx_pml
        A += s * (R - 2 * r) / r**3 * (x * u0x + y * u0y) * (x * v2x +
                                                             y * v2y) * dx_pml
        A += -s**3 * (r - R)**2 / (R * r) * u0 * v2 * dx_pml

        A += grad(u1) * grad(v2) * dx_int
        A += self.V * u1 * v2 * dx_int
        A += 2 * (r - R) / r**3 * (x * u1x + y * u1y) * (x * v2x +
                                                         y * v2y) * dx_pml
        A += 1 / r**2 * (x * u1x + y * u1y) * v2 * dx_pml
        A += -2 * s * s * (r - R) / r * u1 * v2 * dx_pml

        A += R / s / r**3 * (x * u2x + y * u2y) * (x * v2x + y * v2y) * dx_pml
        A += -R * s / r * u2 * v2 * dx_pml

        B += u0 * v0 * dx + u1 * v1 * dx
        B += u2 * v2 * dx_int

        with ng.TaskManager():
            try:
                A.Assemble()
                B.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                B.Assemble()

        P = SpectralProjNG(X3,
                           A.mat,
                           B.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           inverse=inverse)
        Y = NGvecs(X3, nspan, M=B.mat)
        Yl = Y.create()
        Y.setrandom()
        Yl.setrandom()

        z, Y, history, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)

        Yg = Y.gridfun()
        Ylg = Y.gridfun()
        y = NGvecs(X, Y.m)
        yl = NGvecs(X, Y.m)
        for i in range(Y.m):
            y._mv[i].data = Yg.components[0].vecs[i]
            yl._mv[i].data = Ylg.components[0].vecs[i]
        y.centernormalize(self.mesh(0, 0))
        yl.centernormalize(self.mesh(0, 0))
        maxbdrnrm = np.max(self.boundarynorm(y))
        print('Mode boundary norm = %.1e' % maxbdrnrm)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return z, yl, y, P, Yl, Y

    # ###################################################################
    # NGSOLVE AUTOMATIC PML #############################################

    def autopmlsystem(self, p, alpha=1):
        """
        Set up PML by NGSolve's automatic PML using in-built
        mesh transformations.
        """
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')

        radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
        self.mesh.SetPML(radial, 'Outer')
        self.ngspmlset = True
        print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
              'and thickness=%.3f' % (self.Rout - self.R))
        X = ng.H1(self.mesh, order=p, complex=True)

        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (grad(u) * grad(v) + self.V * u * v) * dx
        b += u * v * dx
        with ng.TaskManager():
            try:
                a.Assemble()
                b.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                a.Assemble()
                b.Assemble()

        self.mesh.UnSetPML(definedon='Outer')
        self.ngspmlset = False

        return a, b, X

    def leakymode_auto(self,
                       p,
                       radiusZ2=0.1,
                       centerZ2=4,
                       alpha=1,
                       npts=8,
                       nspan=5,
                       seed=1,
                       within=None,
                       rhoinv=0.0,
                       verbose=True,
                       quadrule='circ_trapez_shift',
                       inverse='umfpack',
                       **feastkwargs):
        """
        Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent automatic PML mesh map of NGSolve
        and using non-selfadjoint FEAST.

        INPUTS:

        * radiusZ2, centerZ2:
            Capture modes whose non-dimensional resonance value Z²
            is such that Z*Z is contained within the circular contour
            centered at "centerZ2" of radius "radiusZ2" in the complex
            plane.
        * Remaining inputs are as documented in leakymode(..).

        OUTPUTS:   zsqr, Yr, Yl, P

        * zsqr: computed resonance values Z²
        * Yl, Yr: left and right eigenspans
        * P: spectral projector object that computed Y, Yl
        """

        print('ModeSolver.leakymode called on object with these settings:\n',
              self)
        self.p = p
        a, b, X = self.autopmlsystem(p, alpha=alpha)

        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=radiusZ2,
                           center=centerZ2,
                           checks=False,
                           npts=npts,
                           within=within,
                           rhoinv=rhoinv,
                           verbose=verbose,
                           quadrule=quadrule,
                           inverse=inverse)

        Y = NGvecs(X, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, _, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)
        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))
        maxbdrnrm = np.max(self.boundarynorm(Y))
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        return zsqr, Y, Yl, beta, P

    # ###################################################################
    # SMOOTHER HANDMADE PML #############################################

    # ###################################################################
    # # SYMBOLICS AND RESOLVENT #########################################

    def smoothpmlsymb(self, alpha, pmlbegin, pmlend):
        """
        Symbolic pml functions useful for debugging/visualization of pml.
        ---
        We compute a radial PML function φ(r) = α * φ(r) and the derived
        functions τ(r) = μ(r) = 1 + αφ(r) (called taut in the code),
        and τ_mapped(r) = r * τ(r) = r * μ(r) = η(r) = r * (1 + αφ(r)).
        Remaining terms are also computed. Notation is not unified.
        Reader is advised to consult the code for the exact meaning of
        the symbols used.
        Cf. Kim and Pasciak; Gopalakrishnan et al.
        """
        # symbolically derive the radial PML functions
        s, t, r0, r1 = sm.symbols('s t R_0 R_1')
        nr = sm.integrate((s - r0)**2 * (s - r1)**2, (s, r0, t)).factor()
        dr = nr.subs(t, r1).factor()
        phi = alpha * nr / dr  # called α * φ in the docstring
        phi = phi.subs(r0, pmlbegin).subs(r1, pmlend)
        # Remaining terms
        sigma = sm.diff(t * phi, t).factor()
        tau = 1 + 1j * sigma
        taut = 1 + 1j * phi  # called μ in the docstring
        mappedt = t * taut  # called η in the docstring
        g = (tau / taut).factor()  # this is what appears in the mapped system
        return g, mappedt, tau, taut

    def symb_to_cf(self, symb, r=None):
        """
        Convert a symbolic expression to an ngsolve coefficient function.
        If r is None, then the symbolic expression is assumed to be the radius.
        Otherwise, r is assumed to be a valid ngsolve coefficient function.
        Assumes that the symbolic expression is a function of t, and that
        the imaginary unit is I.
        """
        x = ng.x
        y = ng.y
        if r is None:
            r = ng.sqrt(x * x + y * y)
        strng = str(symb).replace('I', '1j').replace('t', 'r')
        cf = eval(strng)
        return cf

    def set_vecpml_coeff(self, alpha, pmlbegin, pmlend, **kwargs):
        """
        Set the PML coefficients. Function defined to reduce redundancy
        and improve readability.
        Adds the following attributes to the class:
        * self.detj
        * self.detj_conj
        * self.kappa
        * self.kappa_conj
        * self.gamma
        * self.gamma_conj
        Check documentation of CF.Compile for kwargs.
        Recommended realcompile=True and wait=True.
        """
        # Standard ngsolve imports
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)
        # Get symbolic functions
        _, eta_sym, _, mu_sym = self.smoothpmlsymb(alpha, pmlbegin, pmlend)
        t = sm.symbols('t')
        eta_dt = sm.diff(eta_sym, t).factor()
        mu_dt = sm.diff(mu_sym, t).factor()
        # mu_dt = sm.diff(t * mu_sym, t).factor()
        # Make coefficient functions
        mu_ = self.symb_to_cf(mu_sym)
        eta_ = self.symb_to_cf(eta_sym)
        mu_dr_ = self.symb_to_cf(mu_dt)
        eta_dr_ = self.symb_to_cf(eta_dt)
        # Main terms, after truncating at pmlbegin
        mu = ng.IfPos(r - pmlbegin, mu_, 1)
        eta = ng.IfPos(r - pmlbegin, eta_, r)
        mu_dr = ng.IfPos(r - pmlbegin, mu_dr_, 0)
        eta_dr = ng.IfPos(r - pmlbegin, eta_dr_, 1)
        # Determinant of Jacobian
        detj = mu * eta_dr
        # Jacobian, left as a reminder
        # # j00 = mu + (mu_dr / r) * x * x
        # # j01 = - (mu_dr / r) * x * y
        # # j11 = mu + (mu_dr / r) * y * y
        # Inverse of Jacobian
        jinv00 = 1 / eta_dr + (mu_dr / (eta_dr * eta)) * y * y
        jinv01 = -(mu_dr / (eta_dr * eta)) * x * y
        jinv11 = 1 / eta_dr + (mu_dr / (eta_dr * eta)) * x * x
        # Conjugate the main terms
        mu_conj = ng.Conj(mu)
        eta_conj = ng.Conj(eta)
        mu_dr_conj = ng.Conj(mu_dr)
        eta_dr_conj = ng.Conj(eta_dr)
        detj_conj = ng.Conj(detj)
        jinv00_conj = ng.Conj(jinv00)
        jinv01_conj = ng.Conj(jinv01)
        jinv11_conj = ng.Conj(jinv11)
        # Compile into coefficient functions
        # Only compile the main terms
        jinv00.Compile(**kwargs)
        jinv01.Compile(**kwargs)
        jinv11.Compile(**kwargs)
        detj.Compile(**kwargs)
        detj_conj.Compile(**kwargs)
        jinv00_conj.Compile(**kwargs)
        jinv01_conj.Compile(**kwargs)
        jinv11_conj.Compile(**kwargs)
        # Construct jacobians
        # jac = ng.CoefficientFunction((j00, j01, j01, j11), dims=(2, 2))
        jacinv = ng.CoefficientFunction((jinv00, jinv01, jinv01, jinv11),
                                        dims=(2, 2))
        jacinv_conj = ng.CoefficientFunction(
            (jinv00_conj, jinv01_conj, jinv01_conj, jinv11_conj), dims=(2, 2))
        # Construct gamma, gamma_conj, kappa, kappa_conj
        gamma = detj * (jacinv * jacinv)
        gamma_conj = detj_conj * (jacinv_conj * jacinv_conj)
        kappa = (mu / eta_dr**3) * (1 + (mu_dr * r**2) / eta)**2
        kappa_conj = (mu_conj / eta_dr_conj**3) * \
            (1 + (mu_dr_conj * r**2) / eta_conj)**2

        # Adding  terms to the class as needed
        if self.gamma is not None:
            raise RuntimeError('PML coefficients already set.'
                               ' Check code logic.')
        gamma.Compile(**kwargs)
        gamma_conj.Compile(**kwargs)
        kappa.Compile(**kwargs)
        kappa_conj.Compile(**kwargs)
        # Set the coefficients
        setattr(self, 'detj', detj)
        setattr(self, 'detj_conj', detj_conj)
        setattr(self, 'kappa', kappa)
        setattr(self, 'kappa_conj', kappa_conj)
        setattr(self, 'gamma', gamma)
        setattr(self, 'gamma_conj', gamma_conj)

    def make_resolvent_maxwell(self,
                               m,
                               a,
                               b,
                               c,
                               d,
                               X,
                               Y,
                               inverse='umfpack',
                               autoupdate=True):
        """
        Create resolvent for Maxwell's equations.
        INPUTS:
        * a, b, c, d: bilinear forms of the system, blockwise form
        * X, Y: FE spaces
        * inverse: inverse type
        * autoupdate: ngsolve autoupdate
        OUTPUTS:
        * ResolventVectorMode: resolvent
        * dinv: inverse of d
        """
        print('ModeSolver.make_resolvent_maxwell called...\n')
        # Retrive coefficient functions
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)

        mu = self.mu
        mu_dr = self.mu_dr
        eta = self.eta
        eta_dr = self.eta_dr
        detj = self.detj
        jacinv = self.jacinv

        # Create inverse of d
        dinv = d.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)

        # resolvent class definition begins here ------------------------------
        class ResolventVectorMode():
            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY,
                                   name='wrk1',
                                   autoupdate=autoupdate,
                                   nested=autoupdate)
            wrk2 = ng.GridFunction(XY,
                                   name='wrk2',
                                   autoupdate=autoupdate,
                                   nested=autoupdate)
            tmpY1 = ng.GridFunction(Y,
                                    name='tmpY1',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)
            tmpY2 = ng.GridFunction(Y,
                                    name='tmpY2',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)
            tmpX1 = ng.GridFunction(X,
                                    name='tmpX1',
                                    autoupdate=autoupdate,
                                    nested=autoupdate)

            def __init__(selfr, z, V, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])

                (E, phi), (F, psi) = XY.TnT()

                selfr.Z = ng.BilinearForm(XY, condense=True)
                selfr.ZH = ng.BilinearForm(XY, condense=True)

                # m - a - c - b + (-d)
                selfr.Z += (z * detj * (jacinv * E) * (jacinv * F) -
                            (mu / eta_dr**3) *
                            (1 + (mu_dr * r**2) / eta)**2 * curl(E) * curl(F) -
                            V * detj * (jacinv * E) * (jacinv * F) - detj *
                            (jacinv * grad(phi)) * (jacinv * F) - n2 * detj *
                            (jacinv * E) *
                            (jacinv * grad(psi)) + n2 * detj * phi * psi) * dx
                selfr.ZH += (np.conjugate(z) * detj * (jacinv * F) *
                             (jacinv * E) - (mu / eta_dr**3) *
                             (1 +
                              (mu_dr * r**2) / eta)**2 * curl(F) * curl(E) -
                             V * detj * (jacinv * F) *
                             (jacinv * E) - n2 * detj * (jacinv * F) *
                             (jacinv * grad(phi)) - detj *
                             (jacinv * grad(psi)) *
                             (jacinv * E) + n2 * detj * phi * psi) * dx

                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(
                        selfr.XY.FreeDofs(coupling=True), inverse=inverse)

            def act(selfr, v, Rv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]

                with ng.TaskManager():
                    Mv[:] = m.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.Z.harmonic_extension_trans * \
                            selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.harmonic_extension * selfr.wrk2.vec

                        Rv._mv[i][:] = selfr.wrk2.components[0].vec

            def adj(selfr, v, RHv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]
                with ng.TaskManager():
                    Mv[:] = m.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.ZH.harmonic_extension_trans * \
                            selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I.H * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.harmonic_extension * selfr.wrk2.vec

                        RHv._mv[i][:] = selfr.wrk2.components[0].vec

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
                             workspace=None):
                """
                Return qAq[i, j] = (𝒜 qr[j], ql[i]) with
                𝒜 =  (A - C D⁻¹ B) E
                and qBq[i, j] = (M qr[j], ql[i]).
                """
                if workspace is None:
                    Aqr = ng.MultiVector(qr._mv[0], qr.m)
                else:
                    Aqr = workspace._mv[:qr.m]

                with ng.TaskManager():
                    if qAq is not None:
                        Aqr[:] = a.mat * qr._mv
                        for i in range(qr.m):
                            # TODO: Static condensation caused issues when
                            #       not using TaskManager
                            selfr.tmpY1.vec.data = b.mat * qr._mv[i]
                            selfr.tmpY1.vec.data += \
                                d.harmonic_extension_trans * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data = dinv * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                d.inner_solve * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                d.harmonic_extension * selfr.tmpY2.vec

                            selfr.tmpX1.vec.data = c.mat * selfr.tmpY2.vec
                            Aqr[i].data -= selfr.tmpX1.vec
                        qAq = ng.InnerProduct(Aqr, ql._mv).NumPy().T

                    if qBq is not None:
                        Bqr = Aqr
                        Bqr[:] = m.mat * qr._mv
                        qBq = ng.InnerProduct(Bqr, ql._mv).NumPy().T

                return (qAq, qBq)

            def rayleigh(selfr, q, workspace=None):
                return selfr.rayleigh_nsa(q, q, workspace=workspace)

            def update_system(selfr, verbose=False):
                """
                Update the system matrices.
                For internal use only.
                This should be redundant with the autoupdate feature of
                the GridFunction objects and recreating the resolvent
                object should not be necessary.
                Consider removing this method and simplyfing __init__.
                """
                warn(
                    'This should be redundant with the autoupdate feature of'
                    ' the GridFunction objects and recreating the resolvent'
                    ' object should not be necessary.',
                    PendingDeprecationWarning)
                if verbose:
                    print('Updating system matrices...\n')
                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        if verbose:
                            print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(
                        selfr.XY.FreeDofs(coupling=True),
                        inverse=selfr.inverse)

        # end of class ResolventVectorMode --------------------------------

        return ResolventVectorMode, dinv

    # ###################################################################
    # # PML SYSTEMS #####################################################

    def smoothpmlsystem(self,
                        p,
                        alpha=1,
                        pmlbegin=None,
                        pmlend=None,
                        autoupdate=False):
        """
        Make the matrices needed for formulating the leaky mode
        eigensystem with frequency-independent C² PML map
            mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r.
        """

        print('ModeSolver.leakymode_smooth called on:\n', self)
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with smooth.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout

        G, mappedt, tau, taut = self.smoothpmlsymb(alpha, pmlbegin, pmlend)

        # symbolic -> ngsolve coefficient
        x = ng.x
        y = ng.y
        r = ng.sqrt(x * x + y * y)
        gstr = str(G).replace('I', '1j').replace('t', 'r')
        ttstr = str(tau * taut).replace('I', '1j').replace('t', 'r')
        self.ttstr = ttstr
        self.taut = str(taut).replace('I', '1j').replace('t', 'r')
        self.mappedr = str(mappedt).replace('I', '1j').replace('t', 'r')
        g0 = eval(gstr)
        tt0 = eval(ttstr)
        g = ng.IfPos(r - pmlbegin, g0, 1)
        tt = ng.IfPos(r - pmlbegin, tt0, 1)

        gi = 1.0 / g
        cs = x / r
        sn = y / r
        A00 = gi * cs * cs + g * sn * sn
        A01 = (gi - g) * cs * sn
        A11 = gi * sn * sn + g * cs * cs
        g.Compile()
        gi.Compile()
        tt.Compile()
        A00.Compile()
        A01.Compile()
        A11.Compile()
        A = ng.CoefficientFunction((A00, A01, A01, A11), dims=(2, 2))
        self.pml_A = A
        self.pml_B = tt

        # Make linear eigensystem
        X = ng.H1(self.mesh, order=p, complex=True, autoupdate=autoupdate)
        u, v = X.TnT()
        a = ng.BilinearForm(X)
        b = ng.BilinearForm(X)
        a += (self.pml_A * grad(u) * grad(v) +
              self.V * self.pml_B * u * v) * dx
        b += self.pml_B * u * v * dx

        with ng.TaskManager():
            try:
                a.Assemble()
                b.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                a.Assemble()
                b.Assemble()

        return a, b, X

    def smoothvecpmlsystem_compound(self,
                                    p,
                                    alpha=1,
                                    pmlbegin=None,
                                    pmlend=None,
                                    deg=2,
                                    autoupdate=True):
        """
        Make the matrices needed for formulating the vector
        leaky mode eigensystem with frequency-independent
        C² PML map mapped_x = x * (1 + 1j * α * φ(r)) where
        φ is a C² function of the radius r.
        Using the compound finite element space X*Y.
        INPUTS:
        * p: polynomial degree of finite elements
        * alpha: PML strength
        * pmlbegin: radius where PML begins
        * pmlend: radius where PML ends
        * deg: degree of the PML polynomial
        * autoupdate: whether to use autoupdate in NGSolve
        OUTPUTS:
        * aa: bilinear form for the LHS
        * mm: bilinear form for the RHS
        * Z: finite element space
        """
        if self.ngspmlset:
            raise RuntimeError(
                'NGSolve PML set. Cannot combine with smooth PML.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout
        if self.gamma is None:
            self.set_vecpml_coeff(alpha, pmlbegin, pmlend, maxderiv=3)
        self.p = p

        # Get symbolic functions
        detj = self.detj
        kappa = self.kappa
        gamma = self.gamma

        # Make linear eigensystem, cf. self.vecmodesystem
        n2 = self.index * self.index
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True,
                     autoupdate=autoupdate)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True,
                  autoupdate=autoupdate)

        Z = X * Y
        (E, phi), (F, psi) = Z.TnT()

        aa = ng.BilinearForm(Z)
        mm = ng.BilinearForm(Z)

        aa += ((kappa * curl(E)) * curl(F) + self.V * (gamma * E) * F + n2 *
               (gamma * E) * grad(psi) +
               (gamma * grad(phi)) * F - n2 * detj * phi * psi) * dx
        mm += (gamma * E) * F * dx

        with ng.TaskManager():
            try:
                aa.Assemble()
                mm.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                aa.Assemble()
                mm.Assemble()

        return aa, mm, Z

    def smoothvecpmlsystem_resolvent(self,
                                     p,
                                     alpha=1,
                                     pmlbegin=None,
                                     pmlend=None,
                                     deg=2,
                                     inverse='umfpack',
                                     autoupdate=True):
        """
        Make the matrices needed for formulating the vector
        leaky mode eigensystem with frequency-independent
        C² PML map mapped_x = x * (1 + 1j * α * φ(r)) where
        φ is a C² function of the radius r.
        Using the resolvent T = A - C * D⁻¹ * B.
        INPUTS:
        * p: polynomial degree of finite elements
        * alpha: PML strength
        * pmlbegin: radius where PML begins
        * pmlend: radius where PML ends
        * deg: degree of the PML polynomial
        * inverse: inverse method to use in spectral projector
        * autoupdate: whether to use autoupdate in NGSolve
        OUTPUTS:
        * ResolventVectorMode: resolvent
        * m: bilinear form for the LHS
        * a, b, c, d: bilinear forms for the block matrices for the RHS
        * dinv: inverse of d
        """
        print('ModeSolver.smoothvecpmlsystem_resolvent called...\n')
        # raise NotImplementedError('This is not working yet.')
        if self.ngspmlset:
            raise RuntimeError('NGSolve pml set. Cannot combine with smooth.')
        if abs(alpha.imag) > 0 or alpha < 0:
            raise ValueError('Expecting PML strength alpha > 0')
        if pmlbegin is None:
            pmlbegin = self.R
        if pmlend is None:
            pmlend = self.Rout

        if self.gamma is None:
            self.set_vecpml_coeff(alpha, pmlbegin, pmlend, maxderiv=3)

        # Get symbolic functions
        detj = self.detj
        kappa = self.kappa
        gamma = self.gamma

        # Make linear eigensystem, cf. self.vecmodesystem
        n2 = self.index * self.index
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True,
                     autoupdate=autoupdate)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True,
                  autoupdate=autoupdate)

        E, F = X.TnT()
        phi, psi = Y.TnT()

        m = ng.BilinearForm(X)
        a = ng.BilinearForm(X)
        c = ng.BilinearForm(trialspace=Y, testspace=X)
        b = ng.BilinearForm(trialspace=X, testspace=Y)
        # d = ng.BilinearForm(Y)
        d = ng.BilinearForm(Y, condense=True)

        m += (gamma * E) * F * dx
        a += ((kappa * curl(E)) * curl(F) + self.V * (gamma * E) * F) * dx
        c += (gamma * grad(phi)) * F * dx
        b += n2 * (gamma * E) * grad(psi) * dx
        d += -n2 * detj * phi * psi * dx

        with ng.TaskManager():
            try:
                m.Assemble()
                a.Assemble()
                c.Assemble()
                b.Assemble()
                d.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                m.Assemble()
                a.Assemble()
                c.Assemble()
                b.Assemble()
                d.Assemble()
            res, dinv = self.make_resolvent_maxwell(m,
                                                    a,
                                                    b,
                                                    c,
                                                    d,
                                                    X,
                                                    Y,
                                                    inverse=inverse,
                                                    autoupdate=autoupdate)

        return res, m, a, b, c, d, dinv

    # ###################################################################
    # # LEAKY MODES #####################################################

    def leakymode_smooth(self,
                         p,
                         radiusZ2=0.1,
                         centerZ2=4,
                         pmlbegin=None,
                         pmlend=None,
                         alpha=1,
                         npts=8,
                         nspan=5,
                         seed=1,
                         within=None,
                         rhoinv=0.0,
                         quadrule='circ_trapez_shift',
                         inverse='umfpack',
                         verbose=True,
                         **feastkwargs):
        """
        Compute leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Then a non-selfadjoint FEAST is run on the system.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """

        a, b, X = self.smoothpmlsystem(p,
                                       alpha=alpha,
                                       pmlbegin=pmlbegin,
                                       pmlend=pmlend)
        # OMIT m computation

        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=radiusZ2,
                           center=centerZ2,
                           npts=npts,
                           checks=False,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           verbose=verbose,
                           inverse=inverse)

        Y = NGvecs(X, nspan)
        Yl = NGvecs(X, nspan)
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        zsqr, Y, history, Yl = P.feast(Y,
                                       Yl=Yl,
                                       hermitian=False,
                                       **feastkwargs)
        ewhist, cgd = history[-2], history[-1]

        beta = self.betafrom(zsqr)
        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        bdrnrm = self.boundarynorm(Y)
        if np.max(bdrnrm) > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': bdrnrm,
            'converged': cgd
        }

        return zsqr, Y, Yl, beta, P, moreoutputs

    def leakyvecmodes_smooth_compound(self,
                                      p=None,
                                      radius=None,
                                      center=None,
                                      pmlbegin=None,
                                      pmlend=None,
                                      alpha=None,
                                      npts=None,
                                      nspan=None,
                                      seed=1,
                                      within=None,
                                      rhoinv=0.0,
                                      quadrule='circ_trapez_shift',
                                      inverse='umfpack',
                                      verbose=True,
                                      **feastkwargs):
        """
        Compute vector leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Using the compound finite element space X*Y and compound
        bilinear forms.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """
        print('ModeSolver.leakyvecmodes_smooth_compound called on:\n', self)
        # Check validity of inputs
        if p is None or radius is None or center is None:
            raise ValueError('Missing input(s)')
        # Get compound system
        aa, mm, Z = self.smoothvecpmlsystem_compound(p,
                                                     alpha=alpha,
                                                     pmlbegin=pmlbegin,
                                                     pmlend=pmlend,
                                                     deg=2,
                                                     autoupdate=True)
        # Create spectral projector
        P = SpectralProjNG(Z,
                           aa.mat,
                           mm.mat,
                           radius=radius,
                           center=center,
                           npts=npts,
                           checks=False,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           verbose=verbose,
                           inverse=inverse)

        # Set up NGvecs
        E_phi_r = NGvecs(Z, nspan)
        E_phi_l = NGvecs(Z, nspan)
        E_phi_r.setrandom(seed=seed)
        E_phi_l.setrandom(seed=seed)

        # Use FEAST
        zsqr, E_phi_r, history, E_phi_l = P.feast(E_phi_r,
                                                  Yl=E_phi_l,
                                                  hermitian=False,
                                                  **feastkwargs)

        # Compute betas, extract relevant variables
        ewhist, cgd = history[-2], history[-1]
        beta = self.betafrom(zsqr)

        print('Results:\n Z²:', zsqr)
        print(' beta:', beta)
        print(' CL dB/m:', 20 * beta.imag / np.log(10))

        # Unpack E_phi_r, E_phi_l into E_r, E_l, phi_r, phi_l
        X, Y = Z.components
        E_r = NGvecs(X, E_phi_r.m)
        E_l = NGvecs(X, E_phi_l.m)
        phi_r = NGvecs(Y, E_phi_r.m)
        phi_l = NGvecs(Y, E_phi_l.m)

        for i in range(E_phi_r.m):
            E_r._mv[i].data = E_phi_r[i].components[0].vec.data
            E_l._mv[i].data = E_phi_l[i].components[0].vec.data
            phi_r._mv[i].data = E_phi_r[i].components[1].vec.data
            phi_l._mv[i].data = E_phi_l[i].components[1].vec.data

        maxbdrnrm_r = np.max(self.boundarynorm(E_r))
        maxbdrnrm_l = np.max(self.boundarynorm(E_l))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': maxbdrnrm,
            'converged': cgd,
        }

        return zsqr, E_r, E_l, phi_r, phi_l, beta, P, moreoutputs

    def leakyvecmodes_smooth_resolvent(self,
                                       p=None,
                                       radius=None,
                                       center=None,
                                       pmlbegin=None,
                                       pmlend=None,
                                       alpha=None,
                                       npts=None,
                                       nspan=None,
                                       seed=1,
                                       within=None,
                                       rhoinv=0.0,
                                       quadrule='circ_trapez_shift',
                                       inverse='umfpack',
                                       verbose=True,
                                       **feastkwargs):
        """
        Compute vector leaky modes by solving a linear eigenproblem using
        the frequency-independent C²  PML map
           mapped_x = x * (1 + 1j * α * φ(r))
        where φ is a C² function of the radius r. The coefficients of
        the mapped eigenproblem are used to make the eigensystem.
        Using the resolvent T = A - C * D⁻¹ * B.

        Inputs and outputs are as documented in leakymode_auto(...). The
        only difference is that here you may override the starting and
        ending radius of PML by providing pmlbegin, pmlend.
        """
        print('ModeSolver.leakyvecmodes_smooth_resolvent called on:\n', self)
        # Check validity of inputs
        if p is None or radius is None or center is None:
            raise ValueError('Missing input(s)')
        # Get compound system
        res, m, a, b, c, d, dinv = self.smoothvecpmlsystem_resolvent(
            p,
            alpha=alpha,
            pmlbegin=pmlbegin,
            pmlend=pmlend,
            deg=2,
            inverse=inverse,
            autoupdate=True)

        # Create spectral projector
        P = SpectralProjNGR(
            lambda z: res(z, self.V, self.index, inverse=inverse),
            radius=radius,
            center=center,
            npts=npts,
            checks=False,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose,
            inverse=inverse)

        # Unpack spaces from resolvent
        X, Y = res.XY.components
        # Set up NGvecs
        E_r = NGvecs(X, nspan, M=m)
        E_l = NGvecs(X, nspan, M=m)
        E_r.setrandom(seed=seed)
        E_l.setrandom(seed=seed)

        print('Using FEAST to search for vector leaky modes in')
        print(f'circle of radius {radius} centered at {center}')
        print(f'assuming not more than {nspan} modes in this interval')
        print(f'System size: {E_r.n} x {E_r.n}  Inverse type: {inverse}')

        # Use FEAST
        zsqr, E_r, history, E_l = P.feast(E_r,
                                          Yl=E_l,
                                          hermitian=False,
                                          **feastkwargs)

        # Compute betas, extract relevant variables
        ewhist, cgd = history[-2], history[-1]
        beta = self.betafrom(zsqr)

        print(f'Results:\n\tZ²: {zsqr}')
        print(f'\tbeta: {beta}')
        print(f'\tCL dB/m: {20 * beta.imag / np.log(10)}')

        maxbdrnrm_r = np.max(self.boundarynorm(E_r))
        maxbdrnrm_l = np.max(self.boundarynorm(E_l))
        maxbdrnrm = max(maxbdrnrm_r, maxbdrnrm_l)
        if maxbdrnrm > 1e-6:
            print('*** Mode boundary L2 norm > 1e-6!')

        moreoutputs = {
            'ewshistory': ewhist,
            'bdrnorm': maxbdrnrm,
            'converged': cgd,
        }

        # TODO Compute phi_r, phi_l

        return zsqr, E_r, E_l, beta, P, moreoutputs

    # ###################################################################
    # GUIDED LP MODES FROM SELFADJOINT EIGENPROBLEM #####################

    def selfadjsystem(self, p):

        if self.ngspmlset:
            raise RuntimeError('NGSolve pml mesh trafo set.')

        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()
        A = ng.BilinearForm(X)
        A += grad(u) * grad(v) * dx + self.V * u * v * dx
        B = ng.BilinearForm(X)
        B += u * v * dx

        with ng.TaskManager():
            try:
                A.Assemble()
                B.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                B.Assemble()

        return A, B, X

    def selfadjmodes(self,
                     interval=(-10, 0),
                     p=3,
                     seed=1,
                     npts=20,
                     nspan=15,
                     within=None,
                     rhoinv=0.0,
                     quadrule='circ_trapez_shift',
                     verbose=True,
                     inverse='umfpack',
                     **feastkwargs):
        """
        Search for guided modes in a given "interval", which is to be
        input as a tuple: interval=(left, right). These modes solve

        -Δu + V u = Z² u

        with zero dirichlet boundary conditions (no PML, no loss) at the
        outer boundary of the computational domain.

        The computation is done using Lagrangre finite elements of degree "p"
        (with no PML) using selfadjoint FEAST with a random span of "nspan"
        vectors (and using the remaining parameters, which are simply
        passed to feast).

        OUTPUTS:

        betas, Zsqrs, Y:
            betas[i] give the i-th real-valued propagation constant, and
            Zsqrs[i] gives the feast-computed i-th nondimensional Z² value
            in "interval". The corresponding eigenmode is i-th component
            of the span object Y.

        """

        a, b, X = self.selfadjsystem(p)
        left, right = interval
        print('Running selfadjoint FEAST to capture guided modes in ' +
              '({},{})'.format(left, right))
        print('assuming not more than nspan=%d modes in this interval' % nspan)
        ctr = (right + left) / 2
        rad = (right - left) / 2
        P = SpectralProjNG(X,
                           a.mat,
                           b.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           reduce_sym=True,
                           within=within,
                           rhoinv=rhoinv,
                           quadrule=quadrule,
                           inverse=inverse,
                           verbose=verbose)
        Y = NGvecs(X, nspan, M=b.mat)
        Y.setrandom(seed=seed)
        Zsqrs, Y, history, _ = P.feast(Y, hermitian=True, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        return betas, Zsqrs, Y

    # ###################################################################
    # VECTOR MODES

    def vecmodesystem(self, p, alpha=None, inverse=None):
        """
        Prepare eigensystem and resolvents for solving for vector modes.

        INPUTS:

        p: Determines degree of Nedelec x Lagrange space system.
           This should be an integer >= 0.

        alpha: If alpha is None, prepare system for vector guided modes.
           If alpha is a positive number, use it as PML strength and
           prepare system for leaky modes using NGSolve's automatic
           mesh-based PML.
        """

        if alpha is not None:
            self.ngspmlset = True
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))
        elif self.ngspmlset:
            raise RuntimeError('Unexpected NGSolve pml mesh trafo here.')

        n = self.index
        n2 = n * n
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True)
        E, v = X.TnT()
        phi, psi = Y.TnT()

        A = ng.BilinearForm(X)
        A += (curl(E) * curl(v) + self.V * E * v) * dx
        M = ng.BilinearForm(X)
        M += E * v * dx
        C = ng.BilinearForm(trialspace=Y, testspace=X)
        C += grad(phi) * v * dx
        B = ng.BilinearForm(trialspace=X, testspace=Y)
        B += -n2 * E * grad(psi) * dx
        D = ng.BilinearForm(Y, condense=True)
        D += n2 * phi * psi * dx

        with ng.TaskManager():
            try:
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            # Dinv = D.mat.Inverse(Y.FreeDofs(), inverse=inverse)
            Dinv = D.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)

        # resolvent of the vector mode problem --------------------------
        class ResolventVectorMode():

            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY)
            wrk2 = ng.GridFunction(XY)
            tmpY1 = ng.GridFunction(Y)
            tmpY2 = ng.GridFunction(Y)
            tmpX1 = ng.GridFunction(X)

            def __init__(selfr, z, V, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])
                (E, phi), (v, psi) = XY.TnT()

                # selfr.zminusOp = ng.BilinearForm(XY)
                # selfr.zminusOp += (z * E * v - curl(E) * curl(v)
                #                    - V * E * v - grad(phi) * v
                #                    - n2 * phi * psi + n2 * E * grad(psi))
                # * dx
                # with ng.TaskManager():
                #     try:
                #         selfr.zminusOp.Assemble()
                #     except Exception:
                #         print('*** Trying again with larger heap')
                #         ng.SetHeapSize(int(1e9))
                #         selfr.zminusOp.Assemble()
                #     selfr.R = selfr.zminusOp.mat.Inverse(XY.FreeDofs(),
                #                                          inverse=inverse)

                selfr.Z = ng.BilinearForm(XY, condense=True)
                selfr.Z += (z * E * v - curl(E) * curl(v) - V * E * v -
                            grad(phi) * v - n2 * phi * psi +
                            n2 * E * grad(psi)) * dx
                selfr.ZH = ng.BilinearForm(XY, condense=True)
                selfr.ZH += (np.conjugate(z) * E * v - curl(E) * curl(v) -
                             V * E * v - grad(psi) * E - n2 * phi * psi +
                             n2 * v * grad(phi)) * dx
                with ng.TaskManager():
                    try:
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.Z.Assemble()
                        selfr.ZH.Assemble()
                    selfr.R_I = selfr.Z.mat.Inverse(XY.FreeDofs(coupling=True),
                                                    inverse=inverse)

            def act(selfr, v, Rv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]

                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.Z.harmonic_extension_trans * selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.Z.harmonic_extension * selfr.wrk2.vec

                        Rv._mv[i][:] = selfr.wrk2.components[0].vec

            def adj(selfr, v, RHv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]
                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0

                        # selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec

                        selfr.wrk1.vec.data += \
                            selfr.ZH.harmonic_extension_trans * selfr.wrk1.vec
                        selfr.wrk2.vec.data = selfr.R_I.H * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.inner_solve * selfr.wrk1.vec
                        selfr.wrk2.vec.data += \
                            selfr.ZH.harmonic_extension * selfr.wrk2.vec

                        RHv._mv[i][:] = selfr.wrk2.components[0].vec

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
                             workspace=None):
                """
                Return qAq[i, j] = (𝒜 qr[j], ql[i]) with 𝒜 =  (A - C D⁻¹ B) E
                and qBq[i, j] = (M qr[j], ql[i]). """

                if workspace is None:
                    Aqr = ng.MultiVector(qr._mv[0], qr.m)
                else:
                    Aqr = workspace._mv[:qr.m]

                with ng.TaskManager():
                    if qAq is not None:
                        Aqr[:] = A.mat * qr._mv
                        for i in range(qr.m):
                            selfr.tmpY1.vec.data = B.mat * qr._mv[i]
                            # selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec

                            selfr.tmpY1.vec.data += \
                                D.harmonic_extension_trans * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                D.inner_solve * selfr.tmpY1.vec
                            selfr.tmpY2.vec.data += \
                                D.harmonic_extension * selfr.tmpY2.vec

                            selfr.tmpX1.vec.data = C.mat * selfr.tmpY2.vec
                            Aqr[i].data -= selfr.tmpX1.vec
                        qAq = ng.InnerProduct(Aqr, ql._mv).NumPy().T

                    if qBq is not None:
                        Bqr = Aqr
                        Bqr[:] = M.mat * qr._mv
                        qBq = ng.InnerProduct(Bqr, ql._mv).NumPy().T

                return (qAq, qBq)

            def rayleigh(selfr, q, workspace=None):
                return selfr.rayleigh_nsa(q, q, workspace=workspace)

        # resolvent class definition done -------------------------------

        return ResolventVectorMode, M.mat, A.mat, B.mat, C.mat, D, Dinv

    def guidedvecmodes(self,
                       rad,
                       ctr,
                       p=3,
                       seed=None,
                       npts=8,
                       nspan=20,
                       within=None,
                       rhoinv=0.0,
                       quadrule='circ_trapez_shift',
                       verbose=True,
                       inverse='umfpack',
                       **feastkwargs):
        """
        Capture guided vector modes whose non-dimensional resonance value Z²
        is such that Z*Z is within the interval (ctr-rad, ctr+rad).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p, inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        E.setrandom(seed=seed)

        print('Using FEAST to search for vector guided modes in')
        print(f'circle of radius {rad} centered at {ctr}')
        print(f'assuming not more than {nspan} modes in this interval.')
        print(f'System size: {E.n} x {E.n}  Inverse type: {inverse}')

        P = SpectralProjNGR(
            lambda z: R(z, self.V, self.index, inverse=inverse),
            radius=rad,
            center=ctr,
            npts=npts,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose)
        Zsqrs, E, history, _ = P.feast(E, **feastkwargs)
        betas = self.betafrom(Zsqrs)

        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv

        BE._mv[:] += D.harmonic_extension_trans * BE._mv
        phi._mv[:] = Dinv * BE._mv
        phi._mv[:] += D.inner_solve * BE._mv
        phi._mv[:] += D.harmonic_extension * phi._mv

        return betas, Zsqrs, E, phi, R

    def leakyvecmodes(self,
                      rad,
                      ctr,
                      alpha=1,
                      p=3,
                      seed=1,
                      npts=8,
                      nspan=20,
                      within=None,
                      rhoinv=0.0,
                      quadrule='circ_trapez_shift',
                      verbose=True,
                      inverse='umfpack',
                      **feastkwargs):
        """
        Capture leaky vector modes whose non-dimensional resonance value Z²
        is contained  within the circular contour centered at "ctr"
        of radius "rad" in the Z² complex plane (not the Z-plane!).
        """

        R, M, A, B, C, D, Dinv = self.vecmodesystem(p,
                                                    alpha=alpha,
                                                    inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        El = E.create()
        E.setrandom(seed=seed)
        El.setrandom(seed=seed)

        print('Using FEAST to search for vector leaky modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(
            lambda z: R(z, self.V, self.index, inverse=inverse),
            radius=rad,
            center=ctr,
            npts=npts,
            within=within,
            rhoinv=rhoinv,
            quadrule=quadrule,
            verbose=verbose)

        Zsqrs, E, history, El = P.feast(E,
                                        Yl=El,
                                        hermitian=False,
                                        **feastkwargs)
        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv

        BE._mv[:] += D.harmonic_extension_trans * BE._mv
        phi._mv[:] = Dinv * BE._mv
        phi._mv[:] += D.inner_solve * BE._mv
        phi._mv[:] += D.harmonic_extension * phi._mv

        betas = self.betafrom(Zsqrs)
        print('Results:\n Z²:', Zsqrs)
        print(' beta:', betas)
        print(' CL dB/m:', 20 * betas.imag / np.log(10))

        return betas, Zsqrs, E, phi, R

    # ###################################################################
    # BENT MODES

    def guidedhelicalmodes(self,
                           a,
                           b,
                           center,
                           radius,
                           p=4,
                           npts=4,
                           nspan=6,
                           seed=1,
                           verbose=True,
                           **feastkwargs):
        """
        Find scalar (Helmholtz) guided modes propagating through a
        helically coiled fiber following the theory in the paper
        [Gopalakrishnan & Neunteufel, Guided modes of helical waveguides,
        Wave Motion, 2025. https://doi.org/10.1016/j.wavemoti.2025.103621]

        PARAMETERS
        ----------
        a: radius of the helix (bend radius) in meters

        b: pitch of the helix (can be 0) in meters

        center, radius: of circle in complex plane to search for eigenvalues

        p: Lagrange finite element degree

        npts, nspan, feastkwargs: number of quadrature points, intial span
        dimension, and further keyword arguments to pass to feast eigensolver.
        """

        A, B, C, X = self.guidedhelicalsystem(a, b, p=p)
        P = SpectralProjNGPoly([A, B, C],
                               X,
                               radius=radius,
                               center=center,
                               npts=npts,
                               within=None,
                               rhoinv=0.0,
                               quadrule="circ_trapez_shift",
                               verbose=verbose,
                               checks=False)
        Y = NGvecs(X**2, nspan)
        Yl = Y.create()
        Y.setrandom(seed=seed)
        Yl.setrandom(seed=seed)
        ews, Y, hist, Yl = P.feast(Y, Yl=Yl, hermitian=False, **feastkwargs)
        if not hist[-1]:
            warn('*** Feast iterations did not converge')
        y = P.first(Y)
        yl = P.last(Yl)

        bdrnrm = self.boundarynorm(y)
        if np.max(bdrnrm) > 1e-6:
            warn('*** Mode boundary L2 norm > 1e-6!')

        print('Results:\n ews:', ews)

        return ews, y, yl, P

    def guidedhelicalsystem(self, a, b, p=4):
        """
        Output A, B, C operators on finite element space X so that the
        guided helical mode u is an eigenfuntion of the quadratic eigenvalue
        problem (A + β B + β² C) u = 0  in X.
        """
        if self.ngspmlset:
            raise RuntimeError('Do not use with ngsolve pml.')

        ll = sqrt(a**2 + b**2)
        T = 1 / ll * CF((
            -a * sin(ng.z / ll),  # tangent of helical centerline
            a * cos(ng.z / ll),
            b))
        N = -CF((
            cos(ng.z / ll),  # normal of helical fiber centerline
            sin(ng.z / ll),
            0))
        B = ng.Cross(T, N)  # binormal vector of fiber centerline

        gamma = CF((
            a * cos(ng.z / ll),  # parameterization of helix curve
            a * sin(ng.z / ll),
            b * ng.z / ll))

        Phi = gamma + ng.x * N + ng.y * B  # parameterization of helix pipe

        # Jacobian of untwisting map
        F = CF((N, B, Phi.Diff(ng.z)), dims=(3, 3)).trans
        C_inv = ng.Inv(F.trans * F)
        d = C_inv * CF((0, 0, 1))
        J = ng.Det(F)

        X = ng.H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)
        u, v = X.TnT()

        with ng.TaskManager():

            A = ng.BilinearForm(X)
            A += (J * (C_inv[:2, :2] * grad(u)) * grad(v) -
                  J * self.k**2 * self.index**2 * u * v) * dx(bonus_intorder=5)
            A.Assemble()
            B = ng.BilinearForm(X)
            B += J * 1j * (u * d[:2] * grad(v) -
                           v * d[:2] * grad(u)) * dx(bonus_intorder=5)
            B.Assemble()
            C = ng.BilinearForm(1 / J * u * v * dx(bonus_intorder=5))
            C.Assemble()

        return A, B, C, X

    # OLD STUFF BELOW: SLATED FOR REMOVAL !

    def bentscalarmodes(self,
                        rad,
                        ctr,
                        R_bend,
                        p=2,
                        alpha=None,
                        npts=6,
                        nspan=10,
                        within=None,
                        rhoinv=0.0,
                        niterations=10,
                        quadrule='circ_trapez_shift',
                        verbose=True,
                        inverse='umfpack',
                        nrestarts=0,
                        **feastkwargs):
        """Find bent modes using scalar method.

        Bending radius R_bend should be non-dimensional. If alpha is provided,
        radial pml from ngsolve is set."""

        if alpha is not None:
            self.ngspmlset = True
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))

        r = ng.x + R_bend
        k = self.k * self.index * self.L

        X = ng.H1(self.mesh, order=p, complex=True)

        u, v = X.TnT()

        A0 = ng.BilinearForm(X, check_unused=False)
        A0 += -r * ng.grad(u) * ng.grad(v) * ng.dx
        A0 += k**2 * r * u * v * ng.dx

        A1 = ng.BilinearForm(X, check_unused=False)
        A1 += R_bend**2 / r * u * v * ng.dx

        AA = [A0, A1]

        with ng.TaskManager():
            for i in range(len(AA)):
                try:
                    AA[i].Assemble()
                except Exception:
                    print('*** Trying again with larger heap')
                    ng.SetHeapSize(int(1e9))
                    AA[i].Assemble()

        P = SpectralProjNG(X,
                           A0.mat,
                           A1.mat,
                           radius=rad,
                           center=ctr,
                           npts=npts,
                           rhoinv=rhoinv,
                           quadrule=quadrule)

        Y = NGvecs(X, nspan)
        Yl = Y.create()
        Y.setrandom(seed=1)
        Yl.setrandom(seed=1)

        print('Using FEAST to search for scalar bent modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval.' % nspan)
        print('System size:', X.ndof, ' x ', X.ndof)
        print('  Inverse type:', inverse)

        Nu_sqrs, Y, hist, _ = P.feast(Y,
                                      Yl,
                                      hermitian=False,
                                      nrestarts=nrestarts,
                                      niterations=niterations,
                                      **feastkwargs)

        nus = (Nu_sqrs**.5) * R_bend
        CLs = 20 * nus.imag / np.log(10)
        print('Results:\n Nu²:', Nu_sqrs)
        print(' Nus:', nus)
        print(' CL dB/m:', CLs)

        return Nu_sqrs, Y, P, hist, CLs

    def bentmodesystem(self, p, R_bend, alpha=None, inverse=None):
        """
        Prepare eigensystem and resolvents for solving for bent vector modes.

        INPUTS:

        p: Determines degree of Nedelec x Lagrange space system.
           This should be an integer >= 0.

        R_bend: Non-dimensionalized bend radius.  Typical values around
            800 * R_clad.

        alpha: If alpha is None, prepare system for vector guided modes.
           If alpha is a positive number, use it as PML strength and
           prepare system for leaky modes using NGSolve's automatic
           mesh-based PML.
        """

        if alpha is not None:
            self.ngspmlset = True
            radial = ng.pml.Radial(rad=self.R, alpha=alpha * 1j, origin=(0, 0))
            self.mesh.SetPML(radial, 'Outer')
            print('Set NGSolve automatic PML with p=', p, ' alpha=', alpha,
                  'and thickness=%.3f' % (self.Rout - self.R))

        n = self.index
        n2 = n * n
        r = ng.x + R_bend  # r is NOT sqrt(x^2 + y^2)
        X = ng.HCurl(self.mesh,
                     order=p + 1 - max(1 - p, 0),
                     type1=True,
                     dirichlet='OuterCircle',
                     complex=True)
        Y = ng.H1(self.mesh,
                  order=p + 1,
                  dirichlet='OuterCircle',
                  complex=True)
        E, v = X.TnT()
        phi, psi = Y.TnT()

        A = ng.BilinearForm(X)
        A += r * (curl(E) * curl(v) - (self.L * self.k)**2 * n2 * E * v) * dx
        M = ng.BilinearForm(X)
        M += -R_bend**2 / r * E * v * dx
        C = ng.BilinearForm(trialspace=Y, testspace=X)
        C += R_bend * grad(phi) * v * dx
        C += R_bend / r * phi * v[0] * dx
        B = ng.BilinearForm(trialspace=X, testspace=Y)
        B += -n2 * r * E * grad(psi) * dx
        D = ng.BilinearForm(Y)
        # D = ng.BilinearForm(Y, condense=True)
        D += n2 * R_bend * phi * psi * dx

        with ng.TaskManager():
            try:
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            except Exception:
                print('*** Trying again with larger heap')
                ng.SetHeapSize(int(1e9))
                A.Assemble()
                M.Assemble()
                B.Assemble()
                C.Assemble()
                D.Assemble()
            Dinv = D.mat.Inverse(Y.FreeDofs(), inverse=inverse)
            # Dinv = D.mat.Inverse(Y.FreeDofs(coupling=True), inverse=inverse)
        # resolvent of the vector mode problem --------------------------

        class ResolventVectorMode():

            # static resolvent class attributes, same for all class objects
            XY = ng.FESpace([X, Y])
            wrk1 = ng.GridFunction(XY)
            wrk2 = ng.GridFunction(XY)
            tmpY1 = ng.GridFunction(Y)
            tmpY2 = ng.GridFunction(Y)
            tmpX1 = ng.GridFunction(X)

            def __init__(selfr, z, n, inverse=None):
                n2 = n * n
                XY = ng.FESpace([X, Y])
                (E, phi), (v, psi) = XY.TnT()
                selfr.zminusOp = ng.BilinearForm(XY)
                selfr.zminusOp += (
                    -z * R_bend**2 / r * E * v - r * curl(E) * curl(v) +
                    (self.L * self.k)**2 * n2 * r * E * v - R_bend *
                    (grad(phi) * v + 1 / r * phi * v[0]) -
                    R_bend * n2 * phi * psi + n2 * r * E * grad(psi)) * dx
                with ng.TaskManager():
                    try:
                        selfr.zminusOp.Assemble()
                    except Exception:
                        print('*** Trying again with larger heap')
                        ng.SetHeapSize(int(1e9))
                        selfr.zminusOp.Assemble()
                    selfr.R = selfr.zminusOp.mat.Inverse(XY.FreeDofs(),
                                                         inverse=inverse)

            def act(selfr, v, Rv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]

                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0
                        selfr.wrk2.vec.data = selfr.R * selfr.wrk1.vec
                        Rv._mv[i][:] = selfr.wrk2.components[0].vec

            def adj(selfr, v, RHv, workspace=None):
                if workspace is None:
                    Mv = ng.MultiVector(v._mv[0], v.m)
                else:
                    Mv = workspace._mv[:v.m]
                with ng.TaskManager():
                    Mv[:] = M.mat * v._mv
                    for i in range(v.m):
                        selfr.wrk1.components[0].vec[:] = Mv[i]
                        selfr.wrk1.components[1].vec[:] = 0
                        selfr.wrk2.vec.data = selfr.R.H * selfr.wrk1.vec
                        RHv._mv[i][:] = selfr.wrk2.components[0].vec

            def rayleigh_nsa(selfr,
                             ql,
                             qr,
                             qAq=not None,
                             qBq=not None,
                             workspace=None):
                """
                Return qAq[i, j] = (𝒜 qr[j], ql[i]) with 𝒜 =  (A - C D⁻¹ B) E
                and qBq[i, j] = (M qr[j], ql[i]). """

                if workspace is None:
                    Aqr = ng.MultiVector(qr._mv[0], qr.m)
                else:
                    Aqr = workspace._mv[:qr.m]

                with ng.TaskManager():
                    if qAq is not None:
                        Aqr[:] = A.mat * qr._mv
                        for i in range(qr.m):
                            selfr.tmpY1.vec.data = B.mat * qr._mv[i]
                            selfr.tmpY2.vec.data = Dinv * selfr.tmpY1.vec
                            selfr.tmpX1.vec.data = C.mat * selfr.tmpY2.vec
                            Aqr[i].data -= selfr.tmpX1.vec
                        qAq = ng.InnerProduct(Aqr, ql._mv).NumPy().T

                    if qBq is not None:
                        Bqr = Aqr
                        Bqr[:] = M.mat * qr._mv
                        qBq = ng.InnerProduct(Bqr, ql._mv).NumPy().T

                return (qAq, qBq)

            def rayleigh(selfr, q, workspace=None):
                return selfr.rayleigh_nsa(q, q, workspace=workspace)

        # resolvent class definition done -------------------------------

        return ResolventVectorMode, M.mat, A.mat, B.mat, C.mat, D.mat, Dinv

    def bentvecmodes(self,
                     rad,
                     ctr,
                     R_bend,
                     p=2,
                     alpha=None,
                     seed=1,
                     npts=6,
                     nspan=10,
                     rhoinv=0.0,
                     niterations=15,
                     nrestarts=0,
                     quadrule='circ_trapez_shift',
                     inverse='umfpack',
                     **feastkwargs):
        """
        Capture bent vector modes whose scaled propagation constants have real
        part near or in the interval L^2k_0^2 [n_clad^2, n_core^2].
        """

        R, M, A, B, C, D, Dinv = self.bentmodesystem(p,
                                                     R_bend,
                                                     alpha=alpha,
                                                     inverse=inverse)
        X, Y = R.XY.components
        E = NGvecs(X, nspan, M=M)
        E.setrandom(seed=seed)

        print('Using FEAST to search for vector bent modes in')
        print('circle of radius', rad, 'centered at ', ctr)
        print('assuming not more than %d modes in this interval.' % nspan)
        print('System size:', E.n, ' x ', E.n, '  Inverse type:', inverse)

        P = SpectralProjNGR(lambda z: R(z, self.index, inverse=inverse),
                            radius=rad,
                            center=ctr,
                            npts=npts,
                            rhoinv=rhoinv,
                            quadrule=quadrule)

        Nu_sqrs, E, history, _ = P.feast(E,
                                         hermitian=False,
                                         niterations=niterations,
                                         nrestarts=nrestarts,
                                         **feastkwargs)

        phi = NGvecs(Y, E.m)
        BE = phi.zeroclone()
        BE._mv[:] = -B * E._mv
        phi._mv[:] = Dinv * BE._mv

        nus = (Nu_sqrs**.5) * R_bend
        CLs = 20 * nus.imag / np.log(10)
        print('Results:\n Nu²:', Nu_sqrs)
        print(' Nus:', nus)
        print(' CL dB/m:', CLs)

        return Nu_sqrs, E, phi, R, CLs
