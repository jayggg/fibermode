"""
This class models RADIALLY SYMMETRIC STEP-INDEX optical fibers and
provides method to calculate propagation constants, transverse guided
modes, and leaky modes/resonances using (semi)ANALYTIC calculations.
"""

from ..utilities import named_stepindex_fibers
from math import pi, sqrt, atan2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, bisect
from scipy.special import hankel1, jv, kv, jvp, kvp, jn_zeros
from cxroots import Rectangle, Circle
import sympy as sm
import logging


class StepIndexExact:
    """ A class representing step-index cylindrical fibers, with methods to
    compute propagation constants, (scalar and vector) guided modes
    and leaky modes (semi)analytically. """

    def __init__(self,
                 name=None,
                 rcore=None,
                 rclad=None,
                 nclad=None,
                 ncore=None,
                 NA=None,
                 ks=None,
                 wavelen=None):
        """
        A step-index fiber object can be made either by giving the name of
        a preprogrammed fiber, or by specifying individual properties.

        To initialize by name, give the name as argument:
           F = StepIndexExact('Nufern_Yb')
        Known names include 'Nufern_Yb' (ytterbium-doped fibers which were sold
        by Nufern), 'liekki_1', 'corning_smf_28_1', etc. All preprogrammed
        names are listed in fibermode.named_stepindex_fibers.

        To initialize by other arguments, provide the minimal information
        needed to know the geometry and material property by giving a
        subset of these possible keyword arguments:

            * Both these arguments are required:
              rcore:   core cross-section radius [meters]
              rclad:   cladding cross-section radius [meters]

            * Two of these arguments are required, the remaining is inferred:
              nclad:   cladding refractive index [no dimension]
              ncore:   core refractive index [no dimension]
              NA:      numerical aperture of fiber [no dimension]

            * One of these arguments is required, the remaining is inferred:
              wavelen: signal wavelength [meter]
              ks:      signal wavenumber [1/meter]
        """

        if name is not None:

            self.__init__(**named_stepindex_fibers[name])

        else:

            args_to_provide = """Recall from constructor doc:

            * Both these arguments are required:
              rcore:   core cross-section radius [meters]
              rclad:   cladding cross-section radius [meters]

            * Two of these arguments are required, the remaining is inferred:
              nclad:   cladding refractive index [no dimension]
              ncore:   core refractive index [no dimension]
              NA:      numerical aperture of fiber [no dimension]

            * One of these arguments is required, the remaining is inferred:
              wavelen: signal wavelength [meter]
              ks:      signal wavenumber [1/meter]
            """

            if (rcore is None or rclad is None) or \
               (nclad is None and ncore is None and NA is None) or \
               (wavelen is None and ks is None):

                raise ValueError(args_to_provide)

            self.rcore = rcore
            if rclad > rcore:
                self.rclad = rclad
            else:
                ValueError('Radius of cladding must be larger than core')
            if ks is None:
                self.wavelen = wavelen
            else:
                self.ks = ks

            if ncore is not None and nclad is not None:
                # then ignore any NA value provided
                self._ncore = ncore
                self._nclad = nclad
                self._NA = sqrt(self._ncore**2 - self._nclad**2)
            elif ncore is not None and NA is not None:
                self._ncore = ncore
                self._NA = NA
                self._nclad = sqrt(self._ncore**2 - self._NA**2)
            else:  # then nclad and NA must have been provided
                self._nclad = nclad
                self._NA = NA
                self._ncore = sqrt(self._nclad**2 + self._NA**2)

    def set_from_dict(self, d):
        for key, value in d.items():
            setattr(self, key, value)

    @property
    def ncore(self):
        return self._ncore

    @ncore.setter
    def ncore(self, ncore):  # Set ncore and change NA accordingly
        self._ncore = ncore
        self._NA = sqrt(self._ncore**2 - self._nclad**2)

    @property
    def nclad(self):
        return self._nclad

    @nclad.setter
    def nclad(self, nclad):  # Set nclad and change NA accordingly
        self._nclad = nclad
        self._NA = sqrt(self._ncore**2 - self._nclad**2)

    @property
    def NA(self):  # read only property (that depends on ncore & nclad)
        return self._NA

    @property
    def ks(self):
        return self._ks

    @ks.setter
    def ks(self, kvalue):
        self._ks = kvalue
        self._wavelen = 2 * pi / kvalue

    @property
    def wavelen(self):
        return self._wavelen

    @wavelen.setter
    def wavelen(self, wavelen):
        self._wavelen = wavelen
        self._ks = 2 * pi / wavelen

    def __repr__(self):
        lines = ['STEP INDEX FIBER ATTRIBUTES: ' + '-' * 43]
        lines += [
            'ks:         %20g' % self.ks + '{:>40}'.format('signal wavenumber')
        ]
        lines += [
            'wavelength: %20g' % (2 * pi / self.ks) +
            '{:>40}'.format('signal wavelength')
        ]
        lines += [
            'V:          %20g' % (self.fiberV()) +
            '{:>40}'.format('V-number of fiber')
        ]
        lines += [
            'ncore:      %20g' % (self.ncore) +
            '{:>40}'.format('core refractive index')
        ]
        lines += [
            'nclad:      %20g' % (self.nclad) +
            '{:>40}'.format('cladding refractive index')
        ]
        lines += [
            'NA:         %20g' % (self.NA) +
            '{:>40}'.format('numerical aperture')
        ]
        lines += [
            'rcore:      %20g' % (self.rcore) + '{:>40}'.format('core radius')
        ]
        lines += [
            'rclad:      %20g' % (self.rclad) +
            '{:>40}'.format('cladding radius')
        ]
        lines += ['-' * 72]

        return "\n".join(lines)

    # PROPAGATION CHARACTERISTICS

    def fiberV(self):
        """
        Returns the nondimensional V number of the fiber.
        (See e.g., [section 5.2.1, Rieder, Photonics: An introduction].)
        """
        V = self._NA * self._ks * self.rcore
        return V

    def XtoBeta(self, X, v=None):
        """
        Convert nondimensionalized roots X to actual propagation
        constants Beta of guided modes.
        """
        V = self.fiberV() if v is None else v
        a = self.rcore
        ks = V / (self._NA * a)
        kappas = [x / a for x in X]
        betas = [
            sqrt(self.ncore * self.ncore * ks * ks - kappa * kappa)
            for kappa in kappas
        ]
        return betas

    def ZtoBeta(self, Z, v=None):
        """
        Convert nondimensional Z in the complex plane to complex propagation
        constants Beta of leaky modes.
        """
        V = self.fiberV() if v is None else v
        a = self.rcore
        ks = V / (self._NA * a)
        return np.sqrt((ks * self.nclad)**2 - (Z / a)**2)

    def visualize_mode(self, ell, m):
        """X, Y, Z, LPmode = StepIndexExact.visualize_mode(ell, m)

        returns a plot data set X, Y, Z to visualize the LP mode (ell,m),
        as well as a function LPmode(x, y) representing the mode.
        Note that ell and m are both indexed to start from 0, so for
        example, the traditional LP_01 and LP_11 modes are obtained by
        calling LPmode(0, 0) and LPmode(1, 0), respectively.

        To plot the mode (Z) values using matplotlib:

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.plot_surface(X, Y, Z)
        """

        X = self.propagation_constants(ell)
        if len(X) < m:
            raise ValueError('For ell=%d, only %d fiber modes computed' %
                             (ell, len(X)))
        kappa = X[m] / self.rcore
        k0 = self.ks
        beta = sqrt(self.ncore * self.ncore * k0 * k0 - kappa * kappa)
        gamma = sqrt(beta * beta - self.nclad * self.nclad * k0 * k0)
        Jkrcr = jv(ell, kappa * self.rcore)
        Kgrcr = kv(ell, gamma * self.rcore)

        def mode(x, y):
            r = sqrt(x * x + y * y)
            theta = atan2(y, x)
            if r < self.rcore:
                u = Kgrcr * jv(ell, kappa * r)
            else:
                u = Jkrcr * kv(ell, gamma * r)

            u = u * np.cos(ell * theta)
            return u

        lim = self.rcore * 1.5
        X = np.arange(-lim, lim, 2 * lim / 100)
        Y = np.arange(-lim, lim, 2 * lim / 100)
        X, Y = np.meshgrid(X, Y)
        vmode = np.vectorize(mode)
        Z = vmode(X, Y)

        return X, Y, Z, vmode

    def propagation_constants(self,
                              ell,
                              maxnroots=50,
                              v=None,
                              loglevel=logging.WARNING):
        """
        Given mode index "ell", attempt to find all propagation constants by
        bisection or other nonlinear root finders. (Circularly
        symmetric modes are obtained with ell=0.  Modes with higher ell
        have angular variations.) When loglevel=logging.INFO, verbose
        outputs are logged.
        """
        logging.basicConfig(format='%(message)s', level=loglevel)
        logging.captureWarnings(True)

        if v is None:
            ks = self.ks
            V = self.fiberV()
        else:
            ks = v / (self._NA * self.rcore)
            V = v

        # Case of empty fibers:

        if abs(self.NA) < 1.e-15:

            # If NA is 0, then this is an empty fiber (with V=0) whose
            # propagation constants are given by
            #         beta = sqrt( ks^2 - (jz/rclad)^2 )
            # where jz are roots of l-th Bessel function. Below we reverse
            # engineer X so that this beta is produced by later formulae.

            jz = jn_zeros(ell, maxnroots)
            jz = jz[np.where(jz < ks * self.rclad)[0]]
            if len(jz) == 0:
                logging.info(
                    'There are no propagating modes for wavenumber ks!')
                logging.info('   ks = %g' % ks)
            # Return X adjusted so that later formulas
            # like kappa = X[m] / rcore   are still valid.
            X = jz * (self.rcore / self.rclad)

            # we are done:
            return X

        # Case of non-empty fibers (V>0):

        _, f1, f2, g1, g2, f, g = self.VJKfun(ell, v=V)

        # Collect Bessel roots appended with 0 and V:

        jz = jn_zeros(ell, maxnroots)
        jz = jz[np.where(jz < V)[0]]
        jz = np.insert(jz, 0, 0)
        jz = np.append(jz, V)
        X = []
        logging.info('\nSEARCHING FOR ROOTS in [0,V] (V = %7g) ' % (V) +
                     '-' * 29)

        def root_check(x):  # Subfunction used for checking if x is a root
            if abs(f(x)) < 1.e-9 and abs(g(x)) < 1.e-9:
                logging.info('  Root x=%g found with f(x)=%g and g(x)=%g' %
                             (x, f(x), g(x)))

                return True
            else:
                return False

        for i in range(1, len(jz)):

            # Try bisection based on intervals between bessel roots

            a = jz[i - 1] + 1.e-15
            b = jz[i] - 1.e-15
            if np.sign(g(a) * g(b)) < 0:
                logging.info(' SEARCH %1d: Sign change in [%g, %g]' %
                             (i, a, b))
                x = bisect(g, a, b)
                if root_check(x):
                    X += [x]
                else:
                    logging.info('  Bisection on g did not find a root!\n')
                    x = bisect(f, a, b)
                    if root_check(x):
                        X += [x]
                    else:
                        logging.info(
                            '  Bisection on f also did not find a root!')

            # Check if end points are roots:

            elif root_check(a):
                X += [a]

            elif root_check(b):
                X += [b]

            # Try nonlinear/gradient-based root finding:

            else:

                x0 = 0.5 * (a + b)
                logging.info(
                    '  Trying nonlinear solve on g with initial guess %g' % x0)
                x = fsolve(g, x0, xtol=1.e-10)[0]
                if root_check(x):
                    X += [x]
                else:
                    logging.info('  Nonlinear solve on g did not find a root.')
                    logging.info(
                        '  Trying nonlinear solve on f with initial guess %g' %
                        x0)
                    x = fsolve(f, x0, xtol=1.e-10)[0]
                    if root_check(x):
                        X += [x]
                    else:
                        logging.info(
                            '  Nonlinear solve on f did not find a root.')
                        logging.info(
                            '  Giving up on finding roots in [%g, %g].' %
                            (a, b))

        logging.info(' ROOTS FOUND: ' + ' '.join(map(str, X)))
        return X

    def visualize_roots(self, ell):
        """
        Visualize two different functions f = f1 - f2 and g = g1 - g2
        whose roots are used to compute propagation constants for
        each mode index "l".
        """

        V, f1, f2, g1, g2, f, g = self.VJKfun(ell)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        plt.grid(True)
        plt.rc('text', usetex=True)
        fig.suptitle(r'Mode index $ell=$%1d:' % ell +
                     r'$\beta$ found by roots of $f$ or $g$',
                     fontsize=14)
        xx = np.arange(0, V, V / 500.0)

        axes[0, 0].plot(xx, [f(x) for x in xx])
        axes[0, 0].grid(True)
        axes[0, 0].plot(xx, [0] * len(xx), 'b--')
        axes[0, 0].set_ylim(-V, V)
        axes[0, 0].set_title('$f = f_1 - f_2$')

        axes[1, 0].plot(xx, [f1(x) for x in xx])
        axes[1, 0].grid(True)
        axes[1, 0].plot(xx, [f2(x) for x in xx])
        axes[1, 0].set_ylim(-2 * V, 2 * V)
        axes[1, 0].legend(['$f_1$', '$f_2$'])

        axes[0, 1].plot(xx, [g(x) for x in xx])
        axes[0, 1].grid(True)
        axes[0, 1].plot(xx, [0] * len(xx), 'b--')
        axes[0, 1].set_ylim(-V, V)
        axes[0, 1].set_title('$g = g_1 - g_2$')

        axes[1, 1].plot(xx, [g1(x) for x in xx])
        axes[1, 1].grid(True)
        axes[1, 1].plot(xx, [g2(x) for x in xx])
        axes[1, 1].set_ylim(-V, V)
        axes[1, 1].legend(['$g_1$', '$g_2$'])

        plt.ion()
        plt.show()
        plt.show(block=False)

    def VJKfun(self, ell, v=None):
        """
        For the "ell"-th mode index of the fiber, return the nonlinear
        functions whose roots give propagation constants.
        v: V-number of the fiber
        """

        V = self.fiberV() if v is None else v
        J = jv
        K = kv

        def jl(X):
            Jlx = J(ell, X)
            if abs(Jlx) < 1.e-15:
                if Jlx < 0:
                    Jlx = Jlx - 1e-15
                else:
                    Jlx = Jlx + 1e-15
            return Jlx

        def f1(X):
            JlX = jl(X)
            return J(ell + 1, X) * X / JlX

        def y(X):
            if X > V:
                Y = 0
            else:
                Y = sqrt(V * V - X * X)
            return Y

        def f2(X):
            Y = y(X)
            return K(ell + 1, Y) * Y / K(ell, Y)

        def f(X):  # Propagation constant is a root of f(X)
            return f1(X) - f2(X)

        def g1(X):
            JlX = jl(X)
            Y = y(X)
            return J(ell + 1, X) * K(ell, Y) / (JlX * K(ell + 1, Y))

        def g2(X):
            return y(X) / max(X, 1e-15)

        def g(X):  # Propagation constant is also a root of g(X)
            return g1(X) - g2(X)

        return V, f1, f2, g1, g2, f, g

    def VJHfuns(self, ell):
        """
        For the "ell"-th mode index, return a nonlinear function of a
        nondimensional variable Z whose nondimensionalized roots give
        leaky modes.  The function is returned as a string (with letter Z)
        which can be evaluated for specific Z values later.
        """
        z, nu = sm.symbols('z nu')
        V = self.fiberV()
        x = sm.sqrt(V * V + z * z)
        g = z*sm.besselj(ell, x)*sm.hankel1(ell+1, z) - \
            x*sm.besselj(ell+1, x)*sm.hankel1(ell, z)
        dg = g.diff(z).expand()

        dgstr = str(dg).replace('z', 'Z').     \
            replace('besselj', 'jv').          \
            replace('nu', 'ell').               \
            replace('sqrt', 'np.sqrt')
        gstr = str(g).replace('z', 'Z').       \
            replace('besselj', 'jv').          \
            replace('nu', 'ell').               \
            replace('sqrt', 'np.sqrt')
        return gstr, dgstr

    def leaky_propagation_constants(self, ell, xran=None, yran=None,
                                    loglevel=logging.WARNING):
        """
        Given a mode index "ell" indicating angular variation (the
        radially symmetric case being ell=0), search the following
        rectangular region (given by tuples xran, yran)
              [xran[0], xran[1]]  x  [yran[0], yran[1]]
        of the complex plane for roots that yield leaky outgoing modes.
        (Be warned that the root finder is not as robust as the real
        line root searching algorithm for guided modes.)
        """
        logging.basicConfig(format='%(message)s', level=loglevel)

        if xran is None:
            xran = (0.01, 3 * self.fiberV())
        if yran is None:
            yran = (-2, 0)
        logging.info('Searching region (%g, %g) x (%g, %g) in complex plane' %
                     (xran[0], xran[1], yran[0], yran[1]))
        gstr, dgstr = self.VJHfuns(ell)
        try:
            rect = Rectangle(xran, yran)
            r = rect.roots(lambda Z: eval(gstr),
                           lambda Z: eval(dgstr),
                           root_err_tol=1.e-13,
                           newton_step_tol=1.e-15)
        except RuntimeError as err:
            logging.warning('Root search failed:\n', err.__str__())
            dx = xran[1] - xran[0]
            dy = yran[1] - yran[0]
            xran2 = (xran[0] + 0.01 * dx, xran[1] - 0.01 * dx)
            yran2 = (yran[0] + 0.01 * dy, yran[1] - 0.01 * dy)
            logging.info('Retrying in adjusted search region ' +
                         '(%g, %g) x (%g, %g)' %
                         (xran2[0], xran2[1], yran2[0], yran2[1]))
            r = self.leaky_propagation_constants(ell, xran=xran2, yran=yran2)
        return r.roots

    def visualize_leaky_mode(self, Z, ell, corelim=2):
        """
        Given a complex propagation constant Z obtained from
        self.leaky_propagation_constants(ell),  compute the corresponding
        leaky mode. Return its values F at a meshgrid of points (X, Y) and
        plot it. If "corelim" is given, this grid discretizes the xy region
        [-lim, lim] x [-lim, lim] where lim is corelim x core radius.
        """

        V = self.fiberV()
        X = np.sqrt(Z * Z + V * V)
        a = self.rcore
        alpha0 = Z / a
        alpha1 = X / a
        B = jv(ell, X)
        A = hankel1(ell, Z)

        def modefun(x, y):
            r = np.sqrt(x * x + y * y)
            theta = atan2(y, x)
            if r < a:
                u = A * jv(ell, alpha1 * r)
            else:
                u = B * hankel1(ell, alpha0 * r)
            u = u * np.cos(ell * theta)
            return u

        lim = a * corelim
        X = np.arange(-lim, lim, 2 * lim / 300)
        Y = np.arange(-lim, lim, 2 * lim / 300)
        X, Y = np.meshgrid(X, Y)
        vmode = np.vectorize(modefun)
        F = vmode(X, Y)

        return X, Y, F, modefun

    def bendloss_Marcuse(self,
                         rbend,
                         ell,
                         m):
        """
        Return bend-induced losses of LP_ell,m guided mode of step-index
        fiber with bend radius R utilizing the curvature loss formula
        for optical fiber described by Marcuse (1976)

        OUTPUT:

        A float representing 2α, with α being the bend
        loss coefficient (units: 1/m).

        INPUTS:

        rbend: circular bend radius from center of

        ell: azimuthal mode index

        m: radial mode index
        """

        ellbetas = self.XtoBeta(ell)
        beta = ellbetas[m]

        kappa = np.sqrt(self.ncore**2 * self.ks**2 - beta**2)
        gamma = np.sqrt(beta**2 - self.nclad**2 * self.ks**2)

        if ell == 0:
            e_ell = 2
        else:
            e_ell = 1

        V = self.fiberV()

        alpha_num = (np.exp(-1 * (2/3) * (gamma**3 / beta**2) * rbend)
                     * np.sqrt(np.pi) * kappa**2)

        alpha_denom = (e_ell * kv(ell-1, gamma * self.rcore)
                       * kv(ell+1, gamma * self.rcore)
                       * gamma**(3/2) * np.sqrt(rbend) * V**2)

        twoalpha = alpha_num / alpha_denom

        return twoalpha

    def vec_propagation_constants(self,
                                  m,
                                  delta=0.01,
                                  nrefine=10000,
                                  tol=1e-9,
                                  maxnroots=50,
                                  m0name=None):
        """
        Given mode angular variation index "m", attempt to find all
        propagation constants of vector modes (with m=0 corresponding
        to the circularly symmetric case).

        OUTPUT:

        A list of tuples (ys, a, b) where ys is a list of roots found
        in the interval [a, b].  These nondimensional Y-values in ys are
        related to the physical propagation constant β by
        Y = a * sqrt(β² - k²n₀²).

        INPUTS:

        nrefine: Split the interval [0, V] into nrefine subintervals and
        separately search in each. (Here V is the V-number of the fiber.)

        delta: do not search near +/- delta intervals near BesselJm roots and
        near points 0 and V (where spurious roots can be picked up easily).

        tol: root accepted when f has abs value less than tol.

        maxnroots: an estimate for number of BesselJm's roots that might
        be contained in [0, V].

        m0name: if m=0, this argument should specify either 'TM' or 'TE'.
        """

        k = self.ks
        V = self.fiberV()
        n1 = self.ncore
        n0 = self.nclad
        a = self.rcore
        kn0a2 = (k * n0 * a)**2

        if abs(self.NA) < 1.e-15:
            raise NotImplementedError()

        def f(Y):
            X = np.sqrt(V**2 - Y**2)
            J = jv(m, X)
            JX = J * X
            dJ = jvp(m, X)
            K = kv(m, Y)
            KY = K * Y
            dK = kvp(m, Y)
            if m == 0:
                if m0name == 'TE':
                    fY = dJ * KY + JX * dK
                elif m0name == 'TM':
                    fY = (n1 / n0)**2 * dJ * KY + JX * dK
                else:
                    raise ValueError('m=0 case should specify m0name!')
                return fY
            else:
                fY = kn0a2 * ((X * Y)**2 * ((n1/n0)**2 * dJ * KY + JX * dK) *
                              (dJ * KY + JX * dK)) \
                    - (m**2 * V**4) * ((Y**2 + kn0a2) * (J*K)**2)
                return fY

        # Collect Bessel roots appended with 0 and V:

        jz = jn_zeros(m, maxnroots)
        jz = jz[np.where(jz < V)[0]]
        jz = np.insert(jz, 0, 0)
        jz = np.append(jz, V)
        jy = np.sort(np.sqrt(V**2 - jz**2))  # convert to Y-intervals
        Y = []

        for i in range(1, len(jz)):

            # Try bisection on "nrefine" intervals between bessel roots
            a0 = jy[i - 1] + delta
            b0 = jy[i] - delta
            aa = np.linspace(a0, b0, num=nrefine)
            logging.info('\nSEARCHING FOR ROOTS Y in [%6g, %6g]' % (a0, b0) +
                         '-' * 29)
            ys = []
            for ii in range(nrefine - 1):
                roots = []
                a = aa[ii]
                b = aa[ii + 1]
                if abs(f(a)) < tol:
                    roots += [a]
                elif abs(f(b)) < tol:
                    roots += [b]
                elif np.sign(f(a) * f(b)) < 0:
                    logging.info(' SEARCH %1d: Sign change in [%g, %g]'
                                 % (ii, a, b))
                    y = bisect(f, a, b, xtol=tol, maxiter=10000)
                    if abs(f(y)) < tol:
                        logging.info('  Bisection succeeded.')
                        roots += [y]
                    else:
                        y = fsolve(f, (a + b) / 2, xtol=tol)[0]
                        if abs(f(y)) < tol:
                            logging.info('  Nonlinear solve succeeded.')
                            roots += [y]
                        else:
                            logging.info(
                                '  >>>Neither bisection nor fsolve worked!')
                if len(roots):
                    ys += [(roots, a, b)]

            if len(ys) == 0:
                y = fsolve(f, (a0 + b0) / 2, xtol=tol)[0]
                if abs(f(y)) < tol and abs(y - b0) > delta and abs(y -
                                                                   a0) > delta:
                    logging.info(
                        '  Nonlinear solve b/w Bessel roots succeeded.')
                    ys += [(y, a0, b0)]
                else:
                    logging.info(
                        '  >>>Could not find root in [%6g, %6g]' % (a0, b0))

            Y += ys

        # Report
        logging.info('-' * 64)
        if len(Y):
            logging.info('ROOTS FOR m =', m, ' FOUND:')
            for yy in Y:
                y, a, b = yy
                for yroot in y:
                    logging.info('  %12.10f in [%8.6f, %8.6f]' % (yroot, a, b))
        else:
            logging.info('NO ROOTS FOUND FOR m =', m)
        logging.info('-' * 64)

        return Y

    def vec_symbolic_rootfun(self, m):
        """
        For the "m"-th mode index, return a nonlinear function of a
        nondimensional variable Z whose nondimensionalized roots give
        leaky modes.  The function is returned as a string (with letter Z)
        which can be evaluated for specific Z values later.
        """
        x, y = sm.symbols('x y')
        n1 = self.ncore
        n0 = self.nclad
        a = self.rcore
        k = self.ks
        kn0a2 = (k * n0 * a)**2
        V = self.fiberV()

        J = sm.besselj(m, x)
        JX = J * x
        dJ = J.diff(x).factor()
        K = sm.besselk(m, y)
        KY = K * y
        dK = K.diff(y).factor()

        fY = kn0a2 * ((x * y)**2 * ((n1 / n0)**2 * dJ * KY + JX * dK) *
                      (dJ * KY + JX * dK))
        if m > 0:
            fY -= (m**2 * V**4) * ((y**2 + kn0a2) * (J * K)**2)

        fY = fY.subs(x, sm.sqrt(V**2 - y**2))
        dfY = fY.diff(y).factor()

        dgstr = str(dfY).replace('y', 'Y').    \
            replace('besselj', 'jv').          \
            replace('besselk', 'kv').          \
            replace('sqrt', 'np.sqrt')
        gstr = str(fY).replace('y', 'Y').      \
            replace('besselj', 'jv').          \
            replace('besselk', 'kv').          \
            replace('sqrt', 'np.sqrt')

        return gstr, dgstr

    def vec_confirm_roots(self, m, YY):
        """
        Use cxroots to confirm the roots found within intervals, as output
        by YY = self.vec_propagation_constants(...). The input YY must
        be in the same format (as a 3-tuple) as the output of
        vec_propagation_constants.  Return roots as found by cxroots
        (which might have better precision).
        """

        g, dg = self.vec_symbolic_rootfun(m)
        roots = []
        for yy in YY:
            y, a, b = yy
            for ctr in y:
                rad = max(ctr - a, b - ctr, 1e-6)
                logging.info(' Checking root ', ctr, ' about radius', rad)
                rec = Circle(ctr, rad)
                r = rec.roots(lambda Y: eval(g),
                              lambda Y: eval(dg),
                              root_err_tol=1.e-10,
                              newton_step_tol=1.e-14)
                roots += [r]
        return roots

    def vec_symbolic_Emode(self, m, Y, m0name=None):
        """
        INPUTS:

        * Angular index m (with m=0 indicating circular symmetry).
        * The Y-value corresponding to the input "m", e.g, as output by the
        method vec_propagation_constants(...).
        * When m=0, there are two mode profiles, and it is necessary to
        give m0name as either 'TM' or 'TE' to select one.

        OUTPUTS:  Ecore, Eclad

        Both are a list of evaluable strings representing the mode
        expression as a function of polar coordinates r (radius), t
        (angle). Each list has three complex components, corresponding
        to r, t, and z components of the electric field E. The first
        list represents the expressions in the core and the second in
        the cladding.

        """
        x, y = sm.symbols('x y')
        r, t = sm.symbols('r t')
        n1 = self.ncore
        n0 = self.nclad
        a = self.rcore
        k = self.ks
        V = self.fiberV()
        X = sm.sqrt(V**2 - Y**2)
        J = sm.besselj(m, X)
        K = sm.besselk(m, Y)
        dJ = sm.besselj(m, x).diff(x).subs(x, X)
        dK = sm.besselk(m, y).diff(y).subs(y, Y)
        beta = self.XtoBeta([X])[0]

        if m > 0:
            B1t = 1j * k**2 * (dJ*K*Y*n1**2 + J*X*dK*n0**2) * X*Y \
                / (m * beta * J * K * V**2)
            phase = sm.exp(sm.I * m * t)

            Ezcore = sm.besselj(m, X * r / a) * phase
            Hztcore = B1t * sm.besselj(m, X * r / a) * phase
            Ercore = -sm.I * (a / X)**2 * (beta * Ezcore.diff(r) +
                                           Hztcore.diff(t) / r)
            Etcore = -sm.I * (a / X)**2 * (beta * Ezcore.diff(t) / r -
                                           Hztcore.diff(r))

            Ezclad = (J / K) * sm.besselk(m, Y * r / a) * phase
            Hztclad = B1t * (J / K) * sm.besselk(m, Y * r / a) * phase
            Erclad = sm.I * (a / Y)**2 * (beta * Ezclad.diff(r) +
                                          Hztclad.diff(t) / r)
            Etclad = sm.I * (a / Y)**2 * (beta * Ezclad.diff(t) / r -
                                          Hztclad.diff(r))

        elif m == 0 and m0name == 'TE':

            Ezcore = 0
            Hztcore = sm.besselj(m, X * r / a)
            Ercore = -sm.I * (a / X)**2 * Hztcore.diff(t) / r
            Etcore = sm.I * (a / X)**2 * Hztcore.diff(r)

            Ezclad = 0
            Hztclad = (J / K) * sm.besselk(m, Y * r / a)
            Erclad = sm.I * (a / Y)**2 * Hztclad.diff(t) / r
            Etclad = -sm.I * (a / Y)**2 * Hztclad.diff(r)

        elif m == 0 and m0name == 'TM':

            Ezcore = sm.besselj(m, X * r / a)
            Hztcore = 0 * sm.I
            Ercore = -sm.I * (a / X)**2 * beta * Ezcore.diff(r)
            Etcore = -sm.I * (a / X)**2 * beta * Ezcore.diff(t) / r

            Ezclad = (J / K) * sm.besselk(m, Y * r / a)
            Hztclad = 0 * sm.I
            Erclad = sm.I * (a / Y)**2 * beta * Ezclad.diff(r)
            Etclad = sm.I * (a / Y)**2 * beta * Ezclad.diff(t) / r

        else:
            raise ValueError('Improper input parameters')

        Ecore, Eclad = \
            [[str(e).replace('besselj', 'jv').replace('besselk', 'kv').
              replace('sqrt', 'np.sqrt').replace('I', '1j').
              replace('exp', 'np.exp') for e in ee]
             for ee in [(Ercore, Etcore, Ezcore), (Erclad, Etclad, Ezclad)]]

        return Ecore, Eclad

    def visualize_vec_Emode(self,
                            m,
                            Y,
                            m0name=None,
                            real=False,
                            num=200,
                            block=True):
        """
        An inefficient quick hack for visualizing hybrid electric modes.

        INPUTS: m, Y, m0name are as documented in other methods
        like  vec_symbolic_Emode(..).

        "real": For hybrid modes, real=True and real=False should give
        two distinct mode profiles (each Y-value is a root of
        multiplicity 2, with even and odd angular variation).

        "num": plot on a num x num uniform grid.
        "block": Set to False if you don't want the plot to pause the code.
        """

        Ecore, Eclad = self.vec_symbolic_Emode(m, Y, m0name=m0name)
        a = self.rcore
        x = np.linspace(-2 * a, 2 * a, num=num)
        y = np.linspace(-2 * a, 2 * a, num=num)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        t = np.arctan2(y, x)
        usevars = {'np': np, 'jv': jv, 'kv': kv, 'r': r, 't': t}
        E0 = [eval(e, usevars) for e in Eclad]
        E1 = [eval(e, usevars) for e in Ecore]
        Er = np.select([r < self.rcore, r >= self.rcore], [E1[0], E0[0]])
        Et = np.select([r < self.rcore, r >= self.rcore], [E1[1], E0[1]])
        Ez = np.select([r < self.rcore, r >= self.rcore], [E1[2], E0[2]])
        Ex = Er * np.cos(t) - Et * np.sin(t)
        Ey = Er * np.sin(t) + Et * np.cos(t)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.set_xlim(-1.5 * a, 1.5 * a)
        ax.set_ylim(-1.5 * a, 1.5 * a)
        if real:
            rep = 'real'
            U = Ex.real
            V = Ey.real
            W = Ez.real
            tit = 'Re($E_{xy}$) and Re($E_z$)'
            ctit = 'Re($E_z$)'
            ax.set_title(tit)
            stit = '|Re($E_{xy})|^2$'
        else:
            rep = 'imaginary'
            U = Ex.imag
            V = Ey.imag
            W = Ez.imag
            tit = 'Im($E_{xy}$) and Im($E_z$)'
            ctit = 'Im($E_z$)'
            ax.set_title(tit)
            stit = '|Im($E_{xy})|^2$'

        cs = ax.contourf(x, y, W, cmap='seismic', alpha=0.8)
        cbar = fig.colorbar(cs, location='right')
        cbar.ax.set_ylabel(ctit)
        s0 = np.linspace(-a, a, num=20)
        sl = np.linspace(-1.5 * a, -a, num=3, endpoint=False)
        s = np.concatenate((sl, s0, -sl))
        sx, sy = np.meshgrid(s, s)
        xs = sx.reshape(1, sx.shape[0] * sx.shape[1])
        ys = sy.reshape(1, sy.shape[0] * sy.shape[1])
        seeds = np.concatenate((ys, xs), axis=0)
        Intens = U**2 + V**2
        if np.linalg.norm(Intens) < 1e-20:
            logging.info('**** Refusing to drawing the 0 ' + rep +
                         ' part of transverse E!')
            logging.info('(Have you tried to draw the other part?)')
        else:
            strm = ax.streamplot(x,
                                 y,
                                 U,
                                 V,
                                 start_points=seeds.T,
                                 density=2.5,
                                 linewidth=1.5,
                                 color=Intens,
                                 cmap='PuBuGn')
            sbar = fig.colorbar(strm.lines, location='left')
            sbar.ax.set_ylabel(stit)

        plt.tight_layout()
        plt.show(block=block)

        return fig, ax, x, y, Ex, Ey, Ez
