import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import H1, CF, dx
import fibermode
from fibermode.stepindex import StepIndexExact
from fibermode.solvers import ModeSolver
from pyeigfeast.spectralproj.ngs import NGvecs
from pyeigfeast.spectralproj import splitzoom
from scipy.sparse import coo_matrix


class StepIndex(ModeSolver):
    """Class with facilities to numerically approximate transverse modes
    of a RADIALLY SYMMETRIC STEP-INDEX fiber using a nondimensional
    eigenproblem and FEAST. Guided modes and leaky modes can be computed.
    """

    def __init__(
            self,
            fibername=None,
            fiber=None,
            R=None,  # nondimensional cladding radius
            Rout=None,  # nondimensional outer radius
            geom=None,
            curveorder=3,
            h=3,
            hcore=None,
            refine=0,
            dtemp=None):
        """
        To construct a StepIndex object (for numerical mode computations
        for a radially symmetric step-index fiber), either provide a
        predefined fiber object with name "fibername", or provide a
        StepIndexExact fiber object "fiber", e.g.,

           StepIndex(fibername='Nufern_Yb', Rout=10, R=2)

        The geometry and mesh in StepIndex objects are such that

          * region r < 1, in polar coords, is called "core",
          * region 1 < r < R   is called "clad",
          * region R < r < Rout   is called "pml",
          * when "R" is None, it is set to R = (Rout+1)/2,
          * index of refraction is set using StepIndexExact("fibername")
          * when "Rout" is unspecified, it is taken to match the ratio
            of cladding radius to core radius from StepIndexExact("fibername"),
          * cladding and pml meshsize is "h", while core mesh size
            is "hcore" (set to a default of hcore = h/10),
          * degree "p" finite element space is set on the mesh.
          * when dtemp is specified, the index of refraction is set
            to explicitly depend on temperature and the thermo-optic
            coefficient.

        (Variables beginning with capital R such as "R", "Rout" are
        nondimensional lengths -- in contrast, "rout" found in other classes
        is length in meters.)

        """

        if fibername is None and fiber is None:
            raise ValueError('Need either a fiber or fibername')

        self.makestepindex(fibername,
                           fiber,
                           R=R,
                           Rout=Rout,
                           geom=geom,
                           h=h,
                           hcore=hcore)
        self.makemesh(refine, curveorder)

        self.p = None  # degree of finite elements used in mode calc
        self.a = None
        self.b = None
        self.V = None
        self.ngspmlset = None  # True if ngsolve pml set (then cant reuse mesh)
        self.X = None
        self._curvature = None
        self._bendfactor = None

        self._dtemp = dtemp
        self.dndT = 1.285e-5

        self.setnondimmat()  # sets self.k and self.V
        L = self.fiber.rcore
        n0 = self.fiber.nclad
        super().__init__(self.mesh, L, n0)

    def __repr__(self):

        s = '\nStepIndex (ModeSolver) Object:' + '-' * 42
        s += '\nNondimensional Computational Parameters:'
        s += '\n  Geometry consists of circular core (radius = 1), an annular'
        s += '\n  cladding layer 1<r<R=%g, and an outer layer R<r<Rout=%g.'\
            % (self.R, self.Rout)
        s += '\n  Max mesh sizes: %g (core), %g (cladding), %g (outer)' \
            % (self.hcore, self.hclad, self.hpml)
        s += '\n  Mesh curved by order %d' % self.curveorder
        if self.ngspmlset:
            s += '\n  Mesh has been deformed by ngsolve PML'
        s += '\nPhysical Parameters:' + \
            '\n  Wavelength = %g meters' % (2*np.pi/self.fiber.ks)
        s += '\n  Refractive indices: %g (cladding), %g (core)' % \
            (self.fiber.nclad, self.fiber.ncore)
        if self._curvature is not None:
            s += '\n  StepIndexExact bending curvature = %g with ' % \
                self._curvature
            s += 'bend factor = %g' % self._bendfactor
        else:
            s += '\n  No curvature set'
        if self._dtemp is not None:
            s += '\n  Index modified by temperature differential %g and ' %\
                self._dtemp
            s += 'thermo-optic coefficient %g' % self.dndT
        else:
            s += '\n  No temperature set'

        s += '\nIncluded fiber object:\n%s' % self.fiber.__repr__()
        return s

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, curvature):
        self._curvature = curvature
        if self._bendfactor is None:
            self._bendfactor = 1.28
        self.setnondimmat()

    @property
    def bendfactor(self):
        return self._bendfactor

    @bendfactor.setter
    def bendfactor(self, bendfactor):
        self._bendfactor = bendfactor
        if self._curvature is None:
            self._curvature = 12
        self.setnondimmat()

    @property
    def dtemp(self):
        return self._dtemp

    @dtemp.setter
    def dtemp(self, dtemp):
        self._dtemp = dtemp
        self.setnondimmat()

    # FURTHER INITIALIZATIONS & SETTERS #####################################

    def makestepindex(self,
                      fibername=None,
                      fiber=None,
                      R=None,
                      Rout=None,
                      geom=None,
                      h=4,
                      hcore=None):

        if fibername is not None:
            self.fibername = fibername
            self.fiber = StepIndexExact(fibername)
        elif fiber is not None:
            self.fiber = fiber
        else:
            raise ValueError('Need fiber or fibername')

        if Rout is None:
            Rout = self.fiber.rclad / self.fiber.rcore
        if R is None:
            R = (Rout + 1) / 2
        if R < 1 or R > Rout:
            raise ValueError('Set R between 1 and Rout')
        self.R = R
        self.Rout = Rout

        if hcore is None:
            hcore = h / 10
        self.hcore = hcore
        self.hclad = h
        self.hpml = h

    def makemesh(self, refine=0, curveorder=3):
        self.setstepindexgeom()  # sets self.geo
        ngmesh = self.geo.GenerateMesh()
        for i in range(refine):
            ngmesh.Refine()
        mesh = ng.Mesh(ngmesh.Copy())
        mesh.ngmesh.SetGeometry(self.geo)
        self.curveorder = curveorder
        mesh.Curve(curveorder)
        ng.Draw(mesh)
        self.mesh = mesh

    def setstepindexgeom(self):
        geo = SplineGeometry()
        geo.AddCircle((0, 0),
                      r=self.Rout,
                      leftdomain=1,
                      rightdomain=0,
                      bc='OuterCircle')
        geo.AddCircle((0, 0),
                      r=self.R,
                      leftdomain=2,
                      rightdomain=1,
                      bc='cladbdry')
        geo.AddCircle((0, 0), r=1, leftdomain=3, rightdomain=2, bc='corebdry')
        geo.SetMaterial(1, 'Outer')
        geo.SetMaterial(2, 'clad')
        geo.SetMaterial(3, 'core')

        geo.SetDomainMaxH(1, self.hpml)
        geo.SetDomainMaxH(2, self.hclad)
        geo.SetDomainMaxH(3, self.hcore)

        self.geo = geo

    def setnondimmat(self):  # curvature=12, bendfactor=1.28, dtemp=10.0):
        """
        When a fiber of refractive index n is bent to have the
        input "curvature" (curvature = reciprocal of bending radius,
        since we assume bending along a perfect circle), the changed
        refractive index is modeled by the formula

            nbent = n * (1 + (x * curvature/bendfactor))

        with "bendfactor" as input - see [Schermer and Cole, 2007].
        This dimensional formula is used non-dimensionally below to
        set the internal data member "V", the non-dimensional
        coefficient function for the eigenproblem.

        """

        fib = self.fiber
        self.k = fib.ks

        if self._curvature is None:
            if self._dtemp is None:

                # the standard case with no curvature and no temperature
                V = fib.fiberV()
                self.V = CF([0, 0, -V * V])
                self.index = CF([fib.nclad, fib.nclad, fib.ncore])

            else:
                a = fib.rcore
                n = CF([fib.nclad, fib.nclad, fib.ncore]) + \
                    self.dndT * self._dtemp

                self.V = (a * fib.ks)**2 * (fib.nclad**2 - n**2)
                self.index = n
        else:
            if self._dtemp is None:
                n = CF([fib.nclad, fib.nclad, fib.ncore])
            else:
                n = CF([fib.nclad, fib.nclad, fib.ncore]) + \
                    self.dndT * self._dtemp

            a = fib.rcore
            ka2 = (fib.ks * a)**2
            kan2 = ka2 * (fib.nclad**2)

            nbent = n * (1 + (ng.x * a * self._curvature / self._bendfactor))
            self.index = nbent
            m = kan2 - ka2 * nbent * nbent
            self.V = CF([0, m, m])

    # MODE CALCULATORS AND RELATED FUNCTIONALITIES  #########################

    def Z2toX2(self, Z2, v=None):
        """Convert non-dimensional Z² values to non-dimensional X² values
        through the relation X² - Z² = V². """

        V = self.fiber.fiberV() if v is None else v
        Zsqr = np.array(Z2)
        Vsqr = V**2
        return Zsqr + Vsqr

    def X2toBeta(self, X2, v=None):
        """Convert non-dimensional X² values to dimensional propagation
        constants beta through the relation (ncore*k)² - (X/a)² = beta². """

        V = self.fiber.fiberV() if v is None else v
        a = self.fiber.rcore
        ks = V / (self.fiber.NA * a)
        Xsqr = np.array(X2)

        return np.sqrt((ks * self.fiber.ncore)**2 - Xsqr / a**2)

    def Z2toBeta(self, Z2, v=None):
        """Convert nondimensional Z² (input as "Z2") in the complex plane to
        complex propagation constant Beta. """

        return self.X2toBeta(self.Z2toX2(Z2, v=v), v=v)

    def guidedmodes(self,
                    interval=None,
                    p=3,
                    nquadpts=20,
                    seed=1,
                    nspan=15,
                    verbose=True,
                    **feastkwargs):
        """
        Search for guided modes in interval=(left, right). If interval is None,
        then an automatic choice will be made to include all guided modes.

        The computation is done using Lagrangre finite elements of
        degree "p", with no PML, using selfadjoint FEAST with a random span
        of "nspan" vectors, (and using the remaining parameters, which are
        simply passed to feast).

        OUTPUTS:

        betas, Zsqrs, Y: betas[i] give the i-th real-valued propagation
        constant and Zsqrs[i] gives the feast-computed i-th nondimensional
        Z² value in "interval". The corresponding eigenmode is i-th component
        of the span object Y.

        """

        V = self.fiber.fiberV()
        V = [V]
        k = [self.fiber.ks]
        betas = []
        Zsqrs = []
        Y = None
        fmind = [0]
        self.p = p

        for vnum, kk in zip(V, k):

            self.V = CF([0, 0, -vnum * vnum])
            self.k = kk

            if interval is None:
                # We choose the interval for the nondimensional Z² variable
                # recalling that  for guided modes,
                #         (L k₀ nclad)² < (β L)² < (L k₀ ncore)²,
                # where L is the scaling factor used to nondimensionalize.
                # It follows that Z² = (a α₀)² = (a k₀ nclad)² - (a β)²
                # satisfies
                #         0 > Z² > (a k₀ nclad)² - (a k₀ ncore)² = -V².
                interval = (-vnum * vnum, 0)

            betas_, Zsqrs_, Y_ =  \
                super().selfadjmodes(interval=interval, p=p, seed=seed,
                                     nspan=nspan, npts=nquadpts,
                                     verbose=verbose, **feastkwargs)

            betas = np.append(betas, betas_)
            Zsqrs = np.append(Zsqrs, Zsqrs_)
            if Y is None:
                Y = Y_
            else:
                for ind in range(len(betas_)):
                    Y._mv.Append(Y_._mv[ind])
                Y.m += len(betas_)
                fmind.append(fmind[-1] + len(betas_))
        fmind.append(len(betas))
        self.firstmodeindex = fmind
        self.X = Y.fes

        return betas, Zsqrs, Y

    def name2indices(self, betas, maxl=9, delta=None):
        """Given a numpy 1D array "betas" of approximations to
        propagation constants, produce a dictionary of mode names and
        corresponding exact propagation constants.

        OUTPUT of name2ind, exact = name2indices(betas)

            * name2ind is a dictionary such that beta[name2ind['LP01']]
              gives the beta corresponding to LP01 mod, etc.

            * exact[i] = i-th exact propagation constant obtained
              semi-analytically, to which beta[i] is an approximation.
              (We use from StepIndexExact for this.)

        OPTIONAL INPUTS:

            delta: consider numbers that differ by less than delta as
            approximations of a multiple eigenvalue.

            maxl: assume that betas correspond to LP(l,m) modes where l is
            less than maxl.
        """

        def construct_names(vnum, β):
            """
            constructs and saves LP names of propagation constants in β
            INPUTS:
                vnum: V-number in float
                β   : a numpy array containing propagation constants
            OUTPUTS:
                name2ind, exact: see self.name2indices docstring.
            """

            lft = self.Z2toBeta(0, v=vnum)  # βs must be in (lft, rgt)
            rgt = self.Z2toBeta(-vnum * vnum, v=vnum)
            # roughly identify simple and multiple ew approximants
            sm, ml = splitzoom.simple_multiple_zoom(lft, rgt, β, delta=delta)

            name2ind = {}
            exact = -np.ones_like(β)

            # l=0 case should be simple eigenvalues:
            activesimple = np.arange(len(sm['index']))
            LP0 = self.fiber.XtoBeta(self.fiber.propagation_constants(0,
                                                                      v=vnum),
                                     v=vnum)
            b = β[sm['index']]
            for m in range(len(LP0)):
                ind = np.argmin(abs(LP0[m] - b[activesimple]))
                i2beta = sm['index'][activesimple[ind]]
                name2ind['LP0' + str(m + 1)] = i2beta
                exact[i2beta] = LP0[m]
                activesimple = np.delete(activesimple, [ind])
                if len(activesimple) == 0:
                    break

            # l>0 cases should have multiplicity 2:
            activemultiple = np.arange(len(ml['index']))
            ctrs = np.array(ml['center'])
            for ll in range(1, maxl):
                LPl = self.fiber.XtoBeta(self.fiber.propagation_constants(
                    ll, v=vnum),
                                         v=vnum)
                for m in range(len(LPl)):
                    ind = np.argmin(abs(LPl[m] - ctrs[activemultiple]))
                    i2beta_a = ml['index'][activemultiple[ind]][0]
                    i2beta_b = ml['index'][activemultiple[ind]][1]
                    name2ind['LP' + str(ll) + str(m + 1) +
                             '_a'] = int(i2beta_a)
                    name2ind['LP' + str(ll) + str(m + 1) +
                             '_b'] = int(i2beta_b)
                    exact[i2beta_a] = LPl[m]
                    exact[i2beta_b] = LPl[m]
                    activemultiple = np.delete(activemultiple, ind)
                    if len(activemultiple) == 0:
                        return name2ind, exact
            return name2ind, exact

        V = self.fiber.fiberV()
        name2ind, exact = construct_names(V, betas)
        return name2ind, exact

    def corefraction(self, efs):
        """
        INPUT: "efs" multidimensional gridfunction with eigenmodes.
        OUTPUT: "cfs" list of fractions of energy in the core for each mode
                (where energy of mode u is measured via integral of |u|²).
        """
        cfs = []

        for i in range(len(efs.vecs)):
            total_energy = ng.Integrate(
                ng.InnerProduct(efs.MDComponent(i), efs.MDComponent(i)) *
                dx, self.mesh).real
            core_energy = ng.Integrate(
                ng.InnerProduct(efs.MDComponent(i), efs.MDComponent(i)) *
                dx("core"), self.mesh).real
            cfs.append(core_energy / total_energy)

        return cfs

    # INTERPOLATED MODES ####################################################

    def interpmodes(self, p):
        """
        Return interpolated modes as an NGvecs object
        and propagation constants as a list for supported fibers.

        Nufern Yb-doped: 4 modes and betas
        Nufern Tm-doped: 2 modes and betas
        LLMA Yb-doped: 23 modes and betas
        """
        self.p = p
        self.X = H1(self.mesh, order=p, dirichlet='OuterCircle', complex=True)

        if self.fibername == 'LLMA_Yb':
            simple = list(range(4))
            multi = [
                list(range(1, 9)),
                list(range(1, 7)),
                list(range(1, 5)),
                list(range(1, 2))
            ]
        elif self.fibername == 'Nufern_Yb':
            simple = list(range(2))
            multi = [list(range(1, 3))]
        elif self.fibername == 'Nufern_Tm':
            simple = list(range(1))
            multi = [list(range(1, 2))]
        else:
            errmsg = 'Interp. modes not available for {}'.format(
                self.fibername)
            raise NotImplementedError(errmsg)

        phi, β, n2i = self.modepropn2i(simple, multi)

        gf = ng.GridFunction(self.X)
        n, m = len(gf.vec), len(phi)
        y = np.zeros((n, m), dtype=complex)
        for j, f in enumerate(phi):
            gf = ng.GridFunction(self.X)
            gf.Set(f)
            y[:, j] = gf.vec.FV().NumPy()[:]
        Y = NGvecs(self.X, m)
        Y.fromnumpy(y)
        return β, n2i, Y

    def modepropn2i(self, simple, multi):
        """
        INPUTS:
        simple: list of 'm' indices for simple modes ('l'=0)
        multi: nested list of 'l' indices for multiple modes,
               where 'm' is implied by the ordering of sublists.

        OUTPUTS:
        modes: CFs for the fiber modes
        betas: Propagation constants
        name2ind: A 'name to index' dict which places propagation
                  constants in descending order.
        """
        simple_pairs = [self.interpmodeLP(0, i) for i in simple]
        simple_names = ['LP0{}'.format(i + 1) for i in simple]
        multi_pairs = [
            self.interpmodeLP(j, i) for i, lst in enumerate(multi) for j in lst
        ]
        multi_names = [
            'LP{}{}'.format(j, i + 1) for i, lst in enumerate(multi)
            for j in lst
        ]
        betas, modes = zip(*(simple_pairs + multi_pairs))
        triples = sorted(list(zip(betas, modes, simple_names + multi_names)),
                         reverse=True)
        betas, modes, names = zip(*triples)  # lists ordered by betas
        name2ind = dict(zip(names, range(len(names))))
        return modes, betas, name2ind

    # CONVENIENCE & DEBUGGING ###############################################

    def scipymats(self):
        """ Return scipy versions of matrices StepIndex.a and StepIndex.b,
        if these data members exist. (Also uses StepIndex.X freedofs.)"""

        if self.a is None or self.b is None or self.X is None:
            raise RuntimeError('Set a, b, and X before calling scipymats()')

        free = np.array(self.X.FreeDofs())
        freedofs = np.where(free)[0]
        i, j, avalues = self.a.mat.COO()
        A = coo_matrix((avalues.NumPy(), (i, j)))
        i, j, bvalues = self.b.mat.COO()
        B = coo_matrix((bvalues.NumPy(), (i, j)))
        A = A.tocsc()[:, freedofs]
        A = A.tocsr()[freedofs, :]
        B = B.tocsc()[:, freedofs]
        B = B.tocsr()[freedofs, :]
        return A, B, freedofs


# END OF CLASS DEFINITION ###################################################

# MODULE END #############################################################
