"""
This class models antiresonant hollow-core microstructured fibers
(HC ARF or ARF).  They are typically made of 6 or 8 thin tubes
surrounding a "core" region of air.
"""

import netgen.geom2d as geom2d
import ngsolve as ng
import numpy as np
from pyeigfeast.spectralproj.ngs import NGvecs
from fiberamp.fiber.modesolver import ModeSolver
from fiberamp.fiber import sellmeier
import pickle


class ARF(ModeSolver):

    def __init__(self, name=None, freecapil=False,
                 outermaterials=None, curve=3, refine=0,
                 e=None, **kwargs):
        """
        PARAMETERS:

           name: If None, default to using the ARF microstructure fiber
                with 6 capillary tubes. Otherwise, set geometric fibers
                for the fiber based on the name provided.

           freecapil: If True, capillary tubes in the microstructure will
                be modeled as free standing in the hollow region.
                Otherwise, they will be embedded into the glass sheath.

           outermaterials: A string (or iterable) containing the material
                specification for each outer region 'OuterAir' and
                'Outer'. The index of refraction of each such region is
                set accordingly. If a string or an iterable of length 1
                is given, then we assume all outer regions to have the
                index of refraction of that material. Otherwise, None
                gives the default setting of all air outside of the physical
                fiber cross-section.

           kwargs: Override default values of updatable length attributes.
                Give length values in units of micrometers, e.g.,
                    ARF(touter=15)
                yields a PML thickness of 15 micrometers in physical units.
                A keyword argument 'scaling', if given, will also divide
                the updatable length attributes by 'scaling'.
        """

        self.freecapil = freecapil

        # Set the fiber parameters.
        self.set(name=name, e=e)

        # Updatable length attributes. All lengths are in micrometers.

        self.updatablelengths = ['Rc', 'Rto', 'Rti', 't', 'd', 'tclad',
                                 'touter', 'touterair']

        # Physical parameters

        self.n_air = 1.00027717     # refractive index of air
        self.n_si = sellmeier.index(self.wavelength, material='FusedSilica')
        self.NA_pol = 0.46
        self.n_pol = np.sqrt(self.n_si**2 - self.NA_pol**2)

        # UPDATE attributes set in ARF.set() using given inputs

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set the scaling as requested by the user.
        if 'scaling' in kwargs:
            self.scaling = kwargs['scaling']

        # Scale updatablelengths and store scaled values in class attributes
        # whose name has an 's' appended. Only the scaled values are used
        # for geometry and mesh construction.
        for key in self.updatablelengths:
            setattr(self, key + 's', getattr(self, key)/self.scaling)

        # attributes in addition to updatablelengths for reconstructing obj
        #    (don't save scaling: avoid re-re-scaling!)
        self.savableattr = ['freecapil', 'n_air', 'n_si', 'wavelength',
                            'e', 's', 'scaling', 'refined']

        if self.freecapil:
            # outer radius of glass sheath
            self.Rclado = self.Rcs + 2 * self.Rtos + self.tclads
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + 2 * self.Rtos
        else:
            # radius where glass sheath (cladding) begins
            self.Rcladi = self.Rcs + self.ts + 2*self.Rtis + self.ts*(1-self.e)
            # outer radius of glass sheath
            self.Rclado = self.Rcladi + self.tclads

        # final radius where geometry ends is Rout
        self.R = self.Rclado + self.touterairs
        self.Rout = self.R + self.touters

        # BOUNDARY & MATERIAL NAMES

        self.material = {
            'Outer': 1,          # outer most annular layer (PML)
            'OuterAir': 2,       # air outside jacket
            'Si': 3,             # cladding & capillaries are glass
            'CapillaryEncl': 4,  # air regions enclosed by capillaries
            'InnerCore': 5,      # inner hollow core (air) region r < Rc
            'FillAir': 6,        # remaining intervening spaces (air)
        }
        mat = self.material
        self.boundary = {
            #              [left domain, right domain]    while going ccw
            'OuterCircle': [mat['Outer'], 0],

            #              [left domain, right domain]    while going ccw
            'AirCircle': [mat['OuterAir'], mat['Outer']],

            # circle  separating outer most layer from cladding
            'OuterClad':   [mat['Si'], mat['OuterAir']],

            # inner circular boundary of capillary tubes
            'CapilInner':  [mat['CapillaryEncl'], mat['Si']],

            # artificial inner circular core boundary
            'Inner':       [mat['InnerCore'], mat['FillAir']],

            # outer boundary of capillaries and the inner boundary of
            # sheath/cladding together forms one curve in the case
            # when capillaries are pushed into  the cladding (this curve
            # is absent in the freestanding case):
            'CapilOuterCladInner': [mat['FillAir'], mat['Si']],

            # in the freestanding capillary case, the inner boundary of
            # cladding/sheath is disconnected  from the outer boundaries of
            # capillaries, so we have these two curves (which do not
            # exist in the embedded capillary case).
            'CladInner':  [mat['FillAir'], mat['Si']],
            'CapilOuter': [mat['Si'], mat['FillAir']],
        }

        # Before creating the mesh, check to make sure that the geometric
        # parameters lead to a mesh with non-tangent and non-overlapping
        # subdomains (overlapping and tangent subdomains can cause netgen
        # to crash).
        self._check_geometric_parameters()

        # CREATE GEOMETRY & MESH
        if self.freecapil:
            self.geo = self.geom_freestand_capillaries()
        else:
            self.geo = self.geom_embedded_capillaries()

        # Set the materials for the domain.
        for material, domain in self.material.items():
            self.geo.SetMaterial(domain, material)

        # Set the maximum mesh sizes in subdomains
        self.geo.SetDomainMaxH(mat['Outer'], self.outer_maxhs)
        self.geo.SetDomainMaxH(mat['OuterAir'], self.outer_maxhs)
        self.geo.SetDomainMaxH(mat['Si'], self.glass_maxhs)
        self.geo.SetDomainMaxH(mat['CapillaryEncl'], self.air_maxhs)
        self.geo.SetDomainMaxH(mat['InnerCore'], self.inner_core_maxhs)
        self.geo.SetDomainMaxH(mat['FillAir'], self.air_maxhs)

        # Generate Mesh
        if 'ngmesh' in kwargs:
            print('  Using input mesh.')
            self.mesh = ng.Mesh(kwargs['ngmesh'])
            self.mesh.ngmesh.SetGeometry(self.geo)
        else:
            print('  Generating new mesh.')
            ngmesh = self.geo.GenerateMesh()
            self.mesh = ng.Mesh(ngmesh)

        self.refine(n=refine, curve=curve)

        # MATERIAL COEFFICIENTS

        # Check the outer materials to see if any are valid.
        acceptable_materials = ('air', 'silica', 'polymer')

        # Make the outer materials tuple contain two elements as needed.
        if outermaterials is None:
            self.outermaterials = ('air',) * 2
        elif isinstance(outermaterials, str):
            self.outermaterials = (outermaterials,) * 2
        elif len(outermaterials) == 0:
            self.outermaterials = ('air',) * 2
        elif len(outermaterials) == 1:
            self.outermaterials = (outermaterials[0],) * 2
        else:
            self.outermaterials = outermaterials[:2]

        if (not (self.outermaterials[0] in acceptable_materials)) or \
           (not (self.outermaterials[1] in acceptable_materials)):
            raise ValueError('Outer materials must be one of \'air\',' +
                             '\'polymer\', or \'silica\'.')

        # Set the material parameters based on the outermaterials tuple.
        n_outerair = self.n_air
        if self.outermaterials[0] == 'silica':
            n_outerair = self.n_si
        elif self.outermaterials[0] == 'polymer':
            n_outerair = self.n_pol

        n_outer = self.n_air
        if self.outermaterials[1] == 'silica':
            n_outer = self.n_si
        elif self.outermaterials[1] == 'polymer':
            n_outer = self.n_pol

        # index of refraction
        self.indexdict = {'Outer':         n_outer,
                          'OuterAir':      n_outerair,
                          'Si':            self.n_si,
                          'CapillaryEncl': self.n_air,
                          'InnerCore':     self.n_air,
                          'FillAir':       self.n_air}
        self.index = ng.CoefficientFunction(
            [self.indexdict[mat] for mat in self.mesh.GetMaterials()])

        self.setnondimmat()  # sets self.V and self.k
        L = self.scaling * 1e-6
        n0 = self.indexdict['Outer']
        super().__init__(self.mesh, L, n0)


    @classmethod
    def from_dict(cls, d):
        """
        Create an ARF object from a dictionary.
        """
        return cls(**d)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, lam):
        self._wavelength = lam
        self.n_si = sellmeier.index(self.wavelength, material='FusedSilica')
        self.n_pol = np.sqrt(self.n_si**2 - self.NA_pol**2)

        # Set the material parameters based on the outermaterials tuple.
        n_outerair = self.n_air
        if self.outermaterials[0] == 'silica':
            n_outerair = self.n_si
        elif self.outermaterials[0] == 'polymer':
            n_outerair = self.n_pol

        n_outer = self.n_air
        if self.outermaterials[1] == 'silica':
            n_outer = self.n_si
        elif self.outermaterials[1] == 'polymer':
            n_outer = self.n_pol

        # Reset index of refraction
        self.indexdict = {'Outer':         n_outer,
                          'OuterAir':      n_outerair,
                          'Si':            self.n_si,
                          'CapillaryEncl': self.n_air,
                          'InnerCore':     self.n_air,
                          'FillAir':       self.n_air}
        self.index = ng.CoefficientFunction(
            [self.indexdict[mat] for mat in self.mesh.GetMaterials()])

        self.setnondimmat()

    def setnondimmat(self):
        """ set the material cf """

        a = self.scaling * 1e-6
        k = 2 * np.pi / self.wavelength
        idx = self.indexdict
        m = {'Outer':         0,
             'OuterAir':      idx['Outer']**2 - idx['OuterAir']**2,
             'Si':            idx['Outer']**2 - idx['Si']**2,
             'CapillaryEncl': idx['Outer']**2 - idx['CapillaryEncl']**2,
             'InnerCore':     idx['Outer']**2 - idx['InnerCore']**2,
             'FillAir':       idx['Outer']**2 - idx['FillAir']**2}

        self.V = ng.CoefficientFunction(
            [(a*k)**2 * m[mat] for mat in self.mesh.GetMaterials()])
        self.k = k

    def set(self, name=None, e=None):
        """
        Method that sets the geometric parameters of the ARF fiber based
        on the name supplied by the user. Specifying name='poletti'
        yields a six-capillary fiber, while name='kolyadin' specifies
        an 8-capillary fiber.

        Attributes specific to the embedded capillary case:

           e = fraction of the capillary tube thickness that
               is embedded into the adjacent silica layer. When
               e=0, the outer circle of the capillary tube
               osculates the circular boundary of the silica
               layer. When e=1, the inner circle of the capillary
               tube is tangential to the circular boundary (and
               the outer circle is embedded). Value of e must be
               strictly greater than 0 and less than or equal to 1.

        Attributes used only in the freestanding capillary case:

           s = separation of the hollow capillary tubes as a
               percentage of the radial distance from the center of
               the fiber to the center of capillary tubes (which
               would be tangential to the outer glass jacket when s=0).
        """

        # By default, we'll use the 6-capillary fiber.
        self.name = 'poletti' if name is None else name

        if self.name == 'poletti':
            # This case gives the default attributes of the fiber.

            self.Rc = 15                  # core radius
            self.Rto = 12.9               # capillary outer radius
            self.Rti = 12.48              # capillary inner radius
            self.t = self.Rto - self.Rti  # capillary thickness
            self.tclad = 10               # glass jacket (cladding) thickness
            self.touter = 50              # outer PML thickness
            self.touterair = 10
            self.scaling = self.Rc        # scaling for the PDE
            self.num_capillary_tubes = 6  # number of capillaries
            self.s = 0.05
            if e is not None:
                self.e = e
            else:
                self.e = 0.025 / self.t
            self._wavelength = 1.8e-6

            # Compute the capillary tube separation distance.
            half_sector = np.pi / self.num_capillary_tubes
            D = 2 * (self.Rc + self.Rto) * np.sin(half_sector)
            self.d = D - 2 * self.Rto

            # Set the (non-dimensional) mesh sizes.
            self.capillary_maxhs = 0.04
            self.air_maxhs = 0.25
            self.inner_core_maxhs = 0.1
            self.glass_maxhs = 0.33
            self.outer_maxhs = 2.0

            self.refined = 0

        elif self.name == 'kolyadin':
            self.Rc = 59.5                # core radius
            self.Rto = 31.5               # capillary outer radius
            self.Rti = 25.5               # capillary inner radius
            self.t = self.Rto - self.Rti  # capillary thickness
            self.tclad = 1.2 * self.Rti   # glass jacket (cladding) thickness
            self.touter = 100             # outer jacket (PML) thickness
            self.touterair = self.tclad
            self.scaling = self.Rc        # scaling for the PDE
            self.num_capillary_tubes = 8  # number of capillaries
            self.s = 0.05
            if e is not None:
                self.e = e
            else:
                self.e = 2.0 / self.t
            self._wavelength = 5.75e-6

            # Compute the capillary tube separation distance.
            half_sector = np.pi / self.num_capillary_tubes
            D = 2 * (self.Rc + self.Rto) * np.sin(half_sector)
            self.d = D - 2 * self.Rto

            # Set the (non-dimensional) mesh sizes.
            self.capillary_maxhs = 0.02
            self.air_maxhs = 0.2
            self.inner_core_maxhs = 0.1
            self.glass_maxhs = 0.4
            self.outer_maxhs = 0.5

            self.refined = 0
        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

        self.epw = self.eltsperwave()

    def __str__(self):
        s = 'ARF Physical Parameters:' + \
            '\n  Rc = %g x %g x 1e-6 meters' % (self.Rcs, self.scaling)
        s += '\n  tclad = %g x %g x 1e-6 meters' % (self.tclads, self.scaling)
        s += '\n  touter = %g x %g x 1e-6 meters' % \
            (self.touters, self.scaling)
        s += '\n  t = %g x %g x 1e-6 meters' % (self.ts, self.scaling)
        s += '\n  d = %g x %g x 1e-6 meters' % (self.ds, self.scaling)
        s += '\n  Rti = %g x %g x 1e-6 meters' % (self.Rti, self.scaling)
        s += '\n  Rto = %g x %g x 1e-6 meters' % (self.Rto, self.scaling)
        s += '\n  Wavelength = %g meters' % self.wavelength
        s += '\n  Refractive indices: %g (air), %g (Si)' % \
            (self.n_air, self.n_si)

        s += '\nNondimensional Computational Parameters:'
        s += '\n  Divide all lengths above by %g x 1e-6' % self.scaling
        s += '\n  to get the actual computational lengths used.'
        s += '\n  Cladding starts at Rcladi = %g' % self.Rcladi
        s += '\n  PML starts at R = %g and ends at Rout = %g' \
            % (self.R, self.Rout)
        s += '\n  Mesh sizes: %g (capillary), %g (air), %g (inner core)' \
            % (self.capillary_maxhs, self.air_maxhs, self.inner_core_maxhs)
        s += '\n  Mesh sizes: %g (glass), %g (outer)'  \
            % (self.glass_maxhs,   self.outer_maxhs)
        s += '\n  Elements/wavelength:'
        epw = self.epw
        s += '%g (capillary), %g (air), %g (inner core)' \
            % (epw['capillary'], epw['air'], epw['inner'])
        s += '\n  Elements/wavelength: %g (glass), %g (outer)'  \
            % (epw['glass'], epw['outer'])

        if self.refined > 0:
            s += '\n  Uniformly refined %g times.' % self.refined
        if self.freecapil:
            s += '\n  With free capillaries, s = %g.' % self.s
        else:
            s += '\n  With embedded capillaries, e/t = %g.' % self.e
        return s

    def eltsperwave(self):
        epw = {'capillary': 1/self.capillary_maxhs,
               'air': 1/self.air_maxhs,
               'inner': 1/self.inner_core_maxhs,
               'glass': 1/self.glass_maxhs,
               'outer': 1/self.outer_maxhs}
        for key in epw:
            epw[key] = epw[key] * self.wavelength*1e6/self.scaling
        return epw

    # GEOMETRY ########################################################

    def _check_geometric_parameters(self):
        """
        Method that checks to make sure there are no overlapping
        capillary tubes, or that other parameters provided for the
        geometry are not erroneous.
        """

        if self.freecapil:
            # First, given the current geometric parameters, compute
            # an upper bound on the capillary contraction parameters.
            half_sector = np.pi / self.num_capillary_tubes
            frac = self.Rtos / ((self.Rcs + self.Rtos) * np.sin(half_sector))
            sub = 1 - frac

            # Next, given the current geometric parameters, compute
            # an upper bound on the number of capillary tubes and
            # check the current specified number of capillary tubes
            # against this.
            asin_frac = self.Rtos / ((1 - self.s) * (self.Rcs + self.Rtos))
            nub = int(np.floor(np.pi / np.arcsin(asin_frac)))

            # A placeholder for the number of user-specified tubes.
            n = self.num_capillary_tubes

            if n > nub:
                # Set a new upper bound on s that uses the maximum lower
                # bound on the number of capillary tubes.
                frac = self.Rtos / ((self.Rcs + self.Rtos) *
                                    np.sin(np.pi / nub))
                new_sub = sub if sub > 0 else 1 - frac

                err_str = 'Specifying {0:d} capillary tube(s) '.format(n) \
                    + 'results in tangent or overlapping capillary' \
                    + 'subdomains. Consider setting ' \
                    + '\'num_capillary_tubes\' less than or equal ' \
                    + 'to {:d} and then'.format(nub) \
                    + 'setting s < {:.3f},'.format(new_sub) \
                    + 'or adjusting other geometric parameters.'
        else:
            # For the embedded case, we first check to make sure the embed
            # fraction e gives us a resulting geometry that doesn't have
            # tangent subdomain boundaries (e = 0) or capillary tubes that
            # get pushed into the outer glass jacket.
            if self.e <= 0 or self.e > 1:
                err_str = 'Current value of e = {:.3f}'.format(self.e) \
                    + 'results in capillary tubes in invalid ' \
                    + 'positions. The embedding fraction \'e\'' \
                    + 'must be a real number satisfying ' \
                    + '0 < e <= 1.'
                raise ValueError(err_str)

            # Next, given the current geometric parameters, compute
            # an upper limit on the number of capillary tubes and
            # check the current specified number of capillary tubes
            # against this.
            asin_frac = self.Rtos / (self.Rcs + self.Rtos)
            nub = int(np.floor(np.pi / np.arcsin(asin_frac)))

            # A placeholder for the number of user-specified tubes.
            n = self.num_capillary_tubes

            if n > nub:
                err_str = 'Specifying {0:d} capillary tube(s) '.format(n) \
                    + 'results in tangent or overlapping capillary' \
                    + 'subdomains. Consider making ' \
                    + '\'num_capillary_tubes\' less than or equal to ' \
                    + '{0:d}'.format(nub) \
                    + 'or adjusting other geometric parameters.'
                raise ValueError(err_str)

    def geom_freestand_capillaries(self):

        geo = geom2d.SplineGeometry()
        bdr = self.boundary

        # The outermost circle
        geo.AddCircle(c=(0, 0), r=self.Rout,
                      leftdomain=bdr['OuterCircle'][0], rightdomain=0,
                      bc='OuterCircle')

        # The air-pml interface
        geo.AddCircle(c=(0, 0), r=self.R,
                      leftdomain=bdr['AirCircle'][0],
                      rightdomain=bdr['AirCircle'][1], bc='AirCircle')

        # The glass sheath
        geo.AddCircle(c=(0, 0), r=self.Rclado,
                      leftdomain=bdr['OuterClad'][0],
                      rightdomain=bdr['OuterClad'][1], bc='OuterClad')
        geo.AddCircle(c=(0, 0), r=self.Rcladi,
                      leftdomain=bdr['CladInner'][0],
                      rightdomain=bdr['CladInner'][1],
                      bc='CladInner')

        # --------------------------------------------------------------------
        # Add capillary tubes.
        # --------------------------------------------------------------------

        # Spacing for the angles we need to add the inner circles for the
        # capillaries.
        theta = np.pi / 2.0 + np.linspace(0, 2*np.pi,
                                          num=self.num_capillary_tubes,
                                          endpoint=False)

        # The radial distance to the capillary tube centers.
        dist = (1 - self.s) * (self.Rcs + self.Rtos)

        for t in theta:
            c = (dist*np.cos(t), dist*np.sin(t))

            geo.AddCircle(c=c, r=self.Rtis,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)

            geo.AddCircle(c=c, r=self.Rtos,
                          leftdomain=bdr['CapilOuter'][0],
                          rightdomain=bdr['CapilOuter'][1],
                          bc='CapilOuter', maxh=self.capillary_maxhs)

        # Add the circle for the inner core. Since we are scaling back the
        # (original) distance to the capillary tube centers by (1 - s), we
        # necessarily need to do the same for the inner core region.
        radius = 0.9 * self.Rcs * (1 - self.s)
        geo.AddCircle(c=(0, 0), r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxhs)

        return geo

    def geom_embedded_capillaries(self):
        # Grab the boundary dictionary and create the spline geometry.
        bdr = self.boundary
        geo = geom2d.SplineGeometry()

        # The outermost circle
        geo.AddCircle(c=(0, 0), r=self.Rout,
                      leftdomain=bdr['OuterCircle'][0], rightdomain=0,
                      bc='OuterCircle')

        # The air-pml interface
        geo.AddCircle(c=(0, 0), r=self.R,
                      leftdomain=bdr['AirCircle'][0],
                      rightdomain=bdr['AirCircle'][1], bc='AirCircle')

        # Cladding begins here
        geo.AddCircle(c=(0, 0), r=self.Rclado,
                      leftdomain=bdr['OuterClad'][0],
                      rightdomain=bdr['OuterClad'][1], bc='OuterClad')

        # Inner portion:

        # The angle 'phi' corresponds to the polar angle that gives the
        # intersection of the two circles of radius Rcladi and Rto, resp.
        # The coordinates of the intersection can then be recovered as
        # (Rcladi * cos(phi), Rcladi * sin(phi)) and
        # (-Rcladi * cos(phi), Rcladi * sin(phi)).

        numerator = self.Rcladi**2 + (self.Rcs + self.Rtos)**2 - self.Rtos**2
        denominator = 2 * (self.Rcs + self.Rtos) * self.Rcladi
        acos_frac = numerator / denominator
        phi = np.arccos(acos_frac)

        # The angle of a given sector that bisects two adjacent capillary
        # tubes. Visually, this looks like a wedge in the computational domain
        # that contains a half capillary tube on each side of the widest part
        # of the wedge.
        sector = 2 * np.pi / self.num_capillary_tubes

        # Obtain the angle of the arc between two capillaries. This subtends
        # the arc between the two points where two adjacent capillary tubes
        # embed into the outer glass jacket.
        psi = sector - 2 * phi

        # Get the distance to the middle control point for the aforementioned
        # arc.
        D = self.Rcladi / np.cos(psi / 2)

        # The center of the top capillary tube.
        c = (0, self.Rcs + self.Rtos)

        capillary_points = []

        for k in range(self.num_capillary_tubes):
            # Compute the rotation angle needed for rotating the north
            # capillary spline points to the other capillary locations in the
            # domain.
            rotation_angle = k * sector

            # Compute the middle control point for the outer arc subtended by
            # the angle psi + rotation_angle.
            ctrl_pt_angle = np.pi / 2 - phi - psi / 2 + rotation_angle
            capillary_points += [(D * np.cos(ctrl_pt_angle),
                                  D * np.sin(ctrl_pt_angle))]

            # Obtain the control points for the capillary tube immediately
            # counterclockwise from the above control point.
            capillary_points += \
                self.get_capillary_spline_points(c, phi, rotation_angle)

        # Add the capillary points to the geometry
        capnums = [geo.AppendPoint(x, y) for x, y in capillary_points]
        NP = len(capillary_points)    # number of capillary point IDs.
        for k in range(1, NP + 1, 2):  # add the splines.
            geo.Append(
                [
                    'spline3',
                    capnums[k % NP],
                    capnums[(k + 1) % NP],
                    capnums[(k + 2) % NP]
                ],
                leftdomain=bdr['CapilOuterCladInner'][0],
                rightdomain=bdr['CapilOuterCladInner'][1],
                bc='CapilOuterCladInner'
            )

        # --------------------------------------------------------------------
        # Add capillary tubes.
        # --------------------------------------------------------------------

        # Spacing for the angles we need to add the inner circles for the
        # capillaries.
        theta = np.pi / 2.0 + np.linspace(0, 2*np.pi,
                                          num=self.num_capillary_tubes,
                                          endpoint=False)

        # The radial distance to the capillary tube centers.
        dist = self.Rcs + self.Rtos

        for t in theta:
            c = (dist*np.cos(t), dist*np.sin(t))

            geo.AddCircle(c=c, r=self.Rtis,
                          leftdomain=bdr['CapilInner'][0],
                          rightdomain=bdr['CapilInner'][1],
                          bc='CapilInner', maxh=self.capillary_maxhs)

        # Add the circle for the inner core.
        radius = 0.75 * self.Rcs
        geo.AddCircle(c=(0, 0), r=radius,
                      leftdomain=bdr['Inner'][0],
                      rightdomain=bdr['Inner'][1],
                      bc='Inner', maxh=self.inner_core_maxhs)

        return geo

    def get_capillary_spline_points(self, c, phi, rotation_angle):
        """
        Method that obtains the spline points for the interface between one
        capillary tube and inner hollow core. By default, we generate the
        spline points for the topmost capillary tube, and then rotate
        these points to generate the spline points for another tube based
        upon the inputs.

        INPUTS:

        c  = the center of the northern capillary tube

        phi = corresponds to the polar angle that gives theintersection of
          the two circles of radius Rcladi and Rto, respectively. In this
          case, the latter circle has a center of c given as the first
          argument above.

        rotation_angle = the angle by which we rotate the spline points
          obtained for the north circle to obtain the spline
          points for another capillary tube in the fiber.

        OUTPUTS:

        A list of control point for a spline that describes the interface.
        """

        # Start off with an array of x- and y-coordinates. This will
        # make any transformations to the points easier to work with.
        points = np.zeros((2, 9))

        # Compute the angle inside of the capillary tube to determine some
        # of the subsequent spline points for the upper half of the outer
        # capillary tube.
        acos_frac = (self.Rcladi * np.sin(phi) - c[0]) / self.Rtos
        psi = np.arccos(acos_frac)

        # The control points for the first spline.
        points[:, 0] = [np.cos(psi), np.sin(psi)]
        points[:, 1] = [1, (1 - np.cos(psi)) / np.sin(psi)]
        points[:, 2] = [1, 0]

        # Control points for the second and third splines.
        points[:, 3] = [1, -1]
        points[:, 4] = [0, -1]
        points[:, 5] = [-1, -1]

        # Control points for the final spline.
        points[:, 6] = [-1, 0]
        points[:, 7] = [-1, (1 - np.cos(psi)) / np.sin(psi)]
        points[:, 8] = [-np.cos(psi), np.sin(psi)]

        # The rotation matrix needed to generate the spline points for an
        # arbitrary capillary tube.
        R = np.mat(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)]
            ]
        )

        # Rotate, scale, and shift the points.
        points *= self.Rtos
        points[0, :] += c[0]
        points[1, :] += c[1]
        points = np.dot(R, points)

        # Return the points as a collection of tuples.
        capillary_points = []

        for k in range(0, 9):
            capillary_points += [(points[0, k], points[1, k])]

        return capillary_points

    def refine(self, n=1, curve=3):
        """Uniformly refine mesh n times and set mesh curvature."""

        print('  Refining ARF mesh uniformly ' + str(n) + ' times:\
 each element split into four')
        self.refined += n
        for i in range(n):
            self.mesh.ngmesh.Refine()
            self.mesh.ngmesh.SetGeometry(self.geo)
            self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.curve(curve)
        for key in self.epw:
            self.epw[key] = self.epw[key] * (2 ** n)
        s = '  Elements/wavelength revised:'
        s += '%g (capillary), %g (air), %g (inner core)' \
            % (self.epw['capillary'], self.epw['air'], self.epw['inner'])
        s += '\n  Elements/wavelength revised: %g (glass), %g (outer)'  \
            % (self.epw['glass'], self.epw['outer'])
        print(s)

    def curve(self, curve=3):
        self.mesh.Curve(curve)

