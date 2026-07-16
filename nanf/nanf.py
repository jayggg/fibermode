#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:04 2022

@author: pv
"""

import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from fibermode.solvers import ModeSolver
from pyeigfeast.spectralproj.ngs import NGvecs


class NANF(ModeSolver):
    """
    Create Nested Antiresonant Nodeless Fiber.
    """

    def __init__(self, r_core=15e-6,
                 capillary_info=[
                     {'N': 6, 'r': 12.48e-6, 't': .42e-6, 'e': 1},
                     {'N': 1, 'r': 6e-6, 't': .42e-6, 'e': 1},
                     # {'N': 1, 'r': 3e-6, 't': .42e-6, 'e': 1},
                 ],
                 scale=15e-6,
                 t_cladding=10e-6,
                 t_buffer=15e-6,
                 t_outer=20e-6,
                 glass_maxh=.06,
                 air_maxh=.1,
                 core_maxh=.08,
                 buffer_maxh=.2,
                 pml_maxh=.1,
                 wl=1.8e-6,
                 refine=0,
                 curve=3,
                 poly_core=True,
                 core_factor=.8,
                 outer_materials=None,
                 ):

        self.scale = scale
        self.R_core = r_core / scale
        self.R_core_comp = core_factor * self.R_core

        self.Rs = [0]
        self.Ts = [t_cladding/scale]
        self.Ns = [0]
        self.es = [0]

        for i, D in enumerate(capillary_info):
            for key, value in D.items():
                if key == 'r':
                    self.Rs.append(value/scale)
                elif key == 't':
                    self.Ts.append(value/scale)
                elif key == 'N':
                    self.Ns.append(value)
                elif key == 'e':
                    self.es.append(value)
                else:
                    raise ValueError('Key not recognized.')

        self.Rs[0] = self.R_core + 2 * self.Rs[1] + (2 - self.es[1])*self.Ts[1]
        self.Cs = [0] * len(self.Ts)

        for i in range(len(self.Ts) - 1):
            self.Cs[i+1] = self.Cs[i] + self.Rs[i] - self.Rs[i+1] - \
                (1 - self.es[i+1]) * self.Ts[i+1]

        self.check_parameters()

        self.phis = [0] * len(self.Ts)
        self.psis = [0] * len(self.Ts)

        for i in range(len(self.Ts) - 1):

            frac = ((self.Cs[i+1] - self.Cs[i])**2 + self.Rs[i]**2 -
                    (self.Rs[i+1] + self.Ts[i+1])**2) / \
                (2 * (self.Cs[i+1] - self.Cs[i]) * self.Rs[i])

            self.phis[i+1] = np.arccos(frac)

            frac = self.Rs[i] * np.sin(self.phis[i+1]) / \
                (self.Rs[i+1] + self.Ts[i+1])

            self.psis[i+1] = np.arccos(frac)

        self.T_outer = t_outer / scale
        self.T_buffer = t_buffer / scale

        self.wavelength = wl
        # self.n_glass = sellmeier.index(wl, material='FusedSilica')
        self.n_glass = 1.44
        self.n_air = 1

        self.n_buffer = self.n_air
        self.n0 = self.n_air

        self.air_maxh = air_maxh

        self.core_maxh = core_maxh
        self.glass_maxh = glass_maxh

        if outer_materials is not None:
            self.outer_materials = outer_materials
            self.n0 = outer_materials[-1]['n']  # Need to reset n0
        else:
            self.outer_materials = [
                {'material': 'buffer',
                 'n': self.n_buffer,
                 'T': self.T_buffer,
                 'maxh': .5},

                {'material': 'Outer',
                 'n': self.n0,
                 'T': self.T_outer,
                 'maxh': 4}
            ]

        # Create geometry
        self.create_geometry(poly_core=poly_core)
        self.create_mesh(refine=refine, curve=curve)

        # Set physical properties
        self.set_material_properties()

        super().__init__(self.mesh, self.scale, self.n0)

    @classmethod
    def from_dict(cls, d):
        """
        Create NANF from dictionary.
        """
        new_dict = {}
        list_of_keys = ['r_core', 'capillary_info', 'scale', 't_cladding',
                        't_buffer', 't_outer', 'glass_maxh', 'air _maxh',
                        'core_maxh', 'buffer_maxh', 'pml_maxh', 'wl',
                        'refine', 'curve', 'poly_core', 'core_factor',
                        'outer_materials']
        for key in list_of_keys:
            if key in d:
                new_dict[key] = d[key]
        return cls(**d)

    def check_parameters(self):
        """Check to ensure given parameters give valid geometry."""
        es = np.array(self.es)
        if np.any(es[1:] <= 0) or np.any(es[1:] > 1):
            raise ValueError('Embedding parameter e must be strictly greater\
 than zero and less than or equal to one.')

        for i in range(1, len(self.Ns)):
            n = self.Ns[i]
            if n == 1:
                if self.Rs[i-1] <= self.Rs[i] + \
                        (1-.5*self.es[i])*self.Ts[i]:
                    raise ValueError('Single nested capillary larger than \
tube containing it at layer %i' % i)
            if n >= 2:
                if (self.Cs[i]-self.Cs[i-1]) * np.sin(np.pi / n) \
                        <= self.Rs[i] + self.Ts[i]:
                    raise ValueError('Capillary tubes at layer %i \
overlap each other.' % (i))

    def mesh_maxh(self, material=None):

        def maxlength(pts):
            L = np.linalg.norm(
                [pts[1]-pts[0], pts[2] - pts[1], pts[0] - pts[2]], axis=1)
            return np.max(L)

        if material is not None:
            elts = [v for v in self.mesh.Elements() if v.mat == material]
        else:
            elts = [v for v in self.mesh.Elements()]

        diams = [maxlength([np.array(self.mesh[el.vertices[i]].point)
                           for i in range(3)]) for el in elts]

        return max(diams)

    def number_of_elts(self, material=None):
        if material is not None:
            elts = [v for v in self.mesh.Elements() if v.mat == material]
        else:
            elts = [v for v in self.mesh.Elements()]
        return len(elts)

    def elements_per_wavelength(self, material=None):
        lambda_s = self.wavelength / self.scale
        mat_maxh = self.mesh_maxh(material=material)
        return lambda_s / mat_maxh

    def set_material_properties(self):
        """
        Set k0, refractive indices, and V function.
        """
        self.k = 2 * np.pi / self.wavelength

        # Set up inner region refractive indices
        refractive_index_dict = {'core': self.n_air,
                                 'air': self.n_air,
                                 'glass': self.n_glass,
                                 }

        # Add outer material refractive indices
        for i, d in enumerate(self.outer_materials):
            refractive_index_dict[d['material']] = d['n']

        self.refractive_index_dict = refractive_index_dict
        self.index = self.mesh.RegionCF(ng.VOL, refractive_index_dict)

        self.V = (self.scale * self.k)**2 * (self.n0 ** 2 - self.index ** 2)

    def create_mesh(self, refine=0, curve=3):
        """
        Create mesh from geometry.
        """
        self.mesh = ng.Mesh(self.geo.GenerateMesh())

        self.refinements = 0
        for i in range(refine):
            self.mesh.ngmesh.Refine()

        self.refinements += refine
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(curve)

    def create_geometry(self, poly_core=True):

        geo = SplineGeometry()

        N = self.Ns[1]

        if N <= 2 and poly_core:
            print('Polygonal core only available for n_tubes > 2.')
            poly_core = False

        # We build the fiber starting with the core region

        if poly_core:

            R_poly = self.R_core_comp / np.cos(np.pi / N)

            core_points = [(-R_poly * np.sin((2 * i - 1) * np.pi / N),
                            R_poly * np.cos((2 * i - 1) * np.pi / N))
                           for i in range(N)]

            pts = [geo.AppendPoint(x, y) for x, y in core_points]

            for i in range(N - 1):
                geo.Append(["line", pts[i], pts[i+1]], leftdomain=1,
                           rightdomain=2, bc='core_fill_air_interface')

            geo.Append(["line", pts[-1], pts[0]], leftdomain=1,
                       rightdomain=2, bc='core_fill_air_interface')

        else:
            geo.AddCircle(c=(0, 0), r=self.R_core_comp, leftdomain=1,
                          rightdomain=2, bc='core_fill_air_interface')

        # Now we add the microtubes and cladding

        if N == 0:
            geo.AddCircle(c=(0, 0), r=self.Rs[0], leftdomain=2,
                          rightdomain=3, bc='fill_air_cladding_interface')
        else:

            # Recursively build microstructure
            self.make_layer(geo, 0, np.array([0., 0.]), 0)

        self.Rout = self.Rs[0] + self.Ts[0]  # set base Rout

        geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=3,
                      rightdomain=4,
                      bc='cladding_outer_materials_interface')

        self.inner_materials = {
            'core': {'index': 1,
                     'maxh': self.core_maxh},
            'air': {'index': 2,
                    'maxh': self.air_maxh},
            'glass': {'index': 3,
                      'maxh': self.glass_maxh}
        }

        # Set inner material names and maxhs
        for material, info in self.inner_materials.items():
            geo.SetMaterial(info['index'], material)
            geo.SetDomainMaxH(info['index'], info['maxh'])

        # Create and add outer materials including PML region
        n_outer = len(self.outer_materials)
        for i, d in enumerate(self.outer_materials):

            self.Rout += d['T']  # add thickness of materials

            # Take care of boundary naming
            if i < (n_outer-1):
                mat = d['material']
                next_mat = self.outer_materials[i+1]['material']
                bc = mat + '_' + next_mat + '_interface'
                geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=(i+4),
                              rightdomain=(i+5), bc=bc)
                geo.SetMaterial(i+4, d['material'])
                geo.SetDomainMaxH(i+4, d['maxh'])
            else:
                bc = 'OuterCircle'
                geo.AddCircle(c=(0, 0), r=self.Rout, leftdomain=(i+4), bc=bc)
                geo.SetMaterial(i+4, d['material'])
                geo.SetDomainMaxH(i+4, d['maxh'])

        self.R = self.Rout - self.outer_materials[-1]['T']
        self.geo = geo

    def make_layer(self, geo, i, total_center, total_rotation):
        '''
        Recursively build nested microstructures.

        Parameters
        ----------
        geo : netgen.Geom2d type
            Geometry to add to.
        i : int
            Level at which we are working.
        total_center : np.array
            Center at which to place microstructure.
        total_rotation : float
            Amount to rotate microstructure.

        Returns
        -------
        None.

        '''

        if i == len(self.Ns) - 1:
            # We are at base level, add just a circle.
            geo.AddCircle(c=total_center, r=self.Rs[i], leftdomain=2,
                          rightdomain=3, bc='air_glass_interface')
            return

        else:
            # Build interface between current air level and next level
            # of capillaries.
            self.build_outer(geo, self.Cs[i+1]-self.Cs[i], self.Ns[i+1],
                             self.Rs[i+1]+self.Ts[i+1], self.Rs[i],
                             self.phis[i+1], self.psis[i+1],
                             total_center, total_rotation)

            for k in range(self.Ns[i+1]):
                # Now go into each capillary and repeat the process
                # at the correct new center point and rotation.
                new_point = np.array([(self.Cs[i+1] - self.Cs[i]) *
                                      np.cos(total_rotation + np.pi/2),
                                      (self.Cs[i+1] - self.Cs[i]) *
                                      np.sin(total_rotation + np.pi/2)
                                      ])
                # Don't update the total_center, just pass current
                # total_center plus new_point.
                self.make_layer(geo, i+1, total_center+new_point,
                                total_rotation)

                # Update amount to rotate next level of geometry.
                total_rotation += 2*np.pi/self.Ns[i+1]

    def build_outer(self, geo, north_pole_distance,
                    N_tubes_inner, R_inner_cap_total, R_outer, phi, psi,
                    total_center, total_rotation):
        '''
        Build interface between air layer inside of one tube and capillary
        microstructures of next nested layer.

        Parameters
        ----------
        geo : ng.Geom2d object
            Geometry to add to.
        north_pole_distance : float
            Distance to center of top capillary tube, prior to rotation and
            scaling.
        N_tubes_inner : int
            Number of tubes in next level of geometry.
        R_inner_cap_total : float
            Total radius of capillary tubes in next level of geometry, ie the
            sum of the inner capillary radius and capillary thickness
        R_outer : float
            Radius of inside of capillary tube at current level of geometry, ie
            the tube you are currently in.
        phi : float
            Angle clockwise off of vertical from origin to right
            intersection of current capillary with northern subcapillary.
        psi : float
            Angle counter-clockwise off of horizontal from north pole to
            intersection of current capillary with northern subcapillary.
        total_center : np.array
            Center point to translate completed geometry to.
        total_rotation : floa
            Amount to rotate completed geometry.

        Returns
        -------
        None.

        '''

        # The center of the top capillary tube. For nested tubes, need to
        # use Cs[i+1] - Cs[i]
        north_pole = (0, north_pole_distance)

        if N_tubes_inner == 1:

            inner_rotation = 0
            points = []
            points.append([R_outer,
                           R_outer * (1 - np.sin(phi)) / np.cos(phi)])

            points.extend(self.get_capillary_spline_points(north_pole,
                                                           R_inner_cap_total,
                                                           phi, psi,
                                                           inner_rotation))

            points.append([-R_outer,
                           R_outer * (1 - np.sin(phi)) / np.cos(phi)])

            points.extend([[-R_outer, 0], [-R_outer, -R_outer],
                          [0, -R_outer], [R_outer, -R_outer],
                          [R_outer, 0]])

        else:

            # Get the distance to the middle control point for the
            # arc between two capillaries.
            sector = 2 * np.pi / N_tubes_inner
            sigma = sector - 2 * phi
            D = R_outer / np.cos(sigma / 2)

            points = []
            for k in range(N_tubes_inner):

                # Compute the rotation angle needed for rotating the north
                # capillary spline points to the other capillary locations
                # in the domain.
                inner_rotation = k * sector

                # Compute the middle control point for the outer arc
                # subtended by the angle psi + rotation_angle.
                ctrl_pt_angle = np.pi / 2 - phi - sigma / 2 \
                    + inner_rotation

                points.append([D * np.cos(ctrl_pt_angle),
                              D * np.sin(ctrl_pt_angle)])

                # Obtain the control points for the capillary tube
                # immediately counterclockwise from the above control point
                points.extend(self.get_capillary_spline_points(
                    north_pole, R_inner_cap_total, phi, psi, inner_rotation))

        # Rotate and shift points to desired center
        points = np.array(points)
        M = np.array([[np.cos(total_rotation), -np.sin(total_rotation)],
                      [np.sin(total_rotation), np.cos(total_rotation)]])

        points = (M @ points.T).T

        # total center should use Cs[i+1] directly
        points += total_center

        # Add the capillary points to the geometry
        point_ids = [geo.AppendPoint(x, y) for x, y in points]
        NP = len(point_ids)

        for k in range(1, NP + 1, 2):
            bc = 'air_capillary_interface'
            geo.Append(
                ['spline3',
                    point_ids[k % NP],
                    point_ids[(k + 1) % NP],
                    point_ids[(k + 2) % NP]
                 ], leftdomain=2, rightdomain=3,
                bc=bc)

    def get_capillary_spline_points(self, north_pole, r_cap_total,
                                    phi, psi, inner_rotation):
        """
        Method that obtains the spline points for the interface between one
        capillary tube and inner hollow core. By default, we generate the
        spline points for the topmost capillary tube, and then rotate
        these points to generate the spline points for another tube based
        upon the inputs.

        INPUTS:

        north_pole  = the center of the northern capillary tube

        r_cap_total = radius of outer boundary of capillary tube

        phi = corresponds to the polar angle that gives theintersection of
          the two circles of radius Rcladi and Rto, respectively. In this
          case, the latter circle has a center given as the first
          argument above.

        psi = angle (cc from x axis) of intersection point relative to center

        inner_rotation = the angle by which we rotate the spline points
          obtained for the north circle to obtain the spline
          points for another capillary tube in the fiber.

        OUTPUTS:

        A list of control point for a spline that describes the interface.
        """

        # Start off with an array of x- and y-coordinates. This will
        # make any transformations to the points easier to work with.
        points = np.zeros((9, 2))

        # The control points for the first spline.
        points[0] = [np.cos(psi), np.sin(psi)]
        points[1] = [1, (1 - np.cos(psi)) / np.sin(psi)]
        points[2] = [1, 0]

        # Control points for the second and third splines.
        points[3] = [1, -1]
        points[4] = [0, -1]
        points[5] = [-1, -1]

        # Control points for the final spline.
        points[6] = [-1, 0]
        points[7] = [-1, (1 - np.cos(psi)) / np.sin(psi)]
        points[8] = [-np.cos(psi), np.sin(psi)]

        # The rotation matrix needed to generate the spline points for an
        # arbitrary capillary tube.
        # print('rotating inner capillary by ', inner_rotation)
        M = np.array([[np.cos(inner_rotation), -np.sin(inner_rotation)],
                      [np.sin(inner_rotation), np.cos(inner_rotation)]])

        # Rotate, scale, and shift the points.
        points *= r_cap_total
        points += north_pole
        points = (M @ points.T).T

        # Return the points as a collection of tuples.
        point_list = []

        for p in points:
            point_list.append(list(p))

        return point_list

    def E_modes_from_array(self, array, p=1, mesh=None):
        """Create NGvec object containing modes and set data given by array."""
        if mesh is None:
            mesh = self.mesh
        X = ng.HCurl(mesh, order=p+1-max(1-p, 0), type1=True,
                     dirichlet='OuterCircle', complex=True)
        m = array.shape[1]
        E = NGvecs(X, m)
        try:
            E.fromnumpy(array)
        except ValueError:
            raise ValueError("Array is wrong length: make sure your mesh is\
 constructed the same as for input array and that polynomial degree correspond\
ing to array has been passed as keyword p.")
        return E

    def phi_modes_from_array(self, array, p=1, mesh=None):
        """Create NGvec object containing modes and set data given by array."""
        if mesh is None:
            mesh = self.mesh
        Y = ng.H1(mesh, order=p+1, dirichlet='OuterCircle', complex=True)
        m = array.shape[1]
        phi = NGvecs(Y, m)
        try:
            phi.fromnumpy(array)
        except ValueError:
            raise ValueError("Array is wrong length: make sure your mesh is\
 constructed the same as for input array and that polynomial degree correspond\
ing to array has been passed as keyword p.")
        return phi

    # SAVE & LOAD #####################################################

    def save_mesh(self, name):
        """ Save mesh using pickle (allows for mesh curvature). """
        if name[-4:] != '.pkl':
            name += '.pkl'
        with open(name, 'wb') as f:
            pickle.dump(self.mesh, f)

    def save_modes(self, modes, name):
        """Save modes as numpy arrays."""
        if name[-4:] == '.npy':
            name -= '.npy'
        np.save(name, modes.tonumpy())

    def load_mesh(self, name):
        """ Load a saved ARF mesh."""
        if name[-4:] != '.pkl':
            name += '.pkl'
        with open(name, 'rb') as f:
            pmesh = pickle.load(f)
        return pmesh

    def load_E_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        if mode_name[-4:] == '.npy':
            mode_name -= '.npy'
        array = np.load(mode_name+'.npy')
        return self.E_modes_from_array(array, mesh=mesh, p=p)

    def load_phi_modes(self, mesh_name, mode_name, p=8):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name)
        array = np.load(mode_name+'.npy')
        return self.phi_modes_from_array(array, mesh=mesh, p=p)
