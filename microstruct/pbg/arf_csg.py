#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:04:04 2022

@author: pv
"""

import ngsolve as ng
import numpy as np
import pickle
from netgen.geom2d import CSG2d, Circle, Solid2d
from fiberamp.fiber.modesolver import ModeSolver
from pyeigfeast.spectralproj.ngs.spectralprojngs import NGvecs


class ARFcsg(ModeSolver):
    """
    Create an ARF fiber using csg2d
    """

    def __init__(self, name=None, refine=0, curve=3, e=None,
                 poly_core=False, shift_capillaries=False,
                 outer_materials=None):

        # Set and check the fiber parameters.
        self.set_parameters(name=name, shift_capillaries=shift_capillaries,
                            e=e, outer_materials=outer_materials)
        self.check_parameters()

        # Create geometry
        self.create_geometry(poly_core=poly_core)
        self.create_mesh(refine=refine, curve=curve)

        # Set physical properties
        self.set_material_properties()

        super().__init__(self.mesh, self.scale, self.n0)

    def set_parameters(self, name=None, e=None, shift_capillaries=False,
                       outer_materials=None):
        """
        Set fiber parameters.
        """
        # Default is 6-capillary fiber.
        self.name = 'poletti' if name is None else name

        if self.name == 'poletti':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .42 / scaling

            self.T_cladding = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            if shift_capillaries:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                   self.T_tube)

                self.R_tube_center = (self.R_cladding - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                                   self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': 2},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 2}
                ]

            self.inner_air_maxh = .25
            self.fill_air_maxh = .25
            self.tube_maxh = .11
            self.cladding_maxh = .33
            self.core_maxh = .25
            self.glass_maxh = 0  # Overrides maxh in tubes and cladding

            self.wavelength = 1.8e-6

        elif self.name == 'basic':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.06 / scaling
            self.T_tube = .84 / scaling

            self.T_cladding = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 10 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            if shift_capillaries:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                   self.T_tube)

                self.R_tube_center = (self.R_cladding - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                                   self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': 2},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 4}
                ]

            self.core_maxh = .25
            self.fill_air_maxh = .35
            self.tube_maxh = .11
            self.inner_air_maxh = .2
            self.cladding_maxh = .25
            self.glass_maxh = 0  # Overrides maxh in tubes and cladding

            self.wavelength = 1.8e-6

        elif self.name == 'fine_cladding':

            self.n_tubes = 6

            scaling = 15
            self.scale = scaling * 1e-6

            if e is not None:
                self.e = e
            else:
                self.e = .025/.42

            self.R_tube = 12.48 / scaling
            self.T_tube = .42 / scaling

            self.T_cladding = 10 / scaling
            self.T_outer = 30 / scaling
            self.T_buffer = 30 / scaling
            self.T_soft_polymer = 30 / scaling
            self.T_hard_polymer = 30 / scaling

            if shift_capillaries:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - .025/.42) *
                                   self.T_tube)

                self.R_tube_center = (self.R_cladding - self.R_tube -
                                      (1 - self.e) * self.T_tube)

                self.core_factor = .75
                self.R_core = ((self.R_tube_center - self.R_tube -
                                self.T_tube) * self.core_factor)
            else:
                self.R_cladding = (1 + 2 * self.R_tube + (2 - self.e) *
                                   self.T_tube)

                self.R_tube_center = 1 + self.R_tube + self.T_tube
                self.core_factor = .75
                self.R_core = self.core_factor

            self.n_glass = 1.4388164768221814
            self.n_air = 1.00027717
            self.n_soft_polymer = 1.44
            self.n_hard_polymer = 1.56
            self.n_buffer = self.n_air
            self.n0 = self.n_air

            if outer_materials is not None:
                self.outer_materials = outer_materials
            else:
                self.outer_materials = [

                    # {'material': 'soft_polymer',
                    #  'n': self.n_soft_polymer,
                    #  'T': self.T_soft_polymer,
                    #  'maxh': 2},

                    # {'material': 'hard_polymer',
                    #  'n': self.n_hard_polymer,
                    #  'T': self.T_hard_polymer,
                    #  'maxh': 2},

                    {'material': 'buffer',
                     'n': self.n_buffer,
                     'T': self.T_buffer,
                     'maxh': .5},

                    {'material': 'Outer',
                     'n': self.n0,
                     'T': self.T_outer,
                     'maxh': 2}
                ]

            self.inner_air_maxh = .2
            self.fill_air_maxh = .2
            self.tube_maxh = .11
            self.cladding_maxh = .25
            self.core_maxh = .25
            self.glass_maxh = 0.05  # Overrides maxh in tubes and cladding

            self.wavelength = 1.8e-6

        else:
            err_str = 'Fiber \'{:s}\' not implemented.'.format(self.name)
            raise NotImplementedError(err_str)

    def check_parameters(self):
        """Check to ensure given parameters give valid geometry."""
        if self.e <= 0 or self.e >= 1:
            raise ValueError('Embedding parameter e must be strictly between\
 zero and one.')

        if self.n_tubes >= 2:
            if self.R_tube_center * np.sin(np.pi / self.n_tubes) \
                    <= self.R_tube + self.T_tube:
                raise ValueError('Capillary tubes overlap each other.')

    def set_material_properties(self):
        """
        Set k0, refractive indices, and V function.
        """
        self.k = 2 * np.pi / self.wavelength

        # Set up inner region refractive indices
        refractive_index_dict = {'core': self.n_air,
                                 'fill_air': self.n_air,
                                 'glass': self.n_glass,
                                 'inner_air': self.n_air,
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

        self.mesh.ngmesh.SetGeometry(self.spline_geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())

        self.mesh.Curve(curve)

    def create_geometry(self, poly_core=True, fill=None):
        """
        Create geometry and mesh
        """

        geo = CSG2d()

        n_tubes = self.n_tubes
        R_core = self.R_core
        R_cladding = self.R_cladding
        T_cladding = self.T_cladding
        R_tube = self.R_tube
        T_tube = self.T_tube
        R_tube_center = self.R_tube_center

        if self.n_tubes <= 2 and poly_core:
            raise ValueError('Polygonal core only available for n_tubes > 2.')

        if poly_core:

            R_poly = self.R_core / np.cos(np.pi / n_tubes)
            core_points = [(-R_poly * np.sin((2 * i - 1) * np.pi / n_tubes),
                            R_poly * np.cos((2 * i - 1) * np.pi / n_tubes))
                           for i in range(n_tubes)]

            core = Solid2d(core_points, mat="core",
                           bc="core_fill_air_interface")
        else:
            core = Circle(center=(0, 0), radius=R_core,
                          mat="core", bc="core_fill_air_interface")

        circle1 = Circle(center=(0, 0),
                         radius=R_cladding,
                         mat="fill_air", bc="glass_air_interface")
        circle2 = Circle(center=(0, 0),
                         radius=R_cladding+T_cladding,
                         mat="glass", bc="cladding_outer_materials_interface")

        small1 = Circle(center=(0, R_tube_center), radius=R_tube,
                        mat="inner_air", bc="glass_air_interface")
        small2 = Circle(center=(0, R_tube_center), radius=R_tube+T_tube,
                        mat="glass", bc="glass_air_interface")

        # # previous method
        # inner_tubes = small1.Copy()
        # outer_tubes = small2.Copy()

        # for i in range(1, n_tubes):
        #     inner_tubes += small1.Copy().Rotate(360/n_tubes * i,
        #                                         center=(0, 0))
        #     outer_tubes += small2.Copy().Rotate(360/n_tubes * i,
        #                                         center=(0, 0))

        # Second method
        inner_tubes = Solid2d()
        outer_tubes = Solid2d()

        for i in range(0, n_tubes):
            inner_tubes += small1.Copy().Rotate(360/n_tubes * i,
                                                center=(0, 0))
            outer_tubes += small2.Copy().Rotate(360/n_tubes * i,
                                                center=(0, 0))

        cladding = circle2 - circle1
        tubes = outer_tubes - inner_tubes

        tubes.Maxh(self.tube_maxh)
        cladding.Maxh(self.cladding_maxh)
        inner_tubes.Maxh(self.inner_air_maxh)

        glass = cladding + tubes

        glass.Mat('glass')

        if self.glass_maxh > 0:  # setting overrides tube and cladding maxh
            glass.Maxh(self.glass_maxh)

        fill_air = circle1 - outer_tubes - core
        fill_air.Maxh(self.fill_air_maxh)
        fill_air.Mat('fill_air')

        core.Maxh(self.core_maxh)
        core.Mat('core')

        geo.Add(core)
        geo.Add(fill_air)
        geo.Add(glass)

        if n_tubes > 0:  # Meshing fails if you add empty Solid2d instance
            geo.Add(inner_tubes)
            pass

        # Create and add outer materials including PML region

        self.Rout = R_cladding + T_cladding  # set base Rout

        n = len(self.outer_materials)

        for i, d in enumerate(self.outer_materials):

            circle1 = circle2  # increment circles

            self.Rout += d['T']  # add thickness of materials

            # Take care of boundary naming
            if i < (n-1):
                mat = d['material']
                next_mat = self.outer_materials[i+1]['material']
                bc = mat + '_' + next_mat + '_interface'
            else:
                bc = 'OuterCircle'
                mat = 'Outer'

            # Make new outermost circle
            circle2 = Circle(center=(0, 0),
                             radius=self.Rout,
                             mat=mat,
                             bc=bc)

            # Create and add region to geometry
            region = circle2 - circle1
            region.Mat(mat)
            region.Maxh(d['maxh'])
            geo.Add(region)

        self.R = self.Rout - self.outer_materials[-1]['T']
        self.geo = geo
        self.spline_geo = geo.GenerateSplineGeometry()

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

    def save_mesh(self, name, use_pickle=True):
        """ Save mesh using pickle (allows for mesh curvature). """
        if use_pickle:
            with open(name, 'wb') as f:
                pickle.dump(self.mesh, f)
        else:
            self.mesh.ngmesh.Save(name + '.vol')

    def save_modes(self, modes, name):
        """Save modes as numpy arrays."""
        np.save(name, modes.tonumpy())

    def load_mesh(self, name, use_pickle=True):
        """ Load a saved ARF mesh."""
        if use_pickle:
            with open(name, 'rb') as f:
                mesh = pickle.load(f)
        else:
            mesh = ng.Mesh(name + '.vol')

        return mesh

    def load_E_modes(self, mesh_name, mode_name, p=8, use_pickle=True):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name, use_pickle=use_pickle)
        array = np.load(mode_name+'.npy')
        return mesh, self.E_modes_from_array(array, mesh=mesh, p=p)

    def load_phi_modes(self, mesh_name, mode_name, p=8, use_pickle=True):
        """Load transverse vectore E modes and associated mesh"""
        mesh = self.load_mesh(mesh_name, use_pickle=use_pickle)
        array = np.load(mode_name+'.npy')
        return mesh, self.phi_modes_from_array(array, mesh=mesh, p=p)
