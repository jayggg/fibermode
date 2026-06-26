"""Numerical solver for Bragg fiber modes.

Class
-----
Bragg
    Builds a concentric-circle NGSolve mesh and finds leaky modes via
    the eigensolver inherited from ModeSolver.
"""

import numpy as np
import netgen.geom2d as geom2d
import ngsolve as ng

from warnings import warn
from pyeigfeast.spectralproj.ngs import NGvecs
from fibermode.solvers.modesolver import ModeSolver


class Bragg(ModeSolver):
    """Numerical leaky-mode solver for Bragg (concentric-ring) fibers.

    Bragg fibers consist of a circular core surrounded by many concentric
    layers of alternating dielectric material.

    Parameters
    ----------
    scale   : characteristic length (meters); non-dimensionalizes all lengths
    ts      : layer thicknesses (physical, same units as scale)
    mats    : material name per layer; last entry must be ``'Outer'``
    ns      : refractive index per layer (float or callable of wavelength)
    maxhs   : relative mesh size per layer (multiplied by layer outer radius)
    bcs     : boundary condition names; last entry must be ``'OuterCircle'``
    wl      : free-space wavelength (meters)
    ref     : number of uniform mesh refinements
    curve   : geometry curve order for NGSolve
    beta_sq_plane : if True, search in the β² plane instead of Z²
    """

    def __init__(self, scale=5e-5, ts=(5e-5, 1e-5, 2e-5, 2e-5),
                 mats=('air', 'glass', 'air', 'Outer'), ns=(1, 1.44, 1, 1),
                 maxhs=(.2, .025, .08, .1), bcs=None, wl=1.2e-6,
                 ref=0, curve=8, beta_sq_plane=False):

        self.check_parameters(ts, ns, mats, maxhs, bcs)

        self.L = scale
        self.scale = scale
        self.ts = list(ts)
        self.mats = list(mats)

        if bcs is not None:
            self.bcs = list(bcs)
        else:
            self.bcs = ['r' + str(i + 1) for i in range(len(ts))]
            self.bcs[-1] = 'OuterCircle'
            self.bcs[-2] = 'R'

        self.Ts = np.array(ts) / scale
        self.Rs = [sum(self.Ts[:i]) for i in range(1, len(self.Ts) + 1)]
        self.maxhs = np.array(maxhs) * self.Rs

        self.R, self.Rout = self.Rs[-2], self.Rs[-1]
        self.wavelength = wl
        self.ns = list(ns)

        self.create_geometry()
        self.create_mesh(ref=ref, curve=curve)
        self.curveorder = curve
        self.set_material_properties(beta_sq_plane)

        super(Bragg, self).__init__(self.mesh, self.L, self.n0)

    @classmethod
    def from_dict(cls, d):
        """Construct a Bragg instance from a parameter dictionary."""
        keys = ['scale', 'ts', 'mats', 'ns', 'maxhs', 'bcs', 'wl',
                'ref', 'curve', 'beta_sq_plane']
        return cls(**{k: d[k] for k in keys if k in d})

    def check_parameters(self, ts, ns, mats, maxhs, bcs):
        lengths = [len(ts), len(ns), len(mats), len(maxhs)]
        names = ['ts', 'ns', 'mats', 'maxhs']
        if bcs is not None:
            if bcs[-1] != 'OuterCircle':
                raise ValueError('Final boundary condition must ' +
                                 'be "OuterCircle".')
            lengths.append(len(bcs))
            names.append('bcs')
        else:
            print('Boundary names not provided, using default names.')

        if not all(x == lengths[0] for x in lengths):
            msg = 'Parameters must have the same length:\n'
            msg += '\n'.join(f'  {n}: {l}' for n, l in zip(names, lengths))
            raise ValueError(msg)

        if mats[-1] != 'Outer':
            raise ValueError('Final material (PML region) must ' +
                             'be named "Outer".')

    def create_mesh(self, ref=0, curve=8):
        """Generate and curve the NGSolve mesh."""
        self.mesh = ng.Mesh(self.geo.GenerateMesh())
        for _ in range(ref):
            self.mesh.ngmesh.Refine()
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(curve)

    def create_geometry(self):
        """Build the non-dimensionalized concentric-circle geometry."""
        self.geo = geom2d.SplineGeometry()
        for i, R in enumerate(self.Rs[:-1]):
            self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i + 1,
                               rightdomain=i + 2, bc=self.bcs[i])
        self.geo.AddCircle(c=(0, 0), r=self.Rs[-1],
                           leftdomain=len(self.Rs), bc=self.bcs[-1])
        for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
            self.geo.SetMaterial(i + 1, mat)
            self.geo.SetDomainMaxH(i + 1, maxh)

    def set_material_properties(self, beta_sq_plane=False):
        """Set k0, refractive indices, and the index-well coefficient."""
        if beta_sq_plane:
            warn('Using square-beta plane: search centers should be at '
                 '-(beta*L)^2 where L is the scale attribute.')
        self.k = 2 * np.pi / self.wavelength
        self.refractive_indices = [
            n(self.wavelength) if callable(n) else n for n in self.ns]
        self.index = ng.CF(self.refractive_indices)
        self.n0 = self.refractive_indices[-1]
        n0sq = ng.CF([self.n0 ** 2] * len(self.ns))
        self.V = (self.L * self.k) ** 2 * (
            n0sq * (not beta_sq_plane) - self.index ** 2)

    def E_modes_from_array(self, array, p=1, mesh=None):
        """Wrap a numpy array as an NGvecs object in the HCurl space."""
        if mesh is None:
            mesh = self.mesh
        X = ng.HCurl(mesh, order=p + 1 - max(1 - p, 0), type1=True,
                     dirichlet='OuterCircle', complex=True)
        E = NGvecs(X, array.shape[1])
        try:
            E.fromnumpy(array)
        except ValueError:
            raise ValueError(
                'Array shape mismatch: check that the mesh and polynomial '
                'degree match those used when the array was created.')
        return E

    def phi_modes_from_array(self, array, p=1, mesh=None):
        """Wrap a numpy array as an NGvecs object in the H1 space."""
        if mesh is None:
            mesh = self.mesh
        Y = ng.H1(mesh, order=p + 1, dirichlet='OuterCircle', complex=True)
        phi = NGvecs(Y, array.shape[1])
        try:
            phi.fromnumpy(array)
        except ValueError:
            raise ValueError(
                'Array shape mismatch: check that the mesh and polynomial '
                'degree match those used when the array was created.')
        return phi
