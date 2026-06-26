#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:48:03 2022

@author: pv

"""
import numpy as np
import netgen.geom2d as geom2d
import ngsolve as ng
import matplotlib.pyplot as plt

from ngsolve import exp, CF
from copy import deepcopy

from scipy.special import jv, jvp, h1vp, h2vp, yv, yvp
from scipy.special import hankel1 as h1
from scipy.special import hankel2 as h2

from .utilities import r, theta, Jv, Yv, \
    Hankel1, Hankel2, Jvp, Yvp, Hankel1p, Hankel2p


class BraggScalar():
    """
    Create a Bragg type multilayered radial fiber.

    Bragg fibers consist of a circular core region surrounded by many
    concentric circular layers of alternating material, often glass and air.

    """

    def __init__(self, scale=5e-5, ts=[5e-5, 1e-5, 2e-5],
                 mats=['air', 'glass', 'air'], ns=[1, 1.44, 1],
                 maxhs=[.2, .025, .08], bcs=None, no_mesh=False,
                 wl=1.2e-6, ref=0, curve=8):

        # Check inputs for errors
        self.check_parameters(ts, ns, mats, maxhs, bcs)

        self.scale = scale
        self.no_mesh = no_mesh
        self.ref = ref
        self.curve = curve
        self.L = scale
        self.mats = mats
        self.maxhs_in = maxhs

        if bcs is not None:
            self.bcs = bcs
        else:
            self.bcs = ['r'+str(i+1) for i in range(len(ts))]

        self.ts = ts

        self.ns_in = deepcopy(ns)

        self.wavelength = wl

    @property
    def wavelength(self):
        """Get wavelength."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        """ Set wavelength and associated material parameters."""
        self._wavelength = wl
        self.k0 = 2 * np.pi / wl
        N = len(self.ns_in)
        self.ns = np.array([self.ns_in[i](wl)
                            if callable(self.ns_in[i])
                            else self.ns_in[i] for i in range(N)])
        self.ks = self.k0 * self.ns

    @property
    def ts(self):
        """Get ts."""
        return self._ts

    @ts.setter
    def ts(self, ts):
        """ Set radii and maxhs."""
        ts = np.array(ts)
        self._ts = ts
        self.rhos = np.array([sum(ts[:i]) for i in range(1, len(ts)+1)])
        self.maxhs = np.array(self.maxhs_in) * self.rhos / self.scale

        if not self.no_mesh:
            # Create geometry
            self.create_geometry()
            self.create_mesh(ref=self.ref, curve=self.curve)

    def check_parameters(self, ts, ns, mats, maxhs, bcs):

        # Check that all relevant inputs have same length
        lengths = [len(ts), len(ns), len(mats), len(maxhs)]
        names = ['ts', 'ns', 'mats', 'maxhs']
        if bcs is not None:
            lengths.append(len(bcs))
            names.append('bcs')
        else:
            print('Boundary names not provided, using default names.')

        lengths = np.array(lengths)

        same = all(x == lengths[0] for x in lengths)

        if not same:
            string = "Provided parameters not of same length: \n\n"
            for name, length in zip(names, lengths):
                string += name + ': ' + str(length) + '\n'
            raise ValueError(string + "\nModify above inputs as necessary and \
try again.")

    def create_mesh(self, ref=0, curve=8):
        """
        Create mesh from geometry.
        """
        self.mesh = ng.Mesh(self.geo.GenerateMesh())

        self.refinements = 0
        for i in range(ref):
            self.mesh.ngmesh.Refine()

        self.refinements += ref
        self.mesh.ngmesh.SetGeometry(self.geo)
        self.mesh = ng.Mesh(self.mesh.ngmesh.Copy())
        self.mesh.Curve(curve)

    def create_geometry(self):
        """Construct and return Non-Dimensionalized geometry."""
        self.geo = geom2d.SplineGeometry()

        Rs = self.rhos / self.scale
        for i, R in enumerate(Rs[:-1]):
            self.geo.AddCircle(c=(0, 0), r=R, leftdomain=i+1,
                               rightdomain=i+2, bc=self.bcs[i])

        self.geo.AddCircle(c=(0, 0), r=Rs[-1], leftdomain=len(Rs),
                           bc=self.bcs[-1])

        for i, (mat, maxh) in enumerate(zip(self.mats, self.maxhs)):
            self.geo.SetMaterial(i+1, mat)
            self.geo.SetDomainMaxH(i+1, maxh)

    def transfer_matrix(self, beta, nu, rho, n1, n2, zfunc='bessel'):
        """Return transfer matrix from Yeh et al Theory of Bragg Fiber.

        We have scaled the H field such that we make the replacements:

            we ---> k0n^2,  wu ---> k0,  e1/e2 ---> (n1/n2)^2

        We take all u to be idential to u0. This function also assumes
        that provided beta is scaled, but the provided rho is not scaled."""

        beta = np.array(beta, dtype=np.complex128)

        M = np.zeros(beta.shape + (2, 2), dtype=np.complex128)

        k0 = self.k0 * self.scale
        k1 = k0 * n1
        k2 = k0 * n2

        rho = rho / self.scale

        K1 = np.sqrt(k1 ** 2 - beta ** 2, dtype=complex)
        K2 = np.sqrt(k2 ** 2 - beta ** 2, dtype=complex)

        X, Y = K1*rho,  K2*rho

        Ymat = np.zeros_like(M)
        Ymat[..., 0, :] = np.array([Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T]).T

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
            f = 1j * np.pi / 4
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
            f = np.pi / 2
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        F = K1 / K2

        M[..., 0, :] = np.array([(z1(nu, X) * z2p(nu, Y) -
                                  F * z1p(nu, X) * z2(nu, Y)).T,

                                 (z2(nu, X) * z2p(nu, Y) -
                                 F * z2p(nu, X) * z2(nu, Y)).T
                                 ]).T

        M[..., 1, :] = np.array([(F * z1p(nu, X) * z1(nu, Y) -
                                z1(nu, X) * z1p(nu, Y)).T,

                                 (F * z2p(nu, X) * z1(nu, Y) -
                                z2(nu, X) * z1p(nu, Y)).T
                                 ]).T

        return f * Ymat * M

    def state_matrix(self, beta, nu, rho, n, zfunc='bessel', Ktype='kappa'):
        """Return matching matrix from Yeh et al Theory of Bragg Fiber.

        This matrix appears as equation 34 in that paper. Again we have scaled
        the H field such that we make the replacements:

            we ---> k0n^2,  wu ---> k0,  e1/e2 ---> (n1/n2)^2

        We take all u to be idential to u0. This function also assumes
        that provided beta is scaled, but the provided rho is not scaled."""

        beta = np.array(beta, dtype=np.complex128)
        rho /= self.scale

        k0 = self.k0 * self.scale
        k = k0 * n
        if Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta ** 2 - k ** 2, dtype=complex)
        elif Ktype == 'kappa':
            K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)
        else:
            raise TypeError('Ktype must be kappa or i_gamma.')

        L = np.zeros(beta.shape + (2, 2), dtype=complex)

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        L[..., 0, :] = np.array([
            z1(nu, K * rho).T,
            z2(nu, K * rho).T,
        ]).T

        L[..., 1, :] = np.array([
            K * z1p(nu, K * rho).T,
            K * z2p(nu, K * rho).T,
        ]).T

        return L

    def determinant(self, beta, nu=1, outer='h2', Ktype='kappa',
                    return_coeffs=False, return_matrix=False):
        """Return determinant of matching matrix.

        Provided beta should be scaled.  This zeros of this functions are the
        propagation constants for the fiber."""

        if return_coeffs and return_matrix:
            raise ValueError("Only one of return_matrix and return_coeffs\
 can be set to True")

        if outer not in ['h1', 'h2']:
            raise ValueError("Outer must be either 'h1', 'h2'.")

        beta = np.array(beta, dtype=np.complex128)

        rhos = self.rhos
        ns = self.ns
        L = np.zeros(beta.shape + (2, 1), dtype=complex)
        L[..., :, :] = np.eye(2)[:, [0]]  # pick J columns for core

        # apply transfer matrix up to second to last layer
        for i in range(len(rhos)-2):
            nl, nr = ns[i], ns[i+1]
            rho = rhos[i]
            L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L

        # Apply state matrix to get left hand side at last layer
        L = self.state_matrix(beta, nu, rhos[-2], ns[-2]) @ L

        # Now need to match coeffs at second to last layer with last layer
        if outer == 'h2':
            inds = [1]
        elif outer == 'h1':
            inds = [0]
        else:
            raise TypeError(
                "Outer function must be 'h1' (guided) or 'h2' (leaky).")

        R = self.state_matrix(beta, nu, rhos[-2],
                              ns[-1], zfunc='hankel',
                              Ktype=Ktype)[..., inds]

        if return_coeffs:
            A, B = L[..., 0, 0], L[..., 1, 0]
            C, D = R[..., 0, 0], R[..., 1, 0]
            return A, B, C, D
        else:
            if return_matrix:
                M = np.zeros(beta.shape + (2, 2), dtype=complex)
                M[..., : 1] = L
                M[..., 1:] = R
                return M
            else:
                A, B = L[..., 0, 0], L[..., 1, 0]
                C, D = R[..., 0, 0], R[..., 1, 0]
                return B * C - A * D

    def coefficients(self, beta, nu=1, outer='h2', Ktype='kappa'):
        """Return coefficients for fields of bragg fiber.

        Returns an array where each row gives the coeffiencts of the fields
        in the associated region of the fiber."""

        if outer not in ['h1', 'h2']:
            raise ValueError("Outer must be either 'h1', 'h2'.")

        A, B, C, D = self.determinant(beta, nu=nu, outer=outer,
                                      Ktype=Ktype,
                                      return_coeffs=True)

        # A, B, or C,D can be used to make v1,v2 for core, but if mode
        # is transverse one of the pairs is zero, here we account for this
        Vs = np.array([C, D])
        Vn = ((Vs*Vs.conj()).real) ** .5

        imax = np.argmax(Vn)

        if imax == 0:
            v1 = C
            w1 = A

        else:
            v1 = D
            w1 = B

        v = np.array([v1])

        rhos = self.rhos
        ns = self.ns

        M = np.zeros((len(self.rhos), 2), dtype=complex)
        L = np.zeros((2, 1), dtype=complex)

        L[:, :] = np.eye(2)[:, [0]]
        L = L @ v
        M[0, :] = L

        for i in range(len(rhos)-2):
            nl, nr = ns[i], ns[i+1]
            rho = rhos[i]
            L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L
            M[i+1, :] = L

        if outer == 'h2':
            inds = [1]
        elif outer == 'h1':
            inds = [0]
        else:
            raise TypeError("Outer function must be 'h1' (guided) or 'h2'\
 (leaky).")

        M[-1, inds] = w1

        return 1 / v1 * M

    # ------------- Matplotlib Field Visualizations --------------

    def fields_matplot(self, beta, nu=1, outer='h2', Ktype='kappa',
                       pml=None):
        if Ktype not in ['i_gamma', 'kappa']:
            raise TypeError('Ktype must be kappa or i_gamma.')

        M = self.coefficients(beta, nu=nu, outer=outer, Ktype=Ktype)

        rhos = np.concatenate([[0], self.rhos/self.scale])

        ks = self.ks * self.scale

        Ks = np.sqrt(ks ** 2 - beta ** 2, dtype=complex)
        if Ktype == 'i_gamma':
            Ks[-1] = 1j * np.sqrt(beta**2 - ks[-1]**2, dtype=complex)

        def Ez_rad(rs):

            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B = M[i, :]

                if i == 0:
                    ys[y_idx] = A * jv(nu, Ks[i] * rs[r_idx])
                elif i != len(conds)-1:
                    ys[y_idx] = A * jv(nu, Ks[i] * rs[r_idx]) + \
                        B * yv(nu, Ks[i] * rs[r_idx])
                else:
                    ys[y_idx] = A * h1(nu, Ks[i] * rs[r_idx]) + \
                        B * h2(nu, Ks[i] * rs[r_idx])
            return ys

        def Ez(x, y):
            '''Return Ez function.

            x and y should be result of np.meshgrid.'''
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Ez_rad(r) * np.e**(1j * nu * t)

        return {'Ez': Ez, 'Ez_rad': Ez_rad}

    def graphpoints(self, rlist=None, ntheta=101):
        """Create points for 1D and 2D plotting."""
        rhos = np.concatenate([[1e-9], self.rhos/self.scale])
        if rlist is not None:
            if len(rlist) != len(self.rhos):
                raise ValueError('Provided point list has wrong number of\
entries.  Please give a list with same number of entries as regions of fiber.')
            self.rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i+1], rlist[i])
                 for i in range(len(rhos)-1)])
        else:
            self.rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i+1], 101)
                 for i in range(len(rhos)-1)])

        self.thetas = np.linspace(0, 2*np.pi, ntheta)
        self.Rs, self.Thetas = np.meshgrid(self.rs, self.thetas)
        self.Xs, self.Ys = self.Rs * \
            np.cos(self.Thetas), self.Rs * np.sin(self.Thetas)

    def plot1D(self, F, rlist=None, figsize=(8, 6), part='real', nu=1,
               double_r=False, return_vals=False, maxscale=False, **lineargs):
        """Plot 1D function F using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.graphpoints(rlist=rlist)
        if double_r:
            rs = np.concatenate([-np.flip(self.rs), self.rs])
            ys = np.concatenate([np.exp(1j*nu*np.pi)*F(np.flip(self.rs)),
                                 F(self.rs)])
        else:
            rs = self.rs
            ys = F(rs)
        if maxscale:
            ys /= np.max(np.abs(ys))
        if part == 'real':
            ax.plot(rs, ys.real, **lineargs)
        elif part == 'imag':
            ax.plot(rs, ys.imag, **lineargs)
        elif part == 'norm':
            ax.plot(rs, np.abs(ys), **lineargs)
        else:
            raise ValueError('Part must be "real", "imag" or "norm".')

        plt.show()
        if return_vals:
            return rs, ys
        else:
            return fig, ax

    def add1D_plot(self, ax, F, part='real', double_r=False, nu=1,
                   maxscale=False, **lineargs):
        if double_r:
            rs = np.concatenate([-np.flip(self.rs), self.rs])
            ys = np.concatenate([np.exp(1j*nu*np.pi)*F(np.flip(self.rs)),
                                 F(self.rs)])
        else:
            rs = self.rs
            ys = F(rs)
        if maxscale:
            ys /= np.max(np.abs(ys))
        if part == 'real':
            ax.plot(rs, ys.real, **lineargs)
        elif part == 'imag':
            ax.plot(rs, ys.imag, **lineargs)
        elif part == 'norm':
            ax.plot(rs, np.abs(ys), **lineargs)
        else:
            raise ValueError('Part must be "real", "imag" or "norm".')
        plt.show()

    def plot2D_contour(self, F, rlist=None, ntheta=101, figsize=(16, 16),
                       part='real', levels=40, plot_rhos=True, edgecolor='k',
                       cmap='jet', colorbar_scale=.8, colorbar_fontsize=14,
                       linewidth=1.1, **lineargs):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.graphpoints(rlist=rlist, ntheta=ntheta)
        X, Y = self.Xs, self.Ys
        Z = F(X, Y)
        if part == 'real':
            zs = Z.real
        elif part == 'imag':
            zs = Z.imag
        elif part == 'norm':
            zs = np.abs(Z)
        else:
            raise ValueError('Part must be "real", "imag" or "norm".')

        contour = ax.contourf(X, Y, zs, levels=levels, cmap=cmap)

        if plot_rhos:
            for rho in self.rhos/self.scale:
                plt.plot(rho*np.cos(self.thetas), rho *
                         np.sin(self.thetas), color=edgecolor,
                         linewidth=linewidth, **lineargs)
        ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(contour, shrink=colorbar_scale)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        return fig, ax

    # ------------- NGSolve Field Visualizations ------------------

    def regional_fields(self, beta, coeffs, index, nu=1, Ktype='kappa',
                        zfunc='bessel'):
        """Create fields on one region of the fiber."""

        A, B = coeffs[:]

        k0 = self.k0 * self.scale
        n = self.ns[index]
        k = k0 * n

        if Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta ** 2 - k ** 2, dtype=complex)
        elif Ktype == 'kappa':
            K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)
        else:
            raise TypeError('Bad Ktype.')

        if zfunc == 'hankel':
            Z1, Z2 = Hankel1, Hankel2
            Z1p, Z2p = Hankel1p, Hankel2p
        elif zfunc == 'bessel':
            Z1, Z2 = Jv, Yv
            Z1p, Z2p = Jvp, Yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        einu = exp(1j * nu * theta)

        u = (A * Z1(K*r, nu) + B * Z2(K*r, nu)) * einu
        du_dr = K * (A * Z1p(K*r, nu) + B * Z2p(K*r, nu)) * einu

        du_dx = du_dr * ng.x/r - ng.y/r**2 * (1j * nu * u)
        du_dy = du_dr * ng.y/r + ng.x/r**2 * (1j * nu * u)

        return {'u': u, 'du_dr': du_dr, 'du_dtheta': 1j * nu * u,
                'du_dx': du_dx, 'du_dy': du_dy,
                'grad_u': ng.CF((du_dx, du_dy))}

    def all_fields(self, beta, nu=1, outer='h2', Ktype='kappa'):
        """Create total fields for fiber from regional fields."""
        M = self.coefficients(beta, nu=nu, outer=outer, Ktype=Ktype)

        U, grad_U = [], []
        dU_dr, dU_dtheta = [], []
        dU_dx, dU_dy = [], []

        for i in range(len(self.ns)):
            coeffs = M[i]
            if i == len(self.ns) - 1:
                Fs = self.regional_fields(beta, coeffs, i, nu=nu,
                                          zfunc='hankel', Ktype=Ktype)
            else:
                Fs = self.regional_fields(beta, coeffs, i, nu=nu,
                                          zfunc='bessel', Ktype='kappa')

            U.append(Fs['u']), dU_dr.append(Fs['du_dr'])
            dU_dx.append(Fs['du_dx']), dU_dy.append(Fs['du_dy'])
            dU_dtheta.append(Fs['du_dtheta']), grad_U.append(Fs['grad_u'])

        U, grad_U = CF(U), CF(grad_U)
        dU_dr, dU_dtheta = CF(dU_dr), CF(dU_dtheta)
        dU_dx, dU_dy = CF(dU_dx), CF(dU_dy)

        return {'U': U, 'dU_dr': dU_dr, 'dU_dtheta': dU_dtheta,
                'dU_dx': dU_dx, 'dU_dy': dU_dy, 'grad_U': grad_U}
