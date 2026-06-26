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

from warnings import warn
from copy import deepcopy
from ngsolve import x, y, exp, CF

from scipy.special import jv, jvp, h1vp, h2vp, yv, yvp
from scipy.special import hankel1 as h1
from scipy.special import hankel2 as h2

from .utilities import r, theta, Jv, Yv, \
    Hankel1, Hankel2, Jvp, Yvp, Hankel1p, Hankel2p


class BraggExact():
    """
    Create a Bragg type multilayered radial fiber.

    Bragg fibers consist of a circular core region surrounded by many
    concentric circular layers of alternating material, often glass and air.

    """

    def __init__(self, scale=5e-5, ts=[5e-5, 1e-5, 2e-5],
                 mats=['air', 'glass', 'air'], ns=[1, 1.44, 1],
                 maxhs=[.2, .035, .08], bcs=None, no_mesh=False,
                 wl=1.2e-6, ref=0, curve=8):

        # Check inputs for errors
        self.check_parameters(ts, ns, mats, maxhs, bcs)

        self.scale = scale
        self.no_mesh = no_mesh  # don't create mesh if you only need betas
        self.ref = ref
        self.curve = curve
        self.L = scale
        self.mats = mats
        self.maxhs_in = maxhs

        if bcs is not None:
            self.bcs = bcs
        else:
            self.bcs = ['r'+str(i+1) for i in range(len(ts))]

        self.ts = ts  # This is a property that also sets geo and mesh

        self.ns_in = deepcopy(ns)  # Copy input refractive index information

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

        M = np.zeros(beta.shape + (4, 4), dtype=np.complex128)

        k0 = self.k0 * self.scale
        k1 = k0 * n1
        k2 = k0 * n2

        rho = rho / self.scale

        K1 = np.sqrt(k1 ** 2 - beta ** 2, dtype=complex)
        K2 = np.sqrt(k2 ** 2 - beta ** 2, dtype=complex)

        X, Y = K1*rho,  K2*rho

        expr = (1 / Y - Y / X**2)

        F1 = (K2 * n1 ** 2) / (K1 * n2 ** 2)
        F2 = nu * beta / (k0 * n2 ** 2)
        F3 = nu * beta / k0
        F4 = K2 / K1

        Ymat = np.zeros_like(M)
        Ymat[..., 0, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 2, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T
        Ymat[..., 3, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T

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

        M[..., 0, :] = np.array([(z1(nu, X) * z2p(nu, Y) -
                                  F1 * z1p(nu, X) * z2(nu, Y)).T,

                                 (z2(nu, X) * z2p(nu, Y) -
                                 F1 * z2p(nu, X) * z2(nu, Y)).T,

                                 (1j*F2 * z1(nu, X) * z2(nu, Y)*expr).T,

                                 (1j*F2 * z2(nu, X) * z2(nu, Y)*expr).T]).T

        M[..., 1, :] = np.array([(F1 * z1p(nu, X) * z1(nu, Y) -
                                z1(nu, X) * z1p(nu, Y)).T,

                                 (F1 * z2p(nu, X) * z1(nu, Y) -
                                z2(nu, X) * z1p(nu, Y)).T,

                                 (-1j*F2 * z1(nu, X) * z1(nu, Y)*expr).T,

                                 (-1j*F2 * z2(nu, X) * z1(nu, Y)*expr).T]).T

        M[..., 2, :] = np.array([(-1j * F3 * z1(nu, X) * z2(nu, Y) * expr).T,

                                 (-1j * F3 * z2(nu, X) * z2(nu, Y) * expr).T,

                                 (z1(nu, X) * z2p(nu, Y) - F4 *
                                 z1p(nu, X) * z2(nu, Y)).T,

                                 (z2(nu, X) * z2p(nu, Y) - F4 *
                                  z2p(nu, X) * z2(nu, Y)).T]).T

        M[..., 3, :] = np.array([(1j * F3 * z1(nu, X) * z1(nu, Y) * expr).T,

                                 (1j * F3 * z2(nu, X) * z1(nu, Y) * expr).T,

                                 (F4 * z1p(nu, X) * z1(nu, Y) -
                                 z1(nu, X) * z1p(nu, Y)).T,

                                 (F4 * z2p(nu, X) * z1(nu, Y) -
                                 z2(nu, X) * z1p(nu, Y)).T]).T

        return f * Ymat * M

    def state_matrix(self, beta, nu, rho, n, zfunc='bessel', Ktype='kappa',
                     pml=None):
        """Return matching matrix from Yeh et al Theory of Bragg Fiber.

        This matrix appears as equation 34 in that paper. Again we have scaled
        the H field such that we make the replacements:

            we ---> k0n^2,  wu ---> k0,  e1/e2 ---> (n1/n2)^2

        We take all u to be idential to u0. This function also assumes
        that provided beta is scaled, but the provided rho is not scaled."""

        beta = np.array(beta, dtype=np.complex128)
        rho /= self.scale

        # State matrix is only affected here by PML, (since 1 + alpha*i term
        # cancels in all derivatives in matrix).
        if pml is not None:
            alpha, R0 = pml['alpha'], pml['R0']
            R0 /= self.scale
            rho = (rho - R0) * (1 - alpha * 1j) + R0
        else:
            alpha = 0

        k0 = self.k0 * self.scale
        k = k0 * n
        if Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta ** 2 - k ** 2, dtype=complex)
        elif Ktype == 'kappa':
            K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)
        else:
            raise TypeError('Ktype must be kappa or i_gamma.')

        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        Z = np.zeros_like(beta).T

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
            Z,
            Z
        ]).T

        L[..., 1, :] = np.array([
            (k0 * n**2 / (beta * K) * z1p(nu, K * rho)).T,
            (k0 * n**2 / (beta * K) * z2p(nu, K * rho)).T,
            (1j * nu / (K ** 2 * rho) * z1(nu, K * rho)).T,
            (1j * nu / (K ** 2 * rho) * z2(nu, K * rho)).T
        ]).T

        L[..., 2, :] = np.array([
            Z,
            Z,
            z1(nu, K * rho).T,
            z2(nu, K * rho).T
        ]).T

        L[..., 3, :] = np.array([
            (1j * nu / (K ** 2 * rho) * z1(nu, K * rho)).T,
            (1j * nu / (K ** 2 * rho) * z2(nu, K * rho)).T,
            (-k0 / (beta * K) * z1p(nu, K * rho)).T,
            (-k0 / (beta * K) * z2p(nu, K * rho)).T
        ]).T

        return L

    def state_matrix_inverse(self, beta, nu, rho, n, zfunc='bessel'):
        """Return inverse of matching matrix above.

        Same replacements as above.  Inverse found using sympy.  Also uses
        K * rho = x.  Mainly for debugging transfer matrix."""

        beta = np.array(beta, dtype=np.complex128)

        k0 = self.k0 * self.scale
        k = k0 * n
        K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)

        rho /= self.scale

        X = K * rho
        F1 = K * beta / (k0 * n**2)
        F2 = beta * nu / (k0 * n**2 * X)
        F3 = beta * nu / (k0 * X)
        F4 = K * beta / k0

        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        Z = np.zeros_like(beta).T

        if zfunc == 'hankel':
            z1, z1p = h1, h1vp
            z2, z2p = h2, h2vp
        elif zfunc == 'bessel':
            z1, z1p = jv, jvp
            z2, z2p = yv, yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        L[..., 0, :] = np.array([z2p(nu, X).T,
                                 (-F1 * z2(nu, X)).T,
                                 (1j * F2 * z2(nu, X)).T,
                                 Z]).T

        L[..., 1, :] = np.array([(-z1p(nu, X)).T,
                                 (F1 * z1(nu, X)).T,
                                 (-1j * F2 * z1(nu, X)).T,
                                 Z]).T

        L[..., 2, :] = np.array([(-1j * F3 * z2(nu, X)).T,
                                 Z,
                                 z2p(nu, X).T,
                                 (F4 * z2(nu, X)).T]).T

        L[..., 3, :] = np.array([(-1j * F3 * z1(nu, X)).T,
                                 Z,
                                 -z1p(nu, X).T,
                                 (-F4 * z1(nu, X)).T]).T

        L[..., :, :] = np.pi * X / 2 * L[..., :, :]

        return L

    def determinant(self, beta, nu=1, outer='h2', Ktype='kappa', pml=None,
                    return_coeffs=False, return_matrix=False):
        """Return determinant of matching matrix.

        Provided beta should be scaled.  This zeros of this functions are the
        propagation constants for the fiber."""

        if return_coeffs and return_matrix:
            raise ValueError("Only one of return_matrix and return_coeffs\
 can be set to True")

        if outer not in ['h1', 'h2', 'pcb']:
            raise ValueError("Outer must be either 'h1', 'h2' or 'pcb'.")

        if pml is not None and outer != 'pcb':
            warn("Using PML without PCBCs won't change eigenvalues.")

        if pml is not None and self.ns[-1] != self.ns[-2]:
            raise ValueError("Last two regions should have same index of \
refraction when using pml, but have values %3f and %3f." % (self.ns[-2],
                                                            self.ns[-1]))

        beta = np.array(beta, dtype=np.complex128)

        rhos = self.rhos
        ns = self.ns
        L = np.zeros(beta.shape + (4, 2), dtype=complex)
        L[..., :, :] = np.eye(4)[:, [0, 2]]  # pick J columns for core

        if outer != 'pcb':  # not perfect conducting bcs

            # apply transfer matrix up to second to last layer
            for i in range(len(rhos)-2):
                nl, nr = ns[i], ns[i+1]
                rho = rhos[i]
                L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L

            # Apply state matrix to get left hand side at last layer
            L = self.state_matrix(beta, nu, rhos[-2], ns[-2]) @ L

            # Now need to match coeffs at second to last layer with last layer
            if outer == 'h2':
                inds = [1, 3]
                # marcuse_k = True
            elif outer == 'h1':
                inds = [0, 2]
                # marcuse_k = True
            else:
                raise TypeError("Outer function must be 'h1' (guided) or 'h2'\
     (leaky).")

            R = self.state_matrix(beta, nu, rhos[-2],
                                  ns[-1], zfunc='hankel',
                                  Ktype=Ktype)[..., inds]

            a, b, e, f = L[..., 0, 0], L[..., 0, 1], L[..., 1, 0], L[..., 1, 1]
            c, d, g, h = L[..., 2, 0], L[..., 2, 1], L[..., 3, 0], L[..., 3, 1]

            alpha, Beta = R[..., 0, 0], R[..., 2, 1]
            gamma, delta = R[..., 1, 0], R[..., 1, 1]
            epsilon, sigma = R[..., 3, 0], R[..., 3, 1]

            A = e - (a/alpha * gamma + c/Beta * delta)
            B = f - (b/alpha * gamma + d/Beta * delta)
            C = g - (a/alpha * epsilon + c/Beta * sigma)
            D = h - (b/alpha * epsilon + d/Beta * sigma)

            if return_coeffs:
                return A, B, C, D, a, b, c, d, alpha, Beta
            else:
                if return_matrix:
                    M = np.zeros(beta.shape + (4, 4), dtype=complex)
                    M[..., : 2] = L
                    M[..., 2:] = R
                    return M
                else:
                    return C * B - A * D

        else:   # perfectly conducting bcs need one more layer

            for i in range(len(rhos)-1):  # apply transfer one more layer
                nl, nr = ns[i], ns[i+1]
                rho = rhos[i]
                L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L

            # At last rho, Ez and Ephi = 0
            # These equations correspond to rows 0 and 3 in the state matrix
            L = self.state_matrix(beta, nu, rhos[-1], ns[-1], pml=pml,
                                  Ktype=Ktype) @ L

            R = np.zeros(beta.shape + (2, 4), dtype=complex)
            R[..., :, :] = np.eye(4)[[0, 3], :]  # pick Ez and Ephi rows
            L = R @ L
            # Since L was 4x2 before last operation, it's now 2x2 so we can
            # easily take the determinant.
            A, B, C, D = L[..., 0, 0], L[..., 0,
                                         1], L[..., 1, 0], L[..., 1, 1]

            if return_coeffs:
                return A, B, C, D
            else:
                if return_matrix:
                    return L, R
                else:
                    return A * D - B * C

    def coefficients(self, beta, nu=1, outer='h2', Ktype='kappa', pml=None):
        """Return coefficients for fields of bragg fiber.

        Returns an array where each row gives the coeffiencts of the fields
        in the associated region of the fiber."""

        if outer not in ['h1', 'h2', 'pcb']:
            raise ValueError("Outer must be either 'h1', 'h2' or 'pcb'.")

        if pml is not None and outer != 'pcb':
            warn("Using PML without PCBCs won't change eigenvalues.")

        if pml is not None and self.ns[-1] != self.ns[-2]:
            raise ValueError("Last two regions should have same index of \
refraction when using pml, but have values %3f and %3f." % (self.ns[-2],
                                                            self.ns[-1]))

        if outer != 'pcb':

            A, B, C, D, a, b, c, d, \
                alpha, Beta = self.determinant(beta, nu=nu, outer=outer,
                                               Ktype=Ktype, pml=None,
                                               return_coeffs=True)

            # A, B, or C,D can be used to make v1,v2 for core, but if mode
            # is transverse one of the pairs is zero, here we account for this
            Vs = np.array([A, B, C, D])
            Vn = ((Vs*Vs.conj()).real) ** .5

            if max(Vn) < 1e-13:  # Check that not all are too small
                raise ValueError("Error in coefficients: small matrix elements\
     are all too small.")

            imax = np.argmax(Vn)

            if imax in [0, 1]:
                v1, v2 = B, -A  # If C,D too small, assign with B,A

            else:
                v1, v2 = D, -C  # Otherwise use C and D

            v = np.array([v1, v2])

            w1, w2 = (a*v1 + b*v2) / alpha, (c*v1 + d*v2) / Beta

            rhos = self.rhos
            ns = self.ns

            M = np.zeros((len(self.rhos), 4), dtype=complex)
            L = np.zeros((4, 2), dtype=complex)

            L[:, :] = np.eye(4)[:, [0, 2]]
            L = L @ v
            M[0, :] = L

            for i in range(len(rhos)-2):
                nl, nr = ns[i], ns[i+1]
                rho = rhos[i]
                L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L
                M[i+1, :] = L

            if outer == 'h2':
                inds = [1, 3]
            elif outer == 'h1':
                inds = [0, 2]
            else:
                raise TypeError("Outer function must be 'h1' (guided) or 'h2'\
         (leaky).")

            M[-1, inds] = w1, w2

            if nu > 0:  # Hybrid, Ez non zero in core
                return 1 / v1 * M  # Normalize so Ez coeff in core is 1

            else:  # TE or TM,
                # Find non-zero coefficient (either Ez or Hz) and use to scale
                vscale = v[np.argmax((v*v.conj()).real)]
                return 1/vscale * M
        else:
            A, B, C, D = self.determinant(beta, nu, outer, Ktype=Ktype,
                                          pml=pml, return_coeffs=True)

            # A, B, or C,D can be used to make v1,v2 for core, but if mode
            # is transverse one of thes pairs is zero, here we account for this
            Vs = np.array([A, B, C, D])
            Vn = ((Vs*Vs.conj()).real) ** .5

            if max(Vn) < 1e-13:  # Check that not all are too small
                raise ValueError("Error in coefficients: small matrix elements\
       are all too small.")

            imax = np.argmax(Vn)

            if imax in [0, 1]:
                v1, v2 = B, -A  # If C,D too small, assign with B,A

            else:
                v1, v2 = D, -C  # Otherwise use C and D

            v = np.array([v1, v2])

            rhos = self.rhos
            ns = self.ns

            M = np.zeros((len(self.rhos), 4), dtype=complex)
            L = np.zeros((4, 2), dtype=complex)

            L[:, :] = np.eye(4)[:, [0, 2]]
            L = L @ v
            M[0, :] = L

            for i in range(len(rhos)-1):
                nl, nr = ns[i], ns[i+1]
                rho = rhos[i]
                L = self.transfer_matrix(beta, nu, rho, nl, nr) @ L
                M[i+1, :] = L

            if nu > 0:  # Hybrid, Ez non zero in core
                return 1 / v1 * M  # Normalize so Ez coeff in core is 1

            else:  # TE or TM,
                # Find non-zero coefficient (either Ez or Hz) and use to scale
                vscale = v[np.argmax((v*v.conj()).real)]
                return 1/vscale * M

    # ------------- NGSolve Field Visualizations ------------------

    def regional_fields(self, beta, coeffs, index, nu=1, Ktype='kappa',
                        zfunc='bessel', pml=None):
        """Create fields on one region of the fiber."""

        if pml is not None:
            alpha, R0 = pml['alpha'], pml['R0']
            R0 /= self.scale
            rf = (r - R0) * (1 - alpha * 1j) + R0
        else:
            alpha = 0
            rf = r

        A, B, C, D = coeffs[:]

        k0 = self.k0 * self.scale
        n = self.ns[index]
        k = k0 * n

        if Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta ** 2 - k ** 2, dtype=complex)
        elif Ktype == 'kappa':
            K = np.sqrt(k ** 2 - beta ** 2, dtype=complex)
        else:
            raise TypeError('Bad Ktype.')

        F = 1j * beta / (k ** 2 - beta ** 2)

        if zfunc == 'hankel':
            Z1, Z1p = Hankel1, Hankel1p
            Z2, Z2p = Hankel2, Hankel2p
        elif zfunc == 'bessel':
            Z1, Z1p = Jv, Jvp
            Z2, Z2p = Yv, Yvp
        else:
            raise TypeError("zfunc must be 'bessel' or 'hankel'.")

        einu = exp(1j * nu * theta)

        Ez = (A * Z1(K*rf, nu) + B * Z2(K*rf, nu)) * einu

        # If using PML, we would get a factor of 1-alpha j, but it
        # would later be canceled by it's inverse so both are omitted
        dEzdr = K * (A * Z1p(K*rf, nu) + B * Z2p(K*rf, nu)) * einu
        dEzdt = 1j * nu * Ez

        Hz = (C * Z1(K*rf, nu) + D * Z2(K*rf, nu)) * einu

        # Same comment as above, factor of 1 - alpha j is omitted
        dHzdr = K * (C * Z1p(K*rf, nu) + D * Z2p(K*rf, nu)) * einu
        dHzdt = 1j * nu * Hz

        Er = F * (dEzdr + k0 / (beta * rf) * dHzdt)
        Ephi = F * (1 / rf * dEzdt - k0 / beta * dHzdr)

        Hr = F * (dHzdr - k0 * n**2 / (beta * rf) * dEzdt)
        Hphi = F * (1 / rf * dHzdt + k0 * n**2 / beta * dEzdr)

        Ex = (x * Er - y * Ephi) / r  # Note this doesn't work in pml region
        Ey = (y * Er + x * Ephi) / r  # Need to implement that still

        dEzdx = dEzdr * x/r - y/r**2 * (1j * nu * Ez)
        dEzdy = dEzdr * y/r + x/r**2 * (1j * nu * Ez)

        Hx = (x * Hr - y * Hphi) / r
        Hy = (y * Hr + x * Hphi) / r

        Sx = Ey * ng.Conj(Hz) - Ez * ng.Conj(Hy)  # Poynting Vector (time avg)
        Sy = Ez * ng.Conj(Hx) - Ex * ng.Conj(Hz)
        Sz = Ex * ng.Conj(Hy) - Ey * ng.Conj(Hx)

        Sr = Ephi * ng.Conj(Hz) - Ez * ng.Conj(Hphi)
        Sphi = Ez * ng.Conj(Hr) - Er * ng.Conj(Hz)

        return {'Ez': Ez, 'Hz': Hz, 'Er': Er, 'Hr': Hr, 'Ephi': Ephi,
                'Hphi': Hphi,  'Ex': Ex, 'Ey': Ey, 'Hx': Hx, 'Hy': Hy,
                'Sx': Sx, 'Sy': Sy, 'Sz': Sz, 'Sr': Sr, 'Sphi': Sphi,
                'dEzdx': dEzdx, 'dEzdy': dEzdy}

    def all_fields(self, beta, nu=1, outer='h2', Ktype='kappa', pml=None):
        """Create total fields for fiber from regional fields."""
        M = self.coefficients(beta, nu=nu, outer=outer, Ktype=Ktype, pml=pml)

        Ez, Hz = [], []
        dEzdx, dEzdy = [], []
        Er, Hr = [], []
        Ephi, Hphi = [], []
        Ex, Ey, Hx, Hy = [], [], [], []
        Sx, Sy, Sz = [], [], []
        Sr, Sphi = [], []

        for i in range(len(self.ns)):
            coeffs = M[i]
            if i == len(self.ns) - 1 and outer != 'pcb':
                zfunc = 'hankel'
            else:
                zfunc = 'bessel'

            if i == len(self.ns)-1:
                F = self.regional_fields(beta, coeffs, i, nu=nu, pml=pml,
                                         zfunc=zfunc, Ktype=Ktype)
            else:
                F = self.regional_fields(beta, coeffs, i, nu=nu, pml=None,
                                         zfunc=zfunc, Ktype='kappa')

            Ez.append(F['Ez']), Er.append(F['Er'])
            Ex.append(F['Ex']), Ey.append(F['Ey'])
            Ephi.append(F['Ephi'])
            Hz.append(F['Hz']), Hr.append(F['Hr'])
            Hx.append(F['Hx']), Hy.append(F['Hy'])
            Hphi.append(F['Hphi'])
            Sx.append(F['Sx']), Sy.append(F['Sy'])
            Sz.append(F['Sz'])
            Sr.append(F['Sr']), Sphi.append(F['Sphi'])
            dEzdx.append(F['dEzdx']), dEzdy.append(F['dEzdy'])

        Ez, Er, Ephi, Ex, Ey = CF(Ez), CF(Er), CF(Ephi), CF(Ex), CF(Ey)
        Hz, Hr, Hphi, Hx, Hy = CF(Hz), CF(Hr), CF(Hphi), CF(Hx), CF(Hy)
        Sx, Sy, Sz = CF(Sx), CF(Sy), CF(Sz)
        Sr, Sphi = CF(Sr), CF(Sphi)

        # Transverse Fields

        Etv = CF((Ex, Ey))
        Htv = CF((Hx, Hy))
        Stv = CF((Sx, Sy))
        gradEz = CF((dEzdx, dEzdy))
        return {'Ez': Ez, 'Hz': Hz, 'Er': Er, 'Hr': Hr,
                'Ephi': Ephi, 'Hphi': Hphi, 'Etv': Etv, 'Htv': Htv,
                'Ex': Ex, 'Ey': Ey, 'Hx': Hx, 'Hy': Hy,
                'Stv': Stv, 'Sz': Sz, 'Sr': Sr, 'Sphi': Sphi,
                'gradEz': gradEz,
                }

    # ------------- Matplotlib Field Visualizations --------------

    def fields_matplot(self, beta, nu=1, outer='h2', Ktype='kappa',
                       pml=None):
        if Ktype not in ['i_gamma', 'kappa']:
            raise TypeError('Ktype must be kappa or i_gamma.')

        if pml is not None:
            raise NotImplementedError('PML not implemented for matplot\
visualization.  Use self.all_fields (ngsolve based visualization).')

        M = self.coefficients(beta, nu=nu, outer=outer, pml=pml, Ktype=Ktype)

        rhos = np.concatenate([[0], self.rhos/self.scale])

        k0 = self.k0 * self.scale
        ks = self.ks * self.scale
        ns = self.ns

        Ks = np.sqrt(ks ** 2 - beta ** 2, dtype=complex)
        if Ktype == 'i_gamma':
            Ks[-1] = 1j * np.sqrt(beta**2 - ks[-1]**2, dtype=complex)
        Fs = 1j * beta / (ks ** 2 - beta ** 2)

        def Ez_rad(rs):

            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

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

        def Hz_rad(rs):

            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

                if i == 0:
                    ys[y_idx] = C * jv(nu, Ks[i] * rs[r_idx])
                elif i != len(conds)-1:
                    ys[y_idx] = C * jv(nu, Ks[i]*rs[r_idx]) + \
                        D * yv(nu, Ks[i] * rs[r_idx])
                else:
                    ys[y_idx] = C * h1(nu, Ks[i] * rs[r_idx]) + \
                        D * h2(nu, Ks[i] * rs[r_idx])
            return ys

        def Hz(x, y):
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Hz_rad(r) * np.e**(1j * nu * t)

        def Er_rad(rs):
            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

                if i == 0:
                    ys[y_idx] = Fs[i] * (Ks[i] * A * jvp(nu, Ks[i]*rs[r_idx]) +
                                         k0 / (beta * rs[r_idx]) *
                                         1j * nu * C *
                                         jv(nu, Ks[i] * rs[r_idx]))
                elif i != len(conds)-1:
                    ys[y_idx] = Fs[i] * (Ks[i] * (A*jvp(nu, Ks[i]*rs[r_idx]) +
                                         B * yvp(nu, Ks[i] * rs[r_idx])) +
                                         k0/(beta * rs[r_idx]) * 1j * nu *
                                         (C * jv(nu, Ks[i]*rs[r_idx]) +
                                          D * yv(nu, Ks[i] * rs[r_idx])))
                else:
                    ys[y_idx] = Fs[i] * (Ks[i] * (A*h1vp(nu, Ks[i]*rs[r_idx]) +
                                         B * h2vp(nu, Ks[i] * rs[r_idx])) +
                                         k0/(beta * rs[r_idx]) * 1j * nu *
                                         (C * h1(nu, Ks[i]*rs[r_idx]) +
                                          D * h2(nu, Ks[i] * rs[r_idx])))
            return ys

        def Er(x, y):
            '''Return Er function.

            Rs and Thetas should be result of np.meshgrid.'''
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Er_rad(r) * np.e**(1j * nu * t)

        def Ephi_rad(rs):
            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

                if i == 0:
                    ys[y_idx] = Fs[i] * (1j * nu / rs[r_idx] *
                                         A * jv(nu, Ks[i] * rs[r_idx]) -
                                         k0 / beta * Ks[i] * C *
                                         jvp(nu, Ks[i] * rs[r_idx]))
                elif i != len(conds)-1:
                    ys[y_idx] = Fs[i] * (1j*nu/rs[r_idx] *
                                         (A * jv(nu, Ks[i] * rs[r_idx]) +
                                         B * yv(nu, Ks[i] * rs[r_idx])) -
                                         k0 / beta * Ks[i] *
                                         (C * jvp(nu, Ks[i]*rs[r_idx]) +
                                         D * yvp(nu, Ks[i] * rs[r_idx])))
                else:
                    ys[y_idx] = Fs[i] * (1j*nu/rs[r_idx] *
                                         (A * h1(nu, Ks[i] * rs[r_idx]) +
                                         B * h2(nu, Ks[i] * rs[r_idx])) -
                                         k0 / beta * Ks[i] *
                                         (C * h1vp(nu, Ks[i]*rs[r_idx]) +
                                         D * h2vp(nu, Ks[i] * rs[r_idx])))
            return ys

        def Ephi(x, y):
            '''Return Er function.

            Rs and Thetas should be result of np.meshgrid.'''
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Ephi_rad(r) * np.e**(1j * nu * t)

        def Hr_rad(rs):
            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

                if i == 0:
                    ys[y_idx] = Fs[i] * (Ks[i] * C * jvp(nu, Ks[i]*rs[r_idx]) -
                                         k0 * ns[i]**2 / (beta * rs[r_idx]) *
                                         1j * nu * A *
                                         jv(nu, Ks[i] * rs[r_idx]))
                elif i != len(conds)-1:
                    ys[y_idx] = Fs[i] * (Ks[i] * (C*jvp(nu, Ks[i]*rs[r_idx]) +
                                         D * yvp(nu, Ks[i] * rs[r_idx])) -
                                         k0*ns[i]**2/(beta*rs[r_idx])*1j * nu *
                                         (A * jv(nu, Ks[i]*rs[r_idx]) +
                                          B * yv(nu, Ks[i] * rs[r_idx])))
                else:
                    ys[y_idx] = Fs[i] * (Ks[i] * (C*h1vp(nu, Ks[i]*rs[r_idx]) +
                                         D * h2vp(nu, Ks[i] * rs[r_idx])) -
                                         k0*ns[i]**2/(beta*rs[r_idx])*1j * nu *
                                         (A * h1(nu, Ks[i]*rs[r_idx]) +
                                          B * h2(nu, Ks[i] * rs[r_idx])))
            return ys

        def Hr(x, y):
            '''Return Er function.

            Rs and Thetas should be result of np.meshgrid.'''
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Hr_rad(r) * np.e**(1j * nu * t)

        def Hphi_rad(rs):
            conds = [(rhos[i] <= rs)*(rs <= rhos[i+1])
                     for i in range(len(rhos)-1)]
            ys = np.zeros_like(rs, dtype=complex)

            for i in range(len(conds)):

                y_idx, r_idx = np.nonzero(conds[i]), np.where(conds[i])
                A, B, C, D = M[i, :]

                if i == 0:
                    ys[y_idx] = Fs[i] * (1j * nu / rs[r_idx] *
                                         C * jv(nu, Ks[i] * rs[r_idx]) +
                                         k0 * ns[i]**2 / beta * Ks[i] * A *
                                         jvp(nu, Ks[i] * rs[r_idx]))
                elif i != len(conds)-1:
                    ys[y_idx] = Fs[i] * (1j*nu/rs[r_idx] *
                                         (C * jv(nu, Ks[i] * rs[r_idx]) +
                                         D * yv(nu, Ks[i] * rs[r_idx])) +
                                         k0 * ns[i]**2 / beta * Ks[i] *
                                         (A * jvp(nu, Ks[i]*rs[r_idx]) +
                                         B * yvp(nu, Ks[i] * rs[r_idx])))
                else:
                    ys[y_idx] = Fs[i] * (1j*nu/rs[r_idx] *
                                         (C * h1(nu, Ks[i] * rs[r_idx]) +
                                         D * h2(nu, Ks[i] * rs[r_idx])) +
                                         k0 * ns[i]**2 / beta * Ks[i] *
                                         (A * h1vp(nu, Ks[i]*rs[r_idx]) +
                                         B * h2vp(nu, Ks[i] * rs[r_idx])))
            return ys

        def Hphi(x, y):
            '''Return Er function.

            Rs and Thetas should be result of np.meshgrid.'''
            r = (x*x + y*y)**.5
            t = np.arctan2(y, x)
            return Hphi_rad(r) * np.e**(1j * nu * t)

        def Ex(x, y):
            r = (x*x + y*y)**.5
            return (x * Er(x, y) - y * Ephi(x, y)) / r

        def Ey(x, y):
            r = (x*x + y*y)**.5
            return (y * Er(x, y) + x * Ephi(x, y)) / r

        def Hx(x, y):
            r = (x*x + y*y)**.5
            return (x * Hr(x, y) - y * Hphi(x, y)) / r

        def Hy(x, y):
            r = (x*x + y*y)**.5
            return (y * Hr(x, y) + x * Hphi(x, y)) / r

        def Sz_rad(rs):
            return Er_rad(rs) * np.conj(Hphi_rad(rs)) - \
                Ephi_rad(rs) * np.conj(Hr_rad(rs))

        def Sz(x, y):
            return Ex(x, y) * np.conj(Hy(x, y)) - Ey(x, y) * np.conj(Hx(x, y))

        return {'Ez': Ez, 'Ez_rad': Ez_rad,
                'Hz': Hz, 'Hz_rad': Hz_rad,
                'Er': Er, 'Er_rad': Er_rad,
                'Ephi': Ephi, 'Ephi_rad': Ephi_rad,
                'Hr': Hr, 'Hr_rad': Hr_rad,
                'Hphi': Hphi, 'Hphi_rad': Hphi_rad,
                'Ex': Ex, 'Ey': Ey, 'Hx': Hx, 'Hy': Hy,
                'Sz': Sz, 'Sz_rad': Sz_rad
                }

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

    def plot2D_streamlines(self, Fx, Fy, rlist=None, ntheta=101, Nstrm=101,
                           figsize=(16, 16), part='real', levels=40,
                           contourfunc=None, contourpart='norm',
                           colorbar_scale=.8, colorbar_fontsize=14,
                           streamline_color='k', maxd_scaling=1,
                           streamline_width=1.5, arrowsize=3.5,
                           arrowstyle='->',
                           broken_streamlines=True, density=2.2,
                           plot_rhos=True, rho_linewidth=1.1,
                           rho_linestyle='-', plot_seed=False,
                           rho_linecolor='k', seed_nr=None, seed_ntheta=65,
                           **streamplotkwargs):

        if contourfunc is not None:

            fig, ax = self.plot2D_contour(contourfunc, rlist=rlist,
                                          ntheta=ntheta, figsize=figsize,
                                          levels=levels, part=contourpart,
                                          plot_rhos=plot_rhos, cmap='jet',
                                          linewidth=rho_linewidth,
                                          linestyle=rho_linestyle,
                                          edgecolor=rho_linecolor,
                                          colorbar_fontsize=colorbar_fontsize,
                                          colorbar_scale=colorbar_scale)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Build grid for streamplot (must be cartesian and uniform)
        R = self.rhos[-1]/self.scale   # get outer radius
        stream_pts = np.linspace(-R, R, Nstrm, dtype=float)  # base array

        if 0 in stream_pts:  # avoid zero (not implemented/not defined)
            stream_pts = np.linspace(-R, R, Nstrm+1, dtype=float)

        X2, Y2 = np.meshgrid(stream_pts, stream_pts)
        U, V = Fx(X2, Y2), Fy(X2, Y2)

        if part == 'real':
            ex, ey = U.real, V.real
        elif part == 'imag':
            ex, ey = U.imag, V.imag
        else:
            raise ValueError('Part must be real or imag.')

        seed_points = self.seed_points(rlist=seed_nr, ntheta=seed_ntheta)

        try:
            ax.streamplot(X2, Y2, ex, ey, density=density,
                          linewidth=streamline_width, color=streamline_color,
                          broken_streamlines=broken_streamlines,
                          arrowsize=arrowsize, arrowstyle=arrowstyle,
                          start_points=seed_points, maxd_scaling=maxd_scaling,
                          **streamplotkwargs)
        except TypeError:
            ax.streamplot(X2, Y2, ex, ey, density=density,
                          linewidth=streamline_width, color=streamline_color,
                          broken_streamlines=broken_streamlines,
                          arrowsize=arrowsize, arrowstyle=arrowstyle,
                          start_points=seed_points,
                          **streamplotkwargs)
        if plot_seed:
            ax.scatter(seed_points[..., 0], seed_points[..., 1])

        ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def seed_points(self, rlist=None, ntheta=4):
        rhos = np.concatenate([[0], self.rhos/self.scale])
        if rlist is not None:
            if len(rlist) != len(self.rhos):
                raise ValueError('Provided point list has wrong number of\
entries.  Please give a list with same number of entries as regions of fiber.')
            rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i+1], rlist[i]+2)[1:-1]
                 for i in range(len(rhos)-1)])
        else:
            rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i+1], 3+2)[1:-1]
                 for i in range(len(rhos)-1)])

        thetas = np.linspace(0, 2*np.pi, ntheta+1)[:-1]
        Rs, Thetas = np.meshgrid(rs, thetas)
        Xs, Ys = Rs * np.cos(Thetas), Rs * np.sin(Thetas)
        return np.array([Xs.flatten(), Ys.flatten()]).T
