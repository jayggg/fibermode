"""Analytical (transfer-matrix) solution for Bragg fiber modes.

Classes
-------
BraggExactScalar
    Scalar Helmholtz approximation: 2×2 transfer matrix (Yeh et al.).
BraggExactVector
    Full Maxwell solution: 4×4 transfer matrix (Yeh et al.).


References
----------
Yeh, P., Yariv, A., & Marom, E. (1978). Theory of Bragg fiber. JOSA.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from scipy.special import jv, jvp, yv, yvp
from scipy.special import hankel1 as h1, hankel2 as h2
from scipy.special import h1vp, h2vp

# #######################################################################
# Internal helpers #######################################################


def _select_part(Z, part):
    if part == 'real':
        return Z.real
    if part == 'imag':
        return Z.imag
    if part == 'norm':
        return np.abs(Z)
    raise ValueError('part must be "real", "imag", or "norm".')


def _plot_part(ax, xs, ys, part, **kw):
    ax.plot(xs, _select_part(ys, part), **kw)


def _bessel_pair(zfunc):
    if zfunc == 'bessel':
        return jv, jvp, yv, yvp
    if zfunc == 'hankel':
        return h1, h1vp, h2, h2vp
    raise ValueError("zfunc must be 'bessel' or 'hankel'.")


# #########################################################################
# Base class ##############################################################


class _BraggExactBase:
    """Shared initialization, parameter handling, and matplotlib plotting.

    Subclasses provide the transfer-matrix mathematics.
    """

    def __init__(self,
                 scale=5e-5,
                 ts=(5e-5, 1e-5, 2e-5),
                 mats=('air', 'glass', 'air'),
                 ns=(1, 1.44, 1),
                 wl=1.2e-6):
        self._check_parameters(ts, ns, mats)
        self.scale = scale
        self.L = scale
        self.mats = list(mats)
        self.ns_in = deepcopy(list(ns))
        self.ts = ts  # property → sets _ts, rhos
        self.wavelength = wl  # property → sets k0, ns, ks

    # Properties #######################################################

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.k0 = 2 * np.pi / wl
        self.ns = np.array([n(wl) if callable(n) else n for n in self.ns_in])
        self.ks = self.k0 * self.ns

    @property
    def ts(self):
        return self._ts

    @ts.setter
    def ts(self, ts):
        ts = np.array(ts)
        self._ts = ts
        self.rhos = np.array([sum(ts[:i]) for i in range(1, len(ts) + 1)])

    # Validation #######################################################

    def _check_parameters(self, ts, ns, mats):
        lengths = [len(ts), len(ns), len(mats)]
        names = ['ts', 'ns', 'mats']
        if len(set(lengths)) != 1:
            msg = 'Parameters must have the same length:\n'
            msg += '\n'.join(f'  {n}: {l}' for n, l in zip(names, lengths))
            raise ValueError(msg)

    # Propagation-constant ↔ Z² conversions ############################

    def betafrom(self, Z2):
        """Return physical propagation constant β given nondimensional Z².

        Uses the outer refractive index n0 = ns[-1] and the
        characteristic length L = scale:

            β = sqrt((L · k₀ · n0)² − Z²) / L
        """
        k_low = self.scale * self.k0 * self.ns[-1]
        return np.sqrt(k_low**2 - Z2) / self.scale

    def sqrZfrom(self, beta):
        """Return nondimensional Z² given physical propagation constant β.

        Uses the outer refractive index n0 = ns[-1] and the
        characteristic length L = scale:

            Z² = (L · k₀ · n0)² − (L · β)²
        """
        k_low = self.scale * self.k0 * self.ns[-1]
        return k_low**2 - (self.scale * beta)**2

    # Matplotlib plotting helpers ######################################

    def graphpoints(self, rlist=None, ntheta=101):
        """Build radial/angular sample grids for 1-D and 2-D plots.
        """

        rhos = np.concatenate([[1e-9], self.rhos / self.scale])
        n = len(rhos) - 1
        if rlist is not None:
            if len(rlist) != n:
                raise ValueError(f'rlist must have {n} entries ' +
                                 '(one per region).')
            self.rs = np.concatenate([
                np.linspace(rhos[i], rhos[i + 1], rlist[i]) for i in range(n)
            ])
        else:
            self.rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i + 1], 101) for i in range(n)])
        self.thetas = np.linspace(0, 2 * np.pi, ntheta)
        self.Rs, self.Thetas = np.meshgrid(self.rs, self.thetas)
        self.Xs = self.Rs * np.cos(self.Thetas)
        self.Ys = self.Rs * np.sin(self.Thetas)

    def plot1D(self,
               F,
               rlist=None,
               figsize=(8, 6),
               part='real',
               nu=1,
               double_r=False,
               return_vals=False,
               maxscale=False,
               **lineargs):
        """Plot a radial field function F(r) across the fiber diameter.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.graphpoints(rlist=rlist)
        if double_r:
            rs = np.concatenate([-np.flip(self.rs), self.rs])
            ys = np.concatenate(
                [np.exp(1j * nu * np.pi) * F(np.flip(self.rs)),
                 F(self.rs)])
        else:
            rs, ys = self.rs, F(self.rs)
        if maxscale:
            ys /= np.max(np.abs(ys))
        _plot_part(ax, rs, ys, part, **lineargs)
        plt.show()
        return (rs, ys) if return_vals else (fig, ax)

    def add1D_plot(self,
                   ax,
                   F,
                   part='real',
                   double_r=False,
                   nu=1,
                   maxscale=False,
                   **lineargs):
        """Add a radial field curve to an existing Axes.
        """
        self.graphpoints()
        if double_r:
            rs = np.concatenate([-np.flip(self.rs), self.rs])
            ys = np.concatenate(
                [np.exp(1j * nu * np.pi) * F(np.flip(self.rs)),
                 F(self.rs)])
        else:
            rs, ys = self.rs, F(self.rs)
        if maxscale:
            ys /= np.max(np.abs(ys))
        _plot_part(ax, rs, ys, part, **lineargs)

    def plot2D_contour(self,
                       F,
                       rlist=None,
                       ntheta=101,
                       figsize=(16, 16),
                       part='real',
                       levels=40,
                       plot_rhos=True,
                       edgecolor='k',
                       cmap='jet',
                       colorbar_scale=.8,
                       colorbar_fontsize=14,
                       linewidth=1.1,
                       **lineargs):
        """Filled contour plot of a field function F(x, y).
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.graphpoints(rlist=rlist, ntheta=ntheta)
        zs = _select_part(F(self.Xs, self.Ys), part)
        contour = ax.contourf(self.Xs, self.Ys, zs, levels=levels, cmap=cmap)
        if plot_rhos:
            for rho in self.rhos / self.scale:
                plt.plot(rho * np.cos(self.thetas),
                         rho * np.sin(self.thetas),
                         color=edgecolor,
                         linewidth=linewidth,
                         **lineargs)
        ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(contour, shrink=colorbar_scale)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        return fig, ax


# #######################################################################
# Scalar  (2×2 transfer matrix) ##########################################


class BraggExactScalar(_BraggExactBase):
    """Scalar Helmholtz approximation via 2×2 transfer matrix (Yeh et al.).

    Propagation constants are zeros of determinant().
    Use fields_matplot() to obtain callable field functions.
    """

    def transfer_matrix(self, beta, nu, rho, n1, n2, zfunc='bessel'):
        """2×2 transfer matrix across an interface at physical radius *rho*.

        Parameters
        ----------
        beta   : scaled propagation constant (scalar or array)
        nu     : azimuthal order
        rho    : interface radius (physical units, not scaled)
        n1, n2 : refractive indices on left and right of interface
        zfunc  : 'bessel' (J/Y) or 'hankel' (H1/H2)
        """
        beta = np.array(beta, dtype=np.complex128)
        z1, z1p, z2, z2p = _bessel_pair(zfunc)
        f = 1j * np.pi / 4 if zfunc == 'hankel' else np.pi / 2

        k0 = self.k0 * self.scale
        K1 = np.sqrt((k0 * n1)**2 - beta**2, dtype=complex)
        K2 = np.sqrt((k0 * n2)**2 - beta**2, dtype=complex)
        X = K1 * rho / self.scale
        Y = K2 * rho / self.scale
        F = K1 / K2

        M = np.zeros(beta.shape + (2, 2), dtype=np.complex128)
        Ymat = np.zeros_like(M)
        Ymat[..., 0, :] = np.array([Y.T, Y.T]).T
        Ymat[..., 1, :] = np.array([Y.T, Y.T]).T

        M[..., 0, :] = np.array([
            (z1(nu, X) * z2p(nu, Y) - F * z1p(nu, X) * z2(nu, Y)).T,
            (z2(nu, X) * z2p(nu, Y) - F * z2p(nu, X) * z2(nu, Y)).T,
        ]).T
        M[..., 1, :] = np.array([
            (F * z1p(nu, X) * z1(nu, Y) - z1(nu, X) * z1p(nu, Y)).T,
            (F * z2p(nu, X) * z1(nu, Y) - z2(nu, X) * z1p(nu, Y)).T,
        ]).T
        return f * Ymat * M

    def state_matrix(self, beta, nu, rho, n, zfunc='bessel', Ktype='kappa'):
        """2×2 state (matching) matrix at physical radius *rho* in medium *n*.
        """
        beta = np.array(beta, dtype=np.complex128)
        z1, z1p, z2, z2p = _bessel_pair(zfunc)
        rho = rho / self.scale
        k0 = self.k0 * self.scale
        k = k0 * n

        if Ktype == 'kappa':
            K = np.sqrt(k**2 - beta**2, dtype=complex)
        elif Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta**2 - k**2, dtype=complex)
        else:
            raise ValueError('Ktype must be "kappa" or "i_gamma".')

        L = np.zeros(beta.shape + (2, 2), dtype=complex)
        L[..., 0, :] = np.array([z1(nu, K * rho).T, z2(nu, K * rho).T]).T
        L[..., 1, :] = np.array([(K * z1p(nu, K * rho)).T,
                                 (K * z2p(nu, K * rho)).T]).T
        return L

    def determinant(self,
                    beta,
                    nu=1,
                    outer='h1',
                    Ktype='kappa',
                    return_coeffs=False,
                    return_matrix=False):
        """Characteristic determinant; zeros are propagation constants.

        Parameters
        ----------
        beta   : scaled propagation constant (scalar or array)
        nu     : azimuthal order
        outer  : 'h1' (Im β > 0, our convention here in fibermode) or
                 'h2' (Im β < 0, Yeh et al. convention)
        """
        if return_coeffs and return_matrix:
            raise ValueError(
                'Only one of return_coeffs and return_matrix may be True.')
        if outer not in ('h1', 'h2'):
            raise ValueError("outer must be 'h1' or 'h2'.")

        beta = np.array(beta, dtype=np.complex128)
        rhos, ns = self.rhos, self.ns

        L = np.zeros(beta.shape + (2, 1), dtype=complex)
        L[..., :, :] = np.eye(2)[:, [0]]  # J column for core

        for i in range(len(rhos) - 2):
            L = self.transfer_matrix(beta, nu, rhos[i], ns[i], ns[i + 1]) @ L

        L = self.state_matrix(beta, nu, rhos[-2], ns[-2]) @ L
        ind = [0] if outer == 'h1' else [1]
        R = self.state_matrix(beta,
                              nu,
                              rhos[-2],
                              ns[-1],
                              zfunc='hankel',
                              Ktype=Ktype)[..., ind]

        A, B = L[..., 0, 0], L[..., 1, 0]
        C, D = R[..., 0, 0], R[..., 1, 0]

        if return_coeffs:
            return A, B, C, D
        if return_matrix:
            M = np.zeros(beta.shape + (2, 2), dtype=complex)
            M[..., :1] = L
            M[..., 1:] = R
            return M
        return B * C - A * D

    def coefficients(self, beta, nu=1, outer='h1', Ktype='kappa'):
        """Bessel expansion coefficients per layer, shape (n_layers, 2).
        """
        if outer not in ('h1', 'h2'):
            raise ValueError("outer must be 'h1' or 'h2'.")

        A, B, C, D = self.determinant(beta,
                                      nu=nu,
                                      outer=outer,
                                      Ktype=Ktype,
                                      return_coeffs=True)
        Vs = np.array([C, D])
        imax = np.argmax(np.abs(Vs))
        v1, w1 = (C, A) if imax == 0 else (D, B)

        rhos, ns = self.rhos, self.ns
        M = np.zeros((len(rhos), 2), dtype=complex)
        L = np.eye(2)[:, [0]] * v1
        M[0, :] = L.flatten()

        for i in range(len(rhos) - 2):
            L = self.transfer_matrix(beta, nu, rhos[i], ns[i], ns[i + 1]) @ L
            M[i + 1, :] = L.flatten()

        M[-1, 0 if outer == 'h1' else 1] = w1
        return 1 / v1 * M

    def fields_matplot(self, beta, nu=1, outer='h1', Ktype='kappa'):
        """Callable field functions for matplotlib plotting.

        Returns a dict with keys 'Ez' (2-D callable) and 'Ez_rad' (1-D radial).
        """
        M = self.coefficients(beta, nu=nu, outer=outer, Ktype=Ktype)
        rhos = np.concatenate([[0], self.rhos / self.scale])
        ks = self.ks * self.scale
        Ks = np.sqrt(ks**2 - beta**2, dtype=complex)
        if Ktype == 'i_gamma':
            Ks[-1] = 1j * np.sqrt(beta**2 - ks[-1]**2, dtype=complex)
        n_reg = len(rhos) - 1

        def Ez_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B = M[i, :]
                if i == 0:
                    ys[idx] = A * jv(nu, Ks[i] * ri)
                elif i < n_reg - 1:
                    ys[idx] = A * jv(nu, Ks[i] * ri) + B * yv(nu, Ks[i] * ri)
                else:
                    ys[idx] = A * h1(nu, Ks[i] * ri) + B * h2(nu, Ks[i] * ri)
            return ys

        def Ez(x, y):
            r = np.sqrt(x * x + y * y)
            t = np.arctan2(y, x)
            return Ez_rad(r) * np.exp(1j * nu * t)

        return {'Ez': Ez, 'Ez_rad': Ez_rad}


# #######################################################################
# Vector  (4×4 transfer matrix) ##########################################


class BraggExactVector(_BraggExactBase):
    """Full Maxwell solution via 4×4 transfer matrix (Yeh et al.).

    All six field components and transverse Cartesian fields are available
    via fields_matplot().
    """

    def transfer_matrix(self, beta, nu, rho, n1, n2, zfunc='bessel'):
        """4×4 transfer matrix across an interface at physical radius *rho*.
        """
        beta = np.array(beta, dtype=np.complex128)
        z1, z1p, z2, z2p = _bessel_pair(zfunc)
        f = 1j * np.pi / 4 if zfunc == 'hankel' else np.pi / 2

        k0 = self.k0 * self.scale
        K1 = np.sqrt((k0 * n1)**2 - beta**2, dtype=complex)
        K2 = np.sqrt((k0 * n2)**2 - beta**2, dtype=complex)
        rho_sc = rho / self.scale
        X, Y = K1 * rho_sc, K2 * rho_sc

        expr = 1 / Y - Y / X**2
        F1 = (K2 * n1**2) / (K1 * n2**2)
        F2 = nu * beta / (k0 * n2**2)
        F3 = nu * beta / k0
        F4 = K2 / K1

        M = np.zeros(beta.shape + (4, 4), dtype=np.complex128)
        Ymat = np.zeros_like(M)
        for row in range(4):
            Ymat[..., row, :] = np.array([Y.T, Y.T, Y.T, Y.T]).T

        M[..., 0, :] = np.array([
            (z1(nu, X) * z2p(nu, Y) - F1 * z1p(nu, X) * z2(nu, Y)).T,
            (z2(nu, X) * z2p(nu, Y) - F1 * z2p(nu, X) * z2(nu, Y)).T,
            (1j * F2 * z1(nu, X) * z2(nu, Y) * expr).T,
            (1j * F2 * z2(nu, X) * z2(nu, Y) * expr).T,
        ]).T
        M[..., 1, :] = np.array([
            (F1 * z1p(nu, X) * z1(nu, Y) - z1(nu, X) * z1p(nu, Y)).T,
            (F1 * z2p(nu, X) * z1(nu, Y) - z2(nu, X) * z1p(nu, Y)).T,
            (-1j * F2 * z1(nu, X) * z1(nu, Y) * expr).T,
            (-1j * F2 * z2(nu, X) * z1(nu, Y) * expr).T,
        ]).T
        M[..., 2, :] = np.array([
            (-1j * F3 * z1(nu, X) * z2(nu, Y) * expr).T,
            (-1j * F3 * z2(nu, X) * z2(nu, Y) * expr).T,
            (z1(nu, X) * z2p(nu, Y) - F4 * z1p(nu, X) * z2(nu, Y)).T,
            (z2(nu, X) * z2p(nu, Y) - F4 * z2p(nu, X) * z2(nu, Y)).T,
        ]).T
        M[..., 3, :] = np.array([
            (1j * F3 * z1(nu, X) * z1(nu, Y) * expr).T,
            (1j * F3 * z2(nu, X) * z1(nu, Y) * expr).T,
            (F4 * z1p(nu, X) * z1(nu, Y) - z1(nu, X) * z1p(nu, Y)).T,
            (F4 * z2p(nu, X) * z1(nu, Y) - z2(nu, X) * z1p(nu, Y)).T,
        ]).T
        return f * Ymat * M

    def state_matrix(self, beta, nu, rho, n, zfunc='bessel', Ktype='kappa'):
        """4×4 state (matching) matrix; see Yeh et al. eq. 34.
        """
        beta = np.array(beta, dtype=np.complex128)
        z1, z1p, z2, z2p = _bessel_pair(zfunc)
        rho = rho / self.scale
        k0 = self.k0 * self.scale
        k = k0 * n
        if Ktype == 'kappa':
            K = np.sqrt(k**2 - beta**2, dtype=complex)
        elif Ktype == 'i_gamma':
            K = 1j * np.sqrt(beta**2 - k**2, dtype=complex)
        else:
            raise ValueError('Ktype must be "kappa" or "i_gamma".')

        Z = np.zeros_like(beta, dtype=complex)
        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        L[..., 0, :] = np.array([z1(nu, K * rho).T, z2(nu, K * rho).T, Z, Z]).T
        L[..., 1, :] = np.array([
            (k0 * n**2 / (beta * K) * z1p(nu, K * rho)).T,
            (k0 * n**2 / (beta * K) * z2p(nu, K * rho)).T,
            (1j * nu / (K**2 * rho) * z1(nu, K * rho)).T,
            (1j * nu / (K**2 * rho) * z2(nu, K * rho)).T,
        ]).T
        L[..., 2, :] = np.array([Z, Z, z1(nu, K * rho).T, z2(nu, K * rho).T]).T
        L[..., 3, :] = np.array([
            (1j * nu / (K**2 * rho) * z1(nu, K * rho)).T,
            (1j * nu / (K**2 * rho) * z2(nu, K * rho)).T,
            (-k0 / (beta * K) * z1p(nu, K * rho)).T,
            (-k0 / (beta * K) * z2p(nu, K * rho)).T,
        ]).T
        return L

    def state_matrix_inverse(self, beta, nu, rho, n, zfunc='bessel'):
        """Analytic inverse of the 4×4 state matrix (for debugging).
        """
        beta = np.array(beta, dtype=np.complex128)
        z1, z1p, z2, z2p = _bessel_pair(zfunc)
        k0 = self.k0 * self.scale
        K = np.sqrt((k0 * n)**2 - beta**2, dtype=complex)
        rho = rho / self.scale
        X = K * rho
        F1 = K * beta / (k0 * n**2)
        F2 = beta * nu / (k0 * n**2 * X)
        F3 = beta * nu / (k0 * X)
        F4 = K * beta / k0

        Z = np.zeros_like(beta, dtype=complex)
        L = np.zeros(beta.shape + (4, 4), dtype=complex)
        L[..., 0, :] = np.array(
            [z2p(nu, X).T, (-F1 * z2(nu, X)).T, (1j * F2 * z2(nu, X)).T, Z]).T
        L[..., 1, :] = np.array([(-z1p(nu, X)).T, (F1 * z1(nu, X)).T,
                                 (-1j * F2 * z1(nu, X)).T, Z]).T
        L[..., 2, :] = np.array([(-1j * F3 * z2(nu, X)).T, Z,
                                 z2p(nu, X).T, (F4 * z2(nu, X)).T]).T
        L[..., 3, :] = np.array([(-1j * F3 * z1(nu, X)).T, Z, (-z1p(nu, X)).T,
                                 (-F4 * z1(nu, X)).T]).T
        return np.pi * X / 2 * L

    def determinant(self,
                    beta,
                    nu=1,
                    outer='h1',
                    Ktype='kappa',
                    return_coeffs=False,
                    return_matrix=False):
        """Characteristic determinant; zeros are propagation constants.

        Parameters
        ----------
        outer : 'h1', 'h2', or 'pcb' (perfectly-conducting BC)
        """
        if return_coeffs and return_matrix:
            raise ValueError(
                'Only one of return_coeffs and return_matrix may be True.')
        if outer not in ('h1', 'h2', 'pcb'):
            raise ValueError("outer must be 'h1', 'h2', or 'pcb'.")

        beta = np.array(beta, dtype=np.complex128)
        rhos, ns = self.rhos, self.ns

        L = np.zeros(beta.shape + (4, 2), dtype=complex)
        L[..., :, :] = np.eye(4)[:, [0, 2]]  # J columns for core

        if outer != 'pcb':
            for i in range(len(rhos) - 2):
                L = self.transfer_matrix(beta, nu, rhos[i], ns[i],
                                         ns[i + 1]) @ L
            L = self.state_matrix(beta, nu, rhos[-2], ns[-2]) @ L

            inds = [0, 2] if outer == 'h1' else [1, 3]
            R = self.state_matrix(beta,
                                  nu,
                                  rhos[-2],
                                  ns[-1],
                                  zfunc='hankel',
                                  Ktype=Ktype)[..., inds]

            a, b = L[..., 0, 0], L[..., 0, 1]
            c, d = L[..., 2, 0], L[..., 2, 1]
            e, f_ = L[..., 1, 0], L[..., 1, 1]
            g, h_ = L[..., 3, 0], L[..., 3, 1]
            alpha, Beta_ = R[..., 0, 0], R[..., 2, 1]
            gamma, delta = R[..., 1, 0], R[..., 1, 1]
            epsilon, sigma = R[..., 3, 0], R[..., 3, 1]

            A = e - (a / alpha * gamma + c / Beta_ * delta)
            B = f_ - (b / alpha * gamma + d / Beta_ * delta)
            C = g - (a / alpha * epsilon + c / Beta_ * sigma)
            D = h_ - (b / alpha * epsilon + d / Beta_ * sigma)

            if return_coeffs:
                return A, B, C, D, a, b, c, d, alpha, Beta_
            if return_matrix:
                M = np.zeros(beta.shape + (4, 4), dtype=complex)
                M[..., :2] = L
                M[..., 2:] = R
                return M
            return C * B - A * D

        else:  # perfectly-conducting boundary
            for i in range(len(rhos) - 1):
                L = self.transfer_matrix(beta, nu, rhos[i], ns[i],
                                         ns[i + 1]) @ L
            L = self.state_matrix(beta, nu, rhos[-1], ns[-1], Ktype=Ktype) @ L
            R_sel = np.zeros(beta.shape + (2, 4), dtype=complex)
            R_sel[..., :, :] = np.eye(4)[[0, 3], :]
            L = R_sel @ L
            A, B, C, D = (L[..., 0, 0], L[..., 0, 1], L[..., 1, 0], L[..., 1,
                                                                      1])
            if return_coeffs:
                return A, B, C, D
            if return_matrix:
                return L, R_sel
            return A * D - B * C

    def coefficients(self, beta, nu=1, outer='h1', Ktype='kappa'):
        """Bessel expansion coefficients per layer, shape (n_layers, 4).
        """
        if outer not in ('h1', 'h2', 'pcb'):
            raise ValueError("outer must be 'h1', 'h2', or 'pcb'.")

        rhos, ns = self.rhos, self.ns

        if outer != 'pcb':
            A, B, C, D, a, b, c, d, alpha, Beta_ = self.determinant(
                beta, nu=nu, outer=outer, Ktype=Ktype, return_coeffs=True)

            Vs = np.array([A, B, C, D])
            if np.abs(Vs).max() < 1e-13:
                raise ValueError('Determinant matrix elements are too small.')
            imax = np.argmax(np.abs(Vs))
            v1, v2 = (B, -A) if imax in (0, 1) else (D, -C)
            v = np.array([v1, v2])
            w1 = (a * v1 + b * v2) / alpha
            w2 = (c * v1 + d * v2) / Beta_

            M = np.zeros((len(rhos), 4), dtype=complex)
            L = np.eye(4)[:, [0, 2]] @ np.array([[v1], [v2]])
            M[0, :] = L.flatten()
            for i in range(len(rhos) - 2):
                L = self.transfer_matrix(beta, nu, rhos[i], ns[i],
                                         ns[i + 1]) @ L
                M[i + 1, :] = L.flatten()

            inds = (0, 2) if outer == 'h1' else (1, 3)
            M[-1, inds[0]] = w1
            M[-1, inds[1]] = w2
            vscale = v1 if nu > 0 else v[np.argmax(np.abs(v))]
            return 1 / vscale * M

        else:
            A, B, C, D = self.determinant(beta,
                                          nu,
                                          outer,
                                          Ktype=Ktype,
                                          return_coeffs=True)
            Vs = np.array([A, B, C, D])
            if np.abs(Vs).max() < 1e-13:
                raise ValueError('Determinant matrix elements are too small.')
            imax = np.argmax(np.abs(Vs))
            v1, v2 = (B, -A) if imax in (0, 1) else (D, -C)
            v = np.array([v1, v2])

            M = np.zeros((len(rhos), 4), dtype=complex)
            L = np.eye(4)[:, [0, 2]] @ np.array([[v1], [v2]])
            M[0, :] = L.flatten()
            for i in range(len(rhos) - 1):
                L = self.transfer_matrix(beta, nu, rhos[i], ns[i],
                                         ns[i + 1]) @ L
                M[i + 1, :] = L.flatten()

            vscale = v1 if nu > 0 else v[np.argmax(np.abs(v))]
            return 1 / vscale * M

    def fields_matplot(self, beta, nu=1, outer='h1', Ktype='kappa'):
        """Callable field functions for matplotlib plotting.

        Returns a dict of Ez, Hz, Er, Ephi, Hr, Hphi, Ex, Ey, Hx, Hy, Sz
        (2-D callables) plus their _rad counterparts (1-D radial).
        """
        M = self.coefficients(beta, nu=nu, outer=outer, Ktype=Ktype)
        rhos = np.concatenate([[0], self.rhos / self.scale])
        k0 = self.k0 * self.scale
        ks = self.ks * self.scale
        ns = self.ns
        Ks = np.sqrt(ks**2 - beta**2, dtype=complex)
        if Ktype == 'i_gamma':
            Ks[-1] = 1j * np.sqrt(beta**2 - ks[-1]**2, dtype=complex)
        Fs = 1j * beta / (ks**2 - beta**2)
        n_reg = len(rhos) - 1

        # ---- Ez ----
        def Ez_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B = M[i, 0], M[i, 1]
                if i == 0:
                    ys[idx] = A * jv(nu, Ks[i] * ri)
                elif i < n_reg - 1:
                    ys[idx] = (A * jv(nu, Ks[i] * ri) + B * yv(nu, Ks[i] * ri))
                else:
                    ys[idx] = (A * h1(nu, Ks[i] * ri) + B * h2(nu, Ks[i] * ri))
            return ys

        def Ez(x, y):
            r = np.sqrt(x * x + y * y)
            return Ez_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Hz ----
        def Hz_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                C, D = M[i, 2], M[i, 3]
                if i == 0:
                    ys[idx] = C * jv(nu, Ks[i] * ri)
                elif i < n_reg - 1:
                    ys[idx] = (C * jv(nu, Ks[i] * ri) + D * yv(nu, Ks[i] * ri))
                else:
                    ys[idx] = (C * h1(nu, Ks[i] * ri) + D * h2(nu, Ks[i] * ri))
            return ys

        def Hz(x, y):
            r = np.sqrt(x * x + y * y)
            return Hz_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Er ----
        def Er_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B, C, D = M[i, 0], M[i, 1], M[i, 2], M[i, 3]
                if i == 0:
                    ys[idx] = Fs[i] * (
                        Ks[i] * A * jvp(nu, Ks[i] * ri) + k0 /
                        (beta * ri) * 1j * nu * C * jv(nu, Ks[i] * ri))
                elif i < n_reg - 1:
                    ys[idx] = Fs[i] * (
                        Ks[i] *
                        (A * jvp(nu, Ks[i] * ri) + B * yvp(nu, Ks[i] * ri)) +
                        k0 / (beta * ri) * 1j * nu *
                        (C * jv(nu, Ks[i] * ri) + D * yv(nu, Ks[i] * ri)))
                else:
                    ys[idx] = Fs[i] * (
                        Ks[i] *
                        (A * h1vp(nu, Ks[i] * ri) + B * h2vp(nu, Ks[i] * ri)) +
                        k0 / (beta * ri) * 1j * nu *
                        (C * h1(nu, Ks[i] * ri) + D * h2(nu, Ks[i] * ri)))
            return ys

        def Er(x, y):
            r = np.sqrt(x * x + y * y)
            return Er_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Ephi ----
        def Ephi_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B, C, D = M[i, 0], M[i, 1], M[i, 2], M[i, 3]
                if i == 0:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri * A * jv(nu, Ks[i] * ri) -
                        k0 / beta * Ks[i] * C * jvp(nu, Ks[i] * ri))
                elif i < n_reg - 1:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri *
                        (A * jv(nu, Ks[i] * ri) + B * yv(nu, Ks[i] * ri)) -
                        k0 / beta * Ks[i] *
                        (C * jvp(nu, Ks[i] * ri) + D * yvp(nu, Ks[i] * ri)))
                else:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri *
                        (A * h1(nu, Ks[i] * ri) + B * h2(nu, Ks[i] * ri)) -
                        k0 / beta * Ks[i] *
                        (C * h1vp(nu, Ks[i] * ri) + D * h2vp(nu, Ks[i] * ri)))
            return ys

        def Ephi(x, y):
            r = np.sqrt(x * x + y * y)
            return Ephi_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Hr ----
        def Hr_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B, C, D = M[i, 0], M[i, 1], M[i, 2], M[i, 3]
                if i == 0:
                    ys[idx] = Fs[i] * (
                        Ks[i] * C * jvp(nu, Ks[i] * ri) - k0 * ns[i]**2 /
                        (beta * ri) * 1j * nu * A * jv(nu, Ks[i] * ri))
                elif i < n_reg - 1:
                    ys[idx] = Fs[i] * (
                        Ks[i] *
                        (C * jvp(nu, Ks[i] * ri) + D * yvp(nu, Ks[i] * ri)) -
                        k0 * ns[i]**2 / (beta * ri) * 1j * nu *
                        (A * jv(nu, Ks[i] * ri) + B * yv(nu, Ks[i] * ri)))
                else:
                    ys[idx] = Fs[i] * (
                        Ks[i] *
                        (C * h1vp(nu, Ks[i] * ri) + D * h2vp(nu, Ks[i] * ri)) -
                        k0 * ns[i]**2 / (beta * ri) * 1j * nu *
                        (A * h1(nu, Ks[i] * ri) + B * h2(nu, Ks[i] * ri)))
            return ys

        def Hr(x, y):
            r = np.sqrt(x * x + y * y)
            return Hr_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Hphi ----
        def Hphi_rad(rs):
            ys = np.zeros_like(rs, dtype=complex)
            for i in range(n_reg):
                idx = np.where((rhos[i] <= rs) & (rs <= rhos[i + 1]))
                ri = rs[idx]
                A, B, C, D = M[i, 0], M[i, 1], M[i, 2], M[i, 3]
                if i == 0:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri * C * jv(nu, Ks[i] * ri) +
                        k0 * ns[i]**2 / beta * Ks[i] * A * jvp(nu, Ks[i] * ri))
                elif i < n_reg - 1:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri *
                        (C * jv(nu, Ks[i] * ri) + D * yv(nu, Ks[i] * ri)) +
                        k0 * ns[i]**2 / beta * Ks[i] *
                        (A * jvp(nu, Ks[i] * ri) + B * yvp(nu, Ks[i] * ri)))
                else:
                    ys[idx] = Fs[i] * (
                        1j * nu / ri *
                        (C * h1(nu, Ks[i] * ri) + D * h2(nu, Ks[i] * ri)) +
                        k0 * ns[i]**2 / beta * Ks[i] *
                        (A * h1vp(nu, Ks[i] * ri) + B * h2vp(nu, Ks[i] * ri)))
            return ys

        def Hphi(x, y):
            r = np.sqrt(x * x + y * y)
            return Hphi_rad(r) * np.exp(1j * nu * np.arctan2(y, x))

        # ---- Cartesian ----
        def Ex(x, y):
            r = np.sqrt(x * x + y * y)
            return (x * Er(x, y) - y * Ephi(x, y)) / r

        def Ey(x, y):
            r = np.sqrt(x * x + y * y)
            return (y * Er(x, y) + x * Ephi(x, y)) / r

        def Hx(x, y):
            r = np.sqrt(x * x + y * y)
            return (x * Hr(x, y) - y * Hphi(x, y)) / r

        def Hy(x, y):
            r = np.sqrt(x * x + y * y)
            return (y * Hr(x, y) + x * Hphi(x, y)) / r

        def Sz_rad(rs):
            return (Er_rad(rs) * np.conj(Hphi_rad(rs)) -
                    Ephi_rad(rs) * np.conj(Hr_rad(rs)))

        def Sz(x, y):
            return Ex(x, y) * np.conj(Hy(x, y)) - Ey(x, y) * np.conj(Hx(x, y))

        return {
            'Ez': Ez,
            'Ez_rad': Ez_rad,
            'Hz': Hz,
            'Hz_rad': Hz_rad,
            'Er': Er,
            'Er_rad': Er_rad,
            'Ephi': Ephi,
            'Ephi_rad': Ephi_rad,
            'Hr': Hr,
            'Hr_rad': Hr_rad,
            'Hphi': Hphi,
            'Hphi_rad': Hphi_rad,
            'Ex': Ex,
            'Ey': Ey,
            'Hx': Hx,
            'Hy': Hy,
            'Sz': Sz,
            'Sz_rad': Sz_rad,
        }

    def plot2D_streamlines(self,
                           Fx,
                           Fy,
                           rlist=None,
                           ntheta=101,
                           Nstrm=101,
                           figsize=(16, 16),
                           part='real',
                           levels=40,
                           contourfunc=None,
                           contourpart='norm',
                           colorbar_scale=.8,
                           colorbar_fontsize=14,
                           streamline_color='k',
                           maxd_scaling=1,
                           streamline_width=1.5,
                           arrowsize=3.5,
                           arrowstyle='->',
                           broken_streamlines=True,
                           density=2.2,
                           plot_rhos=True,
                           rho_linewidth=1.1,
                           rho_linestyle='-',
                           plot_seed=False,
                           rho_linecolor='k',
                           seed_nr=None,
                           seed_ntheta=65,
                           **streamplotkwargs):
        """Streamline plot of the transverse electric field."""

        if contourfunc is not None:
            fig, ax = self.plot2D_contour(contourfunc,
                                          rlist=rlist,
                                          ntheta=ntheta,
                                          figsize=figsize,
                                          levels=levels,
                                          part=contourpart,
                                          plot_rhos=plot_rhos,
                                          cmap='jet',
                                          linewidth=rho_linewidth,
                                          linestyle=rho_linestyle,
                                          edgecolor=rho_linecolor,
                                          colorbar_fontsize=colorbar_fontsize,
                                          colorbar_scale=colorbar_scale)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        R = self.rhos[-1] / self.scale
        stream_pts = np.linspace(-R, R, Nstrm, dtype=float)
        if 0 in stream_pts:
            stream_pts = np.linspace(-R, R, Nstrm + 1, dtype=float)

        X2, Y2 = np.meshgrid(stream_pts, stream_pts)
        U, V = Fx(X2, Y2), Fy(X2, Y2)
        ex = _select_part(U, part)
        ey = _select_part(V, part)

        seed_points = self.seed_points(rlist=seed_nr, ntheta=seed_ntheta)

        try:
            ax.streamplot(X2,
                          Y2,
                          ex,
                          ey,
                          density=density,
                          linewidth=streamline_width,
                          color=streamline_color,
                          broken_streamlines=broken_streamlines,
                          arrowsize=arrowsize,
                          arrowstyle=arrowstyle,
                          start_points=seed_points,
                          maxd_scaling=maxd_scaling,
                          **streamplotkwargs)
        except TypeError:
            ax.streamplot(X2,
                          Y2,
                          ex,
                          ey,
                          density=density,
                          linewidth=streamline_width,
                          color=streamline_color,
                          broken_streamlines=broken_streamlines,
                          arrowsize=arrowsize,
                          arrowstyle=arrowstyle,
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
        """Generate seed points for streamline plotting."""

        rhos = np.concatenate([[0], self.rhos / self.scale])
        n = len(rhos) - 1
        if rlist is not None:
            if len(rlist) != n:
                raise ValueError(
                    f'rlist must have {n} entries (one per region).')
            rs = np.concatenate([
                np.linspace(rhos[i], rhos[i + 1], rlist[i] + 2)[1:-1]
                for i in range(n)
            ])
        else:
            rs = np.concatenate(
                [np.linspace(rhos[i], rhos[i + 1], 5)[1:-1] for i in range(n)])

        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        Rs, Thetas = np.meshgrid(rs, thetas)
        Xs = Rs * np.cos(Thetas)
        Ys = Rs * np.sin(Thetas)
        return np.array([Xs.flatten(), Ys.flatten()]).T
