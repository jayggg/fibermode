import numpy as np
import matplotlib.pyplot as plt


def plotlogf(f, rmin, rmax, imin, imax, *args, title='',
             levels=25, truncate=False, h=2, equal=False,
             colorbar=True, rref=20, iref=20, figsize=(14, 7),
             log_off=False, loop=False, three_D=False,
             phase=False, R2plane=False):
    """Create contour plot of complex function f on given range."""
    Xr = np.linspace(rmin, rmax, num=rref)
    Xi = np.linspace(imin, imax, num=iref)
    xr, xi = np.meshgrid(Xr, Xi)
    if not R2plane:
        zs = xr + 1j * xi
    if not loop:
        if not R2plane:
            fx = f(zs, *args)
        else:
            fx = f(xr, xi, *args)

    else:
        if not R2plane:
            fx = np.zeros_like(zs)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(zs[i, j], *args)
        else:
            fx = np.zeros_like(xr)
            for i in range(zs.shape[0]):
                for j in range(zs.shape[1]):
                    fx[i, j] = f(xr[i, j], xi[i, j], *args)
        # i, j = 0, 0
        # while i < zs.shape[0]:
        #     fx[i, j] = f(zs[i, j], *args)
        #     j += 1
        #     if (j == zs.shape[1]):
        #         j = 0
        #         i += 1
    if truncate:
        fx[np.abs(fx) > h] = h  # ignore large values to focus on roots

    if phase:
        if R2plane:
            raise ValueError('Working with R2plane=True; output assumed real, \
hence has no phase.  If complex output desired set R2plane=False (default).')
        ys = np.angle(fx)
    else:
        if R2plane:
            ys = fx
        else:
            ys = np.abs(fx)

    if not log_off and phase:
        raise ValueError("Need log_off when phase is turned on.")

    if not log_off:
        ys = np.log(np.abs(fx))

    fig = plt.figure(figsize=figsize)

    if three_D:
        ax = fig.add_subplot(projection='3d')
        lims = (np.ptp(Xr), np.ptp(Xi), min(np.ptp(Xr), np.ptp(Xi)))
        ax.set_box_aspect(lims)
        ax.set_axis_off()
        # ax.set_xlim3d(auto=True)
        # ax.set_ylim3d(auto=True)

        # ax.autoscale_view(tight=True)

    else:
        ax = fig.add_subplot()
        ax.grid(True)
        ax.set_facecolor('grey')

    if equal:
        ax.axis('equal')
        # plt.figure(figsize=(1.2 * (rmax - rmin), imax - imin))

    im = ax.contour(xr, xi, ys, levels=levels)

    plt.title(title)
    if colorbar:
        plt.colorbar(im)


def plotlogf_real(f, x_min, x_max, *args, n=1000, figsize=(12, 8),
                  level=np.e, log_off=False, truncate=False, height=2,
                  bounds=None):
    """Create contour plot of complex function f on given range."""

    xs = np.linspace(x_min, x_max, num=n)
    fx = f(xs, *args)

    plt.figure(figsize=figsize)

    if log_off:
        if truncate:
            fx[np.abs(fx) > height] = height  # truncate to height
        ys = np.abs(fx)
        plt.plot(xs, ys)
    else:
        ys = np.zeros_like(fx, dtype=float)
        ys[np.abs(fx) >= level] = np.log(np.abs(fx[np.abs(fx) >= level]))
        ys[np.abs(fx) < level] = np.abs(
            fx[np.abs(fx) < level]) * (np.log(level) / level)
        if truncate:
            ys[np.abs(ys) > height] = height
        plt.plot(xs, ys)

    bottom, top = plt.ylim()
    plt.ylim(0, top)

    if bounds is not None:
        l, r = bounds[:]
        plt.plot([l, l], [0, top], color='orange', linestye=':', linewidth=.75)
        plt.plot([r, r], [0, top], color='orange', linestye=':', linewidth=.75)
    plt.grid(True)
