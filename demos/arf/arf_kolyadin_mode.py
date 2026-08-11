"""
Computing some modes of a 8-tube ARF structure in Kolyadin's paper
"""

from fibermode import ARF
from math import pi

if __name__ == '__main__':
    a = ARF(name='kolyadin', freecapil=False)
    p = 3  # finite element degree
    a.refine()
    # Note that for p = 3 and above, we need to do at least one refinement
    # from prior experience.

    #       LP01, LP11, LP21  LP02
    ctrs = (2.29, 3.64, 4.86, 5.20)
    radi = (0.02, 0.02, 0.02, 0.02)

    Zs, Ys, Yls, betas = [], [], [], []
    for ctr, rad in zip(ctrs, radi):
        Z, y, yl, beta, P, moreoutputs = a.leakymode(p=p,
                                                     ctr=ctr,
                                                     rad=rad,
                                                     npts=4,
                                                     alpha=5,
                                                     eta_tol=1.e-12,
                                                     stop_tol=1e-8)
        Zs.append(Z)
        Ys.append(y)
        Yls.append(yl)
        betas.append(beta)

    print('All computes eigenvalues:\n')
    print(' Zs =', Zs)
    print(' betas =', betas)

    # Compute the effective indices.
    k = 2 * pi / a.wavelength
    effective_indices_real = [b.real / k for beta in betas for b in beta]
    effective_indices_imag = [b.imag / k for beta in betas for b in beta]
    print(' effective indices (real part) =', effective_indices_real)
    print(' effective indices (imag part) =', effective_indices_imag)

    # visualize and / or save into file:

    Ys[-1].draw()  # visualize in netgen window
