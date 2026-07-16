"""
Computing some modes of a 8-tube ARF structure in Kolyadin's paper
"""

from arf import ARF
from math import pi


if __name__ == '__main__':
    a = ARF(name='kolyadin', freecapil=False)
    p = 3         # finite element degree
    a.refine()

    # Note that for p = 3 and above, we need to do at least one mesh
    # refinement. Otherwise, the center for the LP01 mode falls back
    # to having real(Z) ~ 2.29 for p = 2, 3.

    #       LP01, LP11, LP21  LP02
    ctrs = (2.29, 3.64, 4.86, 5.20)
    radi = (0.02, 0.02, 0.02, 0.02)

    # compute:
    Zs, Ys, Yls, betas, P, _, _, _ =  \
        a.polyeig(p=p, ctrs=ctrs, radi=radi,
                  npts=4, alpha=5,
                  eta_tol=1.e-12, stop_tol=1e-12)
    print('All computes eigenvalues:\n')
    print(' Zs =', Zs)
    print(' betas =', betas)

    # Compute the effective indices.
    k = 2 * pi / a.wavelength
    effective_indices_real = [b.real / k for b in betas]
    effective_indices_imag = [b.imag / k for b in betas]
    print(' effective indices (real part) =', effective_indices_real)
    print(' effective indices (imag part) =', effective_indices_imag)

    # visualize and / or save into file:

    Ys[-1].draw()  # visualize in netgen window

