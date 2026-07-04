"""Demo: Bragg fiber leaky scalar mode via semi-analytical
approach of transfer-matrix implemented in  BraggExactScalar.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from fibermode.bragg import BraggExactScalar

A = BraggExactScalar(
    ts=[5e-5, 1e-5, 2e-5],
    ns=[1, 1.44, 1],
    mats=['air', 'glass', 'air'],
    wl=1.2e-6,
    scale=1e-6,
)

# Complex root finding

nu = 0  # scalar fundamental mode is azimuthally symmetric
outer = 'h1'  # outgoing Hankel; gives Im(beta) > 0 for leaky modes
k_low = A.k0 * A.ns[0] * A.scale  # Visual identification in Notebook 2.1
guess = (.999955 * k_low + .99996 * k_low) / 2
beta1 = newton(A.determinant, guess, args=(nu, outer), tol=1e-15)
print(f'Scaled beta = {beta1}')
print(f'Residual    = {abs(A.determinant(beta1, nu, outer)):.2e}')

# Physical beta and Z² eigenvalue (Im(Z) < 0 for leaky modes)

beta_phys = beta1 / A.scale  # Undo the L-scaling to recover physical beta
Z2 = A.sqrZfrom(beta_phys)  # The  Z^2 nondimensional eigenvalue
print(f'beta (physical)  = {beta_phys}')
print(f'Z^2              = {Z2}')
print(f'Confinement loss = {20 * beta_phys.imag / np.log(10):.4f} dB/m')
# Archived print outputs:
#    Scaled beta = (5.2357654056629634+2.6356530509847056e-08j)
#    Residual    = 1.71e-12
#    beta (physical)  = (5235765.405662963+0.026356530509847056j)
#    Z^2              = (0.0023283976667194395-2.759932213135152e-07j)
#    Confinement loss = 0.2289 dB/m

# Assemble and plot the field

F = A.fields_matplot(beta1, nu, outer)
A.plot2D_contour(F['Ez'], figsize=(10, 7))
plt.show()
A.plot1D(F['Ez_rad'],
         double_r=True,
         rlist=[400, 10000, 400],
         nu=nu,
         maxscale=True,
         linewidth=1.5,
         color='k',
         figsize=(6, 7))
