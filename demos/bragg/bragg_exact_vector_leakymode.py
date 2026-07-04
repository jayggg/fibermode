"""Demo: Bragg fiber leaky VECTOR mode via semi-analytical
approach of transfer-matrix implemented in  BraggExactVector.

We find a vector mode in approximately the same Z-location
as the scalar mode in bragg_exact_scalar_leakymode.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from fibermode.bragg import BraggExactVector

B = BraggExactVector(
    ts=[5e-5, 1e-5, 2e-5],
    ns=[1, 1.44, 1],
    mats=['air', 'glass', 'air'],
    wl=1.2e-6,
    scale=1e-6,
)

# Complex root finding:

nu = 1  # HE11 fundamental vector mode needs nu=1 (contrast scalar's nu=0)
outer = 'h1'
k_low = B.k0 * B.ns[0] * B.scale  # Visual identification in Notebook 2.1
guess = np.array(.99995 * k_low)
beta2 = newton(B.determinant, guess, args=(nu, outer), tol=1e-15)
print(f'Scaled beta = {beta2}')
print(f'Residual    = {abs(B.determinant(beta2, nu, outer)):.2e}')

# Physical beta and Z² eigenvalue (Im(Z) < 0 for leaky modes)

beta_phys = beta2 / B.scale  # Undo the L-scaling to recover physical beta
Z2 = B.sqrZfrom(beta_phys)  # The  Z^2 nondimensional eigenvalue
print(f'beta (physical)  = {beta_phys}')
print(f'Z^2              = {Z2}')
print(f'Confinement loss = {20 * beta_phys.imag / np.log(10):.4f} dB/m')
# Archived print outputs:
#    Scaled beta = (5.235764623018904+6.891942784408688e-08j)
#    Residual    = 5.34e-07
#    beta (physical)  = (5235764.623018904+0.06891942784408689j)
#    Z^2              = (0.002336593147489907-7.216918042895482e-07j)
#    Confinement loss = 0.5986 dB/m

# Assemble and plot the field:

F = B.fields_matplot(beta2, nu, outer)
B.plot2D_contour(F['Ez'], figsize=(10, 7))
plt.show()
B.plot1D(F['Ez_rad'],
         double_r=True,
         rlist=[400, 10000, 400],
         nu=nu,
         maxscale=True,
         linewidth=1.5,
         color='k',
         figsize=(6, 7))
plt.show()
