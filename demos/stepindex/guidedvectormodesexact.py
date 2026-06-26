"""
Run semi-analytical root finder for guided vector (Maxwell) modes of
a radially symmetric step-index fiber.

(Ensure matplotlib is installed and use "python thisfilename.py" to
visualize vector modes.)
"""

import warnings
import logging
from fibermode import StepIndexExact

logging.getLogger('matplotlib.axes._base').setLevel(logging.ERROR)

f = StepIndexExact('Nufern_Yb')

Ys = []  # compute list of roots for each m in Ys[m]:

with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)

    m = 0    # special handling of m=0 case to determine if TE or TM:
    Y0TE = f.vec_propagation_constants(0, m0name='TE')
    Y0TM = f.vec_propagation_constants(0, m0name='TM')
    Ys += [Y0TE + Y0TM]

    for m in range(1, 6):   # try a few higher m-values too:
        Ys += [f.vec_propagation_constants(m)]

    #   Ys = a list of tuples (ys, a, b) where ys is a list of roots found
    #   in the interval [a, b]. These nondimensional Y-values in ys are
    #   related to the physical propagation constant β by
    #   Y = a * sqrt(β² - k²n₀²).

print('\n\n' + 'SUMMARY OF EXACT ROOTS FOUND: ' + '='*40)
print('          Y-values        Z-squared values' +  # Note that Y = iZ
      '         beta-values\n' + 70*'-')
if len(Y0TE):
    print('  CASE m =', 0, ' TE')
    for yab in Y0TE:
        for y in yab[0]:
            print('  %20.12f %20.12f  %24.12f'
                  % (y, -y**2, f.ZtoBeta(1j*y).real))
if len(Y0TM):
    print('  CASE m =', 0, ' TM')
    for yab in Y0TM:
        for y in yab[0]:
            print('  %20.12f %20.12f  %24.12f'
                  % (y, -y**2, f.ZtoBeta(1j*y).real))
for m in range(1, 6):
    if len(Ys[m]):
        print('  CASE m =', m, ' HYBRID (EACH OF MULTIPLICITY 2)')
        for yab in Ys[m]:
            for y in yab[0]:
                print('  %20.12f %20.12f  %24.12f'
                      % (y, -y**2, f.ZtoBeta(1j*y).real))
print('='*70)

# Two visualization examples:
#     Can visualize the exact mode profile corresponding to any one
#     of the above computed prop.constant
print('\nEXAMPLES OF COMPUTED MODE VISUALIZATIONS:\n\n')
f.visualize_vec_Emode(2, Ys[2][0][0][0], real=True)
f.visualize_vec_Emode(0, Ys[0][1][0][0], m0name='TM')


# Check the above roots using slower cxroots (may get more accurate roots)

reconfirm = True   # only do it if this is turned on

if reconfirm:
    print('RECONFIRMING BY CXROOTS:')
    roots = []
    for m in range(6):
        roots += f.vec_confirm_roots(m, Ys[m])
    print('Summary:')
    for r in roots:
        print(r)
