"""Confinement-loss-vs-wavelength study for a PBG fiber.


Tracks a single mode of `lyr6cr2` (see fiber_dicts/lyr6cr2.py) across
a range of wavelengths, by re-running the scalar leakymode() FEAST
search at each wavelength and using the previous wavelength's
converged Z as the next search center.

`starting_centers` gives, for each named mode, the Z value (in the
Z-plane, nondimensional) known to contain that mode at wl_min
(found by a preliminary search at a single wavelength, e.g. following
technique of starting from `A.sqrZfrom(.9998 * A.k * A.n_core)`
and narrowing down from there, or just trial and error as in
demos/pbg/pbg_mode.py). For convenience, we use 'LP01' for the
fundamental mode and 'LP11' is the next higher-order mode (even
thought this is not a step-index fiber).

Note on `radius`/`nspan`: these need to be tight enough that the
search contour doesn't sweep up a second, unrelated mode alongside
the one being tracked -- if it does, `leakymode` returns more than
one eigenvalue, and naively continuing the sweep from the whole
returned array (rather than the single mode of interest) will point
the next iteration's search at the wrong place, or crash outright
building an invalid multi-valued contour center. See the nearest-to-
previous-center selection below for how this is guarded against.
Even so, radius/nspan may need retuning if a mode crossing happens
within the swept wavelength range.
"""

import matplotlib.pyplot as plt
from fibermode import PBG
from fibermode.pbg.fiber_dicts.lyr6cr2 import params
import numpy as np
import os

# Fiber and mode names on which to perform study. ##
fiber_name = 'lyr6cr2'  # Note: change import above to correspond.
mode_name = 'LP01'

# Center to find mode at first wavelength (wl_min). ################
starting_centers = {
    'LP01': .93,  # reset to .93 to start at 1.16e-6
    'LP11': 1.93487063 - 8.699515e-08j
}

# Folder setup: outputs saved next to this script. ##################
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(folder, exist_ok=True)

# Set wavelength range, polynomial degree and refinements. #############
wl_min, wl_max, n = 1.16e-6, 2.25e-6, 5

wavelengths = np.linspace(wl_min, wl_max, n)
p = 2
ref = 0

# Center, radius and span size for FEAST. ##############################
center = starting_centers[mode_name]
radius = .15
nspan = 4  # Number of initial eigenvectors
npts = 2  # Number of quadrature points

if __name__ == '__main__':

    print("Beginning wavelength loss study.\n")
    print('Using polynomial degree %i.\n' % p)
    print('Building fiber object and performing %i refinements.\n' % ref)

    A = PBG(params)
    for i in range(ref):
        A.refine()

    CLs = []
    zs = []
    used_wavelengths = []  # wavelengths that actually produced a mode;
    # may end up shorter than `wavelengths` if the
    # sweep stops early or a MemoryError is hit
    for i, w in enumerate(wavelengths):

        print("Beginning iteration %i. \n\
Setting object's' wavelength to %e.\n" % (i + 1, w))
        A.wavelength = w

        print('Finding mode.\n')
        try:
            z, y, _, beta, _, extras = A.leakymode(p,
                                                   rad=radius,
                                                   ctr=center,
                                                   alpha=A.alpha,
                                                   niterations=200,
                                                   stop_tol=1e-6,
                                                   npts=npts,
                                                   nspan=nspan,
                                                   verbose=False,
                                                   nrestarts=0)

            # Stop the sweep rather than continue: an unconverged
            # result isn't a reliable Z, so re-centering the next
            # iteration's search on it would just compound the error
            # (and burn time chasing a contour that may not even
            # contain the right mode). Whatever converged so far is
            # still saved below.
            if not extras['converged']:
                print('*** FEAST did not converge at wavelength %e --'
                      ' stopping sweep.\n' % w)
                break

            # leakymode may return more than one eigenvalue if the
            # contour caught an extra mode alongside the one being
            # tracked. Pick whichever converged value is closest to
            # where we searched -- presumed to be the continuation of
            # the same physical mode from the previous wavelength --
            # rather than assuming there's exactly one result

            i_mode = int(np.argmin(np.abs(z - center)))

            CL = 20 * beta[i_mode].imag / np.log(10)
            CLs.append(CL)
            zs.append(z[i_mode])
            used_wavelengths.append(w)
            center = z[i_mode].real  # move center as wl increases
        except MemoryError:
            print("Unable to find modes due to MemoryError.")

    # Note: `y` (the mode field GridFunction) isn't saved here -- this
    # study only tracks Z/CL vs wavelength, and NGSolve GridFunctions
    # aren't plain numeric arrays, so np.savez can't box a list of
    # them into a single array without allow_pickle machinery that's
    # fragile across NGSolve versions. Keep the object returned by
    # leakymode() in memory (e.g. in a notebook) if you need the
    # field itself for a specific wavelength.
    d = {
        'zs': np.array(zs),
        'CLs': np.array(CLs),
        'wavelengths': np.array(used_wavelengths)
    }

    print('Saving data.\n')
    filename = 'wl_range_' + str(wl_min) + '_to_' + str(wl_max) + \
        '_length_' + str(n)
    filepath = os.path.abspath(folder + '/' + filename)
    np.savez(filepath, **d)

# %%
plt.plot(used_wavelengths, CLs)
plt.yscale("log")
