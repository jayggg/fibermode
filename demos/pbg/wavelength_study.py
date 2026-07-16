#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:36:47 2021.

@author: pv
"""

import matplotlib.pyplot as plt
from fiberamp.fiber.microstruct.pbg import PBG
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params
import numpy as np
import os


# Fiber and mode names on which to perform study. ##
fiber_name = 'lyr6cr2'   # Note: change import above to correspond.
mode_name = 'LP01'


# Center to find mode at first wavelength (wl_min). ################
starting_centers = {'LP01': .93,    # reset to .93 to start at 1.16e-6
                    'LP11': 1.93487063 - 8.699515e-08j}


# Folder setup.  Enter your path to pbg folder. ##################
pbg_home = '/home/pv/local/fiberamp/fiber/microstruct/pbg/wavelength_study/'
folder = pbg_home + 'outputs/'

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("%s is not a directory. Make this \
directory and begin again." % folder)

# Set wavelength range, polynomial degree and refinements. #############
wl_min, wl_max, n = 1.16e-6, 2.25e-6, 5

wavelengths = np.linspace(wl_min, wl_max, n)
p = 3
ref = 0

# Center, radius and span size for FEAST. ##############################
center = starting_centers[mode_name]
radius = .15
nspan = 2                      # Number of initial eigenvectors
npts = 4                       # Number of quadrature points


if __name__ == '__main__':

    print("Beginning wavelength loss study.\n")
    print('Using polynomial degree %i.\n' % p)
    print('Building fiber object and performing %i refinements.\n' % ref)

    A = PBG(params)
    for i in range(ref):
        A.refine()

    CLs = []
    zs = []
    ys = []
    for i, w in enumerate(wavelengths):

        print("Beginning iteration %i. \n\
Setting object's' wavelength to %e.\n" % (i+1, w))
        A.wavelength = w

        print('Finding mode.\n')
        try:
            z, y, _, beta, _, _ = A.leakymode(p, rad=radius, ctr=center,
                                              alpha=A.alpha, niterations=40,
                                              npts=npts, nspan=nspan,
                                              nrestarts=0)

            CL = 20 * beta.imag / np.log(10)
            CLs.append(CL)
            zs.append(z)
            ys.append(y)
            center = z.real   # reset center so it moves as wl increases
        except MemoryError("Unable to find modes due to MemoryError."):
            pass

    d = {'zs': zs, 'ys': ys, 'CLs': CLs, 'wavelengths': wavelengths}

    print('Saving data.\n')
    filename = 'wl_range_' + str(wl_min) + '_to_' + str(wl_max) + \
        '_length_' + str(n)
    filepath = os.path.abspath(folder + '/' + filename)
    np.savez(filepath, **d)

# %%
plt.plot(wavelengths, CLs)
plt.yscale("log")
