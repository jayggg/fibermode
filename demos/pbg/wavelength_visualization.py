#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:12:54 2021.

@author: pv
"""
from fiberamp.fiber.microstruct.pbg.fiber_dicts.lyr6cr2 import params
from fiberamp.fiber.microstruct.pbg import PBG
import os
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

# %%


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=40)  # fontsize of the figure title

# %%
# Fiber and mode names on which to perform study. ##
fiber_name = 'lyr6cr2'   # Note: change import above to correspond.
mode_name = 'LP01'

# Folder setup.  Enter your path to pbg folder. ##################
pbg_home = '/home/pv/local/fiberamp/fiber/microstruct/pbg/'
folder = pbg_home + '/outputs/' + fiber_name + \
    '/' + mode_name + '/' + 'wavelengths'   # Make this directory

if not os.path.isdir(os.path.relpath(folder)):
    raise FileNotFoundError("%s is not a directory. Make this \
directory and begin again." % folder)

# Set wavelength range, polynomial degree and refinements. #############
wl_min, wl_max, n = 1.16e-6, 2.25e-6, 50

filename = 'wl_range_' + str(wl_min) + '_to_' + str(wl_max) + \
    '_length_' + str(n)
filepath = os.path.abspath(folder + '/' + filename)
d = np.load(filepath + '.npz', allow_pickle=True)

CLs = d['CLs']
wavelengths = d['wavelengths']
ys = d['ys']
# plt.plot(wavelengths, CLs, 'b-', linewidth=2.5)
# plt.title('Confinment Loss vs. Wavelength.\n')

# # plt.xlabel('$\lambda$ (m)')
# plt.ylabel("CL (dB/m)")
# plt.yscale("log")


# %%

A = PBG(params)


# %%

x = np.linspace(-7, 7, 100)
y = np.linspace(-7, 7, 100)

X, Y = np.meshgrid(x, y)


# %%

Z_all = np.zeros((len(ys), 100, 100), dtype=complex)
for k, y in enumerate(ys):
    func = y[0]
    for i in range(len(X)):
        for j in range(len(X)):
            Z_all[k, i, j] = func(A.mesh(X[i, j], Y[i, j]))

# %%
for q in range(len(ys)):
    plt.figure(q)
    plt.contourf(X, Y, Z_all[q].real, 100)
    plt.axis('square')
    plt.colorbar()
# %%
fig = plt.figure()
camera = Camera(fig)
plt.axis('square')
for q in range(len(ys)):
    plt.contourf(X, Y, Z_all[q].real, 100)
    camera.snap()
animation = camera.animate()

# %%

animation.save('mode_animation.mp4')
