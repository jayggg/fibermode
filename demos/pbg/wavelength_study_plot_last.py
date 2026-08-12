"""Plot the confinement-loss-vs-wavelength curve saved by
wavelength_study.py (see that file for how the data was produced).

Loads the most recently saved .npz in demos/pbg/outputs/ and plots
CL [dB/m] vs wavelength [m] on a log scale.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
files = sorted(glob.glob(os.path.join(folder, 'wl_range_*.npz')),
               key=os.path.getmtime)
if not files:
    raise FileNotFoundError(
        f"No saved results found in {folder}. Run wavelength_study.py first.")
filepath = files[-1]  # most recently saved

d = np.load(filepath)
wavelengths, CLs = d['wavelengths'], d['CLs']

plt.plot(wavelengths, CLs, 'o-')
plt.xlabel('wavelength [m]')
plt.ylabel('CL [dB/m]')
plt.yscale('log')
plt.title(os.path.basename(filepath))
plt.show()
