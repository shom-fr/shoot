#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect eddies from satellite sea level
======================================
"""

# %%
# Initialisations
# -----------------

# Import needed stuff.
import os
import time
import matplotlib.pyplot as plt
import xarray as xr

from shoot.eddies import Eddies
from shoot.plot import create_map, pcarr

xr.set_options(display_style="text")

# %%
# Read data
# ---------

root_path = '../data'
path = os.path.join(root_path, 'jan2024_ionian_sea_duacs.nc')
# select one specific date for this example
ds = xr.open_dataset(path).isel(time=0)

# %%
# Detect eddies
# -------------

# Parameters
# ~~~~~~~~~~


# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50

# %%
# Window size in km to fit SSH and make other diagnostics like contours : 10Rd suggested
window_fit = 120

# %%
# Minimal radius of an eddy to retain it
# Mind that the radius is defined as the radius of the maximum speed contour
# Preconised around Rossby radius of deformation
min_radius = 10

# %%
# Ellipse error
# This is the percentage error from an ellipse.
# For surface field it is conseil to let 5% to 10%
ellipse_error = 0.05  # percentage error from an ellipse

# %%
# Detection
# ~~~~~~~~~
# performed through Eddies class
# parallelisation is possible but should be perfomed with caution (refer to the docs)
####

start = time.time()
eddies = Eddies.detect_eddies(
    ds.ugos,
    ds.vgos,
    window_center,
    window_fit=window_fit,
    ssh=ds.adt,
    min_radius=min_radius,
    paral=False,
    ellipse_error=ellipse_error,
)
end = time.time()
print('it takes %.1f s' % (end - start))

# %%
# Plots
# -----
#
fig, ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
ds.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap="Spectral_r", alpha=0.6)
plt.quiver(ds.longitude.values, ds.latitude.values, ds.ugos.values, ds.vgos.values, transform=pcarr)
for eddy in eddies.eddies:
    # for eddy in eddies_ssh:
    eddy.plot(transform=pcarr, lw=1)
plt.title('w_center %i km, w_fit %ikm min_rad %ikm' % (window_center, window_fit, min_radius))
plt.tight_layout()
