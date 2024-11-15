#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect eddies from satellite sea level
======================================
"""

# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
from shoot.eddies import detect_eddies
from shoot.plot import create_map, pcarr

xr.set_options(display_style="text")

# %%
# Read data
ds = xr.open_dataset("ssh-sat-med.nc").isel(time=0)
ds

# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers
window_center = 50

# %%
# Window size in km to fit SSH and make other diagnostics like contours
window_fit = 120

# %%
# Minimal radius of an eddy to retain it
min_radius = 20

# %%
# Detection
# ~~~~~~~~~
eddies = detect_eddies(
    ds.ugos, ds.vgos, window_center, window_fit=window_fit, ssh=ds.adt, min_radius=min_radius
)
print("Number of detected eddies", len(eddies))

# %%
# Plots
# -----
#
ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
ds.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap="cmo.gray")
for eddy in eddies:
    eddy.plot(transform=pcarr, lw=1)
plt.title(ds.adt.long_name)
plt.tight_layout()
