#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect CROCO-GIGATL1 eddies at 1000m
====================================

In this example, eddies are detected from CROCO model currents interpolated to 1000 m and collocated at RHO points.

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
from shoot.dyn import get_relvort

xr.set_options(display_style="text")

# %%
# Read data
ds = xr.open_dataset("gigatl1-1000m.nc").isel(time=0)
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
eddies = detect_eddies(ds.u, ds.v, window_center, window_fit=window_fit, min_radius=min_radius)
print("Number of detected eddies", len(eddies))

# %%
# Plots
# -----
#
# We plot eddies with the relative vorticity as background.
#
ax = create_map(ds.lon_rho, ds.lat_rho, figsize=(8, 5))
get_relvort(ds.u, ds.v).plot(
    x="lon_rho", y="lat_rho", cmap="cmo.curl", ax=ax, add_colorbar=False, transform=pcarr
)
for eddy in eddies:
    eddy.plot(transform=pcarr, lw=1)
plt.title("Relative vorticity")
plt.tight_layout()
