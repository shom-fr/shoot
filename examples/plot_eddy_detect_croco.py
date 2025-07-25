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
import os
import cmocean as cm
import time
import matplotlib.pyplot as plt
import xarray as xr
import xoa.cf as xcf

from shoot.eddies.eddies2d import Eddies2D
from shoot.plot import create_map, pcarr
from shoot.dyn import get_relvort

xr.set_options(display_style="text")

# %%
# Load croco-specific naming conventions to find dims, coords and variables
xcf.set_cf_specs("croco.cfg")

# %%
# Read data

# Mind that the dataarray should be 2D : single time and single depth
root_path = "../data"
path = os.path.join(root_path, "gigatl1-1000m.nc")
ds = xr.open_dataset(path).isel(time=0)


# %%
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
min_radius = 10

# %%
# Ellipse error
# preconised 1% for deep field, 5% to 10% to surface field

ellipse_error = 0.05
# %%
# Detection
# ---------

start = time.time()
eddies = Eddies2D.detect_eddies(
    ds.u,
    ds.v,
    window_center,
    window_fit=window_fit,
    min_radius=min_radius,
    paral=False,
    ellipse_error=ellipse_error,
)
end = time.time()
print(
    "Number of detected eddies %i in %.1f s"
    % (len(eddies.eddies), end - start)
)

# %%
# Plots
# -----
#
# We plot eddies with the relative vorticity as background.
#

fig, ax = create_map(ds.lon_rho, ds.lat_rho, figsize=(8, 5))
get_relvort(ds.u, ds.v).plot(
    x="lon_rho",
    y="lat_rho",
    cmap=cm.cm.curl,
    ax=ax,
    add_colorbar=False,
    transform=pcarr,
)
nj = 5
plt.quiver(
    ds.lon_rho[::nj, ::nj].values,
    ds.lat_rho[::nj, ::nj].values,
    ds.u[::nj, ::nj].values,
    ds.v[::nj, ::nj].values,
    transform=pcarr,
)

for eddy in eddies.eddies:
    eddy.plot(transform=pcarr, lw=1)
plt.title("Relative vorticity at 1000m")
plt.tight_layout()
