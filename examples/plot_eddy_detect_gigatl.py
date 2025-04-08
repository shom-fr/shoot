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
import os, sys
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import xoa.cf as xcf

sys.path.append('/home/shom/jbroust/Documents/CODE/SHOOT_LIB/')
from shoot.eddies import detect_eddies, Eddies
from shoot.plot import create_map, pcarr
from shoot.dyn import get_relvort
from shoot.contours import get_lnam_peaks

xr.set_options(display_style="text")

# %%
# Load croco-specific naming conventions to find dims, coords and variables
xcf.set_cf_specs("croco.cfg")

# %%
# Read data
root_path = '/local/tmp/jbroust/DATA/CROCO'
root_path = '/home/shom/sraynaud/Src/shoot/examples'
path = os.path.join(root_path, 'gigatl1-1000m.nc')
ds = xr.open_dataset(path).isel(time=0)
ds

# ds = ds.sel(lat_rho=slice(39,40.5), lon_rho=slice(-15,-13))
# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers
window_center = 25  # 25  # 50

# %%
# Window size in km to fit SSH and make other diagnostics like contours
window_fit = 120  # 120

# %%
# Minimal radius of an eddy to retain it
min_radius = 20

# %%
# Detection
# ~~~~~~~~~
import time

start = time.time()
# eddies, centers, lnam, ow, extrema= detect_eddies(ds.u, ds.v, window_center, window_fit=window_fit, min_radius=min_radius
#                                                   , ssh_method = 'streamline')
eddies = Eddies.detect_eddies(
    ds.u,
    ds.v,
    window_center,
    window_fit=window_fit,
    min_radius=min_radius,
    paral=False,
)
end = time.time()
print("Number of detected eddies %i in %.1f s" % (len(eddies.eddies), end - start))

# %%
# Plots
# -----
#
# We plot eddies with the relative vorticity as background.
#
fig, ax = create_map(ds.lon_rho, ds.lat_rho, figsize=(8, 5))
get_relvort(ds.u, ds.v).plot(
    x="lon_rho", y="lat_rho", cmap="cmo.curl", ax=ax, add_colorbar=False, transform=pcarr
)
nj = 3
plt.quiver(
    ds.lon_rho[::nj, ::nj].values,
    ds.lat_rho[::nj, ::nj].values,
    ds.u[::nj, ::nj].values,
    ds.v[::nj, ::nj].values,
    transform=pcarr,
)
for eddy in eddies.eddies:
    eddy.plot(transform=pcarr, lw=1)
plt.title("Relative vorticity")
plt.tight_layout()

# %% tests
# ax = create_map(ds.lon_rho, ds.lat_rho, figsize=(8, 5))
# lnam.plot(
#     x="lon_rho", y="lat_rho", ax=ax, transform=pcarr, add_colorbar=False, cmap=cmocean.cm.dense
# )
# ow.plot.contour(x="lon_rho", y="lat_rho", ax=ax, transform=pcarr, levels=[0], color='k')
# # plt.scatter(lnam.lon_rho[extrema[:,0]], lnam.lat_rho[extrema[:,1]], transform=pcarr, c='k')
# ii = extrema[:, 0]
# jj = extrema[:, 1]
# plt.scatter(lnam.lon_rho.values[jj, ii], lnam.lat_rho.values[jj, ii], transform=pcarr, c='k')

# %% test new function
# the main issue with this methods is the closed contour criteria as we have law resolution
# it should be tested with increased resolution
# minima, maxima, lines = get_lnam_peaks(lnam, K=0.9)
# ax = create_map(ds.lon_rho, ds.lat_rho, figsize=(8, 5))
# lnam.plot(
#     x="lon_rho", y="lat_rho", ax=ax, transform=pcarr, add_colorbar=False, cmap=cmocean.cm.dense
# )
# abs(lnam).plot.contour(x="lon_rho", y="lat_rho", ax=ax, transform=pcarr, levels=[0.9], color='k')
# ow.plot.contour(x="lon_rho", y="lat_rho", ax=ax, transform=pcarr, levels=[0.0], color='k')
# plt.scatter(
#     lnam.lon_rho.values[minima[:, 1], minima[:, 0]],
#     lnam.lat_rho.values[minima[:, 1], minima[:, 0]],
#     s=10,
#     transform=pcarr,
#     c='r',
# )
# plt.scatter(
#     lnam.lon_rho.values[maxima[:, 1], maxima[:, 0]],
#     lnam.lat_rho.values[maxima[:, 1], maxima[:, 0]],
#     s=10,
#     transform=pcarr,
#     c='b',
# )
# for l in lines:
#     plt.plot(l[0], l[1], transform=pcarr, c='g')
