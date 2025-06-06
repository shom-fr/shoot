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
import multiprocessing as mp
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import xoa.cf as xcf

sys.path.append('/home/shom/jbroust/Documents/CODE/SHOOT_LIB/')
# from shoot.eddies import Eddies
from shoot.eddies_debug import find_eddy_centers, Eddies
from shoot.plot import create_map, pcarr
from shoot.dyn import get_relvort
from shoot.contours import get_lnam_peaks


# if __name__ == "__main__": ## A faire pour être très propre avce de gros jeux de données
#     mp.set_start_method("spawn")

xr.set_options(display_style="text")

# %%
# Load croco-specific naming conventions to find dims, coords and variables
xcf.set_cf_specs("croco.cfg")

# %%
# Read data
root_path = '/local/tmp/jbroust/DATA/CROCO/GIGATL'
# root_path = '/home/shom/sraynaud/Src/shoot/examples'
# path = os.path.join(root_path, 'gigatl1-1000m.nc')
path = os.path.join(root_path, 'gigatl1_1h_tides_iberia2_daily_2008-11-01_at_z.nc')
ds = xr.open_dataset(path)
ds = ds.isel(depth_rho=2).squeeze()

ds = ds.sel(x_rho=slice(600, 1100), y_rho=slice(900, 1200))
# ds = ds.sel(x_rho=slice(800, 1100), y_rho=slice(1050, 1200))

# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers
window_center = 50  # 25  # 50

# %%
# Window size in km to fit SSH and make other diagnostics like contours
window_fit = 120  # 120

# %%
# Minimal radius of an eddy to retain it
min_radius = 15

# %%
# Detection centres
# ~~~~~~~~~

centers, lnam, ow, extrema = find_eddy_centers(ds.u, ds.v, window_center, paral=False)

# %%
# Detection eddies


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
    ellipse_error=0.01,
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
nj = 10
plt.quiver(
    ds.lon_rho[::nj, ::nj].values,
    ds.lat_rho[::nj, ::nj].values,
    ds.u[::nj, ::nj].values,
    ds.v[::nj, ::nj].values,
    transform=pcarr,
)

plt.scatter(centers.lon, centers.lat, c='k', transform=pcarr)
for eddy in eddies.eddies:
    eddy.plot(transform=pcarr, lw=1)
# plt.title("Relative vorticity at 1000m")
plt.title("Ellipse Error 1%")
plt.tight_layout()
