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
import os, sys
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

sys.path.append('/home/shom/jbroust/Documents/CODE/SHOOT_LIB/')
from shoot.eddies import detect_eddies, Eddies
from shoot.plot import create_map, pcarr
from shoot.contours import get_lnam_peaks
from shoot.rossby import Rossby
from shoot.grid import get_dx_dy

xr.set_options(display_style="text")

# %%
# Read data
root_path = '/local/tmp/jbroust/DATA/DUACS_MED'
path = os.path.join(root_path, '2023_jan.nc')
path = os.path.join(root_path, '2019_global.nc')  # 8eme degre
ds = xr.open_dataset(path).isel(time=15)
ds

## zoom sur Alboran Sea
# ds = ds.sel(latitude=slice(37, 39.5), longitude=slice(5.5, 8)) #Tourbilon lybien
# ds = ds.sel(latitude=slice(42.5, 44), longitude=slice(7, 10))  # Nord ouest med
# ds = ds.sel(latitude=slice(40.5, 44), longitude=slice(6, 12))
# ds = ds.sel(latitude=slice(42.5, 44), longitude=slice(7, 10))
# ds = ds.sel(latitude=slice(37, 39), longitude=slice(2, 5))
# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50  # 50  # 100#50 C'est assez sensible à ce paramètre

## Rossby from Chelton atlas
# ro = Rossby().get_ro(ds.longitude.mean().values, ds.latitude.mean().values)
ro = (
    Rossby().get_ro_avg(
        ds.longitude.min().values,
        ds.longitude.max().values,
        ds.latitude.min().values,
        ds.latitude.max().values,
    )
    / 1e3
)
print('Rd %.1f (km)' % (ro))

dxdy = get_dx_dy(ds.ugos)
tau = ro / (np.max(dxdy) / 1e3)
# window_center = max(2 * ro / tau, 3 * ro)  # Where tau is the grid parameter

print(window_center)
# %%
# Window size in km to fit SSH and make other diagnostics like contours : 10Rd suggested
window_fit = 120  # 120

##
window_fit = 10 * ro  # Optimal value used in Ameda
window_fit

# %%
# Minimal radius of an eddy to retain it
min_radius = 20

# %%
# Detection via la classe Eddies
####
import time

start = time.time()
eddies = Eddies.detect_eddies(
    ds.ugos,
    ds.vgos,
    window_center,
    window_fit=window_fit,
    ssh=ds.adt,
    min_radius=min_radius,
    paral=True,
)
end = time.time()
print('it takes %.1f s' % (end - start))

# eddies.save_data(root_path + '/test.csv')
# eddies.save_eddies(root_path + '/test.json')
# %% reconstruction
path = root_path + '/test.csv'
import pandas as pd

data = pd.read_csv(path)
eddies_r = Eddies.reconstruct(data)


# %%
# Plots
# -----
#
fig, ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
ds.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap="gray")
plt.quiver(ds.longitude.values, ds.latitude.values, ds.ugos.values, ds.vgos.values, transform=pcarr)
for eddy in eddies.eddies:
    # for eddy in eddies_ssh:
    eddy.plot(transform=pcarr, lw=1)
plt.title('w_center %i km, w_fit %ikm min_rad %ikm' % (window_center, window_fit, min_radius))
plt.tight_layout()
