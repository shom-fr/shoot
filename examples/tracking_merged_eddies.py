#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect eddies from different files and merge tracking
=====================================================
"""

# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import os
import xarray as xr
import time

from shoot.eddies.eddies2d import EvolEddies2D
from shoot.eddies.track import track_eddies
from shoot.samples import get_sample_file

xr.set_options(display_style="text")

# %%
# Read data
root_path = "OBS/SATELLITE/jan2024_ionian_sea_duacs.nc"
path = get_sample_file(root_path)
# partition into 2 dataset continuous in time
ds1 = xr.open_dataset(path).isel(time=slice(0, 10))
ds2 = xr.open_dataset(path).isel(time=slice(10, 20))

# %%
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50  # 20  # 100#50

# %%
# Window size in km to fit SSH and make other diagnostics like contours : 10Rd suggested
window_fit = 120  # 100  # 120

# %%
# Minimal radius of an eddy to retain it
min_radius = 20

# %%
# Ellipse error
ellipse_error = 0.05

# %%
# Detection
# ~~~~~~~~~

start = time.time()
eddies1 = EvolEddies2D.detect_eddies(
    ds1,
    window_center,
    window_fit,
    min_radius,
    ssh="adt",
    ellipse_error=ellipse_error,
)
eddies2 = EvolEddies2D.detect_eddies(
    ds2,
    window_center,
    window_fit,
    min_radius,
    ssh="adt",
    ellipse_error=ellipse_error,
)
end = time.time()
time_detect = end - start
print("nb days %i in %.1f min" % (len(eddies1.eddies), time_detect / 60))

# %%
# Tracking
# ~~~~~~~~

nbackward = 10  # number of admitted time step without detection

start = time.time()
tracks1 = track_eddies(eddies1, nbackward)  # 10*dt
tracks2 = track_eddies(eddies2, nbackward)  # 10*dt
end = time.time()
print("duree totale du calcul sur 1 mois : %.1f s" % (end - start))

# %% Save partial tracking
tracks1.to_netcdf(root_path + "/eddies_med_test1.nc")
tracks2.to_netcdf(root_path + "/eddies_med_test2.nc")

# %% load and merge
ds1 = xr.open_dataset(root_path + "/eddies_med_test1.nc")
ds2 = xr.open_dataset(root_path + "/eddies_med_test2.nc")

# %% Merged step
eddies = EvolEddies2D.merge_ds([ds1, ds2])
eddies
print(len(eddies.eddies))

# %% Track on the complete time
tracks = track_eddies(eddies, 10)

# %% Save the full tracking

tracks.to_netcdf(root_path + "/eddies_mest_test_merged.nc")

# %% Test the merged dataset
ds_merged = xr.open_dataset(root_path + "/eddies_mest_test_merged.nc")
