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
import os
import xarray as xr
import time

from shoot.eddies import EvolEddies
from shoot.track import track_eddies

xr.set_options(display_style="text")

# %%
# Read data
root_path = './data'
path = os.path.join(root_path, 'jan2024_ionian_sea_duacs.n')
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
eddies1 = EvolEddies.detect_eddies(
    ds1, window_center, window_fit, min_radius, ssh='adt', ellipse_error=ellipse_error
)
eddies2 = EvolEddies.detect_eddies(
    ds2, window_center, window_fit, min_radius, ssh='adt', ellipse_error=ellipse_error
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
tracks1.save(root_path + '/eddies_med_test1.nc')
tracks2.save(root_path + '/eddies_med_test2.nc')

# %% load and merge
ds1 = xr.open_dataset(root_path + '/eddies_med_test1.nc')
ds2 = xr.open_dataset(root_path + '/eddies_med_test2.nc')

# %% Merged step
eddies = EvolEddies.merge_ds([ds1, ds2])
eddies
print(len(eddies.eddies))

# %% Track on the complete time
tracks = track_eddies(eddies, 10)

# %% Save the full tracking

tracks.save(root_path + '/eddies_mest_test_merged.nc')

# %% Test the merged dataset
ds_merged = xr.open_dataset(root_path + '/eddies_mest_test_merged.nc')
