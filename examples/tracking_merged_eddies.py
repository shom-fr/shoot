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
import multiprocessing as mp
import cmocean
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import numpy as np


sys.path.append('/home/shom/jbroust/Documents/CODE/SHOOT_LIB/')
from shoot.eddies import Eddies, EvolEddies
from shoot.track import track_eddies, update_tracks, Tracks
from shoot.plot import create_map, pcarr
from shoot.contours import get_lnam_peaks
from shoot.rossby import Rossby
from shoot.grid import get_dx_dy


if __name__ == "__main__":
    mp.set_start_method("spawn")
    xr.set_options(display_style="text")

    # %%
    # Read data
    root_path = '/local/tmp/jbroust/DATA/DUACS_MED'
    path = os.path.join(root_path, '2019.nc')  # 16 eme degre
    path = os.path.join(root_path, '2019_global.nc')  # 8eme degre
    ds1 = xr.open_dataset(path).isel(time=slice(0, 10))
    ds2 = xr.open_dataset(path).isel(time=slice(10, 20))

    # %% D
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
    # Detection
    # ~~~~~~~~~
    import time

    start = time.time()
    # eddies = EvolEddies.detect_eddies(
    #     ds.sel(time=slice('2019-01-01', '2019-01-30')), window_center, window_fit, min_radius, ssh='adt'
    # )
    eddies1 = EvolEddies.detect_eddies(ds1, window_center, window_fit, min_radius, ssh='adt')
    eddies2 = EvolEddies.detect_eddies(ds2, window_center, window_fit, min_radius, ssh='adt')
    end = time.time()
    time_detect = end - start
    print("nb days %i in %.1f min" % (len(eddies1.eddies), time_detect / 60))

    # %%
    # Tracking
    # ~~~~~ $

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
    eddies = EvolEddies.merge_ds(ds1, ds2)
    eddies
    print(len(eddies.eddies))

    # %% Track on the complete time
    tracks = track_eddies(eddies, 10)

    # %% Save the full tracking

    tracks.save(root_path + '/eddies_mest_test_merged.nc')

    # %% Test the merged dataset

    ds_merged = xr.open_dataset(root_path + '/eddies_mest_test_merged.nc')
