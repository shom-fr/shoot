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
import matplotlib.animation as animation
import xarray as xr
import numpy as np
import pandas as pd

from shoot.profiles.profiles import Profiles, Profile

from shoot.eddies.eddies2d import Eddies2D, EvolEddies2D
from shoot.eddies.track import track_eddies, update_tracks, Tracks
from shoot.plot import create_map, pcarr
from shoot.contours import get_lnam_peaks
from shoot.rossby import Rossby
from shoot.grid import get_dx_dy
from shoot.samples import get_sample_file


# %%
# Read data
root_path = "OBS/SATELLITE/jan2024_ionian_sea_duacs.nc"
path = get_sample_file(root_path)
ds = xr.open_dataset(path)

# %%
## %% Download profiles

data_path = "/local/tmp/jbroust/DATA/CORA"

prf = Profiles.from_ds(
    ds,
    root_path=data_path,
    data_types=["PF"],
    download=False,
)
print(prf.ds.time)

# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50  # 50  # 100#50 C'est assez sensible à ce paramètre

# Window size in km to fit SSH and make other diagnostics like contours : 10Rd suggested
window_fit = 120  # 100  # 120

# Minimal radius of an eddy to retain it
min_radius = 20

# Ellipse error

ellipse_error = 0.05

# %%
# Detection
# ~~~~~~~~~
import time

start = time.time()
eddies = EvolEddies2D.detect_eddies(
    ds,
    window_center,
    window_fit,
    min_radius,
    ssh="adt",
    u="ugos",
    v="vgos",
    paral=True,
    ellipse_error=ellipse_error,
)
end = time.time()
print("Temps de calcul pour %i pas de temps : %.2f s" % (len(ds.time), end - start))
# %%
# Tracking
# ~~~~~ $

nbackward = 10  # number of admitted time step without detection

tracks = track_eddies(eddies, nbackward)  # 10*dt
tracked_eddies = tracks.track_eddies

# %%
# Associate to floats
# ~~~~~~~~~~~~~~~~~~~

prf.associate(eddies)
nb_inside = 0
for eddy in eddies.eddies:
    for e in eddy.eddies:
        if e.p_id:
            nb_inside += 1
print(nb_inside)

nb_prf_inside = 0
for i in prf.ds.profil:
    if prf.ds.isel(profil=i).eddy_pos > 0:
        # print(prf.ds.time.isel(profil=i), prf.ds.isel(profil=i).eddy_pos)
        nb_prf_inside += 1
print(nb_prf_inside)


# %%
# Plots
# -----
#
date = "2024-01-15"
n = 14
dss = ds.sel(time=date)

ind_day = np.where(
    (prf.ds.time.values >= np.datetime64(date))
    & (prf.ds.time.values < np.datetime64(date) + np.timedelta64(1, "D"))
)[0]

day_profil = prf.ds.isel(profil=ind_day)

lat_min, lat_max = ds.latitude.min(), ds.latitude.max()
lon_min, lon_max = ds.longitude.min(), ds.longitude.max()

ind_geo = np.where(
    (day_profil.lat <= lat_max)
    & (day_profil.lat >= lat_min)
    & (day_profil.lon <= lon_max)
    & (day_profil.lon >= lon_min)
)[0]

day_profil = day_profil.isel(profil=ind_geo)


dss = ds.isel(time=n)

fig, ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))

dss.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap="cmo.dense")

# plt.quiver(
#     dss.longitude.values,
#     dss.latitude.values,
#     dss.ugos.values,
#     dss.vgos.values,
#     transform=pcarr,
# )


for eddy in eddies.eddies[n].eddies:
    color = "k" if eddy.p_id else "w"
    eddy.plot(transform=pcarr, lw=1)
    plt.text(eddy.glon, eddy.glat, eddy.track_id, c=color, transform=pcarr)
    # track = tracked_eddies[eddy.track_id]
    # lon, lat = [], []
    # for e in track.eddies:
    #     lon.append(e.glon)
    #     lat.append(e.glat)
    # cm = plt.plot(lon, lat, transform=pcarr, c="gray", linewidth=2)

colors = ["k" if i == -1 else "green" for i in day_profil.eddy_pos]
plt.scatter(
    day_profil.lon,
    day_profil.lat,
    marker="*",
    c=colors,
    s=20,
    transform=pcarr,
)
plt.scatter([], [], c="k", marker="*", label="outside profiles")
plt.scatter([], [], c="green", marker="*", label="inside profiles")
plt.legend()
plt.title(dss.time.dt.strftime("%Y-%m-%d").values)
plt.tight_layout()

# %%
# Plot eddy profile

eddy_id = 9  # Pelops id in this case

ind_id = np.where(prf.ds.eddy_pos == eddy_id)[0]
ds_id = prf.ds.isel(profil=ind_id)
ds_id

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(
    ds_id.temp.T,
    -ds_id.depth.broadcast_like(ds_id.temp).T,
    label=[t.dt.strftime("%y-%m-%d").values for t in ds_id.time],
)
plt.legend()
plt.subplot(122)
plt.plot(ds_id.sal.T, -ds_id.depth.broadcast_like(ds_id.sal).T)
