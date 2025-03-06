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
import matplotlib.dates as mdates
import matplotlib.animation as animation
import xarray as xr
import numpy as np


from shoot.eddies import detect_eddies, Eddies, EvolEddies
from shoot.track import track_eddies, update_tracks, Tracks
from shoot.plot import create_map, pcarr
from shoot.contours import get_lnam_peaks
from shoot.rossby import Rossby
from shoot.grid import get_dx_dy

xr.set_options(display_style="text")

# %%
# Read data
root_path = './data'
path = os.path.join(root_path, 'jan2024_ionian_sea_duacs.nc')
ds = xr.open_dataset(path)


# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50  # 50  # 100#50 C'est assez sensible à ce paramètre

# Rossby from Chelton atlas
# rHere need to change the path in the rossby.py
# Rosby Chelton Atlas can be download on the AMEDA github page
# https://github.com/briaclevu/AMEDA/blob/master/Rossby_radius.tar
rossby_param = False
if rossby_param:
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
    window_center = max(2 * ro / tau, 3 * ro)  # Where tau is the grid parameter


print(window_center)
# %%
# Window size in km to fit SSH and make other diagnostics like contours : 10Rd suggested
window_fit = 120  # 100  # 120

# with Rossby
if rossby_param:
    window_fit = 10 * ro

print(window_fit)
# %%
# Minimal radius of an eddy to retain it
min_radius = 20

# %%
# Detection
# ~~~~~~~~~

eddies = EvolEddies.detect_eddies(ds, window_center, window_fit, min_radius, ssh='adt')

# %%
# Tracking
# ~~~~~ $

nbackward = 10  # number of admitted time step without detection


tracks = track_eddies(eddies, nbackward)  # 10*dt
tracked_eddies = tracks.track_eddies


# sauvegarde du tracking complet
tracks.save(os.path.join(root_path, 'track_ionian_sea_duacs_jan2024.nc'))


# %%
# Plots
# -----
#
fig, ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
n = 30  # 297
dss = ds.isel(time=n)
dss.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap="nipy_spectral")

plt.quiver(
    dss.longitude.values, dss.latitude.values, dss.ugos.values, dss.vgos.values, transform=pcarr
)
for eddy in eddies.eddies[n].eddies:
    eddy.plot(transform=pcarr, lw=1)
    plt.text(eddy.glon, eddy.glat, eddy.track_id, c='w', transform=pcarr)
    track = tracked_eddies[eddy.track_id]
    lon, lat = [], []
    for e in track.eddies:
        lon.append(e.glon)
        lat.append(e.glat)
    cm = plt.plot(lon, lat, transform=pcarr, c='gray', linewidth=2)

plt.title(ds.adt.long_name)
plt.tight_layout()


# %% Animation

img = []  # some array of images
frames = []  # for storing the generated images

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)


def animate(i):
    plt.cla()
    dss = ds.isel(time=i)
    dss.adt.plot(
        x="longitude",
        y="latitude",
        cmap="nipy_spectral",
        ax=ax,
        add_colorbar=False,
        # transform=pcarr,
        levels=np.linspace(-0.7, 0.6, 50),
    )

    nj = 2
    plt.quiver(
        dss.longitude[::nj].values,
        dss.latitude[::nj].values,
        dss.ugos[::nj, ::nj].values,
        dss.vgos[::nj, ::nj].values,
        # transform=pcarr,
    )

    for eddy in eddies.eddies[i].eddies:
        # eddy.plot(transform=pcarr, lw=1)
        eddy.plot(lw=1)
        plt.text(eddy.glon, eddy.glat, eddy.track_id, c='w')  # , transform=pcarr)
    plt.title(str(dss.time.values)[:10])
    plt.tight_layout()


ani = animation.FuncAnimation(fig, animate, frames=np.arange(len(ds.time)), interval=1000)

# ani.save('/local/tmp/jbroust/OUTPUTS/SHOOT/AMEDA_SHOOT/ionian_sea/tracking_ionian_sea_jan2024.gif')
plt.show()
