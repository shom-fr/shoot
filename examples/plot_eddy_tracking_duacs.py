#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect and Track eddies from satellite sea level
================================================
"""

# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import os
import time
import cmocean as cm
import matplotlib.pyplot as plt
import xarray as xr

from shoot.eddies import EvolEddies
from shoot.track import track_eddies
from shoot.plot import create_map, pcarr


# %%
# Read data
root_path = '../data'
path = os.path.join(root_path, 'jan2024_ionian_sea_duacs.nc')
ds = xr.open_dataset(path)

# %%
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers : Lb
window_center = 50  # 50  # 100#50 C'est assez sensible à ce paramètre


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
eddies = EvolEddies.detect_eddies(
    ds,
    window_center,
    window_fit,
    min_radius,
    ssh='adt',
    u='ugos',
    v='vgos',
    ellipse_error=ellipse_error,
)
end = time.time()
print("Temps de calcul pour %i pas de temps : %.2f s" % (len(ds.time), end - start))
# %%
# Tracking
# ~~~~~~~~

nbackward = 10  # number of admitted time step without detection

tracks = track_eddies(eddies, nbackward)  # 10*dt
tracked_eddies = tracks.track_eddies

# sauvegarde du tracking complet

# tracks.save(os.path.join(root_path, 'track_ionian_sea_duacs_jan2024.nc'))

# %%
# Plots
# -----
#
fig, ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
n = 30  # 297
dss = ds.isel(time=n)
dss.adt.plot(ax=ax, transform=pcarr, add_colorbar=False, cmap=cm.cm.dense)

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
