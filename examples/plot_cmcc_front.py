#!/usr/bin/env python3
"""
Detect fronts from sea surface temperature
==========================================

Fronts are detected in the sea surface temperature of the CMCC Mediterranean
analysis with the four methods of :func:`~shoot.fronts.fronts2d.detect_fronts`.
"""

# %%
# Initialisations
# ---------------
#
# Import needed stuff.
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from shoot.core.fronts import boa
from shoot.fronts.fronts2d import detect_fronts
from shoot.samples import get_sample_file

xr.set_options(display_style="text")

# %%
# Read data
# ---------
#
# Surface temperature on the 28th of October 2025 between Tunisia, Sicily and Libya.
path = get_sample_file("MODELS/CMEMS-MED/cmcc_med_surface_20251028-30.nc")
sst = xr.open_dataset(path).thetao.isel(time=0)
sst

# %%
# Detect fronts
# -------------
#
# The methods are:
#
# - ``"cca"``: Cayula and Cornillon (1992) histogram and cohesion criteria
#   on overlapping windows;
# - ``"cca_sliding"``: the same criteria on sliding windows centered every
#   ``step`` grid points;
# - ``"boa"``: Belkin and O'Reilly (2009) gradient magnitude, after a contextual
#   median filter, greater than a ``threshold``;
# - ``"canny"``: Canny edge detector.
#
# Parameters of the methods are passed as keywords, like the minimal difference
# of temperature between the two sides of a Cayula-Cornillon front.
params = {
    "cca": dict(min_pop_mean_diff=0.4),
    "cca_sliding": dict(min_pop_mean_diff=0.4, step=10),
    "boa": dict(threshold=0.3),
    "canny": dict(sigma=5),
}
fronts = {}
for method, kwargs in params.items():
    start = time.time()
    fronts[method] = detect_fronts(sst, method=method, **kwargs)
    print(f"{method}: {int(fronts[method].sum())} front points in {time.time() - start:.2f} s")

# %%
# Plot the fronts
# ---------------
#
# Front points are drawn in black over the temperature.
aspect = 1 / np.cos(np.radians(sst.latitude.mean().item()))
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
for ax, (method, mask) in zip(axs.flat, fronts.items()):
    pm = ax.pcolormesh(sst.longitude, sst.latitude, sst, cmap="RdYlBu_r")
    ax.pcolormesh(sst.longitude, sst.latitude, mask.where(mask), cmap="binary", vmin=0, vmax=1)
    ax.set_title(method)
    ax.set_aspect(aspect)
fig.colorbar(pm, ax=axs, label="Temperature [°C]", shrink=0.8)

# %%
# Gradient magnitude of the BOA method
# ------------------------------------
#
# The numeric function :func:`shoot.core.fronts.boa` gives the normalized
# gradient magnitude, which :func:`~shoot.fronts.fronts2d.detect_fronts` thresholds.
magnitude = boa(sst.values)
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
pm = ax.pcolormesh(sst.longitude, sst.latitude, magnitude, cmap="magma_r", vmax=1)
ax.contour(sst.longitude, sst.latitude, magnitude, [params["boa"]["threshold"]], colors="c", linewidths=0.5)
ax.set_aspect(aspect)
fig.colorbar(pm, label="Normalized gradient magnitude")
ax.set_title("BOA gradient magnitude and front threshold")
