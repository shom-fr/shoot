#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics prior to eddy detection
===================================
"""

# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
from shoot.grid import get_dx_dy
from shoot.dyn import get_relvort, get_lnam, get_okuboweiss, get_geos
from shoot.plot import create_map, pcarr

xr.set_options(display_style="text")

# %%
# Read data
ds = xr.open_dataset("ssh-sat-med.nc").isel(time=0)
ds = ds.sel(longitude=slice(24, 34), latitude=slice(31, 35))

# %%
# Compute grid metrics only once
dx, dy = get_dx_dy(ds)

# %%
# Geostrophic currents
# --------------------
#
# ADT
ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
ds.adt.plot(ax=ax, cmap="cmo.balance", add_colorbar=False, transform=pcarr)
ds.adt.plot.contour(ax=ax, transform=pcarr, colors="k", levels=20)

# %%
# Compute gestrophic current
ugeos, vgeos = get_geos(ds.adt, dx=dx, dy=dy)

# %%
# Compare them to dataset currents
ax0 = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
kwqv = dict(units="dots", width=1, scale_units="dots", scale=1 / 20, transform=pcarr)
ds.plot.quiver(
    x="longitude", y="latitude", u="ugos", v="vgos", color="k", ax=ax0, label="dataset", **kwqv
)
ax0.quiver(
    ds.longitude.values,
    ds.latitude.values,
    ugeos.values,
    vgeos.values,
    color="tab:orange",
    label="computed",
    **kwqv,
)
plt.title("Geostrophic currents")
plt.legend()


# %%
# Local normalized angular momentum
# ----------------------------------
#
# The normalized angular momentum is computed at the center of 2D scanning window.
#
# Window in km
window_lnam = 50

# %%
# Local normalized angular momentum
lnam = get_lnam(ds.ugos, ds.vgos, window_lnam, dx=dx, dy=dy)

# %%
# Plot
ax1 = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
lnam.plot(cmap="cmo.diff", ax=ax1, add_colorbar=False, transform=pcarr)
plt.title(f"Local angular momentum [{window_lnam}km]")


# %%
# Okubo-Weiss
# -----------
#
# Diagnostic
ow = get_okuboweiss(ds.ugos, ds.vgos, dx=dx, dy=dy)

# %%
# Plot
ax = create_map(ds.longitude, ds.latitude, figsize=(8, 5))
lnam.plot(cmap="cmo.delta", ax=ax, add_colorbar=False, transform=pcarr)
lnam.plot.contour(levels=[0], colors="k", transform=pcarr)
plt.title("Okubo-Weiss")
