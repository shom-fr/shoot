#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid utilities
"""
import numpy as np
import xarray as xr

from . import geo as sgeo
from . import cf as scf


def get_dx_dy(da, dx=None, dy=None):
    """Get the local resolution in meters along X and Y"""
    if dx is not None and dy is not None:
        return dx, dy

    lon = scf.get_lon(da)
    lat = scf.get_lat(da)

    lat2d, lon2d = xr.broadcast(lat, lon)
    dlonx = np.gradient(lon2d.values, axis=-1)
    dlony = np.gradient(lon2d.values, axis=-2)
    dlatx = np.gradient(lat2d.values, axis=-1)
    dlaty = np.gradient(lat2d.values, axis=-2)

    # cf = scf.get_cf_specs()
    kw = dict(format_coords=False, rename_dims=False)

    if dx is None:
        dx = sgeo.deg2m(dlonx, lat2d.values) ** 2
        dx += sgeo.deg2m(dlatx) ** 2
        dx = np.sqrt(dx)
        dx = xr.DataArray(dx, dims=lon2d.dims, coords=lon2d.coords, name="dx", attrs={"units": "m"})
        # dx = cf.format_data_var(dx, "dx", **kw)

    if dy is None:
        dy = sgeo.deg2m(dlony, lat2d.values) ** 2
        dy += sgeo.deg2m(dlaty) ** 2
        dy = np.sqrt(dy)
        dy = xr.DataArray(dy, dims=lon2d.dims, coords=lon2d.coords, name="dy", attrs={"units": "m"})
        # dy = cf.format_data_var(dy, "dy", **kw)

    return dx, dy


def get_wx_wy(window, dx, dy):
    """Get the window size in grid points along X and Y"""
    dx = np.nanmean(dx)
    dy = np.nanmean(dy)
    wx = 2 * (int(np.ceil(window * 1e3 / dx)) // 2) + 1
    wy = 2 * (int(np.ceil(window * 1e3 / dy)) // 2) + 1
    return wx, wy
