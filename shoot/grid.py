#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid utilities for computing spatial resolutions and window sizes
"""
import numpy as np
import xarray as xr

from . import geo as sgeo
from . import meta as smeta


def get_dx_dy(da, dx=None, dy=None):
    """Compute local grid resolution in meters

    Parameters
    ----------
    da : xarray.DataArray
        Data array with longitude and latitude coordinates.
    dx : xarray.DataArray, optional
        X resolution in meters. Computed if not provided.
    dy : xarray.DataArray, optional
        Y resolution in meters. Computed if not provided.

    Returns
    -------
    dx : xarray.DataArray
        Resolution along X in meters.
    dy : xarray.DataArray
        Resolution along Y in meters.
    """
    if dx is not None and dy is not None:
        return dx, dy

    lon = smeta.get_lon(da)
    lat = smeta.get_lat(da)

    lat2d, lon2d = xr.broadcast(lat, lon)
    dlonx = np.gradient(lon2d.values, axis=-1)
    dlony = np.gradient(lon2d.values, axis=-2)
    dlatx = np.gradient(lat2d.values, axis=-1)
    dlaty = np.gradient(lat2d.values, axis=-2)

    # cf = smeta.get_cf_specs()
    kw = dict(format_coords=False, rename_dims=False)

    if dx is None:
        dx = sgeo.deg2m(dlonx, lat2d.values) ** 2
        dx += sgeo.deg2m(dlatx) ** 2
        dx = np.sqrt(dx)
        dx = xr.DataArray(dx, dims=lon2d.dims, coords=lon2d.coords, name="dx", attrs={"units": "m"})

    if dy is None:
        dy = sgeo.deg2m(dlony, lat2d.values) ** 2
        dy += sgeo.deg2m(dlaty) ** 2
        dy = np.sqrt(dy)
        dy = xr.DataArray(dy, dims=lon2d.dims, coords=lon2d.coords, name="dy", attrs={"units": "m"})

    return dx, dy


def get_wx_wy(window, dx, dy):
    """Convert window size from km to grid points

    Parameters
    ----------
    window : float
        Window size in kilometers.
    dx : float or xarray.DataArray
        Grid resolution along X in meters.
    dy : float or xarray.DataArray
        Grid resolution along Y in meters.

    Returns
    -------
    wx : int
        Window width in grid points (odd number).
    wy : int
        Window height in grid points (odd number).
    """
    dx = np.nanmean(dx)
    dy = np.nanmean(dy)
    wx = 2 * (int(np.ceil(window * 1e3 / dx)) // 2) + 1
    wy = 2 * (int(np.ceil(window * 1e3 / dy)) // 2) + 1
    return wx, wy
