#!/usr/bin/env python3
"""
Numerical utilities

The pure numeric routines live in :mod:`shoot.core.num` and are re-exported here.
"""

import numba
import numpy as np
import xarray as xr

from . import meta as smeta
from .core.num import (  # noqa: F401
    any_points_in_polygon,
    find_signed_peaks_2d,
    point_in_polygon,
    points_in_polygon,
)


# @numba.njit
def find_signed_peaks_2d_jb_old(lnam, closed_lines):
    """Find peaks within closed contour lines (old implementation)

    Parameters
    ----------
    lnam : xarray.DataArray
        2D LNAM field.
    closed_lines : list
        List of closed contour lines.

    Returns
    -------
    minima : ndarray
        Array of (i, j) indices for local minima.
    maxima : ndarray
        Array of (i, j) indices for local maxima.
    """
    lon = smeta.get_lon(lnam)
    lat = smeta.get_lat(lnam)
    ny, nx = lnam.shape
    mask = np.isnan(lnam)

    # find inside indexes
    dict_line = {nl: [] for nl in range(len(closed_lines))}
    for j in numba.prange(ny):
        for i in range(nx):
            if mask[j, i]:
                continue
            for nl, line in enumerate(closed_lines):
                if points_in_polygon([lon[i], lat[j]], line):
                    dict_line[nl].append([i, j])

    # find maximum
    maxima = np.empty((0, 2), dtype=np.int64)
    minima = np.empty((0, 2), dtype=np.int64)
    for nl in dict_line:
        ind_lat = xr.DataArray(dict_line[nl][:, 1], dims="latitude")
        ind_lon = xr.DataArray(dict_line[nl][:, 1], dims="longitude")
        imax, jmax = abs(lnam).isel(latitude=ind_lat, longitude=ind_lon).argmax()
        if lnam[imax, jmax] > 0:
            maxima.append(maxima, np.array([[imax, jmax]]), axis=0)
        else:
            minima.append(minima, np.array([[imax, jmax]]), axis=0)

    return minima, maxima
