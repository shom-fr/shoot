#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical utilities

Optimized numeric functions for peak detection and geometric computations.
"""

import numba
import numpy as np
import xarray as xr

from . import meta as smeta


@numba.njit
def _find_signed_peaks_2d_ref(data, wx, wy):
    """Reference implementation for finding 2D peaks (serial, not parallel)

    Parameters
    ----------
    data : ndarray
        2D data array to search for peaks.
    wx : int
        Window width in x direction.
    wy : int
        Window width in y direction.

    Returns
    -------
    minima : ndarray
        Array of (i, j) indices for local minima.
    maxima : ndarray
        Array of (i, j) indices for local maxima.
    """
    ny, nx = data.shape
    mask = np.isnan(data)
    maxima = np.empty((0, 2), dtype=np.int64)
    minima = np.empty((0, 2), dtype=np.int64)
    wx2 = wx // 2
    wy2 = wy // 2
    for j in numba.prange(1, ny - 1):
        for i in range(1, nx - 1):
            if mask[j, i]:
                continue
            i0 = max(0, i - wx2)
            i1 = min(nx, i + wx2 + 1)
            j0 = max(0, j - wy2)
            j1 = min(ny, j + wy2 + 1)
            if mask[j - 1 : j + 2, i - 1 : i + 2].all():
                continue
            imax = -1
            jmax = -1
            imin = -1
            jmin = -1
            vmax = -np.inf
            vmin = np.inf
            for jl in range(j0, j1):
                for il in range(i0, i1):
                    if mask[jl, il]:
                        continue
                    if data[jl, il] > vmax:
                        imax = il
                        jmax = jl
                        vmax = data[jl, il]
                    if data[jl, il] < vmin:
                        imin = il
                        jmin = jl
                        vmin = data[jl, il]

            if imin == i and jmin == j:
                minima = np.append(minima, np.array([[i, j]]), axis=0)
            if imax == i and jmax == j:
                maxima = np.append(maxima, np.array([[i, j]]), axis=0)

    return minima, maxima


@numba.njit(parallel=True)
def _find_signed_peaks_2d_paral_save(data, wx, wy):
    """Parallel implementation for finding 2D peaks (memory-safe version)

    Parameters
    ----------
    data : ndarray
        2D data array to search for peaks.
    wx : int
        Window width in x direction.
    wy : int
        Window width in y direction.

    Returns
    -------
    minima : ndarray
        3D array of (nx, ny, 2) with peak indices, -1 for non-peaks.
    maxima : ndarray
        3D array of (nx, ny, 2) with peak indices, -1 for non-peaks.
    """
    ny, nx = data.shape
    mask = np.isnan(data)
    maxima = np.ones((nx, ny, 2), dtype=np.int64) * -1
    minima = np.ones((nx, ny, 2), dtype=np.int64) * -1
    wx2 = wx // 2
    wy2 = wy // 2

    for k in numba.prange(0, (ny - 2) * (nx - 2)):
        j = k // (nx - 2) + 1
        i = k + 1 - (j - 1) * (nx - 2)
        if mask[j, i]:
            continue
        i0 = max(0, i - wx2)
        i1 = min(nx, i + wx2 + 1)
        j0 = max(0, j - wy2)
        j1 = min(ny, j + wy2 + 1)
        if mask[j - 1 : j + 2, i - 1 : i + 2].all():
            continue
        imax = -1
        jmax = -1
        imin = -1
        jmin = -1
        vmax = -np.inf
        vmin = np.inf
        for jl in range(j0, j1):
            for il in range(i0, i1):
                if mask[jl, il]:
                    continue
                if data[jl, il] > vmax:
                    imax = il
                    jmax = jl
                    vmax = data[jl, il]
                if data[jl, il] < vmin:
                    imin = il
                    jmin = jl
                    vmin = data[jl, il]
        if imin == i and jmin == j:
            minima[i, j] = [i, j]
        if imax == i and jmax == j:
            maxima[i, j] = [i, j]
    return minima, maxima


@numba.njit(parallel=True)
def _find_signed_peaks_2d_paral(data, wx, wy):
    """Parallel implementation for finding 2D peaks

    Note: May miss extrema on the right edge of the domain.

    Parameters
    ----------
    data : ndarray
        2D data array to search for peaks.
    wx : int
        Window width in x direction.
    wy : int
        Window width in y direction.

    Returns
    -------
    minima : ndarray
        3D array of (nx, ny, 2) with peak indices, -1 for non-peaks.
    maxima : ndarray
        3D array of (nx, ny, 2) with peak indices, -1 for non-peaks.
    """
    ny, nx = data.shape
    mask = np.isnan(data)

    maxima = np.ones((nx, ny, 2), dtype=np.int64) * -1  # np.ones((nmax,2),dtype=np.int64)*-1#
    minima = np.ones((nx, ny, 2), dtype=np.int64) * -1
    wx2 = wx // 2
    wy2 = wy // 2

    for j in numba.prange(1, ny - 1):
        for i in range(1, nx - 1):
            if mask[j, i]:
                continue
            i0 = max(0, i - wx2)
            i1 = min(nx, i + wx2 + 1)
            j0 = max(0, j - wy2)
            j1 = min(ny, j + wy2 + 1)
            if mask[j - 1 : j + 2, i - 1 : i + 2].all():
                continue
            imax = -1
            jmax = -1
            imin = -1
            jmin = -1
            vmax = -np.inf
            vmin = np.inf
            for jl in range(j0, j1):
                for il in range(i0, i1):
                    if mask[jl, il]:
                        continue
                    if data[jl, il] > vmax:
                        imax = il
                        jmax = jl
                        vmax = data[jl, il]
                    if data[jl, il] < vmin:
                        imin = il
                        jmin = jl
                        vmin = data[jl, il]
            if imin == i and jmin == j:
                minima[i, j] = [i, j]
            if imax == i and jmax == j:
                maxima[i, j] = [i, j]
    return minima, maxima


def find_signed_peaks_2d(data, wx, wy, paral=False):
    """Find local extrema (minima and maxima) in 2D field

    Parameters
    ----------
    data : ndarray
        2D data array to search for peaks.
    wx : int
        Window width in x direction for local extrema detection.
    wy : int
        Window width in y direction for local extrema detection.
    paral : bool, default False
        Use parallel implementation if True, serial if False.

    Returns
    -------
    minima : ndarray
        Array of (i, j) indices for local minima.
    maxima : ndarray
        Array of (i, j) indices for local maxima.
    """
    if paral:
        minima, maxima = _find_signed_peaks_2d_paral(data, wx, wy)
        ny, nx = data.shape
        # reshape
        minima = minima.reshape((ny * nx, 2))
        maxima = maxima.reshape((ny * nx, 2))

        return (
            minima[np.where(~np.all(minima == -1, axis=1))[0]],
            maxima[np.where(~np.all(maxima == -1, axis=1))[0]],
        )
    else:
        return _find_signed_peaks_2d_ref(data, wx, wy)


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


@numba.guvectorize(
    [(numba.float64[:], numba.float64[:, :], numba.boolean[:])],
    "(nd),(ne,nd)->()",
)
def points_in_polygon(point, poly, inside):
    """

    Parameters
    ----------
    points: array(npts, 2)
    """

    n = poly.shape[0]
    inside[0] = False

    p1x, p1y = poly[0]
    x, y = point[:2]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside[0] = not inside[0]
        p1x, p1y = p2x, p2y


def get_coord_name(data):
    """Extract longitude and latitude coordinate names from xarray object

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Data object with coordinates.

    Returns
    -------
    lon_name : str or None
        Name of longitude coordinate.
    lat_name : str or None
        Name of latitude coordinate.
    """
    coords_name = [k for k in data.coords.keys()]
    lon_name = None
    lat_name = None
    for var in coords_name:
        if var[:3] in ['LON', 'Lon', 'lon', 'longitude', 'Longitude', 'lon_rho']:
            lon_name = var
        if var[:3] in ['LAT', 'Lat', 'lat', 'latitude', 'Latitude', 'lat_rho']:
            lat_name = var
    return lon_name, lat_name
