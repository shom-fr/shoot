#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure numeric utilities
======================
"""

import numba
import numpy as np
import xoa.coords as xcoords
import xarray as xr


@numba.njit  # this version can't be paralellized : today's reference
def _find_signed_peaks_2d_ref(data, wx, wy):
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
            vmax = 0.0
            vmin = 0.0
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


@numba.njit(
    parallel=True
)  # PB : parcoure tout le domaine mais voit pas les extrema à droite du domaine
def _find_signed_peaks_2d_paral_save(data, wx, wy):
    ny, nx = data.shape
    mask = np.isnan(data)
    # nmax = (nx//wx)*(ny//wy)
    maxima = np.ones((nx, ny, 2), dtype=np.int64) * -1  # np.ones((nmax,2),dtype=np.int64)*-1#
    minima = (
        np.ones((nx, ny, 2), dtype=np.int64) * -1
    )  # np.ones((nmax,2),dtype=np.int64)*-1#np.ones((ny,nx,2),dtype=np.int64)*-1
    # cmp_min, cmp_max = 0,0
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
        vmax = 0.0
        vmin = 0.0
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


@numba.njit(
    parallel=True
)  # PB : parcoure tout le domaine mais voit pas les extrema à droite du domaine
def _find_signed_peaks_2d_paral(data, wx, wy):
    ny, nx = data.shape
    mask = np.isnan(data)
    # nmax = (nx//wx)*(ny//wy)
    maxima = np.ones((nx, ny, 2), dtype=np.int64) * -1  # np.ones((nmax,2),dtype=np.int64)*-1#
    minima = (
        np.ones((nx, ny, 2), dtype=np.int64) * -1
    )  # np.ones((nmax,2),dtype=np.int64)*-1#np.ones((ny,nx,2),dtype=np.int64)*-1
    # cmp_min, cmp_max = 0,0
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
            vmax = 0.0
            vmin = 0.0
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


def find_signed_peaks_2d(data, wx, wy, paral=False):  # Problem with the
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
    lon = xcoords.get_lon(lnam)
    lat = xcoords.get_lat(lnam)
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
    coords_name = [k for k in data.coords.keys()]
    lon_name = None
    lat_name = None
    for var in coords_name:
        if var[:3] in ['LON', 'Lon', 'lon', 'longitude', 'Longitude', 'lon_rho']:
            lon_name = var
        if var[:3] in ['LAT', 'Lat', 'lat', 'latitude', 'Latitude', 'lat_rho']:
            lat_name = var
    return lon_name, lat_name
