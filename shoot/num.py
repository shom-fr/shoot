#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure numeric utilities
"""

import numba
import numpy as np


# @numba.njit(numba.int64[:](numba.int64, numba.int64[:]), cache=False)
# def unravel_index(i, shape):
#     ir = i
#     ndim = len(shape)
#     ii = np.zeros(ndim, np.int64)
#     for o in range(ndim):
#         if o != ndim - 1:
#             base = np.prod(shape[o + 1 :])
#         else:
#             base = 1
#         ii[o] = ir // base
#         ir -= ii[o] * base
#     return ii


@numba.njit
def find_signed_peaks_2d(data, wx, wy):
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

            # ijmax = unravel_index(np.argmax(data[j0:j1, i0:i1]), np.array([j1 - j0, i1 - i0]))
            # if ijmax[1] + i0 == i and ijmax[0] + j0 == j:
            #     maxima = np.append(maxima, np.array([[i, j]]), axis=0)

            # ijmin = unravel_index(np.argmin(data[j0:j1, i0:i1]), np.array([j1 - j0, i1 - i0]))
            # if ijmin[1] + i0 == i and ijmin[0] + j0 == j:
            #     minima = np.append(minima, np.array([[i, j]]), axis=0)

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
