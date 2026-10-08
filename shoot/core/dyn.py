#!/usr/bin/env python3
"""
Ocean dynamics numeric kernels

Kinematic quantities computed on 2D arrays of shape (ny, nx).
Grid resolutions ``dx`` and ``dy`` are in meters, either scalars or arrays.
"""

import math

import numba
import numpy as np

GRAVITY = 9.81
OMEGA = 2 * np.pi / 86400


@numba.guvectorize(
    [(numba.float64[:, :], numba.float64[:, :], numba.int64, numba.float64, numba.float64[:, :])],
    "(ny,nx),(ny,nx),(),()->(ny,nx)",
)
def _lnam_(uu, vv, wx, dx2dy, lnam):
    ny, nx = uu.shape
    mask = np.isnan(uu) | np.isnan(vv)
    wx2 = wx // 2
    wy = (int(np.ceil(wx / dx2dy)) // 2) * 2 + 1
    wy2 = wy // 2
    lnam[:, :] = np.nan
    for j in numba.prange(wy2, ny - wy2 - 1):
        for i in range(wx2, nx - wx2 - 1):
            if mask[j - wy2 : j + wy2 + 1, i - wx2 : i + wx2 + 1].any():
                continue
            denom1 = 0.0
            denom2 = 0.0
            lnam[j, i] = 0.0
            for jl in range(-wy2, wy2 + 1):
                for il in range(-wx2, wx2 + 1):
                    lnam[j, i] += il * vv[j + jl, i + il]
                    lnam[j, i] -= jl * uu[j + jl, i + il] * dx2dy
                    denom1 += il * uu[j + jl, i + il]
                    denom1 += jl * vv[j + jl, i + il] * dx2dy
                    denom2 += math.sqrt(uu[j + jl, i + il] ** 2 + vv[j + jl, i + il] ** 2) * math.sqrt(
                        il**2 + (jl * dx2dy) ** 2
                    )
            if (denom1 + denom2) > 1e-6:
                lnam[j, i] /= denom1 + denom2


def get_window_size(window, dx):
    """Convert a window size in km to an odd number of grid points

    Parameters
    ----------
    window : float
        Window size in kilometers.
    dx : float
        Grid resolution in meters.

    Returns
    -------
    int
    """
    return (int(np.ceil(window * 1e3 / dx)) // 2) * 2 + 1


def lnam(u, v, wx, dx2dy):
    """Local normalized angular momentum

    Parameters
    ----------
    u, v : ndarray
        Velocity components.
    wx : int
        Window size along X in grid points (odd).
    dx2dy : float
        Ratio of the mean resolutions ``dy / dx``.

    Returns
    -------
    ndarray
    """
    return _lnam_(np.asarray(u, dtype="d"), np.asarray(v, dtype="d"), int(wx), float(dx2dy))


def div(u, v, dx, dy):
    """Horizontal divergence in s-1"""
    sx = np.gradient(u, axis=-1) / dx
    sy = np.gradient(v, axis=-2) / dy
    div = sx + sy
    div[np.isnan(u) | np.isnan(v)] = np.nan
    return div


def okuboweiss(u, v, dx, dy):
    """Okubo-Weiss parameter in s-2"""
    sn = np.gradient(u, axis=-1) / dx - np.gradient(v, axis=-2) / dy
    ss = np.gradient(v, axis=-1) / dx + np.gradient(u, axis=-2) / dy
    om = np.gradient(v, axis=-1) / dx - np.gradient(u, axis=-2) / dy
    ow = sn**2 + ss**2 - om**2
    ow[np.isnan(u) | np.isnan(v)] = np.nan
    return ow


def relvort(u, v, dx, dy):
    """Relative vorticity in s-1"""
    rv = np.gradient(v, axis=-1) / dx
    rv -= np.gradient(u, axis=-2) / dy
    rv[np.isnan(u) | np.isnan(v)] = np.nan
    return rv


def coriolis(lat):
    """Coriolis parameter (f = 2Ω sin(lat)) in s-1 from latitudes in degrees"""
    return 2 * OMEGA * np.sin(np.radians(lat))


def geos(ssh, dx, dy, corio):
    """Geostrophic currents in m/s from sea surface height in meters

    Parameters
    ----------
    ssh : ndarray
        Sea surface height of shape (ny, nx).
    dx, dy : float or ndarray
        Grid resolutions in meters.
    corio : ndarray
        Coriolis parameter, either 1D along Y or 2D.

    Returns
    -------
    ugeos, vgeos : ndarray
    """
    if corio.ndim == 1:
        corio = corio.reshape(corio.shape[0], 1)
    dhdx = np.gradient(ssh, axis=-1) / (dx * corio)
    dhdy = np.gradient(ssh, axis=-2) / (dy * corio)
    bad = np.isnan(ssh)
    dhdx[bad] = np.nan
    dhdy[bad] = np.nan
    return -dhdy * GRAVITY, dhdx * GRAVITY
