#!/usr/bin/env python3
"""
Contouring numeric routines

Contour lines are handled in grid-index space as ``(n, 2)`` arrays of
fractional ``(i, j)`` indices, as returned by contourpy.
"""

import warnings

import contourpy as cpy
import numpy as np
import scipy.ndimage as scin
from scipy.interpolate import make_interp_spline

from . import geo as sgeo
from . import num as snum


def nearest_index(lon2d, lat2d, lon, lat):
    """Grid indices of the point nearest to a position

    Parameters
    ----------
    lon2d, lat2d : ndarray
        2D longitudes and latitudes.
    lon, lat : float
        Position.

    Returns
    -------
    i, j : int
        Indices along X and Y.
    """
    j, i = np.unravel_index(np.argmin((lon2d - lon) ** 2 + (lat2d - lat) ** 2), lon2d.shape)
    return int(i), int(j)


def find_closed_contours(z, ic, jc, nlevels=50, robust=0.03):
    """Find closed contours of a 2D array enclosing a grid point

    Everything is performed in grid-index space: a contour is retained
    if it is closed, contains the center and contains no NaN point.

    Parameters
    ----------
    z : ndarray
        2D field to contour, with NaNs as land points.
    ic, jc : int
        Grid indices of the center along X and Y.
    nlevels : int, default 50
        Maximum number of contour levels.
    robust : float, default 0.03
        Quantile threshold to exclude extreme values.

    Returns
    -------
    list of (float, ndarray)
        Contour level and line of fractional (i, j) indices of shape (n, 2).
    """
    z = np.asarray(z, dtype="d")
    ic, jc = float(ic), float(jc)

    # Land points in index space
    bad = np.isnan(z)
    nan_points = np.ascontiguousarray(np.argwhere(bad)[:, ::-1], dtype="d") if bad.any() else None

    cont_gen = cpy.contour_generator(z=z)
    vmin, vmax = np.nanquantile(z, [robust, 1 - robust])
    if len(np.arange(vmin, vmax + 0.005, 0.005)) < nlevels:
        levels = np.arange(vmin, vmax + 0.005, 0.005)
    else:
        levels = np.linspace(vmin, vmax, nlevels)
    contours = []
    for level in levels:
        for line in cont_gen.lines(level):
            if not (line[0] == line[-1]).all():  # check if it is closed contour
                continue
            if not snum.point_in_polygon(ic, jc, line):  # Check if it contains the center
                continue
            if nan_points is not None and snum.any_points_in_polygon(nan_points, line):
                continue  # it contains land points inside
            contours.append((level, line))
    return contours


def contour_lines(z, level, x=None, y=None):
    """Contour lines of a 2D array at a single level

    Lines are computed with the "mpl2014" algorithm of contourpy,
    as :func:`matplotlib.pyplot.contour`. NaNs are masked.

    Parameters
    ----------
    z : ndarray
        2D field.
    level : float
        Contour level.
    x, y : ndarray, optional
        1D or 2D coordinates. Defaults to the grid indices.

    Returns
    -------
    list of ndarray
        Lines as (n, 2) arrays of (x, y) coordinates.
    """
    cont_gen = cpy.contour_generator(x, y, np.asarray(z, dtype="d"), name="mpl2014", line_type="SeparateCode")
    return list(cont_gen.lines(level)[0])


def find_open_contours(z, level, x=None, y=None, min_points=7):
    """Open contour lines of a 2D array at a single level

    Parameters
    ----------
    z : ndarray
        2D field.
    level : float
        Contour level.
    x, y : ndarray, optional
        1D or 2D coordinates. Defaults to the grid indices.
    min_points : int, default 7
        Minimum number of points of a line.

    Returns
    -------
    list of ndarray
        Open lines as (n, 2) arrays of (x, y) coordinates.
    """
    return [
        line
        for line in contour_lines(z, level, x=x, y=y)
        if len(line) >= min_points and not (line[0] == line[-1]).all()
    ]


def interp_to_line(data, line):
    """Interpolate 2D field values along a contour line

    Parameters
    ----------
    data : ndarray
        2D field to interpolate.
    line : ndarray
        Contour line coordinates of shape (n, 2).

    Returns
    -------
    ndarray
        Interpolated values along the contour.
    """
    coords = line.T[::-1]
    mask = np.isnan(data).astype("d")
    dataf = np.nan_to_num(data)
    lm = scin.map_coordinates(mask, coords)
    ldata = scin.map_coordinates(dataf, coords)
    lbad = ~np.isclose(lm + 1, 1.0)
    ldata[lbad] = np.nan
    return ldata


def contour_velocity(line, lon, lat, u, v, lon_center, lat_center):
    """Velocity and angular momentum along a contour

    Parameters
    ----------
    line : ndarray
        Contour line of fractional (i, j) indices of shape (n, 2).
    lon, lat : ndarray
        Contour coordinates in degrees.
    u, v : ndarray
        2D velocity components in m/s.
    lon_center, lat_center : float
        Center position in degrees.

    Returns
    -------
    uc, vc, am : ndarray
        Velocity components and angular momentum along the contour.
    mean_velocity, mean_angular_momentum : float
        Averages that ignore NaNs.
    radius : float
        Mean distance to the center in meters.
    """
    uc = interp_to_line(u, line)
    vc = interp_to_line(v, line)
    xdist = sgeo.deg2m(lon - lon_center, lat_center)
    ydist = sgeo.deg2m(lat - lat_center)
    am = xdist * vc - ydist * uc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_velocity = float(np.nanmean(np.sqrt(uc**2 + vc**2)))
        mean_angular_momentum = float(np.nanmean(am))
    radius = float(np.mean(np.sqrt(xdist**2 + ydist**2)))
    return uc, vc, am, mean_velocity, mean_angular_momentum, radius


def contour_steps(lon, lat):
    """Steps along a contour and its length

    Parameters
    ----------
    lon, lat : ndarray
        Contour coordinates in degrees.

    Returns
    -------
    dx, dy : ndarray
        Steps in meters.
    length : float
        Length in meters.
    """
    dx = sgeo.deg2m(np.gradient(lon), lat.mean())
    dy = sgeo.deg2m(np.gradient(lat))
    return dx, dy, float(np.sqrt(dx**2 + dy**2).sum())


def smooth_contour(lon, lat, npts=50, tol=0.0):
    """Resample a closed contour with a periodic cubic spline

    Parameters
    ----------
    lon, lat : ndarray
        Contour coordinates, the last point being equal to the first one.
    npts : int, default 50
        Number of output points.
    tol : float, default 0.0
        Consecutive points closer than this (in lon + lat) are dropped.

    Returns
    -------
    lon_int, lat_int : ndarray
    """
    ok = np.where(np.abs(np.diff(lon)) + np.abs(np.diff(lat)) > tol)[0]
    ok = np.concatenate([ok, [len(lon) - 1]])
    lon = lon[ok]
    lat = lat[ok]
    t = np.linspace(0, 1, len(lon))
    spl_lon = make_interp_spline(t, lon, k=3, bc_type="periodic")
    spl_lat = make_interp_spline(t, lat, k=3, bc_type="periodic")
    t_new = np.linspace(0, 1, npts)
    return spl_lon(t_new), spl_lat(t_new)
