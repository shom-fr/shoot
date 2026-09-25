#!/usr/bin/env python3
"""
Eddy detection numeric routines

Arrays are of shape (ny, nx), grid indices are (i, j) along (X, Y).
"""

from typing import NamedTuple

import numba
import numpy as np

from . import contours as scontours
from . import dyn as sdyn
from . import fit as sfit
from . import num as snum


class Centers(NamedTuple):
    """Eddy center candidates"""

    #: Grid indices along X
    i: np.ndarray
    #: Grid indices along Y
    j: np.ndarray
    #: Local normalized angular momentum at centers
    lnam: np.ndarray
    #: Okubo-Weiss parameter at centers
    ow: np.ndarray
    #: Peak search window size along X in grid points
    wx: int
    #: Peak search window size along Y in grid points
    wy: int


class EddyContour(NamedTuple):
    """A closed contour around an eddy center that fits an ellipse"""

    level: float
    #: Line of fractional (i, j) indices of shape (n, 2)
    line: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    #: Velocity components and angular momentum along the contour
    u: np.ndarray
    v: np.ndarray
    am: np.ndarray
    #: Steps along the contour in meters
    dx: np.ndarray
    dy: np.ndarray
    mean_velocity: float
    mean_angular_momentum: float
    #: Mean distance to the center in meters
    radius: float
    #: Length in meters
    length: float
    #: Fitted ellipse parameters: dict with lon, lat, a, b (km) and angle (degrees)
    ellipse: dict
    #: Ellipse fit error
    fit_error: float


def find_centers(u, v, dx, dy, window, paral=False):
    """Find eddy center candidates

    Centers are the peaks of the local normalized angular momentum
    in regions where the Okubo-Weiss parameter is negative.

    Parameters
    ----------
    u, v : ndarray
        2D velocity components.
    dx, dy : float
        Mean grid resolutions in meters.
    window : float
        Window size in kilometers.
    paral : bool, default False
        Use the parallel peak finder.

    Returns
    -------
    Centers
    """
    lnam = sdyn.lnam(u, v, sdyn.get_window_size(window, dx), float(dy / dx))
    ow = sdyn.okuboweiss(u, v, dx, dy)
    lnam = np.where(ow < 0, lnam, np.nan)

    wx = sdyn.get_window_size(window, dx)
    wy = sdyn.get_window_size(window, dy)
    minima, maxima = snum.find_signed_peaks_2d(lnam, wx, wy, paral=paral)
    extrema = np.vstack((minima, maxima))
    ii = extrema[:, 0]
    jj = extrema[:, 1]
    return Centers(ii, jj, lnam[jj, ii], ow[jj, ii], wx, wy)


def find_eddy_contours(ssh, u, v, lon2d, lat2d, i, j, nlevels=100, robust=0.03, max_ellipse_error=0.01):
    """Find the closed contours around a center that are well fitted by an ellipse

    Parameters
    ----------
    ssh : ndarray
        2D sea surface height or streamfunction.
    u, v : ndarray
        2D velocity components.
    lon2d, lat2d : ndarray
        2D longitudes and latitudes.
    i, j : int
        Grid indices of the center.
    nlevels : int, default 100
        Maximum number of contour levels.
    robust : float, default 0.03
        Quantile threshold to exclude extreme values.
    max_ellipse_error : float, default 0.01
        Maximum allowed ellipse fit error.

    Returns
    -------
    list of EddyContour
    """
    lon_center = float(lon2d[j, i])
    lat_center = float(lat2d[j, i])
    contours = []
    for level, line in scontours.find_closed_contours(ssh, i, j, nlevels=nlevels, robust=robust):
        lon = scontours.interp_to_line(lon2d, line)
        lat = scontours.interp_to_line(lat2d, line)
        ellipse, fit_error = sfit.fit_ellipse_from_coords(lon, lat, get_fit=True)
        # check if ellipse center fall inside the eddy contour
        if not snum.point_in_polygon(ellipse["lon"], ellipse["lat"], np.array([lon, lat]).T):
            continue
        if fit_error < max_ellipse_error:
            uc, vc, am, mean_velocity, mean_am, radius = scontours.contour_velocity(
                line, lon, lat, u, v, lon_center, lat_center
            )
            dx, dy, length = scontours.contour_steps(lon, lat)
            contours.append(
                EddyContour(
                    level,
                    line,
                    lon,
                    lat,
                    uc,
                    vc,
                    am,
                    dx,
                    dy,
                    mean_velocity,
                    mean_am,
                    radius,
                    length,
                    ellipse,
                    fit_error,
                )
            )
    return contours


def argmax_first(values):
    """Index of the first maximum, NaNs being never selected except at index 0

    Parameters
    ----------
    values : sequence of float

    Returns
    -------
    int
    """
    best = 0
    for k in range(1, len(values)):
        if values[k] > values[best]:
            best = k
    return best


@numba.njit(cache=True)
def _filter_intersecting_(points, offsets, speeds):
    n = offsets.size - 1
    bbmin = np.empty((n, 2))
    bbmax = np.empty((n, 2))
    for k in range(n):
        poly = points[offsets[k] : offsets[k + 1]]
        bbmin[k, 0] = poly[:, 0].min()
        bbmin[k, 1] = poly[:, 1].min()
        bbmax[k, 0] = poly[:, 0].max()
        bbmax[k, 1] = poly[:, 1].max()
    keep = np.ones(n, dtype=np.bool_)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                bbmin[j, 0] > bbmax[i, 0]
                or bbmin[j, 1] > bbmax[i, 1]
                or bbmax[j, 0] < bbmin[i, 0]
                or bbmax[j, 1] < bbmin[i, 1]
            ):
                continue
            if snum.any_points_in_polygon(
                points[offsets[j] : offsets[j + 1]], points[offsets[i] : offsets[i + 1]]
            ):
                if speeds[i] > speeds[j]:
                    keep[j] = False
                else:
                    keep[i] = False
    return keep


def filter_intersecting(polygons, speeds):
    """Remove eddies whose contour intersects the contour of a faster one

    Eddy `j` intersects eddy `i` when one of its points is inside the polygon of `i`.
    All pairs are processed in order, and the slowest eddy of a pair is removed.

    Parameters
    ----------
    polygons : list of ndarray
        Contours as (n, 2) arrays of (lon, lat).
    speeds : array-like
        Mean speeds along the contours.

    Returns
    -------
    ndarray of bool
        Eddies to keep.
    """
    if not len(polygons):
        return np.ones(0, dtype=bool)
    offsets = np.zeros(len(polygons) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(poly) for poly in polygons])
    points = np.ascontiguousarray(np.concatenate(polygons), dtype="d")
    return _filter_intersecting_(points, offsets, np.asarray(speeds, dtype="d"))
