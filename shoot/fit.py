#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization and fitting routines

Functions for fitting geometric shapes to spatial data.
"""
import numpy as np
from scipy.optimize import least_squares

from . import geo as sgeo


GRAVITY = 9.81
OMEGA = 2 * np.pi / 86400

# %%
# Ellipse Mean Square fit
# -----------------------


def _residuals(params, points):
    """Compute residuals between points and ellipse

    Parameters
    ----------
    params : array-like
        Ellipse parameters [xc, yc, a, b, theta].
    points : ndarray
        Point coordinates of shape (n, 2).

    Returns
    -------
    ndarray
        Normalized distances minus 1.
    """
    xc, yc, a, b, theta = params
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shifted = points - [xc, yc]
    rotated = shifted @ R
    normed = rotated / [a, b]
    distances = np.linalg.norm(normed, axis=1)
    return distances - 1


def fit_ellipse_from_coords(lons, lats, get_fit=False):
    """Fit ellipse to geographic coordinates

    Uses least-squares optimization to fit an ellipse to a set of points
    given in geographic coordinates.

    Parameters
    ----------
    lons : array-like
        Longitude coordinates in degrees.
    lats : array-like
        Latitude coordinates in degrees.
    get_fit : bool, default False
        If True, return fit error along with parameters.

    Returns
    -------
    dict or tuple
        Dictionary with keys:
        - lon : Center longitude in degrees
        - lat : Center latitude in degrees
        - a : Semi-major axis in kilometers
        - b : Semi-minor axis in kilometers
        - angle : Orientation angle in degrees

        If get_fit=True, returns (dict, error) tuple.
    """

    lons = np.array(lons)
    lats = np.array(lats)

    lon0 = lons.mean()
    lat0 = lats.mean()

    x = sgeo.deg2m(lons - lon0, lat0)
    y = sgeo.deg2m(lats - lat0)

    points = np.column_stack((x, y))

    # 4. Estimation initiale simple
    xc0, yc0 = np.mean(points, axis=0)
    a0, b0 = (np.ptp(points, axis=0)) / 2
    theta0 = 0.0
    x0 = [xc0, yc0, a0, b0, theta0]

    # 5. Fit avec least_squares
    result = least_squares(_residuals, x0, args=(points,))
    xc, yc, a, b, theta = result.x
    error = np.mean(result.fun**2)

    lat = lat0 + sgeo.m2deg(yc)
    lon = lon0 + sgeo.m2deg(xc, lat=lat0)

    if b > a:
        a, b = b, a
        theta += np.pi / 2

    out = dict(lon=lon, lat=lat, a=a / 1e3, b=b / 1e3, angle=np.degrees(theta))
    if get_fit:
        out = out, error
    return out
