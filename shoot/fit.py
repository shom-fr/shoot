# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization routines
"""
import numpy as np
import scipy.optimize as scio
import xoa.coords as xcoords
import xoa.cf as xcf
import xoa.geo as xgeo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from . import grid as sgrid


GRAVITY = 9.81
OMEGA = 2 * np.pi / 86400
EARTH_RADIUS = 6371e3

# %% Ellipse avec Méthode mean square


# Fonction de distance à l’ellipse
def _residuals(params, points):
    xc, yc, a, b, theta = params
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shifted = points - [xc, yc]
    rotated = shifted @ R
    normed = rotated / [a, b]
    distances = np.linalg.norm(normed, axis=1)
    return distances - 1  # on veut des distances = 1 pour les points sur l’ellipse


def fit_ellipse_from_coords(lons, lats, get_fit=False):
    """Fit an allipse to a contour line
    Parameters
    ----------
    lons: array(npts)
        Longitudes in degrees
    lats: array(npts)
        Latitudes in degrees

    Returns
    -------
    dict:
        lon: center lon in degrees
        lat: center lat in degrees
        a: semi-major axis in km
        b: semi-minor axis in km
        angle: angle in degrees
    """

    lons = np.array(lons)
    lats = np.array(lats)

    lon0 = lons.mean()
    lat0 = lats.mean()

    x = xgeo.deg2m(lons - lon0, lat0)
    y = xgeo.deg2m(lats - lat0)

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

    lat = lat0 + xgeo.m2deg(yc)
    lon = lon0 + xgeo.m2deg(xc, lat=lat0)

    if b > a:
        a, b = b, a
        theta += np.pi / 2

    out = dict(lon=lon, lat=lat, a=a / 1e3, b=b / 1e3, angle=np.degrees(theta))
    if get_fit:
        out = out, error
    return out
