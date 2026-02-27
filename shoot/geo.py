#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic utilities
"""

import numpy as np

#: Earth radius in meters
EARTH_RADIUS = 6371e3


def deg2m(deg, lat=None, radius=EARTH_RADIUS):
    """Convert angular distance in degrees to meters

    Parameters
    ----------
    deg : float or array-like
        Angular distance in degrees.
    lat : float or array-like, optional
        Reference latitude for zonal distance. If None, meridional distance.
    radius : float, default EARTH_RADIUS
        Planet radius in meters.

    Returns
    -------
    float or array-like
        Distance in meters.

    Example
    -------
    >>> from shoot.geo import deg2m
    >>> deg2m(1)  # 1 degree of latitude in meters
    111194.92...
    >>> deg2m(1, lat=45)  # 1 degree of longitude at 45°N
    78626.18...
    """
    dd = deg * np.pi * radius / 180.0
    if lat is not None:
        dd *= np.cos(np.radians(lat))
    return dd


def m2deg(met, lat=None, radius=EARTH_RADIUS):
    """Convert distance in meters to angular distance in degrees

    Parameters
    ----------
    met : float or array-like
        Distance in meters.
    lat : float or array-like, optional
        Reference latitude for zonal distance. If None, meridional distance.
    radius : float, default EARTH_RADIUS
        Planet radius in meters.

    Returns
    -------
    float or array-like
        Angular distance in degrees.

    Example
    -------
    >>> from shoot.geo import m2deg
    >>> m2deg(111195)  # ~111 km in degrees of latitude
    1.000...
    >>> m2deg(111195, lat=45)  # ~111 km in degrees of longitude at 45°N
    1.414...
    """
    dd = met * 180 / (np.pi * radius)
    if lat is not None:
        dd /= np.cos(np.radians(lat))
    return dd
