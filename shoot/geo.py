#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic utilities
"""
import numpy as np

#: Earth radius in meters
EARTH_RADIUS = 6371e3


def deg2m(deg, lat=None, radius=EARTH_RADIUS):
    """Convert to meters a zonal or meridional distance in degrees

    Parameters
    ----------
    deg: float
        Longitude step
    lat: float
        Latitude for a zonal distance

    Return
    ------
    float
    """
    dd = deg * np.pi * radius / 180.0
    if lat is not None:
        dd *= np.cos(np.radians(lat))
    return dd


def m2deg(met, lat=None, radius=EARTH_RADIUS):
    """Convert to degrees a zonal or meridional distance in meters

    Parameters
    ----------
    met: float
        Longitude step
    lat: float
        Latitude for a zonal distance

    Return
    ------
    float
    """
    dd = met * 180 / (np.pi * radius)
    if lat is not None:
        dd /= np.cos(np.radians(lat))
    return dd
