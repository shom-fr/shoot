#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphic utilities
"""
import numpy as np
import matplotlib.pyplot as plt

import xoa.geo as xgeo


def plot_ellipse(lon, lat, a, b, angle, ax=None, npts=100, **kwargs):
    if ax is None:
        ax = plt.gca()

    theta = np.radians(angle)
    ca = np.cos(theta)
    sa = np.sin(theta)

    angles = np.linspace(0, 360.0, npts)
    alphas = np.radians(angles)
    cas = np.cos(alphas)
    sas = np.sin(alphas)

    am = a * 1e3
    bm = b * 1e3
    lons = lon + xgeo.m2deg(am * ca * cas - bm * sa * sas, lat)
    lats = lat + xgeo.m2deg(am * sa * cas + bm * ca * sas)

    pe = ax.plot(lons, lats, **kwargs)
    return pe
