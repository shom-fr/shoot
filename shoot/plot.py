#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphic utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xoa.geo as xgeo

pmerc = ccrs.Mercator()
pcarr = ccrs.PlateCarree()


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


def create_map(
    lons,
    lats,
    margin=0.0,
    square=False,
    coastlines=True,
    emodnet=False,
    projection=pmerc,
    title=None,
    **kwargs,
):
    """Create a simple decorated cartopy map"""
    # lons, lats = lons.values, lats.values
    xmin, xmax = np.min(lons), np.max(lons)
    ymin, ymax = np.min(lats), np.max(lats)
    dx, dy = xmax - xmin, ymax - ymin
    x0, y0 = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    if square:
        aspect = dx / dy * np.cos(np.radians(y0))
        if aspect > 1:
            dy *= aspect
        else:
            dx /= aspect
    xmargin = margin * dx
    ymargin = margin * dy
    xmin = x0 - 0.5 * dx - xmargin
    xmax = x0 + 0.5 * dx + xmargin
    ymin = y0 - 0.5 * dy - ymargin
    ymax = y0 + 0.5 * dy + ymargin

    fig, ax = plt.subplots(1, subplot_kw=dict(projection=projection), **kwargs)
    ax.set_extent([xmin, xmax, ymin, ymax])
    ax.gridlines(
        draw_labels=["bottom", "left"],
        linewidth=1,
        color='k',
        alpha=0.5,
        linestyle='--',
        rotate_labels=False,
    )
    if coastlines:
        ax.coastlines()
    if emodnet:
        ax.add_wms("https://ows.emodnet-bathymetry.eu/wms", "emodnet:mean_atlas_land")
    if title:
        ax.set_title(title)
    return ax
