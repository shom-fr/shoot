#!/usr/bin/env python3
"""
Plotting utilities

Cartographic and visualization functions for oceanographic data.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import geo as sgeo


def __getattr__(name):
    """Lazy cartopy projections: :data:`pcarr` (PlateCarree) and :data:`pmerc` (Mercator)

    Cartopy is an optional dependency, only imported when a projection is needed.
    """
    if name in ("pcarr", "pmerc"):
        import cartopy.crs as ccrs

        globals()["pcarr"] = ccrs.PlateCarree()
        globals()["pmerc"] = ccrs.Mercator()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def plot_ellipse(lon, lat, a, b, angle, ax=None, npts=100, **kwargs):
    """Plot geographic ellipse

    Parameters
    ----------
    lon : float
        Center longitude in degrees.
    lat : float
        Center latitude in degrees.
    a : float
        Semi-major axis in kilometers.
    b : float
        Semi-minor axis in kilometers.
    angle : float
        Orientation angle in degrees.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Uses current axes if None.
    npts : int, default 100
        Number of points for ellipse outline.
    **kwargs
        Additional plot styling arguments.

    Returns
    -------
    list
        Matplotlib line objects.
    """
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
    lons = lon + sgeo.m2deg(am * ca * cas - bm * sa * sas, lat)
    lats = lat + sgeo.m2deg(am * sa * cas + bm * ca * sas)

    pe = ax.plot(lons, lats, **kwargs)
    return pe


def create_map(
    lons,
    lats,
    margin=0.0,
    square=False,
    coastlines=True,
    emodnet=False,
    projection=None,
    title=None,
    **kwargs,
):
    """Create cartographic map for oceanographic data

    Parameters
    ----------
    lons : array-like
        Longitude coordinates in degrees.
    lats : array-like
        Latitude coordinates in degrees.
    margin : float, default 0.0
        Fractional margin around data extent.
    square : bool, default False
        Force square aspect ratio.
    coastlines : bool, default True
        Draw coastlines.
    emodnet : bool, default False
        Use EMODnet bathymetry background.
    projection : cartopy.crs.Projection, optional
        Map projection. Defaults to :data:`pmerc` (Mercator).
    title : str, optional
        Map title.
    **kwargs
        Additional arguments for plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Cartographic axes.
    """
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

    if projection is None:
        projection = __getattr__("pmerc")
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
    return fig, ax
