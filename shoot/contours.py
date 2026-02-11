#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contouring utilities

Functions for extracting and analyzing closed contours from 2D fields.
"""

import numpy as np
import contourpy as cpy
import scipy.ndimage as scin
import xarray as xr

from . import meta as smeta
from . import num as snum
from . import geo as sgeo


def get_closed_contours(lon_center, lat_center, ssh, nlevels=50, robust=0.03):
    """Extract closed contours enclosing a center point

    Parameters
    ----------
    lon_center : float
        Center longitude in degrees.
    lat_center : float
        Center latitude in degrees.
    ssh : xarray.DataArray
        2D field to contour (typically SSH).
    nlevels : int, default 50
        Maximum number of contour levels.
    robust : float, default 0.03
        Quantile threshold to exclude extreme values.

    Returns
    -------
    list of xarray.Dataset
        Each dataset contains a closed contour with coordinates and metadata.
    """
    lon = smeta.get_lon(ssh)
    lat = smeta.get_lat(ssh)
    lat2d, lon2d = xr.broadcast(lat, lon)

    cont_gen = cpy.contour_generator(z=ssh.values)
    vmin, vmax = np.nanquantile(ssh, [robust, 1 - robust])
    point = np.array([lon_center, lat_center])
    dss = []
    if len(np.arange(vmin, vmax + 0.005, 0.005)) < nlevels:
        ran = np.arange(vmin, vmax + 0.005, 0.005)
    else:
        ran = np.linspace(vmin, vmax, nlevels)
    for level in ran:
        for line in cont_gen.lines(level):
            if (line[0] == line[-1]).all():  # chek if it is closed contour
                xx = interp_to_line(lon2d.values, line)
                yy = interp_to_line(lat2d.values, line)
                if snum.points_in_polygon(
                    point, np.array([xx, yy]).T
                ):  # Check if it contains the center
                    if np.any(np.isnan(ssh)):  # Chek if it contains land points inside
                        nan_indexes = np.where(np.isnan(ssh))
                        nan_points = np.array(
                            [
                                [lon2d[i, j], lat2d[i, j]]
                                for i, j in zip(nan_indexes[0], nan_indexes[1])
                            ]
                        )
                        if np.any(snum.points_in_polygon(nan_points, np.array([xx, yy]).T)):
                            continue
                    dss.append(
                        xr.Dataset(
                            {"line": (("npts", "ncoords"), line)},
                            coords={
                                "lon": (
                                    "npts",
                                    xx,
                                    {"long_name": "Longitude"},
                                ),
                                "lat": ("npts", yy, {"long_name": "Latitude"}),
                            },
                            attrs={
                                "ssh": level,
                                "lon_center": lon_center,
                                "lat_center": lat_center,
                            },
                        )
                    )
    return dss


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


def add_contour_uv(ds, u, v):
    """Add velocity components and angular momentum to contour dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Contour dataset with 'line' variable.
    u : ndarray
        Zonal velocity field.
    v : ndarray
        Meridional velocity field.

    Returns
    -------
    xarray.Dataset
        Input dataset with added velocity and momentum variables.
    """
    if "u" not in ds:
        uc = interp_to_line(u, ds.line.values)
        vc = interp_to_line(v, ds.line.values)
        ds["u"] = ("npts", uc, {"long_name": "Velocity along X"})
        ds["v"] = ("npts", vc, {"long_name": "Velocity along Y"})
        xdist = sgeo.deg2m(ds.lon - ds.lon_center, ds.lat_center).values
        ydist = sgeo.deg2m(ds.lat - ds.lat_center).values
        am = xdist * vc - ydist * uc
        ds["am"] = (
            "npts",
            am,
            {"long_name": "Angular momentum", "units": "m2.s-2"},
        )
        ds.attrs["mean_velocity"] = float(np.sqrt(ds.u**2 + ds.v**2).mean())
        ds.attrs["mean_angular_momentum"] = float(ds.am.mean())
        ds.attrs["radius"] = float(np.mean(np.sqrt(xdist**2 + ydist**2)))
    return ds


def add_contour_dx_dy(ds):
    """Add spatial metrics to contour dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Contour dataset with lon/lat coordinates.

    Returns
    -------
    xarray.Dataset
        Input dataset with added dx, dy, and length attributes.
    """
    if "dx" not in ds:
        dx = sgeo.deg2m(np.gradient(ds.lon.values), ds.lat.values.mean())
        dy = sgeo.deg2m(np.gradient(ds.lat.values))
        ds["dx"] = ("npts", dx, {"units": "m"})
        ds["dy"] = ("npts", dy, {"units": "m"})
        ds.attrs["length"] = float(np.sqrt(dx**2 + dy**2).sum())
    return ds


class ContourMeanSpeedGetter:
    """Callable class to compute mean speed along a contour

    Parameters
    ----------
    u : ndarray
        Zonal velocity field.
    v : ndarray
        Meridional velocity field.
    """

    def __init__(self, u, v):
        """Initialize with velocity fields

        Parameters
        ----------
        u : ndarray
            Zonal velocity field.
        v : ndarray
            Meridional velocity field.
        """
        self.u, self.v = u, v

    def __call__(self, ds):
        """Compute mean speed along contour

        Parameters
        ----------
        ds : xarray.Dataset
            Contour dataset with 'line' variable.

        Returns
        -------
        float
            Mean speed along the contour.
        """
        add_contour_uv(ds.line.values, self.u, self.v)
        return float(np.sqrt(ds.u**2 + ds.v**2).mean())


def get_lnam_peaks(lnam, K=0.7):
    """Find local extrema of LNAM field within closed contours

    Parameters
    ----------
    lnam : xarray.DataArray
        2D LNAM (Local Normalized Angular Momentum) field.
    K : float, default 0.7
        Contour level threshold for detecting closed regions.

    Returns
    -------
    minima : ndarray
        Array of (i, j) indices for LNAM minima (anticyclones).
    maxima : ndarray
        Array of (i, j) indices for LNAM maxima (cyclones).
    Lines_coords : list
        List of contour line coordinates [lon, lat] for each closed contour.
    """
    # compute lines
    lon = smeta.get_lon(lnam)
    lat = smeta.get_lat(lnam)
    lat2d, lon2d = xr.broadcast(lat, lon)

    lon_name, lat_name = snum.get_coord_name(lnam)

    cont_gen = cpy.contour_generator(z=abs(lnam))
    lines = cont_gen.lines(K)
    maxima = np.empty((0, 2), dtype=np.int64)
    minima = np.empty((0, 2), dtype=np.int64)

    Lines_coords = []
    for i, line in enumerate(lines):
        if not (line[0] == line[-1]).all():  # skip not closed contours
            continue

        xx = interp_to_line(lon2d.values, line)
        yy = interp_to_line(lat2d.values, line)
        Lines_coords.append([xx, yy])

        # get extremun of the polygon
        lon_min = xx.min()
        lon_max = xx.max()
        lat_min = yy.min()
        lat_max = yy.max()

        # get nearest x,y values
        jmin = (abs(lat2d - lat_min) + abs(lon2d - lon_min)).argmin(lnam.dims)[
            lnam.dims[0]
        ].data - 1
        imin = (abs(lat2d - lat_min) + abs(lon2d - lon_min)).argmin(lnam.dims)[
            lnam.dims[1]
        ].data - 1

        jmax = (abs(lat2d - lat_max) + abs(lon2d - lon_max)).argmin(lnam.dims)[
            lnam.dims[0]
        ].data + 1
        imax = (abs(lat2d - lat_max) + abs(lon2d - lon_max)).argmin(lnam.dims)[
            lnam.dims[1]
        ].data + 1

        # compute max inside the polygon
        ijmax = (
            abs(lnam)
            .isel(
                {
                    lnam.dims[0]: slice(jmin, jmax),
                    lnam.dims[1]: slice(imin, imax),
                }
            )
            .argmax(lnam.dims)
        )
        jmax_in = ijmax[lnam.dims[0]].data  # lat
        imax_in = ijmax[lnam.dims[1]].data  # lon

        lat_center = abs(lnam).isel(
            {lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)}
        )[jmax_in, imax_in][lat_name]
        lon_center = abs(lnam).isel(
            {lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)}
        )[jmax_in, imax_in][lon_name]

        jcenter = (
            (abs(lat2d - lat_center) + abs(lon2d - lon_center)).argmin(lnam.dims)[lnam.dims[0]].data
        )
        icenter = (
            (abs(lat2d - lat_center) + abs(lon2d - lon_center)).argmin(lnam.dims)[lnam.dims[1]].data
        )

        if lnam[jcenter, icenter] > 0:
            maxima = np.append(maxima, np.array([[icenter, jcenter]]), axis=0)
        else:
            minima = np.append(minima, np.array([[icenter, jcenter]]), axis=0)

    return minima, maxima, Lines_coords


def area(ds):
    """Compute area enclosed by a contour

    Parameters
    ----------
    ds : xarray.Dataset
        Contour dataset with lon, lat coordinates and lon_center, lat_center attributes.

    Returns
    -------
    float
        Area enclosed by the contour in m².
    """
    xdist = sgeo.deg2m(ds.lon - ds.lon_center, ds.lat_center).values
    ydist = sgeo.deg2m(ds.lat - ds.lat_center).values

    xydist = np.sqrt(xdist**2 + ydist**2)
    xyavg = 0.5 * (xydist[:-1] + xydist[1:])

    dx = np.sqrt(
        sgeo.deg2m(ds.lon[:-1] - ds.lon[1:], ds.lat_center).values ** 2
        + sgeo.deg2m(ds.lat[:-1] - ds.lat[1:]).values ** 2
    )

    theta = np.arccos((-dx + xydist[1:] + xydist[:-1]) / (xydist[1:] + xydist[:-1]))
    return np.sum(xyavg * theta)
