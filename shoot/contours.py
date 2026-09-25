#!/usr/bin/env python3
"""
Contouring utilities

Functions for extracting and analyzing closed contours from 2D fields.
"""

import contourpy as cpy
import numpy as np
import xarray as xr

from . import geo as sgeo
from . import meta as smeta
from .core import contours as ccontours
from .core.contours import interp_to_line  # noqa: F401


def contour_to_dataset(level, line, lon2d, lat2d, lon_center, lat_center):
    """Convert a contour line in index space to a dataset with lon/lat coordinates

    Parameters
    ----------
    level : float
        Contour level.
    line : ndarray
        Contour line of fractional (i, j) indices of shape (n, 2).
    lon2d, lat2d : ndarray
        2D longitudes and latitudes of the contoured field.
    lon_center, lat_center : float
        Center position.

    Returns
    -------
    xarray.Dataset
    """
    return xr.Dataset(
        {"line": (("npts", "ncoords"), line)},
        coords={
            "lon": ("npts", ccontours.interp_to_line(lon2d, line), {"long_name": "Longitude"}),
            "lat": ("npts", ccontours.interp_to_line(lat2d, line), {"long_name": "Latitude"}),
        },
        attrs={
            "ssh": level,
            "lon_center": lon_center,
            "lat_center": lat_center,
        },
    )


def get_closed_contours(lon_center, lat_center, ssh, nlevels=50, robust=0.03):
    """Extract closed contours enclosing a center point

    The center is the grid point nearest to (`lon_center`, `lat_center`).
    See :func:`shoot.core.contours.find_closed_contours` for the numerical part.

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
        Each dataset contains a closed contour with coordinates and metadata,
        including lon/lat coordinates, SSH level, and center position.

    Example
    -------
    >>> from shoot.contours import get_closed_contours
    >>> contours = get_closed_contours(5.0, 43.0, ssh_field)  # doctest: +SKIP
    >>> len(contours)  # number of nested closed contours  # doctest: +SKIP
    """
    lat2d, lon2d = xr.broadcast(smeta.get_lat(ssh), smeta.get_lon(ssh))
    lon2d, lat2d = lon2d.values, lat2d.values
    ic, jc = ccontours.nearest_index(lon2d, lat2d, lon_center, lat_center)
    return [
        contour_to_dataset(level, line, lon2d, lat2d, lon_center, lat_center)
        for level, line in ccontours.find_closed_contours(ssh.values, ic, jc, nlevels=nlevels, robust=robust)
    ]


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
        uc, vc, am, mean_velocity, mean_am, radius = ccontours.contour_velocity(
            ds.line.values, ds.lon.values, ds.lat.values, u, v, ds.lon_center, ds.lat_center
        )
        ds["u"] = ("npts", uc, {"long_name": "Velocity along X"})
        ds["v"] = ("npts", vc, {"long_name": "Velocity along Y"})
        ds["am"] = ("npts", am, {"long_name": "Angular momentum", "units": "m2.s-2"})
        ds.attrs.update(mean_velocity=mean_velocity, mean_angular_momentum=mean_am, radius=radius)
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
        dx, dy, length = ccontours.contour_steps(ds.lon.values, ds.lat.values)
        ds["dx"] = ("npts", dx, {"units": "m"})
        ds["dy"] = ("npts", dy, {"units": "m"})
        ds.attrs["length"] = length
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
    lon_name, lat_name = lon.name, lat.name

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

        # get extremum of the polygon
        lon_min = xx.min()
        lon_max = xx.max()
        lat_min = yy.min()
        lat_max = yy.max()

        # get nearest x,y values
        jmin = (abs(lat2d - lat_min) + abs(lon2d - lon_min)).argmin(lnam.dims)[lnam.dims[0]].data - 1
        imin = (abs(lat2d - lat_min) + abs(lon2d - lon_min)).argmin(lnam.dims)[lnam.dims[1]].data - 1

        jmax = (abs(lat2d - lat_max) + abs(lon2d - lon_max)).argmin(lnam.dims)[lnam.dims[0]].data + 1
        imax = (abs(lat2d - lat_max) + abs(lon2d - lon_max)).argmin(lnam.dims)[lnam.dims[1]].data + 1

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

        lat_center = abs(lnam).isel({lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)})[
            jmax_in, imax_in
        ][lat_name]
        lon_center = abs(lnam).isel({lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)})[
            jmax_in, imax_in
        ][lon_name]

        jcenter = (abs(lat2d - lat_center) + abs(lon2d - lon_center)).argmin(lnam.dims)[lnam.dims[0]].data
        icenter = (abs(lat2d - lat_center) + abs(lon2d - lon_center)).argmin(lnam.dims)[lnam.dims[1]].data

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
