#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contouring utilities
"""

import numpy as np
import contourpy as cpy
import scipy.ndimage as scin
import xarray as xr

from . import cf as scf
from . import num as snum
from . import geo as sgeo


def get_closed_contours(lon_center, lat_center, ssh, nlevels=50, robust=0.03):
    """Get closed contours around a center
    Parameters
    ----------
    lon_center: float
    lat_center: float
    ssh: xarray.DataArray
    nlevels: maximum number of values for contour (optional)
    robust: percentage quantile to avoid looking at extreme ssh values

    Returns
    -------
    list(xarray.Dataset)
    """
    lon = scf.get_lon(ssh)
    lat = scf.get_lat(ssh)
    lat2d, lon2d = xr.broadcast(lat, lon)

    cont_gen = cpy.contour_generator(z=ssh.values)
    # lines = []
    # lons = []
    # lats = []
    vmin, vmax = np.nanquantile(ssh, [robust, 1 - robust])
    point = np.array([lon_center, lat_center])
    dss = []
    # for level in np.linspace(vmin, vmax, nlevels):
    if len(np.arange(vmin, vmax + 0.005, 0.005)) < nlevels:
        ran = np.arange(vmin, vmax + 0.005, 0.005)
    else:
        ran = np.linspace(vmin, vmax, nlevels)
    for level in ran:  # parcoure tous les demi-centimètre
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
                                "lon": ("npts", xx, {"long_name": "Longitude"}),
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
    """Interpolate a 2D field to series of relative coordinates"""
    coords = line.T[::-1]
    mask = np.isnan(data).astype("d")
    dataf = np.nan_to_num(data)
    lm = scin.map_coordinates(mask, coords)
    ldata = scin.map_coordinates(dataf, coords)
    lbad = ~np.isclose(lm + 1, 1.0)
    ldata[lbad] = np.nan
    return ldata


def add_contour_uv(ds, u, v):
    """Interpolate and add u and v along contours"""
    if "u" not in ds:
        uc = interp_to_line(u, ds.line.values)
        vc = interp_to_line(v, ds.line.values)
        ds["u"] = ("npts", uc, {"long_name": "Velocity along X"})
        ds["v"] = ("npts", vc, {"long_name": "Velocity along Y"})
        xdist = sgeo.deg2m(ds.lon - ds.lon_center, ds.lat_center).values
        ydist = sgeo.deg2m(ds.lat - ds.lat_center).values
        am = xdist * vc - ydist * uc
        ds["am"] = ("npts", am, {"long_name": "Angular momentum", "units": "m2.s-2"})
        ds.attrs["mean_velocity"] = float(np.sqrt(ds.u**2 + ds.v**2).mean())
        ds.attrs["mean_angular_momentum"] = float(ds.am.mean())
        ds.attrs["radius"] = float(np.mean(np.sqrt(xdist**2 + ydist**2)))
    return ds


def add_contour_dx_dy(ds):
    """Add x and y metrics to contours"""
    if "dx" not in ds:
        dx = sgeo.deg2m(np.gradient(ds.lon.values), ds.lat.values.mean())
        dy = sgeo.deg2m(np.gradient(ds.lat.values))
        ds["dx"] = ("npts", dx, {"units": "m"})
        ds["dy"] = ("npts", dy, {"units": "m"})
        ds.attrs["length"] = float(np.sqrt(dx**2 + dy**2).sum())
        # radius based on the area
        # a = area(ds)
        # ds['area'] = a

    return ds


class ContourMeanSpeedGetter:
    def __init__(self, u, v):
        self.u, self.v = u, v

    def __call__(self, ds):
        add_contour_uv(ds.line.values, self.u, self.v)
        return float(np.sqrt(ds.u**2 + ds.v**2).mean())


def get_lnam_peaks(lnam, K=0.7):
    # compute lines
    lon = scf.get_lon(lnam)
    lat = scf.get_lat(lnam)
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
        # ijmax = abs(lnam).isel({lnam.dims[0]:slice(lat_min,lat_max), lnam.dims[1]:slice(lon_min,lon_max)}).argmax(lnam.dims)
        ijmax = (
            abs(lnam)
            .isel({lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)})
            .argmax(lnam.dims)
        )
        jmax_in = ijmax[lnam.dims[0]].data  # lat
        imax_in = ijmax[lnam.dims[1]].data  # lon

        # lat_center = abs(lnam).sel({lnam.dims[0]:slice(lat_min,lat_max), lnam.dims[1]:slice(lon_min,lon_max)}).latitude[jmax_in]
        # lon_center = abs(lnam).sel({lnam.dims[0]:slice(lat_min,lat_max), lnam.dims[1]:slice(lon_min,lon_max)}).longitude[imax_in]
        lat_center = abs(lnam).isel(
            {lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)}
        )[jmax_in, imax_in][lat_name]
        lon_center = abs(lnam).isel(
            {lnam.dims[0]: slice(jmin, jmax), lnam.dims[1]: slice(imin, imax)}
        )[jmax_in, imax_in][lon_name]
        # assert(snum.points_in_polygon([lon_center, lat_center], np.array([xx, yy]).T))

        # jcenter = lnam.indexes[lat_name].get_loc(float(lat_center.data))
        # icenter = lnam.indexes[lon_name].get_loc(float(lon_center.data))

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
