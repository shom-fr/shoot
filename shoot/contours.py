#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contouring utilities
"""
import numpy as np
import contourpy as cpy
import scipy.ndimage as scin
import xarray as xr
import xoa.coords as xcoords
import xoa.geo as xgeo

from . import num as snum
from . import fit as sfit


def get_closed_contours(lon_center, lat_center, ssh, nlevels=100, robust=0.03):
    """Closed contours around a center

    Parameters
    ----------
    lon_center: float
    lat_center: float
    ssh: xarray.DataArray

    Returns
    -------
    list(xarray.Dataset)
    """
    lon = xcoords.get_lon(ssh)
    lat = xcoords.get_lat(ssh)
    lat2d, lon2d = xr.broadcast(lat, lon)

    cont_gen = cpy.contour_generator(z=ssh.values)
    # lines = []
    # lons = []
    # lats = []
    vmin, vmax = np.quantile(ssh, [robust, 1 - robust])
    point = np.array([lon_center, lat_center])
    dss = []
    for level in np.linspace(vmin, vmax, nlevels):
        for line in cont_gen.lines(level):
            if (line[0] == line[-1]).all():
                xx = interp_to_line(lon2d.values, line)
                yy = interp_to_line(lat2d.values, line)
                if snum.points_in_polygon(point, np.array([xx, yy]).T):
                    # lines.append(line)
                    # lons.append(xx)
                    # lats.append(yy)
                    dss.append(
                        xr.Dataset(
                            {"line": (("npts", "ncoords"), line)},
                            coords={
                                "lon": ("npts", xx, {"long_name": "Longitude"}),
                                "lat": ("npts", yy, {"long_name": "Latitude"}),
                            },
                            attrs={"ssh": level, "lon_center": lon_center, "lat_center": lat_center},
                        )
                    )
    return dss

    # return lines, lons, lats
    # centers = np.array([l.mean(axis=0) for l in lines])
    # center = np.median(centers, axis=0)
    # cdists = np.sqrt([((c - center) ** 2).sum() for c in centers])
    # mdist = np.median(centers)
    # return [lines[i] for i, cdist in enumerate(cdists) if cdist / mdist < tol]


# class ContourLengthGetter:
#     def __init__(self, dx, dy):
#         self.dx, self.dy = dx, dy

#     def __call__(self, ds):
#         xdiff = xgeo.deg2m(np.diff(ds.lon.values), ds.lat.values.mean())
#         ydiff = xgeo.deg2m(np.diff(ds.lat.values))
#         return np.sqrt(xdiff**2 + ydiff**2).sum()


# def contour_length_getter(ds):
#     xdiff = xgeo.deg2m(np.diff(ds.lon.values), ds.lat.values.mean())
#     ydiff = xgeo.deg2m(np.diff(ds.lat.values))
#     return np.sqrt(xdiff**2 + ydiff**2).sum()


def interp_to_line(data, line):
    coords = line.T[::-1]
    mask = np.isnan(data).astype("d")
    dataf = np.nan_to_num(data)
    lm = scin.map_coordinates(mask, coords)
    ldata = scin.map_coordinates(dataf, coords)
    lbad = ~np.isclose(lm + 1, 1.0)
    ldata[lbad] = np.nan
    return ldata


def add_contour_uv(ds, u, v):
    if "u" not in ds:
        uc = interp_to_line(u, ds.line.values)
        vc = interp_to_line(v, ds.line.values)
        ds["u"] = ("npts", uc, {"long_name": "Velocity along X"})
        ds["v"] = ("npts", vc, {"long_name": "Velocity along Y"})
        xdist = xgeo.deg2m(ds.lon - ds.lon_center, ds.lat_center).values
        ydist = xgeo.deg2m(ds.lat - ds.lat_center).values
        am = xdist * vc - ydist * uc
        ds["am"] = ("npts", am, {"long_name": "Angular momentum", "units": "m2.s-2"})
        ds.attrs["mean_velocity"] = float(np.sqrt(ds.u**2 + ds.v**2).mean())
        ds.attrs["mean_angular_momentum"] = float(ds.am.mean())
    return ds


def add_contour_dx_dy(ds):
    if "dx" not in ds:
        dx = xgeo.deg2m(np.gradient(ds.lon.values), ds.lat.values.mean())
        dy = xgeo.deg2m(np.gradient(ds.lat.values))
        ds["dx"] = ("npts", dx, {"units": "m"})
        ds["dy"] = ("npts", dy, {"units": "m"})
        ds.attrs["length"] = float(np.sqrt(dx**2 + dy**2).sum())
    return ds


class ContourMeanSpeedGetter:
    def __init__(self, u, v):
        self.u, self.v = u, v

    def __call__(self, ds):
        add_contour_uv(ds.line.values, self.u, self.v)
        return float(np.sqrt(ds.u**2 + ds.v**2).mean())
