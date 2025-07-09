#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for finding data_vars and coords using CF conventions
"""

import cf_xarray  # noqa

from .__init__ import ShootError, shoot_warn

STANDARD_NAMES = {
    "u": [
        "sea_water_x_velocity",
        "surface_sea_water_x_velocity",
        "eastward_sea_water_velocity",
        "geostrophic_eastward_sea_water_velocity",
        "surface_geostrophic_eastward_sea_water_velocity",
        "surface_geostrophic_sea_water_x_velocity",
        "surface_geostrophic_eastward_sea_water_velocity_assuming_mean_sea_level_for_geoid",
    ],
    "v": [
        "sea_water_y_velocity",
        "surface_sea_water_y_velocity",
        "northward_sea_water_velocity",
        "geostrophic_northward_sea_water_velocity",
        "surface_geostrophic_northward_sea_water_velocity",
        "surface_geostrophic_sea_water_y_velocity",
        "surface_geostrophic_northward_sea_water_velocity_assuming_mean_sea_level_for_geoid",
    ],
    "ssh": [
        "sea_surface_height_above_geoid",
        "sea_surface_height_above_geopotential_datum",
        "sea_surface_height_above_mean_sea_level",
        "sea_surface_height_above_reference_ellipsoid",
    ],
    "depth": ["depth", "depth_below_geoid", "altitude"],
}


def get_cf_item(container, targets, name=None, errors="raise"):
    """Search for a unique netcdf item"""
    if not isinstance(targets, list):
        targets = [targets]
    if name is None:
        name = targets[0]
    values = []
    for target in targets:
        try:
            values.append(container[target])
        except KeyError:
            continue
    if len(values) == 0:
        msg = f"{name} not found"
        if errors == "raise":
            raise ShootError(msg)
        if errors == "warn":
            shoot_warn(msg)
        return
    if len(values) > 1:
        msg = f"Multiple {name}s found"
        if errors == "raise":
            raise ShootError(msg)
        if errors == "warn":
            shoot_warn(msg)
    return values[0]


def get_lon(obj):
    """Get longitude coordinate data array"""
    return get_cf_item(obj.cf, "longitude")


def get_lat(obj):
    """Get latitude coordinate data array"""
    return get_cf_item(obj.cf, "latitude")


def get_depth(obj):
    """Get depth coodinate data array"""
    try:
        get_cf_item(obj.cf, "vertical")
    except ShootError:
        return _get_from_standard_names_(obj.cf, "depth")


def get_time(obj, errors="ignore"):
    """Get time coordinate data array"""
    return get_cf_item(obj.cf.coords, "time", errors=errors)


def get_xdim(da):
    """Get the x dimension name of a data array"""
    if "X" in da.cf.axes:
        return get_cf_item(da.cf.axes, "X", "x dimension")
    for dim in da.dims:
        if dim.lower().startswith("x"):
            return dim
    return da.dims[-1]


def get_ydim(da):
    """Get the y dimension name of a data array"""
    if "Y" in da.cf.axes:
        return get_cf_item(da.cf.axes, "Y", "y dimension")
    for dim in da.dims:
        if dim.lower().startswith("y"):
            return dim
    return da.dims[-2]


def get_zdim(da):
    """Get the z dimension name of a data array"""
    if "Z" in da.cf.axes:
        return get_cf_item(da, "Z", "z dimension")
    for dim in da.dims:
        dim = dim.lower()
        if (
            dim.startswith("z")
            or dim.startwith("s_")
            or dim.startswith("dep")
            or dim.startswith("lev")
        ):
            return dim
    return da.dims[-2] if da.ndim > 2 else da.dims[-1]


def _get_from_standard_names_(obj, target, errors):
    standard_names = STANDARD_NAMES[target]
    return get_cf_item(obj.cf, standard_names, f"{target} array", errors)


def get_u(ds, errors="raise"):
    """Get the velocity along X"""
    return _get_from_standard_names_(ds, "u", errors)


def get_v(ds, errors="raise"):
    """Get the velocity along Y"""
    return _get_from_standard_names_(ds, "v", errors)


def get_ssh(ds, errors="warn"):
    """Get the sea level"""
    return _get_from_standard_names_(ds, "ssh", errors)
