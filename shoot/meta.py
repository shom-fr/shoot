#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metadata utilities for finding variables and coordinates using CF conventions

This module provides wrappers around xoa functions to retrieve variables and
coordinates from xarray datasets using CF metadata conventions and custom specs.
"""
import os

import xoa.meta as xmeta
import xoa.coords as xcoords
import xoa.dyn as xdyn
import xoa.thermdyn as xthermdyn

# from .__init__ import ShootError, shoot_warn

CFG_DEFAULT = os.path.join(os.path.dirname(__file__), "meta.cfg")


def register_meta_specs(cfg_file=None):
    """Register custom CF specs with name 'shoot'

    Parameters
    ----------
    cfg_file : str, optional
        Path to config file. Defaults to meta.cfg in package directory.
    """
    if cfg_file is None:
        cfg_file = CFG_DEFAULT
    xmeta.register_meta_specs(shoot=cfg_file)


def set_meta_specs(name):
    """Set active CF specs by name

    Parameters
    ----------
    name : str
        Name of registered CF specs (e.g. "shoot", "croco").
    """
    xmeta.set_meta_specs(name)


def get_lon(obj, errors="raise"):
    """Get longitude coordinate from dataset or data array

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    xarray.DataArray
        Longitude coordinate.
    """
    return xcoords.get_lon(obj, errors=errors)


def get_lat(obj, errors="raise"):
    """Get latitude coordinate from dataset or data array

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    xarray.DataArray
        Latitude coordinate.
    """
    return xcoords.get_lat(obj, errors=errors)


def get_depth(obj, errors="ignore"):
    """Get depth coordinate from dataset or data array

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "ignore"

    Returns
    -------
    xarray.DataArray or None
        Depth coordinate if found.
    """
    return xcoords.get_depth(obj, errors=errors)


def get_time(obj, errors="ignore"):
    """Get time coordinate from dataset or data array

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "ignore"

    Returns
    -------
    xarray.DataArray or None
        Time coordinate if found.
    """
    return xcoords.get_time(obj, errors=errors)


def get_xdim(da, errors="raise"):
    """Get X dimension name from data array

    Parameters
    ----------
    da : xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    str
        Name of X dimension.
    """
    return xcoords.get_xdim(da, allow_positional=True, errors=errors)


def get_ydim(da, errors="raise"):
    """Get Y dimension name from data array

    Parameters
    ----------
    da : xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    str
        Name of Y dimension.
    """
    return xcoords.get_ydim(da, allow_positional=True, errors=errors)


def get_zdim(da, errors="raise"):
    """Get Z dimension name from data array

    Parameters
    ----------
    da : xarray.DataArray
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    str
        Name of Z dimension.
    """
    return xcoords.get_zdim(da, allow_positional=True, errors=errors)


def _get_data_var_(ds, name, errors):
    """Search for data variable using CF specs

    Parameters
    ----------
    ds : xarray.Dataset
    name : str
        Variable name to search for.
    errors : {"raise", "warn", "ignore"}

    Returns
    -------
    xarray.DataArray or None
        Found variable.
    """
    return xmeta.get_meta_specs(ds).search(ds, name, errors=errors)


def get_u(ds, errors="raise"):
    """Get zonal (X) velocity component

    Parameters
    ----------
    ds : xarray.Dataset
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    xarray.DataArray
        Zonal velocity.
    """
    return _get_data_var_(ds, "u", errors)


def get_v(ds, errors="raise"):
    """Get meridional (Y) velocity component

    Parameters
    ----------
    ds : xarray.Dataset
    errors : {"raise", "warn", "ignore"}, default "raise"

    Returns
    -------
    xarray.DataArray
        Meridional velocity.
    """
    return _get_data_var_(ds, "v", errors)


def get_ssh(ds, variant=["adt", "sla", "ssh", "mdt"], errors="warn"):
    """Get sea surface height variable

    Parameters
    ----------
    ds : xarray.Dataset
    variant : list of str, default ["adt", "sla", "ssh", "mdt"]
        Variable names to search for.
    errors : {"raise", "warn", "ignore"}, default "warn"

    Returns
    -------
    xarray.DataArray or None
        Sea surface height variable.
    """
    return xdyn.get_sea_level(ds, variant=variant, errors=errors)


def get_salt(ds, errors="warn"):
    """Get salinity variable

    Parameters
    ----------
    ds : xarray.Dataset
    errors : {"raise", "warn", "ignore"}, default "warn"

    Returns
    -------
    xarray.DataArray or None
        Salinity variable.
    """
    return xthermdyn.get_sal(ds, errors=errors)


def get_temp(ds, errors="warn"):
    """Get temperature variable

    Parameters
    ----------
    ds : xarray.Dataset
    errors : {"raise", "warn", "ignore"}, default "warn"

    Returns
    -------
    xarray.DataArray or None
        Temperature variable.
    """
    return xthermdyn.get_temp(ds, errors=errors)
