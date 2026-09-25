#!/usr/bin/env python3
"""
Ocean dynamics utilities

Functions for computing kinematic quantities from velocity fields including
vorticity, divergence, angular momentum, and geostrophic currents.
"""

import numpy as np
import xarray as xr
import xoa.coords as xcoords

from . import grid as sgrid
from .core import dyn as cdyn
from .core.dyn import GRAVITY, OMEGA  # noqa: F401


def get_lnam(u, v, window, dx=None, dy=None):
    """Compute local normalized angular momentum

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component.
    v : xarray.DataArray
        Meridional velocity component.
    window : float
        Window size in kilometers.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    xarray.DataArray
        Local normalized angular momentum.
    """
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    dxm = np.nanmean(dx)
    dym = np.nanmean(dy)
    wx = cdyn.get_window_size(window, dxm)
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")

    lnam = xr.apply_ufunc(
        cdyn.lnam,
        u,
        v,
        input_core_dims=[[ydim, xdim], [ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        dask="parallelized",
        kwargs={"wx": wx, "dx2dy": float(dym / dxm)},
        vectorize=False,
    )
    return lnam.transpose(*u.dims)


def get_div(u, v, dx=None, dy=None):
    """Compute horizontal divergence

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component.
    v : xarray.DataArray
        Meridional velocity component.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    xarray.DataArray
        Horizontal divergence in s^-1.
    """
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")
    input_core_dims = [[ydim, xdim], [ydim, xdim]]
    if np.shape(dx) == 0:
        input_core_dims.extend([[], []])
    else:
        input_core_dims.extend([[ydim, xdim], [ydim, xdim]])
    div = xr.apply_ufunc(
        cdyn.div,
        u,
        v,
        dx,
        dy,
        join="override",
        input_core_dims=input_core_dims,
        output_core_dims=[[ydim, xdim]],
        dask="parallelized",
    )
    div = div.transpose(*u.dims)
    return div


def get_okuboweiss(u, v, dx=None, dy=None):
    """Compute Okubo-Weiss parameter

    The Okubo-Weiss parameter distinguishes vortex-dominated (OW < 0)
    from strain-dominated (OW > 0) regions.

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component.
    v : xarray.DataArray
        Meridional velocity component.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    xarray.DataArray
        Okubo-Weiss parameter in s^-2.
    """
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")
    input_core_dims = [[ydim, xdim], [ydim, xdim]]
    if np.shape(dx) == 0:
        input_core_dims.extend([[], []])
    else:
        input_core_dims.extend([[ydim, xdim], [ydim, xdim]])
    ow = xr.apply_ufunc(
        cdyn.okuboweiss,
        u,
        v,
        dx,
        dy,
        join="override",
        input_core_dims=input_core_dims,
        output_core_dims=[[ydim, xdim]],
        dask="allowed",  # "allowed",  # "parallelized",
        output_dtypes=[u.dtype],
        # dask_gufunc_kwargs={"meta": np.ones((1))},
    )
    ow = ow.transpose(*u.dims)
    return ow


def get_relvort(u, v, dx=None, dy=None):
    """Compute relative vorticity

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component.
    v : xarray.DataArray
        Meridional velocity component.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    xarray.DataArray
        Relative vorticity in s^-1.

    Example
    -------
    >>> import xarray as xr
    >>> import numpy as np
    >>> lon = xr.DataArray(np.linspace(0, 1, 50), dims="lon",
    ...     attrs={"standard_name": "longitude"})
    >>> lat = xr.DataArray(np.linspace(43, 44, 40), dims="lat",
    ...     attrs={"standard_name": "latitude"})
    >>> u = xr.DataArray(np.random.rand(40, 50), dims=("lat", "lon"),
    ...     coords={"lon": lon, "lat": lat})
    >>> v = xr.DataArray(np.random.rand(40, 50), dims=("lat", "lon"),
    ...     coords={"lon": lon, "lat": lat})
    >>> rv = get_relvort(u, v)  # doctest: +SKIP
    """
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")
    input_core_dims = [[ydim, xdim], [ydim, xdim]]
    if np.shape(dx) == 0:
        input_core_dims.extend([[], []])
    else:
        input_core_dims.extend([[ydim, xdim], [ydim, xdim]])
    rv = xr.apply_ufunc(
        cdyn.relvort,
        u,
        v,
        dx,
        dy,
        input_core_dims=input_core_dims,
        output_core_dims=[[ydim, xdim]],
        join="inner",
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    rv = rv.transpose(*u.dims)
    return rv


def get_coriolis(lat):
    """Compute Coriolis parameter

    Parameters
    ----------
    lat : float or array-like
        Latitude in degrees.

    Returns
    -------
    float or array-like
        Coriolis parameter (f = 2Ω sin(lat)) in s^-1.
    """
    return cdyn.coriolis(lat)


def get_geos_old(ssh, dx=None, dy=None):
    """Compute geostrophic currents from SSH (deprecated)

    Parameters
    ----------
    ssh : xarray.DataArray
        Sea surface height.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    u : xarray.DataArray
        Zonal geostrophic velocity.
    v : xarray.DataArray
        Meridional geostrophic velocity.
    """
    dx, dy = sgrid.get_dx_dy(ssh, dx=dx, dy=dy)
    dims = list(ssh.dims)
    xaxis = dims.index(xcoords.get_xdim(ssh, errors="raise"))
    yaxis = dims.index(xcoords.get_ydim(ssh, errors="raise"))
    dhdx = np.gradient(ssh.values, axis=xaxis) / dx
    dhdy = np.gradient(ssh.values, axis=yaxis) / dy
    corio = get_coriolis(xcoords.get_lat(ssh))
    u = xr.DataArray(-GRAVITY * dhdy, dims=ssh.dims, coords=ssh.coords) / corio
    v = xr.DataArray(GRAVITY * dhdx, dims=ssh.dims, coords=ssh.coords) / corio
    return u, v


def get_geos(ssh, dx=None, dy=None):
    """Compute geostrophic currents from SSH

    Uses geostrophic balance: f×u_g = -g∇η

    Parameters
    ----------
    ssh : xarray.DataArray
        Sea surface height in meters.
    dx : xarray.DataArray, optional
        Grid resolution along X in meters.
    dy : xarray.DataArray, optional
        Grid resolution along Y in meters.

    Returns
    -------
    u : xarray.DataArray
        Zonal geostrophic velocity in m/s.
    v : xarray.DataArray
        Meridional geostrophic velocity in m/s.

    Example
    -------
    >>> import xarray as xr, numpy as np
    >>> from shoot.dyn import get_geos
    >>> lon = xr.DataArray(np.linspace(0, 2, 50), dims="lon",
    ...     attrs={"standard_name": "longitude"})
    >>> lat = xr.DataArray(np.linspace(43, 44, 40), dims="lat",
    ...     attrs={"standard_name": "latitude"})
    >>> ssh = xr.DataArray(np.random.rand(40, 50) * 0.1, dims=("lat", "lon"),
    ...     coords={"lon": lon, "lat": lat})
    >>> u, v = get_geos(ssh)  # doctest: +SKIP
    """
    dx, dy = sgrid.get_dx_dy(ssh, dx=dx, dy=dy)
    xdim = xcoords.get_xdim(ssh, errors="raise")
    ydim = xcoords.get_ydim(ssh, errors="raise")
    corio = get_coriolis(xcoords.get_lat(ssh))
    input_core_dims = [[ydim, xdim]]
    if np.shape(dx) == 0:
        input_core_dims.extend([[], []])
    else:
        input_core_dims.extend([[ydim, xdim], [ydim, xdim]])
    if corio.ndim == 1:
        input_core_dims.append([ydim])
    else:
        input_core_dims.append([ydim, xdim])
    ugeos, vgeos = xr.apply_ufunc(
        cdyn.geos,
        ssh,
        dx,
        dy,
        corio,
        input_core_dims=input_core_dims,
        output_core_dims=[[ydim, xdim], [ydim, xdim]],
        join="inner",
        dask="allowed",  # "allowed",  # "parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        # output_dtypes=[ssh.dtype, ssh.dtype],
    )
    return ugeos.transpose(*ssh.dims), vgeos.transpose(*ssh.dims)
