#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamics
"""
import math
import numpy as np
import numba
import xarray as xr

import xoa.coords as xcoords

from . import grid as sgrid

GRAVITY = 9.81
OMEGA = 2 * np.pi / 86400


@numba.guvectorize(
    [(numba.float64[:, :], numba.float64[:, :], numba.int64, numba.float64, numba.float64[:, :])],
    "(ny,nx),(ny,nx),(),()->(ny,nx)",
)
def _get_lnam_(uu, vv, wx, dx2dy, lnam):
    ny, nx = uu.shape
    mask = np.isnan(uu) | np.isnan(vv)
    wx2 = wx // 2
    wy = (int(np.ceil(wx / dx2dy)) // 2) * 2 + 1
    wy2 = wy // 2
    lnam[:, :] = np.nan
    for j in numba.prange(wy2, ny - wy2 - 1):
        for i in range(wx2, nx - wx2 - 1):
            if mask[j - wy2 : j + wy2 + 1, i - wx2 : i + wx2 + 1].any():
                continue
            # if mask[j, i]:
            #     continue
            # xc = xx[j, i]
            # yc = yy[j, i]
            # print(j, i)
            denom1 = 0.0
            denom2 = 0.0
            # count = 0.0
            lnam[j, i] = 0.0
            for jl in range(-wy2, wy2 + 1):
                for il in range(-wx2, wx2 + 1):
                    # if mask[j + jl, i + il]:
                    #     continue
                    lnam[j, i] += il * vv[j + jl, i + il]
                    lnam[j, i] -= jl * uu[j + jl, i + il] * dx2dy
                    denom1 += il * uu[j + jl, i + il]
                    denom1 += jl * vv[j + jl, i + il] * dx2dy
                    denom2 += math.sqrt(
                        uu[j + jl, i + il] ** 2 + vv[j + jl, i + il] ** 2
                    ) * math.sqrt(il**2 + (jl * dx2dy) ** 2)
                    # count += 1.0
            if (denom1 + denom2) > 1e-6:
                # print(lnam[j, i], denom1, denom2)
                lnam[j, i] /= denom1 + denom2


def _get_lnam_wrapper_(uu, vv, wx, dx2dy):
    uu = uu.astype("d")
    vv = vv.astype("d")
    wx = int(wx)
    dx2dy = float(dx2dy)
    return _get_lnam_(uu, vv, wx, dx2dy)


def get_lnam(u, v, window, dx=None, dy=None):
    """Get the local normalized angular momentum

    Parameters
    ----------
    u: xarray.Dataset
        Velocity along X
    v: xarray.Dataset
        Velocity along X
    window: float
        Window in km
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Return
    ------
    xarray.DataArray
        Local normalized angular momentum
    """
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    dxm = np.nanmean(dx)
    dym = np.nanmean(dy)
    wx = (int(np.ceil(window * 1e3 / dxm)) // 2) * 2 + 1
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")

    lnam = xr.apply_ufunc(
        _get_lnam_wrapper_,
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
    """Get the divergence of the flow

    Parameters
    ----------
    u: xarray.Dataset
        Velocity along X
    v: xarray.Dataset
        Velocity along X
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Return
    ------
    xarray.DataArray
        divergence
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
        _get_div_,
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


def _get_div_(u, v, dx, dy):
    sx = np.gradient(u, axis=-1) / dx
    sy = np.gradient(v, axis=-2) / dy
    div = sx + sy
    div[np.isnan(u) | np.isnan(v)] = np.nan
    return div


def get_okuboweiss(u, v, dx=None, dy=None):
    """Get the Okubo-Weiss parameter
    Parameters
    ----------
    u: xarray.Dataset
        Velocity along X
    v: xarray.Dataset
        Velocity along X
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Return
    ------
    xarray.DataArray
        Okubo-Weiss parameters
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
        _get_okuboweiss_,
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


def _get_okuboweiss_(u, v, dx, dy):
    sn = np.gradient(u, axis=-1) / dx - np.gradient(v, axis=-2) / dy
    ss = np.gradient(v, axis=-1) / dx + np.gradient(u, axis=-2) / dy
    om = np.gradient(v, axis=-1) / dx - np.gradient(u, axis=-2) / dy
    ow = sn**2 + ss**2 - om**2
    ow[np.isnan(u) | np.isnan(v)] = np.nan
    return ow


def get_relvort(u, v, dx=None, dy=None):
    """Get the relative vorticity

    Parameters
    ----------
    u: xarray.Dataset
        Velocity along X
    v: xarray.Dataset
        Velocity along X
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Return
    ------
    xarray.DataArray
        Relative vorticity
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
        _get_relvort_,
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


def _get_relvort_(u, v, dx, dy):
    rv = np.gradient(v, axis=-1) / dx
    rv -= np.gradient(u, axis=-2) / dy
    rv[np.isnan(u) | np.isnan(v)] = np.nan
    return rv


def get_coriolis(lat):
    """Get the coriolis parameter of a dataarray"""
    return 2 * OMEGA * np.sin(np.radians(lat))


def get_geos_old(ssh, dx=None, dy=None):
    """Get the geostrophic current from SSH (old version)"""
    dx, dy = sgrid.get_dx_dy(ssh, dx=dx, dy=dy)
    dims = list(ssh.dims)
    xaxis = dims.index(xcoords.get_xdim(ssh, errors="raise"))
    yaxis = dims.index(xcoords.get_ydim(ssh, errors="raise"))
    dhdx = np.gradient(ssh.values, axis=xaxis) / dx
    dhdy = np.gradient(ssh.values, axis=yaxis) / dy
    # dhdx, dhdy = deriv2d(ssh)
    corio = get_coriolis(xcoords.get_lat(ssh))
    u = xr.DataArray(-GRAVITY * dhdy, dims=ssh.dims, coords=ssh.coords) / corio
    v = xr.DataArray(GRAVITY * dhdx, dims=ssh.dims, coords=ssh.coords) / corio
    return u, v


def get_geos(ssh, dx=None, dy=None):
    """Get the geostrophic current from SSH

    Parameters
    ----------
    ssh: xarray.Dataset
        Sea surface height
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Return
    ------
    xarray.DataArray, xarray.DataArray: U, V
        X and Y geostrophic velocities
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
        _get_geos_,
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


def _get_geos_(ssh, dx, dy, corio):
    if corio.ndim == 1:
        corio = corio.reshape(corio.shape[0], 1)
    dhdx = np.gradient(ssh, axis=-1) / (dx * corio)
    dhdy = np.gradient(ssh, axis=-2) / (dy * corio)
    bad = np.isnan(ssh)
    dhdx[bad] = np.nan
    dhdy[bad] = np.nan
    return -dhdy * GRAVITY, dhdx * GRAVITY
