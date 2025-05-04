# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
acoustic functions
"""
import functools
import math
import numpy as np
from scipy.signal import argrelmax, argrelmin
import numba
import xarray as xr
import gsw
import xoa.coords as xcoords
from . import grid as sgrid
from . import num as snum


## TO BE IMPLEMENTED
def _ilmax(profile):
    return argrelmax(profile)[0]


def _ilmin(profile):
    try:
        return argrelmin(profile)[0]
    except IndexError:
        return np.nan


def _ecs(profile, depth):
    ilmaxs = _ilmax(profile)
    try:
        # return profile.depth[ilmaxs[-1]]
        return depth[ilmaxs[-1]] if profile[ilmaxs[-1]] > profile[-1] else 0
    except IndexError:
        return np.nan


def _iminc(profile, depth):
    pos_iminc = None
    ilmaxs = _ilmax(profile)
    ilmins = _ilmin(profile)
    if len(ilmaxs) > 1:
        maxd = ilmaxs[-2]
        maxs = ilmaxs[-1]
        for l in ilmins:
            if (l > maxd) and (l < maxs):
                pos_iminc = l
                break
    if pos_iminc:
        # return profile.depth.isel(s_rho=pos_iminc)
        return depth[pos_iminc]
    else:
        return np.nan


def _mcp(profile, depth):
    ilmins = _ilmin(profile)
    try:
        return depth[ilmins[0]]
        # return profile.depth.isel(s_rho=ilmins[0])
    except IndexError:
        return np.nan


## ECS
def _get_ecs_(cs, depth, ecs, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            ecs[j, i] = _ecs(cs[:, j, i], depth[:, j, i])
            # ecs[j, i] = _ecs(cs.isel({xdim: i, ydim: j}))
    return ecs


def _get_ecs_wrapper_(cs, depth, xdim, ydim, nx, ny):
    return _get_ecs_(cs, depth, np.empty([ny, nx]), xdim, ydim, nx, ny)


def get_ecs(cs):
    """Get the local ecs

    Parameters
    ----------
    cs: xarray.Dataset
        sound speed
    Return
    ------
    xarray.DataArray
        Epaisseur du chenal de surface
    """
    xdim = xcoords.get_xdim(cs, errors="raise")
    ydim = xcoords.get_ydim(cs, errors="raise")
    zdim = xcoords.get_zdim(cs, errors="raise")
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = xcoords.get_depth(cs)
    ecs = xr.apply_ufunc(
        _get_ecs_wrapper_,
        cs,
        depth,
        input_core_dims=[[zdim, ydim, xdim], [zdim, ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        dask="parallelized",
        vectorize=False,
        kwargs={"xdim": xdim, "ydim": ydim, "nx": nx, "ny": ny},
    )
    return ecs  # .transpose(*cs.dims)


## MCP
def _get_mcp_(cs, depth, mcp, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            mcp[j, i] = _mcp(cs[:, j, i], depth[:, j, i])
            # ecs[j, i] = _ecs(cs.isel({xdim: i, ydim: j}))
    return mcp


def _get_mcp_wrapper_(cs, depth, xdim, ydim, nx, ny):
    return _get_mcp_(cs, depth, np.empty([ny, nx]), xdim, ydim, nx, ny)


def get_mcp(cs):
    """Get the local ecs

    Parameters
    ----------
    cs: xarray.Dataset
        sound speed
    Return
    ------
    xarray.DataArray
        Minimum de célérité profond
    """
    xdim = xcoords.get_xdim(cs, errors="raise")
    ydim = xcoords.get_ydim(cs, errors="raise")
    zdim = xcoords.get_zdim(cs, errors="raise")
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = xcoords.get_depth(cs)
    mcp = xr.apply_ufunc(
        _get_mcp_wrapper_,
        cs,
        depth,
        input_core_dims=[[zdim, ydim, xdim], [zdim, ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        dask="parallelized",
        vectorize=False,
        kwargs={"xdim": xdim, "ydim": ydim, "nx": nx, "ny": ny},
    )
    return mcp  # .transpose(*cs.dims)


# IMINC
## MCP
def _get_iminc_(cs, depth, iminc, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            iminc[j, i] = _iminc(cs[:, j, i], depth[:, j, i])
            # ecs[j, i] = _ecs(cs.isel({xdim: i, ydim: j}))
    return iminc


def _get_iminc_wrapper_(cs, depth, xdim, ydim, nx, ny):
    return _get_iminc_(cs, depth, np.empty([ny, nx]), xdim, ydim, nx, ny)


def get_iminc(cs):
    """Get the local ecs

    Parameters
    ----------
    cs: xarray.Dataset
        sound speed
    Return
    ------
    xarray.DataArray
        Minimum de célérité profond
    """
    xdim = xcoords.get_xdim(cs, errors="raise")
    ydim = xcoords.get_ydim(cs, errors="raise")
    zdim = xcoords.get_zdim(cs, errors="raise")
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = xcoords.get_depth(cs)
    iminc = xr.apply_ufunc(
        _get_iminc_wrapper_,
        cs,
        depth,
        input_core_dims=[[zdim, ydim, xdim], [zdim, ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        dask="parallelized",
        vectorize=False,
        kwargs={"xdim": xdim, "ydim": ydim, "nx": nx, "ny": ny},
    )
    return iminc  # .transpose(*cs.dims)


class ProfileAcous:

    def __init__(self, profile):
        self.profile = profile  # it an cs xarray profile

    @functools.cached_property
    def ilmax(self):
        return argrelmax(self.profile.values)[0]

    @functools.cached_property
    def ilmin(self):
        try:
            return argrelmin(self.profile.values)[0]
        except IndexError:
            return np.nan

    @functools.cached_property
    def ecs(self):
        try:
            return (
                self.profile.depth.isel(s_rho=self.ilmax[-1])
                if self.profile.isel(s_rho=self.ilmax[-1]) > self.profile.isel(s_rho=-1)
                else np.nan
            )
        except IndexError:
            return np.nan
        # warning to be coded with more generality

    @functools.cached_property
    def iminc(self):
        pos_iminc = None
        if len(self.ilmax) > 1:
            maxd = self.ilmax[-2]
            maxs = self.ilmax[-1]
            for l in self.ilmin:
                if (l > maxd) and (l < maxs):
                    pos_iminc = l
                    break
        if pos_iminc:
            return self.profile.depth.isel(s_rho=pos_iminc)
        else:
            return self.mcp

    @functools.cached_property
    def mcp(self):
        try:
            return self.profile.depth.isel(s_rho=self.ilmin[0])
        except IndexError:
            return np.nan


class Acous2D:
    def __init__(self, dens):
        self.dens = dens
        self.xdim = xcoords.get_xdim(self.dens, errors="raise")
        self.ydim = xcoords.get_ydim(self.dens, errors="raise")

    @functools.cached_property
    # @numba.njit(parallel=True)
    def ecs(self):
        print(self.xdim, self.ydim)
        print(self.dens.shape)
        ny, nx = self.dens.shape[-2], self.dens.shape[-1]
        res = np.empty([ny, nx])
        # for j in numba.prange(1, ny - 1):
        for j in range(ny):
            for i in range(nx):
                res[j, i] = ProfileAcous(self.dens.isel({self.xdim: i, self.ydim: j})).ecs
        return res

    def iminc(self):
        return

    def mcp(self):
        return


class AcousEddy:
    def __init__(self, anomaly):
        self.anomaly = anomaly

    @functools.cached_property
    def ecs_inside(self):
        ecs = np.empty(self.anomaly._profils_inside.shape[1])
        for i in range(len(ecs)):
            ecs[i] = ProfileAcous(self.anomaly._profils_inside.isel(nb_profil=i)).ecs

        lon = self.anomaly._profils_inside.lon_rho.values
        lat = self.anomaly._profils_inside.lat_rho.values
        ecs = xr.DataArray(
            data=ecs,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return ecs

    @functools.cached_property
    def ecs_outside(self):
        ecs = np.empty(self.anomaly._profils_outside.shape[1])
        for i in range(len(ecs)):
            ecs[i] = ProfileAcous(self.anomaly._profils_outside.isel(nb_profil=i)).ecs

        lon = self.anomaly._profils_outside.lon_rho.values
        lat = self.anomaly._profils_outside.lat_rho.values
        ecs = xr.DataArray(
            data=ecs,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return ecs

    @functools.cached_property
    def mcp_inside(self):
        mcp = np.empty(self.anomaly._profils_inside.shape[1])
        for i in range(len(mcp)):
            mcp[i] = ProfileAcous(self.anomaly._profils_inside.isel(nb_profil=i)).mcp

        lon = self.anomaly._profils_inside.lon_rho.values
        lat = self.anomaly._profils_inside.lat_rho.values
        mcp = xr.DataArray(
            data=mcp,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return mcp

    @functools.cached_property
    def mcp_outside(self):
        mcp = np.empty(self.anomaly._profils_outside.shape[1])
        for i in range(len(mcp)):
            mcp[i] = ProfileAcous(self.anomaly._profils_outside.isel(nb_profil=i)).mcp

        lon = self.anomaly._profils_outside.lon_rho.values
        lat = self.anomaly._profils_outside.lat_rho.values
        mcp = xr.DataArray(
            data=mcp,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return mcp

    @functools.cached_property
    def iminc_inside(self):
        iminc = np.empty(self.anomaly._profils_inside.shape[1])
        for i in range(len(iminc)):
            iminc[i] = ProfileAcous(self.anomaly._profils_inside.isel(nb_profil=i)).iminc

        lon = self.anomaly._profils_inside.lon_rho.values
        lat = self.anomaly._profils_inside.lat_rho.values
        iminc = xr.DataArray(
            data=iminc,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return iminc

    @functools.cached_property
    def iminc_outside(self):
        iminc = np.empty(self.anomaly._profils_outside.shape[1])
        for i in range(len(iminc)):
            iminc[i] = ProfileAcous(self.anomaly._profils_outside.isel(nb_profil=i)).iminc

        lon = self.anomaly._profils_outside.lon_rho.values
        lat = self.anomaly._profils_outside.lat_rho.values
        iminc = xr.DataArray(
            data=iminc,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return iminc


def acoustic_points(eddies):
    """
    eddies is an Eddies objectv that have been attributed an acoustic anomaly attribute
    The function add an ecs, iminc, mcp points inside and outside every eddies
    """
    for eddy in eddies.eddies:
        acous = AcousEddy(eddy.anomaly)
        eddy.ecs_inside = acous.ecs_inside
        eddy.ecs_outside = acous.ecs_outside
        eddy.iminc_inside = acous.iminc_inside
        eddy.iminc_outside = acous.iminc_outside
        eddy.mcp_inside = acous.mcp_inside
        eddy.mcp_outside = acous.mcp_outside
