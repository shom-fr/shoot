# -*- coding: utf-8 -*-

"""
acoustic functions
==================
"""

import functools
import numpy as np
from scipy.signal import argrelmax, argrelmin
import numba
import xarray as xr

from . import cf as scf


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
        return depth[ilmaxs[-1]] if profile[ilmaxs[-1]] > profile[-1] else 0 * depth[ilmaxs[-1]]
    except IndexError:
        return np.nan * depth[-1]


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
        return np.nan * depth[-1]


def _mcp(profile, depth):
    ilmins = _ilmin(profile)
    try:
        return depth[ilmins[0]]
    except IndexError:
        return np.nan * depth[-1]


## ECS
def _get_ecs_(cs, depth, ecs, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            ecs[j, i] = _ecs(cs[:, j, i], depth[:, j, i])
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
    xdim = scf.get_xdim(cs)
    ydim = scf.get_ydim(cs)
    zdim = scf.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = scf.get_depth(cs)
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
    return ecs


## MCP
def _get_mcp_(cs, depth, mcp, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            mcp[j, i] = _mcp(cs[:, j, i], depth[:, j, i])
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
    xdim = scf.get_xdim(cs)
    ydim = scf.get_ydim(cs)
    zdim = scf.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = scf.get_depth(cs)
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
    return mcp


# IMINC
## MCP
def _get_iminc_(cs, depth, iminc, xdim, ydim, nx, ny):
    for j in numba.prange(ny):
        for i in range(nx):
            iminc[j, i] = _iminc(cs[:, j, i], depth[:, j, i])
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
    xdim = scf.get_xdim(cs)
    ydim = scf.get_ydim(cs)
    zdim = scf.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = scf.get_depth(cs)
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
    return iminc


class ProfileAcous:

    def __init__(self, profile, depth):
        self.profile = profile  # it an cs xarray profile
        self.depth = depth  # scf.get_depth(profile)
        self.depth_dim = self.depth.dims[0]

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
        return _ecs(self.profile.values, self.depth)

    @functools.cached_property
    def iminc(self):
        return _iminc(self.profile.values, self.depth)

    @functools.cached_property
    def mcp(self):
        return _mcp(self.profile.values, self.depth)


class AcousEddy:
    def __init__(self, anomaly):
        self.anomaly = anomaly

    @functools.cached_property
    def ecs_inside(self):
        return ProfileAcous(self.anomaly.mean_profil_inside, self.anomaly.depth_vector).ecs

    @functools.cached_property
    def iminc_inside(self):
        return ProfileAcous(self.anomaly.mean_profil_inside, self.anomaly.depth_vector).iminc

    @functools.cached_property
    def mcp_inside(self):
        return ProfileAcous(self.anomaly.mean_profil_inside, self.anomaly.depth_vector).mcp

    @functools.cached_property
    def ecs_outside(self):
        return ProfileAcous(self.anomaly.mean_profil_outside, self.anomaly.depth_vector).ecs

    @functools.cached_property
    def iminc_outside(self):
        return ProfileAcous(self.anomaly.mean_profil_outside, self.anomaly.depth_vector).iminc

    @functools.cached_property
    def mcp_outside(self):
        return ProfileAcous(self.anomaly.mean_profil_outside, self.anomaly.depth_vector).mcp

    @staticmethod
    def _distance(e1, e2):
        if np.isnan(e1) and np.isnan(e2):  # point doesn exists at all
            return 0
        elif np.isnan(e1) or np.isnan(e2):  # creation/destruction point
            return 1
        elif e1 == 0 and e2 == 0:
            return 0
        else:
            return np.abs(e1 - e2) / np.abs((0.5 * (e1 + e2)))

    @functools.cached_property
    def acoustic_impact(self):
        ecs_in = self.ecs_inside
        mcp_in = self.mcp_inside
        iminc_in = self.iminc_inside
        ecs_out = self.ecs_outside
        mcp_out = self.mcp_outside
        iminc_out = self.iminc_outside

        d_mcp = AcousEddy._distance(mcp_in, mcp_out)
        d_iminc = AcousEddy._distance(iminc_in, iminc_out)
        d_ecs = AcousEddy._distance(ecs_in, ecs_out)

        return d_mcp + d_iminc + d_ecs

    @functools.cached_property
    def ecs_insides(self):
        ecs = np.empty(self.anomaly.profils_inside.shape[0])
        for i in range(len(ecs)):
            ecs[i] = ProfileAcous(
                self.anomaly.profils_inside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).ecs

        lon = scf.get_lon(self.anomaly.profils_inside).values
        lat = scf.get_lat(self.anomaly.profils_inside).values
        ecs = xr.DataArray(
            data=ecs,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return ecs

    @functools.cached_property
    def ecs_outsides(self):
        ecs = np.empty(self.anomaly.profils_outside.shape[0])
        for i in range(len(ecs)):
            ecs[i] = ProfileAcous(
                self.anomaly.profils_outside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).ecs

        lon = scf.get_lon(self.anomaly.profils_outside).values
        lat = scf.get_lat(self.anomaly.profils_outside).values
        ecs = xr.DataArray(
            data=ecs,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return ecs

    @functools.cached_property
    def mcp_insides(self):
        mcp = np.empty(self.anomaly.profils_inside.shape[0])
        for i in range(len(mcp)):
            mcp[i] = ProfileAcous(
                self.anomaly.profils_inside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).mcp

        lon = scf.get_lon(self.anomaly.profils_inside).values
        lat = scf.get_lat(self.anomaly.profils_inside).values
        mcp = xr.DataArray(
            data=mcp,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return mcp

    @functools.cached_property
    def mcp_outsides(self):
        mcp = np.empty(self.anomaly.profils_outside.shape[0])
        for i in range(len(mcp)):
            mcp[i] = ProfileAcous(
                self.anomaly.profils_outside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).mcp

        lon = scf.get_lon(self.anomaly.profils_outside).values
        lat = scf.get_lat(self.anomaly.profils_outside).values
        mcp = xr.DataArray(
            data=mcp,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return mcp

    @functools.cached_property
    def iminc_insides(self):
        iminc = np.empty(self.anomaly.profils_inside.shape[0])
        for i in range(len(iminc)):
            iminc[i] = ProfileAcous(
                self.anomaly.profils_inside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).iminc

        lon = scf.get_lon(self.anomaly.profils_inside).values
        lat = scf.get_lat(self.anomaly.profils_inside).values
        iminc = xr.DataArray(
            data=iminc,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return iminc

    @functools.cached_property
    def iminc_outsides(self):
        iminc = np.empty(self.anomaly.profils_outside.shape[0])
        for i in range(len(iminc)):
            iminc[i] = ProfileAcous(
                self.anomaly.profils_outside.isel(nb_profil=i),
                self.anomaly.depth_vector,
            ).iminc

        lon = scf.get_lon(self.anomaly.profils_outside).values
        lat = scf.get_lat(self.anomaly.profils_outside).values
        iminc = xr.DataArray(
            data=iminc,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return iminc


def acoustic_points(eddies):
    """The function add an ecs, iminc, mcp points inside and outside every eddies

    Parameters
    ----------
    eddies: Eddies object

    Return
    ------
    """
    for eddy in eddies.eddies:
        acous = AcousEddy(eddy.anomaly)
        eddy.ecs_insides = acous.ecs_insides
        eddy.ecs_outsides = acous.ecs_outsides
        eddy.iminc_insides = acous.iminc_insides
        eddy.iminc_outsides = acous.iminc_outsides
        eddy.mcp_insides = acous.mcp_insides
        eddy.mcp_outsides = acous.mcp_outsides
        eddy.acoustic_impact = acous.acoustic_impact
