# -*- coding: utf-8 -*-
"""
Acoustic analysis functions

Functions for computing acoustic parameters from sound speed profiles.
"""

import functools
import numpy as np
from scipy.signal import argrelmax, argrelmin
import numba
import xarray as xr

from . import meta as smeta


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
        if np.abs(depth[0]) > np.abs(depth[-1]):  # bottom to surf case
            return depth[ilmaxs[-1]] if profile[ilmaxs[-1]] > profile[-1] else 0 * depth[ilmaxs[-1]]
        else:  # surface to bottom case
            return depth[ilmaxs[0]] if profile[ilmaxs[0]] > profile[0] else 0 * depth[ilmaxs[0]]
    except IndexError:
        return np.nan * depth[-1]


def _iminc(profile, depth):
    pos_iminc = None
    ilmaxs = _ilmax(profile)
    ilmins = _ilmin(profile)
    if len(ilmaxs) > 1:
        if np.abs(depth[0]) > np.abs(depth[-1]):  # bottom to surf case
            maxd = ilmaxs[-2]
            maxs = ilmaxs[-1]
            for l in ilmins:
                if (l > maxd) and (l < maxs):
                    pos_iminc = l
                    break
        else:  # surface to bottom
            maxd = ilmaxs[1]
            maxs = ilmaxs[0]
            for l in ilmins:
                if (l > maxs) and (l < maxd):
                    pos_iminc = l
                    break
    if pos_iminc:
        return depth[pos_iminc]
    else:
        return np.nan * depth[-1]


def _mcp(profile, depth):
    ilmins = _ilmin(profile)
    try:
        if np.abs(depth[0]) > np.abs(depth[-1]):  # bottom to surface
            return depth[ilmins[0]]
        else:  # surface to bottom
            return depth[ilmins[-1]]
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
    """Compute surface duct thickness

    Parameters
    ----------
    cs : xarray.DataArray
        3D sound speed field.

    Returns
    -------
    xarray.DataArray
        Surface duct thickness (épaisseur du chenal de surface).
    """
    xdim = smeta.get_xdim(cs)
    ydim = smeta.get_ydim(cs)
    zdim = smeta.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = smeta.get_depth(cs)
    if len(depth.shape) == 1 : 
        depth = depth.broadcast_like(cs)
        
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
    """Compute deep sound speed minimum depth (MCP)

    Parameters
    ----------
    cs : xarray.DataArray
        3D sound speed field.

    Returns
    -------
    xarray.DataArray
        Deep sound speed minimum depth (minimum de célérité profond).
    """
    xdim = smeta.get_xdim(cs)
    ydim = smeta.get_ydim(cs)
    zdim = smeta.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = smeta.get_depth(cs)
    if len(depth.shape) == 1 : 
        depth = depth.broadcast_like(cs)
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
    """Compute intermediate sound speed minimum depth (IMINC)

    Parameters
    ----------
    cs : xarray.DataArray
        3D sound speed field.

    Returns
    -------
    xarray.DataArray
        Intermediate minimum depth in two-channel profiles.
    """
    xdim = smeta.get_xdim(cs)
    ydim = smeta.get_ydim(cs)
    zdim = smeta.get_zdim(cs)
    nx = len(cs[xdim])
    ny = len(cs[ydim])
    depth = smeta.get_depth(cs)
    if len(depth.shape) == 1 : 
        depth = depth.broadcast_like(cs)
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
    """Acoustic parameters for a single sound speed profile

    Computes acoustic properties (local maxima/minima, surface duct thickness,
    deep minimum, etc.) from a sound speed profile.

    Parameters
    ----------
    profile : xarray.DataArray
        Sound speed profile.
    depth : xarray.DataArray
        Depth coordinate for the profile.

    Attributes
    ----------
    ilmax : ndarray
        Indices of local maxima in the profile.
    ilmin : ndarray
        Indices of local minima in the profile.
    ecs : float
        Surface duct thickness (épaisseur du chenal de surface).
    iminc : float
        Depth of intermediate minimum in two-channel profiles.
    mcp : float
        Depth of deep sound speed minimum (minimum de célérité profond).
    """

    def __init__(self, profile, depth):
        """Initialize acoustic profile analyzer

        Parameters
        ----------
        profile : xarray.DataArray
            Sound speed profile.
        depth : xarray.DataArray
            Depth coordinate for the profile.
        """
        self.profile = profile  # it an cs xarray profile
        self.depth = depth  # smeta.get_depth(profile)
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
        return _ecs(self.profile.values, self.depth.values)

    @functools.cached_property
    def iminc(self):
        return _iminc(self.profile.values, self.depth.values)

    @functools.cached_property
    def mcp(self):
        return _mcp(self.profile.values, self.depth.values)


class AcousEddy:
    """Acoustic analysis for an eddy anomaly

    Computes acoustic parameters both inside and outside an eddy,
    and calculates the acoustic impact (difference between inside/outside).

    Parameters
    ----------
    anomaly : Anomaly
        Eddy anomaly object containing profile data inside and outside the eddy.

    Attributes
    ----------
    ecs_inside : float
        Surface duct thickness inside the eddy.
    ecs_outside : float
        Surface duct thickness outside the eddy.
    mcp_inside : float
        Deep sound speed minimum depth inside the eddy.
    mcp_outside : float
        Deep sound speed minimum depth outside the eddy.
    iminc_inside : float
        Intermediate minimum depth inside the eddy.
    iminc_outside : float
        Intermediate minimum depth outside the eddy.
    acoustic_impact : float
        Combined acoustic impact metric (sum of relative differences).
    """

    def __init__(self, anomaly):
        """Initialize acoustic eddy analyzer

        Parameters
        ----------
        anomaly : Anomaly
            Eddy anomaly object containing profile data.
        """
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

        lon = smeta.get_lon(self.anomaly.profils_inside).values
        lat = smeta.get_lat(self.anomaly.profils_inside).values
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

        lon = smeta.get_lon(self.anomaly.profils_outside).values
        lat = smeta.get_lat(self.anomaly.profils_outside).values
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

        lon = smeta.get_lon(self.anomaly.profils_inside).values
        lat = smeta.get_lat(self.anomaly.profils_inside).values
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

        lon = smeta.get_lon(self.anomaly.profils_outside).values
        lat = smeta.get_lat(self.anomaly.profils_outside).values
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

        lon = smeta.get_lon(self.anomaly.profils_inside).values
        lat = smeta.get_lat(self.anomaly.profils_inside).values
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

        lon = smeta.get_lon(self.anomaly.profils_outside).values
        lat = smeta.get_lat(self.anomaly.profils_outside).values
        iminc = xr.DataArray(
            data=iminc,
            dims=["nb_profil"],
            coords=dict(lon_rho=(["nb_profil"], lon), lat_rho=(["nb_profil"], lat)),
        )
        return iminc


def acoustic_points(eddies):
    """Compute acoustic impact for all eddies

    Adds acoustic parameters (ecs, iminc, mcp) inside and outside
    each eddy, plus overall acoustic impact metric.

    Parameters
    ----------
    eddies : Eddies2D
        Collection of eddies with anomaly attributes.

    Notes
    -----
    Modifies eddies in-place by adding acoustic attributes.
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
