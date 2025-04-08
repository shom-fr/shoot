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
            return self.profile.depth.isel(s_rho=self.ilmax[-1])
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


class Acous:
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
        acous = Acous(eddy.anomaly)
        eddy.ecs_inside = acous.ecs_inside
        eddy.ecs_outside = acous.ecs_outside
        eddy.iminc_inside = acous.iminc_inside
        eddy.iminc_outside = acous.iminc_outside
        eddy.mcp_inside = acous.mcp_inside
        eddy.mcp_outside = acous.mcp_outside
