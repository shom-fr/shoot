#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:20:12 2025

@author: jbroust
"""

import os, glob
import functools
from .download import Download, load_from_ds
import xarray as xr
import numpy as np
from .. import meta as smeta


class Profile:
    """Individual in-situ profile with temperature and salinity data"""

    def __init__(self, prf):
        # Extract scalar values for single-element arrays
        time_vals = prf.TIME.values
        self.time = time_vals

        lat_vals = prf.LATITUDE.values
        self.lat = lat_vals

        lon_vals = prf.LONGITUDE.values
        self.lon = lon_vals

        self.depth = np.arange(1, 2001)
        self.temp = np.interp(
            self.depth,
            prf.PRES_ADJUSTED,
            prf.TEMP_ADJUSTED,
            left=np.nan,
            right=np.nan,
        )
        self.sal = np.interp(
            self.depth,
            prf.PRES_ADJUSTED,
            prf.PSAL_ADJUSTED,
            left=np.nan,
            right=np.nan,
        )
        self.valid = 1.5 * np.sum(np.isnan(self.temp)) < len(self.temp)


class Profiles:
    """Collection of in-situ profiles with temperature and salinity data

    Manages a collection of Profile objects, providing methods to load from datasets,
    convert to xarray format, save to NetCDF, and associate with eddies.

    Parameters
    ----------
    time : xarray.DataArray
        Time coordinate for the profiles.
    root_path : str
        Root directory path for data storage.
    brut_prf : xarray.Dataset
        Raw profile dataset.

    Attributes
    ----------
    profiles : list of Profile
        List of valid Profile objects.
    years : ndarray
        Array of years covered by the profiles.
    ds : xarray.Dataset
        Profiles converted to xarray Dataset format (cached property).
    """

    def __init__(self, time, root_path, brut_prf):
        """Initialize Profiles collection

        Parameters
        ----------
        time : xarray.DataArray
            Time coordinate for the profiles.
        root_path : str
            Root directory path for data storage.
        brut_prf : xarray.Dataset
            Raw profile dataset.
        """
        self.root_path = root_path
        self.time = time
        self.years = np.arange(self.time.min().dt.year.values, self.time.max().dt.year.values + 1)

        self.profiles = []
        for i in brut_prf.N_PROF:
            prf = Profile(brut_prf.isel(N_PROF=i))
            if prf.valid:
                self.profiles.append(prf)

    @classmethod
    def from_ds(cls, ds, root_path):
        """Create Profiles from an xarray Dataset

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with time coordinate.
        root_path : str
            Root directory path for data storage.

        Returns
        -------
        Profiles
            New Profiles instance.
        """
        brut_prf = load_from_ds(ds, root_path)
        return cls(ds.time, root_path, brut_prf)

    @functools.cached_property
    def ds(self):
        """create profile array"""
        lats = np.array([prf.lat for prf in self.profiles])
        lons = np.array([prf.lon for prf in self.profiles])
        times = np.array([prf.time for prf in self.profiles])
        temp = np.array([prf.temp for prf in self.profiles])
        sal = np.array([prf.sal for prf in self.profiles])
        p_id = np.arange(0, len(lats))

        ds = xr.Dataset(
            {
                "time": (("profil"), times),
                "p_id": (("profil"), p_id),
                "lat": (("profil"), lats),
                "lon": (("profil"), lons),
                "temp": (("profil", "depth"), temp),
                "sal": (("profil", "depth"), sal),
            },
            coords={
                "profil": np.arange(0, len(lats)),
                "depth": self.profiles[0].depth,
            },
        )
        return ds

    def to_netcdf(self, name=None, path=None):
        """Save profiles to NetCDF format"""
        if not path:
            path = self.root_path
        if not name:
            name = "profil_%i_%i.nc" % (
                self.years[0],
                self.years[-1],
            )
        self.ds.to_netcdf(
            os.path.join(
                path,
                name,
            )
        )

    def associate(self, eddies, nlag=2):
        """Associate profiles with eddies

        Attributes profiles to eddies (EvolEddies2D object) and adds info to profiles
        if they are considered colocalized within a structure.
        """
        eddy_pos = np.ones(len(self.ds.profil)) * -1
        for eddy in eddies.eddies:  # list of Eddies2D object object
            tstart = eddy.time - np.timedelta64(nlag, "D")
            tend = eddy.time + np.timedelta64(nlag, "D")
            ind_prf = np.where((self.ds.time >= tstart) & (self.ds.time <= tend))[0]
            prf = self.ds.isel(profil=ind_prf)

            for e in eddy.eddies:  # list of Raw2dEddies object
                e.p_id = []
                inside = e.contains_points(prf.lon, prf.lat)
                for i, b in enumerate(inside):
                    if b:
                        e.p_id.append(prf.isel(profil=i).p_id.values)
                        eddy_pos[prf.isel(profil=i).p_id.values] = e.track_id
        self.ds = self.ds.assign(eddy_pos=("profil", np.array(eddy_pos, dtype=np.int32)))
