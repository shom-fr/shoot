#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:20:12 2025

@author: jbroust
"""

import os, glob
import functools
from .download import load, Download
import xarray as xr
import numpy as np
from .. import cf as scf


def determine_region(lat_min, lat_max, lon_min, lon_max):
    REGIONS = Download.get_regions()
    for region, bounds in REGIONS.items():
        lat_bounds = bounds["lat"]
        lon_bounds = bounds["lon"]
        # check if dataset included in a region
        if (
            lat_min >= lat_bounds[0]
            and lat_max <= lat_bounds[1]
            and lon_min >= lon_bounds[0]
            and lon_max <= lon_bounds[1]
        ):
            return region
    return "global"


class Profile:
    def __init__(self, prf):
        self.time = np.datetime64("1950-01-01") + np.timedelta64(
            int(prf.TIME.values * 86400), "s"
        )
        self.lat = prf.LATITUDE.values
        self.lon = prf.LONGITUDE.values
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
    def __init__(self, time, region, root_path, data_types, download=True):
        self.root_path = os.path.join(root_path, region)
        self.region = region
        self.time = time
        self.years = np.arange(
            self.time.min().dt.year.values, self.time.max().dt.year.values + 1
        )
        self.months = [
            self.time.min().dt.month.values,
            self.time.max().dt.month.values,
        ]

        if download:
            load(
                self.years,
                self.months,
                self.region,
                self.root_path,
                data_types,
            )

        else:
            print("Data already exists")

        file_path = []
        for year in self.years:
            file_path += sorted(
                glob.glob(os.path.join(self.root_path, str(year), "*.nc"))
            )

        self.profiles = []
        for f in file_path:
            tmp_profiles = xr.open_dataset(f, decode_times=False)
            for i in tmp_profiles.N_PROF:
                prf = Profile(tmp_profiles.isel(N_PROF=i))
                if prf.valid:
                    self.profiles.append(prf)

    @classmethod
    def from_ds(
        cls, ds, root_path, region=None, data_types=["XT", "PF"], download=True
    ):

        if not region:
            lat_min = scf.get_lat(ds).min().values
            lat_max = scf.get_lat(ds).max().values
            lon_min = scf.get_lon(ds).min().values
            lon_max = scf.get_lon(ds).max().values
            region = determine_region(lat_min, lat_max, lon_min, lon_max)
            print("Selected region : %s" % region)

        return cls(ds.time, region, root_path, data_types, download=download)

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
                "nb_prof": np.arange(0, len(lats)),
                "depth": self.profiles[0].depth,
            },
        )
        return ds

    def save(self, name=None, path=None):
        if not path:
            path = self.root_path
        if not name:
            name = "profil_%s_%i_%i.nc" % (
                self.region,
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
        """
        This method attributes profile to eddies (EvolEddies2D object) and add info to profiles
        if they are considered colocalised in a structure
        """
        eddy_pos = np.ones(len(self.ds.profil)) * -1
        for eddy in eddies.eddies:  # list of Eddies2D object object
            tstart = eddy.time - np.timedelta64(nlag, "D")
            tend = eddy.time + np.timedelta64(nlag, "D")
            ind_prf = np.where(
                (self.ds.time >= tstart) & (self.ds.time <= tend)
            )[0]
            prf = self.ds.isel(profil=ind_prf)

            for e in eddy.eddies:  # list of Raw2dEddies object
                e.p_id = None
                inside = e.contains_points(prf.lon, prf.lat)
                for i, b in enumerate(inside):
                    if b:
                        e.p_id = prf.isel(profil=i).p_id.values
                        eddy_pos[prf.isel(profil=i).p_id.values] = e.track_id
        self.ds = self.ds.assign(eddy_pos=("profil", eddy_pos))
