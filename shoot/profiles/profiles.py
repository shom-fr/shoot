#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:20:12 2025

@author: jbroust
"""

import os, glob
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
    def __init__(self):
        return


class Profiles:
    def __init__(self, time, region, root_path, data_types, download=True):
        self.root_path = root_path
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

        self.profiles = None
        for f in file_path:
            tmp_profiles = xr.open_dataset(f, decode_times=False)
            for i in tmp_profiles.N_PROF:
                print("to do")
            if not self.profiles:
                print("to do")
            else:
                print("to do")

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

    def associate(self, eddies):
        """
        This method attribute profile to eddies and add info to profiles
        if they are considered colocalised in a structure
        """

        for pf in self.profiles:
            break
