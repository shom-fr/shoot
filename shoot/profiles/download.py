#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:21:12 2025

@author: jbroust
"""

import copernicusmarine as cm
import logging
import os
import numpy as np

import xarray as xr
from argopy import DataFetcher
from .. import meta as smeta

logger = logging.getLogger(__name__)


#### It requires the user to be already logged in the copernicus
#### marine toolbox


class Download:
    """Download and manage Argo profile data

    Fetches Argo profile data from ERDDAP for a specified spatiotemporal region.
    Caches downloaded data by year to avoid repeated downloads.

    Parameters
    ----------
    time : xarray.DataArray
        Time coordinate range for data download.
    lat_min : float
        Minimum latitude.
    lat_max : float
        Maximum latitude.
    lon_min : float
        Minimum longitude.
    lon_max : float
        Maximum longitude.
    root_path : str
        Root directory for caching downloaded data.
    max_depth : float, default 1000
        Maximum depth (m) for profile data.

    Attributes
    ----------
    profiles : xarray.Dataset
        Downloaded Argo profiles concatenated across all years.
    """

    def __init__(
        self,
        time,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        root_path,
        max_depth=1000,
    ):
        """Initialize downloader and fetch data

        Parameters
        ----------
        time : xarray.DataArray
            Time coordinate range for data download.
        lat_min : float
            Minimum latitude.
        lat_max : float
            Maximum latitude.
        lon_min : float
            Minimum longitude.
        lon_max : float
            Maximum longitude.
        root_path : str
            Root directory for caching downloaded data.
        max_depth : float, default 1000
            Maximum depth (m) for profile data.
        """
        self.path = root_path
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.max_depth = max_depth
        self.root_path = root_path
        self.time = time
        years = np.unique(self.time.dt.year.values)
        self.profiles = None
        for year in years:
            path_tmp = os.path.join(self.root_path, f"argo_profile_{year}.nc")
            if os.path.exists(path_tmp):
                logger.info("Data already exists for year %s", year)
                profiles_tmp = xr.open_dataset(path_tmp)
            else:
                tmin = str(time.sel(time=str(year)).min().dt.strftime("%Y-%m-%d").values)
                tmax = str(time.sel(time=str(year)).max().dt.strftime("%Y-%m-%d").values)
                print(tmin, tmax)
                profiles_tmp = self._load(tmin, tmax)
                profiles_tmp.to_netcdf(path_tmp)
            if self.profiles:
                self.profiles = xr.concat([self.profiles, profiles_tmp], dim="N_PROF")
            else:
                self.profiles = profiles_tmp

    def _load(self, tmin, tmax):
        """Download Argo profiles from ERDDAP for a time range

        Parameters
        ----------
        tmin : str
            Start date (YYYY-MM-DD format).
        tmax : str
            End date (YYYY-MM-DD format).

        Returns
        -------
        xarray.Dataset
            Downloaded Argo profiles.
        """
        f = DataFetcher(src='erddap', mode='expert')
        box = [self.lon_min, self.lon_max, self.lat_min, self.lat_max, 0, self.max_depth, tmin, tmax]

        points = f.region(box).to_xarray()
        profiles = points.argo.point2profile()
        return profiles

    @classmethod
    def from_ds(cls, ds, root_path, max_depth=1000):
        """Create Download instance from a dataset's spatiotemporal extent

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with lat, lon, and time coordinates.
        root_path : str
            Root directory for caching downloaded data.
        max_depth : float, default 1000
            Maximum depth (m) for profile data.

        Returns
        -------
        Download
            New Download instance with data fetched for the dataset's extent.
        """
        lat = smeta.get_lat(ds)
        lon = smeta.get_lon(ds)
        lon_min = float(lon.min().values)
        lon_max = float(lon.max().values)
        lat_min = float(lat.min().values)
        lat_max = float(lat.max().values)
        time = smeta.get_time(ds)
        return cls(time, lat_min, lat_max, lon_min, lon_max, root_path, max_depth=max_depth)


def load_from_ds(ds, root_path="/local/tmp/data"):
    """Load Argo profiles for a dataset's spatiotemporal extent

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with lat, lon, and time coordinates.
    root_path : str, default "/local/tmp/data"
        Root directory for caching downloaded data.

    Returns
    -------
    xarray.Dataset
        Downloaded Argo profiles.
    """
    do = Download.from_ds(ds, root_path)
    return do.profiles
