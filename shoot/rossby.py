#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rossby radius utilities

Functions for accessing first baroclinic Rossby radius climatology.
"""

from scipy.io import loadmat
import numpy as np
import xarray as xr

LATS = ['Lat', 'lat', 'Lat_Rd', 'lat_Rd']
LONS = ['Lon', 'lon', 'Lon_Rd', 'lon_Rd']
RD = ['Rd_Chelton', 'Rd_KNTN', 'RR_baroc1']


class Rossby:
    """First baroclinic Rossby radius climatology

    Loads and provides access to Rossby radius data from climatology.

    Parameters
    ----------
    path : str
        Path to Rossby radius MATLAB file.
    """

    def __init__(
        self,
        path='/home/shom/jbroust/Documents/CODE/SHOOT_LIB/inputs/Rossby_radius/Rossby_Radius_WOA_Barocl1.mat',
    ):
        data = loadmat(path)
        name_lat = None
        name_lon = None
        name_rd = None
        for key in data.keys():
            if key in LATS:
                name_lat = key
            elif key in LONS:
                name_lon = key
            elif key in RD:
                name_rd = key
        self._lat = data[name_lat].squeeze()
        self._lon = data[name_lon].squeeze()
        self._rd = data[name_rd].squeeze()
        if len(self._lat.shape) == 2:
            self._lat = self._lat[:, 0]
        if len(self._lon.shape) == 2:
            self._lon = self._lon[0]

        if len(self._lon) == self._rd.shape[0]:
            self._rd = self._rd.T
        self.ds = xr.Dataset(
            {"rd": (("lat", "lon"), self._rd)},
            coords={
                "lon": ("lon", self._lon, {"long_name": "Longitude"}),
                "lat": ("lat", self._lat, {"long_name": "Latitude"}),
            },
        )

    def get_ro(self, lon, lat):
        """Get Rossby radius at specific location

        Parameters
        ----------
        lon : float
            Longitude in degrees.
        lat : float
            Latitude in degrees.

        Returns
        -------
        float
            Rossby radius in kilometers.
        """
        ilat = np.argmin(np.abs(lat - self.ds.lat))
        jlon = np.argmin(np.abs(lon - self.ds.lon))
        return self.ds.rd.isel(lon=jlon, lat=ilat).values

    def get_ro_avg(self, lon_min, lon_max, lat_min, lat_max):
        """Get average Rossby radius over region

        Parameters
        ----------
        lon_min : float
            Minimum longitude in degrees.
        lon_max : float
            Maximum longitude in degrees.
        lat_min : float
            Minimum latitude in degrees.
        lat_max : float
            Maximum latitude in degrees.

        Returns
        -------
        float
            Mean Rossby radius in kilometers.
        """
        return np.nanmean(self.ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max)).rd)
