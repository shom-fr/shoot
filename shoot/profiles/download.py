#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  19 10:21:12 2025

@author: jbroust
"""

import copernicusmarine as cm
import os
import numpy as np
import xarray as xr
from argopy import DataFetcher
from .. import cf as scf


#### It requires the user to be already logged in the copernicus
#### marine toolbox


class Download:
    def __init__(
        self,
        time, 
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        root_path,
        max_depth = 1000, 
    ):
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
        for year in years : 
            path_tmp = os.path.join(self.root_path, f"argo_profile_{year}.nc")
            if os.path.exists(path_tmp):
                print(f"Data already exists for year {year}")
                profiles_tmp = xr.open_dataset(path_tmp)
            else : 
                tmin = str(time.sel(time=str(year)).min().dt.strftime("%Y-%d-%m").values) 
                tmax = str(time.sel(time=str(year)).max().dt.strftime("%Y-%d-%m").values) 
                profiles_tmp = self._load(tmin,tmax)  
                profiles_tmp.to_netcdf(path_tmp) 
            if self.profiles : 
                self.profiles = xr.concat([self.profiles, profiles_tmp],dim="N_PROF")
            else : 
                self.profiles = profiles_tmp
        

    def _load(self, tmin, tmax):
        f = DataFetcher(src='erddap', mode='expert')
        box = [self.lon_min, self.lon_max, self.lat_min, self.lat_max, 0,self.max_depth, tmin, tmax]
        
        points = f.region(box).to_xarray()
        profiles = points.argo.point2profile()
        return profiles


    @classmethod 
    def from_ds(cls, ds, root_path, max_depth=1000): 
        lat = scf.get_lat(ds)
        lon = scf.get_lon(ds)
        lon_min= float(lon.min().values)
        lon_max= float(lon.max().values)
        lat_min= float(lat.min().values)
        lat_max= float(lat.max().values)
        time = scf.get_time(ds)
        return cls(time, lat_min, lat_max, lon_min, lon_max, root_path, max_depth= max_depth) 
        
    

def load_from_ds(ds, root_path="/local/tmp/data"):
    do = Download.from_ds(ds, root_path)
    return do.profiles
