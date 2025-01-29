#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 08:48:47 2025

@author: jbroust
"""

from scipy.io import loadmat
import numpy as np


class Rossby:
    """This class return the first baroclinic Rossby number based on climatology"""

    def __init__(
        self,
        path='/home/shom/jbroust/Documents/CODE/SHOOT_LIB/inputs/Rossby_radius/global_Rossby_radius.mat',
    ):
        data = loadmat(path)
        self.lat = data['lat_Rd'][:, 0]
        self.lon = data['lon_Rd'][0]
        self.ro = data['Rd_Chelton']

    def get_ro(self, lon, lat):
        ilat = np.argmin(np.abs(lat - self.lat))
        jlon = np.argmin(np.abs(lon - self.lon))
        return self.ro[ilat, jlon]
