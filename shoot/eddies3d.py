#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:39:51 2024 by jbroust
"""

import functools
import warnings
import numpy as np
from scipy.interpolate import splprep, make_interp_spline, splev
from scipy.optimize import linear_sum_assignment
import multiprocessing as mp
import itertools
import xarray as xr
import matplotlib.pyplot as plt
import json
import pandas as pd

import xoa.coords as xcoords
import xoa.geo as xgeo

from . import num as snum
from . import dyn as sdyn
from . import grid as sgrid
from . import fit as sfit
from . import streamline as strl
from . import contours as scontours
from . import plot as splot
from . import eddies as eddies


class Associate:
    def __init__(self, parent_eddies, new_eddies):
        self.parent_eddies = parent_eddies  # reference eddies
        self.new_eddies = new_eddies  # next time eddies

    def search_dist(self, eddyj, eddyi):
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        Ravg = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            Ravg += self.track_eddies[eddyj.track_id].eddies[i].vmax_contour.radius
            n += 1
        # print('Dij components', self._C*(1+self._Dt)/2, Ravg/n/1e3 , eddyi.vmax_contour.radius/1e3, eddyi.radius)
        Dij = self._C * (1 + self._Dt) / 2 + Ravg / n + eddyi.vmax_contour.radius
        return Dij

    @functools.cached_property
    def cost(self):
        """ "cost function between each eddy pairs"""
        M = np.zeros((len(self.new_eddies), len(self.parent_eddies)))
        for i in range(len(self.new_eddies)):
            for j in range(len(self.parent_eddies)):
                # print('eddy 1 ', self.parent_eddies[j].glon, self.parent_eddies[j].glat, self.parent_eddies[j].eddy_type)
                # print('eddy 2 ', self.new_eddies[i].glon, self.new_eddies[i].glat, self.new_eddies[i].eddy_type)
                # print('Mij %.2f before'%M[i,j])
                dlat = self.parent_eddies[j].glat - self.new_eddies[i].glat
                dlon = self.parent_eddies[j].glon - self.new_eddies[i].glon
                x = xgeo.deg2m(dlon, self.parent_eddies[j].glat)
                y = xgeo.deg2m(dlat)
                dxy = np.sqrt(x**2 + y**2) / 1000
                M[i, j] = (
                    dxy if self.parent_eddies[j].eddy_type == self.new_eddies[i].eddy_type else 1e6
                )
        return M

    # def order(self):
    #     M = self.cost
    #     raw, col = linear_sum_assignment(M)
    #     for i, j in zip(raw, col):
    #         self.new_eddies[i].z_id = self.parent_eddies[j].z_id

    def order(self):
        M = self.cost
        idel = []
        for i in range(M.shape[0]):
            if (M[i] > 1e3).all():
                idel.append(i)
        Mclean = np.delete(M, idel, axis=0)  # delete impossible solutions
        raw, col = linear_sum_assignment(Mclean)
        for i, j in zip(raw, col):
            np.delete(self.new_eddies, idel)[i].z_id = self.parent_eddies[j].z_id


class Eddies3D:

    def __init__(self, u, v, eddies3d, nb_eddies):
        self.u = u
        self.v = v
        self.eddies3d = eddies3d  # dictionnary with Eddies at each depth
        self.nb_eddies = nb_eddies

    @classmethod
    def detect_eddies_3d(
        cls,
        u,
        v,
        window_center,
        window_fit=None,
        dx=None,
        dy=None,
        min_radius=None,
        ssh_method='streamline',
        paral=False,
        **kwargs,
    ):
        depth = u.depth
        eddies3d = {}
        eddies_tmp = None
        nb_eddies = 0
        for z in range(len(depth) - 1, -1, -1):
            uz = u.isel(depth=z)
            vz = v.isel(depth=z)
            eddies_z = eddies.Eddies.detect_eddies(
                uz,
                vz,
                window_center,
                window_fit=window_fit,
                min_radius=min_radius,
                paral=True,
            )
            if z == len(depth) - 1:
                for i, eddy in enumerate(eddies_z.eddies):
                    eddy.z_id = i
                nb_eddies = len(eddies_z.eddies)
            else:
                Associate(eddies_tmp.eddies, eddies_z.eddies).order()
                for eddy in eddies_z.eddies:
                    try:
                        eddy.z_id
                    except AttributeError:
                        eddy.z_id = nb_eddies
                        nb_eddies += 1

            eddies3d[z] = eddies_z
            eddies_tmp = eddies_z
        return cls(u, v, eddies3d, nb_eddies)
