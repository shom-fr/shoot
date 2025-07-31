#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:24:21 2025

@author: jbroust
"""

import numpy as np 
from scipy.optimize import linear_sum_assignment
import functools

from .. import geo as sgeo 

class Associate:
    def __init__(self, eddies, ref_eddies, dmax):
        self.ref_eddies = ref_eddies  # reference eddies
        self.eddies = eddies  # next time eddies
        self.dmax

    def ro_avg(self, eddyj):
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        ro = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            ro += self.track_eddies[eddyj.track_id].eddies[i].ro
            n += 1
        return ro / n

    def rad_avg(self, eddyj):
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        radius = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            radius += self.track_eddies[eddyj.track_id].eddies[i].radius
            n += 1
        return radius / n

    @functools.cached_property
    def cost(self):
        """ "cost function between each eddy pairs"""
        M = np.zeros((len(self.eddies), len(self.parent_eddies)))
        for i in range(len(self.eddies)):
            for j in range(len(self.ref_eddies)):
                # print('eddy 1 ', self.parent_eddies[j].lon, self.parent_eddies[j].lat, self.parent_eddies[j].eddy_type)
                # print('eddy 2 ', self.new_eddies[i].lon, self.new_eddies[i].lat, self.new_eddies[i].eddy_type)
                # print('Mij %.2f before'%M[i,j])
                dlat = self.ref_eddies[j].lat - self.eddies[i].lat
                dlon = self.ref_eddies[j].lon - self.eddies[i].lon
                x = sgeo.deg2m(dlon, self.ref_eddies[j].lat)
                y = sgeo.deg2m(dlat)

                # Distance term
                dxy = np.sqrt(x**2 + y**2)
                # print('dxy %.2f (km)'%(dxy/1e3))

                M[i, j] = (dxy**2) / (self.dmax**2) if dxy < self.dmax else 1e6
                # print('Mij %.2f disatnce'%M[i,j])

                # dynamical similarity
                roj = self.ro_avg(self.ref_eddies[j])
                rj = self.rad_avg(self.ref_eddies[j])

                DR = (self.ref_eddies[j].radius - self.eddies[i].radius) / (
                    rj + self.eddies[i].radius
                )
                DR0 = (self.ref_eddies[j].ro - self.eddies[i].ro) / (
                    roj + self.eddies[i].ro
                )

                # print('DR %.2f (km)'%(DR/1e3))
                # print('DR0 %2f (km)'%(DR0/1e3))

                # Warning avoid couple cyclone with anticylone
                M[i, j] += (
                    DR**2 + DR0**2
                    if self.ref_eddies[j].eddy_type == self.eddies[i].eddy_type
                    else 1e6
                )
        return np.sqrt(M)

    def order(self):
        M = self.cost
        idel = []
        for i in range(M.shape[0]):
            if (M[i] > 1e3).all():
                idel.append(i)
        Mclean = np.delete(M, idel, axis=0)  # delete impossible solutions
        raw, col = linear_sum_assignment(Mclean)
        for i, j in zip(raw, col):
            np.delete(self.eddies, idel)[i].id = self.ref_eddies[j].id
            
class BiMod : 
    def __init__(self, eddies, ref_eddies, dmax): 
        self.eddies = eddies 
        self.ref_eddies = ref_eddies 
        self.dmax = dmax
        
    @classmethod
    def compare(cls, eddies, ref_eddies, max_distance = 50): 
        ''' add id to ref eddies and associate eddies to this ids when possible '''
        
        ## attribute id to ref eddies 
        for i, eddy in enumerate(ref_eddies.eddies): 
            eddy.id = i 
            
        Associate(eddies, ref_eddies, max_distance).order()
        return cls(eddies, ref_eddies, max_distance)
    
    
        
        