#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:24:21 2025

@author: jbroust
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import functools
from shapely.geometry import Polygon

from .. import geo as sgeo
from .. import num as snum


class Associate:
    def __init__(self, eddies, ref_eddies, dmax):
        self.ref_eddies = ref_eddies  # list reference eddies
        self.eddies = eddies  # list of eddies
        self.dmax = (
            dmax * 1000
        )  # convert to meter as all is performed in meters

    @functools.cached_property
    def cost(self):
        """ "cost function between each eddy pairs"""
        M = np.zeros((len(self.eddies), len(self.ref_eddies)))
        for i in range(len(self.eddies)):
            for j in range(len(self.ref_eddies)):
                dlat = self.ref_eddies[j].lat - self.eddies[i].lat
                dlon = self.ref_eddies[j].lon - self.eddies[i].lon
                x = sgeo.deg2m(dlon, self.ref_eddies[j].lat)
                y = sgeo.deg2m(dlat)

                # Distance term
                dxy = np.sqrt(x**2 + y**2)
                M[i, j] = (dxy**2) / (self.dmax**2) if dxy < self.dmax else 1e6

                # dynamical similarity
                DR = (self.ref_eddies[j].radius - self.eddies[i].radius) / (
                    self.ref_eddies[j].radius + self.eddies[i].radius
                )
                DR0 = (self.ref_eddies[j].ro - self.eddies[i].ro) / (
                    self.ref_eddies[j].ro + self.eddies[i].ro
                )

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
            if Mclean[i, j] > 1e3:
                continue
            np.delete(self.eddies, idel)[i].id = self.ref_eddies[j].id


class BiMod:
    def __init__(self, eddies, ref_eddies, dmax):
        self.eddies = eddies
        self.ref_eddies = ref_eddies
        self.dmax = dmax

    @classmethod
    def compare(cls, eddies, ref_eddies, max_distance=50):
        """add id to ref eddies and associate eddies to this ids when possible"""

        ## attribute id to ref eddies
        for i, eddy in enumerate(ref_eddies.eddies):
            eddy.id = i
        for i, eddy in enumerate(eddies.eddies):
            eddy.id = None

        Associate(eddies.eddies, ref_eddies.eddies, max_distance).order()
        return cls(eddies, ref_eddies, max_distance)

    @property
    def pmatch(self):
        self._intersects()
        nb_nomatch = np.sum(
            [
                # (eddy.id is None) | (not eddy.intersect)
                not eddy.intersect
                for eddy in self.eddies.eddies
            ]
        )
        return (len(self.eddies.eddies) - nb_nomatch) / len(self.eddies.eddies)

    def _intersects(self):
        for eddy in self.eddies.eddies:
            eddy.intersect = False
            if eddy.id is None:
                eddy.intersect = None
                continue
            for reddy in self.ref_eddies.eddies:
                if reddy.id == eddy.id:
                    if hasattr(eddy, "x_vmax"):  # Eddy case
                        points = np.array([eddy.x_vmax, eddy.y_vmax]).T
                        if snum.points_in_polygon(
                            points, np.array([reddy.x_vmax, reddy.y_vmax]).T
                        ).any():
                            eddy.intersect = True
                    else:  # Raw eddy 2D case
                        points = np.array(
                            [eddy.vmax_contour.lon, eddy.vmax_contour.lat]
                        ).T
                        if snum.points_in_polygon(
                            points,
                            np.array(
                                [
                                    reddy.vmax_contour.lon,
                                    reddy.vmax_contour.lat,
                                ]
                            ).T,
                        ).any():
                            eddy.intersect = True
                    break

    @property
    def parea(self):
        area = 0
        nb_eddy = 0
        for eddy in self.eddies.eddies:
            if eddy.id is None:
                continue
            for reddy in self.ref_eddies.eddies:
                if reddy.id == eddy.id:
                    x_ell, y_ell = eddy.ellipse.sample
                    x_rell, y_rell = reddy.ellipse.sample

                    poly = Polygon([(x, y) for x, y in zip(x_ell, y_ell)])
                    rpoly = Polygon([(x, y) for x, y in zip(x_rell, y_rell)])
                    intersection = poly.intersection(rpoly)
                    area += intersection.area / rpoly.area
                    nb_eddy += 1
                    break
        return area / nb_eddy

    @property
    def dist(self):
        ddist = 0
        nb_eddy = 0
        for eddy in self.eddies.eddies:
            if eddy.id is None:
                continue
            for reddy in self.ref_eddies.eddies:
                if reddy.id == eddy.id:
                    dlat = eddy.lat - reddy.lat
                    dlon = eddy.lon - reddy.lon
                    x = sgeo.deg2m(dlon, reddy.lat)
                    y = sgeo.deg2m(dlat)
                    ddist += np.sqrt(x**2 + y**2) / 1000
                    nb_eddy += 1
                    break
        return ddist / nb_eddy
