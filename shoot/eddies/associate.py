#!/usr/bin/env python3
"""
Created on Thu Jan  9 13:24:21 2025

@author: jbroust
"""

import functools

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from .. import geo as sgeo
from .. import num as snum
from ..core import track as ctrack


class Associate:
    """Associate eddies with reference eddies based on spatial proximity and similarity"""

    def __init__(self, eddies, ref_eddies, dmax):
        self.ref_eddies = ref_eddies  # list reference eddies
        self.eddies = eddies  # list of eddies
        self.dmax = dmax * 1000  # convert to meter as all is performed in meters

    @functools.cached_property
    def cost(self):
        """Cost function between each eddy pair"""
        if not len(self.eddies) or not len(self.ref_eddies):
            return np.zeros((len(self.eddies), len(self.ref_eddies)))
        codes = {}
        return ctrack.association_cost(
            [eddy.lon for eddy in self.eddies],
            [eddy.lat for eddy in self.eddies],
            [eddy.radius for eddy in self.eddies],
            [eddy.ro for eddy in self.eddies],
            [codes.setdefault(eddy.eddy_type, len(codes)) for eddy in self.eddies],
            [eddy.lon for eddy in self.ref_eddies],
            [eddy.lat for eddy in self.ref_eddies],
            [eddy.radius for eddy in self.ref_eddies],
            [eddy.ro for eddy in self.ref_eddies],
            [codes.setdefault(eddy.eddy_type, len(codes)) for eddy in self.ref_eddies],
            np.full(len(self.ref_eddies), self.dmax),
        )

    def order(self):
        """Assign eddy IDs using linear sum assignment"""
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
    """Bipartite matching for eddy association with validation metrics

    Performs eddy association and computes validation metrics including
    match proportion, area overlap, and distance between matched eddies.

    Parameters
    ----------
    eddies : Eddies2D
        Eddies to associate with reference.
    ref_eddies : Eddies2D
        Reference eddies for comparison.
    dmax : float
        Maximum distance (km) for association.

    Attributes
    ----------
    pmatch : float
        Proportion of eddies that match with reference (property).
    parea : float
        Average intersection area ratio between matched eddies (property).
    dist : float
        Average distance (km) between matched eddy centers (property).
    """

    def __init__(self, eddies, ref_eddies, dmax):
        """Initialize bipartite matcher

        Parameters
        ----------
        eddies : Eddies2D
            Eddies to associate with reference.
        ref_eddies : Eddies2D
            Reference eddies for comparison.
        dmax : float
            Maximum distance (km) for association.
        """
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
        """Proportion of eddies that match with reference eddies"""
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
        """Check if matched eddies' contours intersect

        Sets the `intersect` attribute for each eddy indicating whether
        its contour intersects with its matched reference eddy's contour.
        """
        for eddy in self.eddies.eddies:
            eddy.intersect = False
            if eddy.id is None:
                eddy.intersect = None
                continue
            for reddy in self.ref_eddies.eddies:
                if reddy.id == eddy.id:
                    if hasattr(eddy, "x_vmax"):  # Eddy case
                        points = np.array([eddy.x_vmax, eddy.y_vmax]).T
                        if snum.points_in_polygon(points, np.array([reddy.x_vmax, reddy.y_vmax]).T).any():
                            eddy.intersect = True
                    else:  # GriddedEddy2D case
                        points = np.array([eddy.vmax_contour.lon, eddy.vmax_contour.lat]).T
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
        """Average intersection area ratio between matched eddies"""
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
        return area / nb_eddy if nb_eddy > 0 else np.nan

    @property
    def dist(self):
        """Average distance between matched eddy centers"""
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
        return ddist / nb_eddy if nb_eddy > 0 else np.nan
