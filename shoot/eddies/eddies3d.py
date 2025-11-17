#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:39:51 2024 by jbroust
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


from .. import geo as sgeo
from . import eddies2d
from .. import plot as splot
from .. import dyn as sdyn
from .. import cf as scf


class Associate:
    def __init__(self, parent_eddies, new_eddies, max_distance=10):
        self.parent_eddies = parent_eddies  # reference eddies
        self.new_eddies = new_eddies  # next time eddies
        self._max_distance = max_distance  # maximum distance for centers to be associated

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
                dlat = self.parent_eddies[j].glat - self.new_eddies[i].glat
                dlon = self.parent_eddies[j].glon - self.new_eddies[i].glon
                x = sgeo.deg2m(dlon, self.parent_eddies[j].glat)
                y = sgeo.deg2m(dlat)
                dxy = np.sqrt(x**2 + y**2) / 1000
                dxy = dxy if dxy < self._max_distance else 1e6
                M[i, j] = (
                    dxy if self.parent_eddies[j].eddy_type == self.new_eddies[i].eddy_type else 1e6
                )
        return M

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


class EddiesByDepth:

    def __init__(self, u, v, depth, eddies3d, nb_eddies):
        self.u = u
        self.v = v
        self.depth = depth  # to change
        self.eddies3d = eddies3d  # dictionnary with Eddies2D at each depth
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
        max_distance=10,
        **kwargs,
    ):
        depth = u.depth
        eddies3d = {}
        eddies_tmp = None
        nb_eddies = 0
        for z in range(len(depth) - 1, -1, -1):
            uz = u.isel(depth=z)
            vz = v.isel(depth=z)
            eddies_z = eddies2d.Eddies2D.detect_eddies(
                uz, vz, window_center, window_fit=window_fit, min_radius=min_radius, paral=paral
            )
            if z == len(depth) - 1:
                for i, eddy in enumerate(eddies_z.eddies):
                    eddy.z_id = i
                nb_eddies = len(eddies_z.eddies)
            else:
                Associate(eddies_tmp.eddies, eddies_z.eddies, max_distance=max_distance).order()
                for eddy in eddies_z.eddies:
                    if not hasattr(eddy, 'z_id'):
                        eddy.z_id = nb_eddies
                        nb_eddies += 1

            eddies3d[z] = eddies_z
            eddies_tmp = eddies_z

        return cls(u, v, depth, eddies3d, nb_eddies)


class RawEddy3D:
    def __init__(self, depths, eddies):
        self.depths = depths
        self.eddies = eddies  # list of RawEddy2D

    @property
    def min_depth(self):
        return min(np.abs(self.depths))

    @property
    def max_depth(self):
        return max(np.abs(self.depths))

    @property
    def vmax(self):
        "return the maximum speed of the eddy"
        return max([e.vmax_contour.mean_velocity for e in self.eddies])

    @property
    def vmax_depth(self):
        """return the depth of the maximum speed"""
        ivmax = np.argmax([e.vmax_contour.mean_velocity for e in self.eddies])
        return np.abs(self.depths[ivmax])


class Eddies3D:
    def __init__(self, u, v, depths, eddies, eddies_byslice):
        self.eddies = eddies
        self.eddies_byslice = eddies_byslice
        self.depths = depths
        self.u = u
        self.v = v

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
        max_distance=10,
        **kwargs,
    ):

        eddies = EddiesByDepth.detect_eddies_3d(
            u,
            v,
            window_center,
            window_fit=window_fit,
            dx=dx,
            dy=dy,
            min_radius=min_radius,
            ssh_method=ssh_method,
            paral=paral,
            max_distance=max_distance,
            **kwargs,
        )

        eddies_3d = []
        for z_id in range(eddies.nb_eddies):
            e_depth = []
            e = []
            for i, k in enumerate(eddies.eddies3d):
                eddies_2d = eddies.eddies3d[k]
                for eddy in eddies_2d.eddies:
                    if eddy.z_id == z_id:
                        e.append(eddy)
                        e_depth.append(eddies.depth[k].values)
            eddies_3d.append(RawEddy3D(np.array(e_depth), e))
        return cls(u, v, eddies.depth.values, eddies_3d, eddies)

    def plot2d(self, depth=0, boundary=False, vmax=False, quiver=False, ns=5):
        """
        plot the 2D vorticity field superimposed with eddies detection at the nearest
        layer depth
        """

        lon = scf.get_lon(self.u)
        lat = scf.get_lat(self.u)
        i = np.argmin(np.abs(depth - np.abs(self.depths)))
        fig, ax = splot.create_map(
            lon,
            lat,
            figsize=(8, 5),
        )
        sdyn.get_relvort(self.u.isel(depth=i), self.v.isel(depth=i)).plot(
            x="lon_rho",
            y="lat_rho",
            cmap="cmo.curl",
            ax=ax,
            add_colorbar=False,
            transform=splot.pcarr,
        )

        if quiver:
            plt.quiver(
                lon[::ns, ::ns].values,
                lat[::ns, ::ns].values,
                self.u.isel(depth=i)[::ns, ::ns].values,
                self.v.isel(depth=i)[::ns, ::ns].values,
                color="k",
                transform=splot.pcarr,
            )

        for eddy3d in self.eddies:
            inearest = np.argmin(np.abs(depth - np.abs(eddy3d.depths)))
            eddy2d = eddy3d.eddies[inearest]
            eddy2d.plot(boundary=boundary, vmax=vmax, transform=splot.pcarr)
            cb = plt.scatter(
                eddy2d.lon,
                eddy2d.lat,
                c=eddy3d.max_depth,
                cmap="Spectral_r",
                vmin=np.min(np.abs(self.depths)),
                vmax=np.max(np.abs(self.depths)),
                transform=splot.pcarr,
            )
        plt.colorbar(cb, label="maximum depth")
