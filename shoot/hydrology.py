# -*- coding: utf-8 -*-
"""
hydrologic functions
"""
import functools
import math
import numpy as np
import numba
import xarray as xr
import xoa.coords as xcoords
from . import grid as sgrid
from . import num as snum


def compute_cs(ds):
    return "TODO difficult to get the name of temp and salt variables"


def compute_dens(ds):
    return "TODO difficult to get the name of temp and salt variables"


class Anomaly:
    def __init__(self, eddy, eddies, dens, depth=None, r_factor=1.2, nz=100):
        self.lon = eddy.glon
        self.lat = eddy.glat
        self.eddy = eddy
        self.radius = (
            eddy.boundary_contour.radius
        )  # eddy.vmax_contour.radius  # eddy.radius in meters
        self.dens = dens.squeeze()
        if not depth is None:
            self.depth = depth.squeeze()
        else:
            self.depth = xcoords.get_depth(dens).squeeze()
        self.xdim = xcoords.get_xdim(self.dens, errors="raise")
        self.ydim = xcoords.get_ydim(self.dens, errors="raise")
        self.nz = nz
        self._jmax = len(self.dens[self.xdim])
        self._imax = len(self.dens[self.ydim])
        self._r = r_factor
        self.eddies = eddies  # The whole eddies file

    @property
    def _i(self):
        lon_name = xcoords.get_lon(self.dens).name
        lat_name = xcoords.get_lat(self.dens).name
        ij = np.where((self.dens[lon_name] == self.lon) & (self.dens[lat_name] == self.lat))
        return ij[0][0]

    @property
    def _j(self):
        lon_name = xcoords.get_lon(self.dens).name
        lat_name = xcoords.get_lat(self.dens).name
        ij = np.where((self.dens[lon_name] == self.lon) & (self.dens[lat_name] == self.lat))
        return ij[1][0]

    @functools.cached_property
    def depth_vector(self):
        depth = self.depth.isel({self.xdim: self._j, self.ydim: self._i})
        # return np.linspace(depth.min().values, depth.max().values, self.nz)
        return np.linspace(depth.values[0], depth.values[-1], self.nz)

    @functools.cached_property
    def profil_inside(self):
        inside = self.dens.isel({self.xdim: self._j, self.ydim: self._i})
        return np.interp(
            self.depth_vector,
            self.depth.isel({self.xdim: self._j, self.ydim: self._i}),
            inside,
        )

    def is_inside(self, x, y):
        lon = [
            xcoords.get_lon(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        lat = [
            xcoords.get_lat(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        points = np.array([lon, lat]).T

        xx = self.eddy.vmax_contour.lon
        yy = self.eddy.vmax_contour.lat
        result = snum.points_in_polygon(points, np.array([xx, yy]).T)
        return result

    @functools.cached_property
    def _xy_inside(self):
        dx, dy = sgrid.get_dx_dy(self.dens)
        dxm = np.nanmean(dx)
        dym = np.nanmean(dy)
        nx = int(self._r * self.radius / dxm)
        ny = int(self._r * self.radius / dym)
        X = [self._j, self._j, min(self._j + nx, self._jmax - 1), max(self._j - nx, 0)]
        Y = [min(self._i + ny, self._imax - 1), max(self._i - ny, 0), self._i, self._i]

        stepx = int(2 * nx / 10)
        stepy = int(2 * ny / 10)
        X = np.arange(max(self._j - nx, 0), min(self._j + nx, self._jmax - 1) + stepx, stepx)
        Y = np.arange(max(self._i - ny, 0), min(self._i + ny, self._imax - 1) + stepy, stepy)

        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        # test validy
        valids = self.is_inside(X, Y)
        X = [X[i] for i in range(len(X)) if valids[i]]
        Y = [Y[i] for i in range(len(Y)) if valids[i]]
        return (X, Y)

    @functools.cached_property
    def _profils_inside(self):
        X = self._xy_inside[0]
        Y = self._xy_inside[1]

        return self.dens.isel(
            {
                self.xdim: xr.DataArray(
                    X,
                    dims="nb_profil",
                ),
                self.ydim: xr.DataArray(
                    Y,
                    dims="nb_profil",
                ),
            }
        )

    def is_valid(self, x, y):
        lon = [
            xcoords.get_lon(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        lat = [
            xcoords.get_lat(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        points = np.array([lon, lat]).T
        result = np.ones(len(x)) * True
        for eddy in self.eddies.eddies:
            xx = eddy.boundary_contour.lon
            yy = eddy.boundary_contour.lat
            result *= np.invert(snum.points_in_polygon(points, np.array([xx, yy]).T))
        return result

    @functools.cached_property
    def _xy_outside(self):
        dx, dy = sgrid.get_dx_dy(self.dens)
        dxm = np.nanmean(dx)
        dym = np.nanmean(dy)
        nx = int(self._r * self.radius / dxm)
        ny = int(self._r * self.radius / dym)
        X = [self._j, self._j, min(self._j + nx, self._jmax - 1), max(self._j - nx, 0)]
        Y = [min(self._i + ny, self._imax - 1), max(self._i - ny, 0), self._i, self._i]

        stepx = int(2 * nx / 10)
        stepy = int(2 * ny / 10)
        X = np.arange(max(self._j - nx, 0), min(self._j + nx, self._jmax - 1) + stepx, stepx)
        Y = np.arange(max(self._i - ny, 0), min(self._i + ny, self._imax - 1) + stepy, stepy)

        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        # test validy
        valids = self.is_valid(X, Y)
        X = [X[i] for i in range(len(X)) if valids[i]]
        Y = [Y[i] for i in range(len(Y)) if valids[i]]
        return (X, Y)

    @functools.cached_property
    def _profils_outside(self):
        X = self._xy_outside[0]
        Y = self._xy_outside[1]

        return self.dens.isel(
            {
                self.xdim: xr.DataArray(
                    X,
                    dims="nb_profil",
                ),
                self.ydim: xr.DataArray(
                    Y,
                    dims="nb_profil",
                ),
            }
        )

    @functools.cached_property
    def _depths_inside(self):
        X = self._xy_inside[0]
        Y = self._xy_inside[1]
        return self.depth.isel(
            {
                self.xdim: xr.DataArray(
                    X,
                    dims="nb_profil",
                ),
                self.ydim: xr.DataArray(
                    Y,
                    dims="nb_profil",
                ),
            }
        )

    @functools.cached_property
    def _depths_outside(self):
        X = self._xy_outside[0]
        Y = self._xy_outside[1]
        return self.depth.isel(
            {
                self.xdim: xr.DataArray(
                    X,
                    dims="nb_profil",
                ),
                self.ydim: xr.DataArray(
                    Y,
                    dims="nb_profil",
                ),
            }
        )

    @functools.cached_property
    def mean_profil_inside(self):
        p = np.empty((len(self._profils_inside.nb_profil), self.nz))
        for i in range(len(self._profils_inside.nb_profil)):
            p[i] = np.interp(
                self.depth_vector,
                self._depths_inside.isel(nb_profil=i),
                self._profils_inside.isel(nb_profil=i),
            )
        return p.mean(axis=0)

    @functools.cached_property
    def std_profil_inside(self):
        p = np.empty((len(self._profils_inside.nb_profil), self.nz))
        for i in range(len(self._profils_inside.nb_profil)):
            p[i] = np.interp(
                self.depth_vector,
                self._depths_inside.isel(nb_profil=i),
                self._profils_inside.isel(nb_profil=i),
            )
        return p.std(axis=0)

    @functools.cached_property
    def mean_profil_outside(self):
        p = np.empty((len(self._profils_outside.nb_profil), self.nz))
        for i in range(len(self._profils_outside.nb_profil)):
            p[i] = np.interp(
                self.depth_vector,
                self._depths_outside.isel(nb_profil=i),
                self._profils_outside.isel(nb_profil=i),
            )
        return p.mean(axis=0)

    @functools.cached_property
    def std_profil_outside(self):
        p = np.empty((len(self._profils_outside.nb_profil), self.nz))
        for i in range(len(self._profils_outside.nb_profil)):
            p[i] = np.interp(
                self.depth_vector,
                self._depths_outside.isel(nb_profil=i),
                self._profils_outside.isel(nb_profil=i),
            )
        return p.std(axis=0)

    @functools.cached_property
    def anomaly(self):  ## Pour l'instant fait hypothèse de niveaux équirépartie
        # return self.profil_inside - self.mean_profil_outside
        return self.mean_profil_inside - self.mean_profil_outside

    @functools.cached_property
    def center_anomaly(self):  ## Pour l'instant fait hypothèse de niveaux équirépartie
        return self.profil_inside - self.mean_profil_outside


def compute_anomalies(eddies, dens, nz=100, r_factor=1.2):
    """
    eddies is an Eddies object
    dens is the density (temp or salt) dataarray containing depth
    The function add an anomaly object for each eddies
    """
    for eddy in eddies.eddies:
        eddy.anomaly = Anomaly(eddy, eddies, dens, r_factor=r_factor, nz=nz)
