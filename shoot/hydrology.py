# -*- coding: utf-8 -*-
"""
hydrologic functions
====================
"""
import functools
import math
import numpy as np
import numba
import xarray as xr
from scipy.signal import argrelmax, argrelmin
import xoa.coords as xcoords
from . import grid as sgrid
from . import num as snum


class Anomaly:
    def __init__(self, eddy, eddies, dens, depth=None, r_factor=1.2, nz=100, eddy_type= True):
        self.lon = eddy.lon
        self.lat = eddy.lat
        self.eddy = eddy
        self.eddy_type= eddy_type #True if anomaly sign based on eddy_type
        if hasattr(eddy, "boundary_contour"):
            self.radius = (
                eddy.boundary_contour.radius
            )  # eddy.vmax_contour.radius  # eddy.radius in meters
        else:
            self.radius = eddy.eff_radius  # boundary contour radius in meters

        self.dens = dens.squeeze()
        if not depth is None:
            self.depth = depth.squeeze()
        else:
            self.depth = xcoords.get_depth(dens).squeeze()
        self.xdim = xcoords.get_xdim(self.dens, errors="raise")
        self.ydim = xcoords.get_ydim(self.dens, errors="raise")
        self.zdim = xcoords.get_zdim(self.dens, errors="raise")
        self.nz = nz
        self._jmax = len(self.dens[self.xdim])
        self._imax = len(self.dens[self.ydim])
        self._r = r_factor
        self.eddies = eddies  # The whole eddies file

        if not hasattr(self.depth, self.xdim):
            self.depth = self.depth.expand_dims(
                {
                    self.xdim: self.dens.sizes[self.xdim],
                    self.ydim: self.dens.sizes[self.ydim],
                }
            ).broadcast_like(self.dens)

    @functools.cached_property
    def _dist(self):
        lon_name = xcoords.get_lon(self.dens).name
        lat_name = xcoords.get_lat(self.dens).name
        dist = np.sqrt(
            (self.dens[lon_name] - self.lon) ** 2 + (self.dens[lat_name] - self.lat) ** 2
        ).values
        return dist

    @property
    def _i(self):
        lon_name = xcoords.get_lon(self.dens).name
        return np.unravel_index(np.argmin(self._dist), self.dens[lon_name].shape)[0]

    @property
    def _j(self):
        lon_name = xcoords.get_lon(self.dens).name
        return np.unravel_index(np.argmin(self._dist), self.dens[lon_name].shape)[1]

    @functools.cached_property
    def depth_vector(self):
        depth = self.depth.isel({self.xdim: self._j, self.ydim: self._i})
        return xr.DataArray(
            np.linspace(depth.values[0], depth.values[-1], self.nz),
            dims="depth_int",
        )

    @functools.cached_property
    def profil_inside(self):
        inside = self.dens.isel({self.xdim: self._j, self.ydim: self._i})
        if self.depth_vector[0] < self.depth_vector[-1]:  # increasing case
            return np.interp(
                self.depth_vector,
                self.depth.isel({self.xdim: self._j, self.ydim: self._i}),
                inside,
            )
        else:  # decreasing order bad for np.interp
            return np.interp(
                self.depth_vector[::-1],
                self.depth.isel({self.xdim: self._j, self.ydim: self._i})[::-1],
                inside[::-1],
            )[::-1]

    def is_inside(self, x, y):
        lon = [
            xcoords.get_lon(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        lat = [
            xcoords.get_lat(self.dens).isel({self.xdim: xi, self.ydim: yi}) for xi, yi in zip(x, y)
        ]
        points = np.array([lon, lat]).T

        if hasattr(self.eddy, "x_vmax"):
            xx = self.eddy.x_vmax
            yy = self.eddy.y_vmax
        else:
            xx = self.eddy.vmax_contour.lon
            yy = self.eddy.vmax_contour.lat
        result = snum.points_in_polygon(points, np.array([xx, yy]).T)
        return result

    def _xy(self, r):
        dx, dy = sgrid.get_dx_dy(self.dens)
        dxm = np.nanmean(dx)
        dym = np.nanmean(dy)
        nx = int(r * self.radius / dxm)
        ny = int(r * self.radius / dym)
        X = [
            self._j,
            self._j,
            min(self._j + nx, self._jmax - 1),
            max(self._j - nx, 0),
        ]
        Y = [
            min(self._i + ny, self._imax - 1),
            max(self._i - ny, 0),
            self._i,
            self._i,
        ]

        stepx = int(2 * nx / 10)
        stepy = int(2 * ny / 10)
        X = np.arange(max(self._j - nx, 0), min(self._j + nx, self._jmax - 1) + 1, stepx)
        Y = np.arange(max(self._i - ny, 0), min(self._i + ny, self._imax - 1) + 1, stepy)

        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()
        return X, Y

    @functools.cached_property
    def _xy_inside(self):
        X, Y = self._xy(self._r)
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
            if hasattr(eddy, "x_eff"):
                xx = eddy.x_eff
                yy = eddy.y_eff
            else:
                xx = eddy.boundary_contour.lon
                yy = eddy.boundary_contour.lat
            result *= np.invert(snum.points_in_polygon(points, np.array([xx, yy]).T))
        return result

    @functools.cached_property
    def _xy_outside(self):
        test = True
        r = self._r
        while test:  # increase r factor if no outside point founded
            X, Y = self._xy(r)
            # test validy
            valids = self.is_valid(X, Y)
            X = [X[i] for i in range(len(X)) if valids[i]]
            Y = [Y[i] for i in range(len(Y)) if valids[i]]
            if len(X) > 0 and len(Y) > 0:
                test = False
            else:
                r *= 2
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

    @staticmethod
    def _interp_profils(depth, var, depth_vector):
        if depth_vector[0] < depth_vector[-1]:
            return np.interp(
                depth_vector,
                depth,
                var,
            )
        else:
            return np.interp(
                depth_vector[::-1],
                depth[::-1],
                var[::-1],
            )[::-1]

    @functools.cached_property
    def profils_inside(self):
        return xr.apply_ufunc(
            self._interp_profils,
            self._depths_inside.transpose("nb_profil", self.zdim),
            self._profils_inside.transpose("nb_profil", self.zdim),
            self.depth_vector,
            input_core_dims=[
                [self.zdim],
                [self.zdim],
                [self.depth_vector.dims[0]],
            ],
            output_core_dims=[[self.depth_vector.dims[0]]],
            dask_gufunc_kwargs={
                "output_sizes": {self.depth_vector.dims[0]: len(self.depth_vector)}
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.profil_inside.dtype],
        ).compute()

    @functools.cached_property
    def profils_outside(self):
        return xr.apply_ufunc(
            self._interp_profils,
            self._depths_outside.transpose("nb_profil", self.zdim),
            self._profils_outside.transpose("nb_profil", self.zdim),
            self.depth_vector,
            input_core_dims=[
                [self.zdim],
                [self.zdim],
                [self.depth_vector.dims[0]],
            ],
            output_core_dims=[[self.depth_vector.dims[0]]],
            dask_gufunc_kwargs={
                "output_sizes": {self.depth_vector.dims[0]: len(self.depth_vector)}
            },
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.profil_inside.dtype],
        ).compute()

    @functools.cached_property
    def mean_profil_inside(self):
        return self.profils_inside.mean(dim="nb_profil")

    @functools.cached_property
    def std_profil_inside(self):
        return self.profils_inside.std(dim="nb_profil")

    @functools.cached_property
    def mean_profil_outside(self):
        return self.profils_outside.mean(dim="nb_profil")

    @functools.cached_property
    def std_profil_outside(self):
        return self.profils_outside.std(dim="nb_profil")

    @functools.cached_property
    def anomaly(self):
        return self.mean_profil_inside - self.mean_profil_outside

    @functools.cached_property
    def center_anomaly(self):
        return self.profil_inside - self.mean_profil_outside

    @functools.cached_property
    def _icore_depth(self):
        if np.isnan(self.anomaly).all():
            return None
        if self.eddy_type : 

            if self.eddy.eddy_type == "anticyclone":
                icore = np.argmin(self.anomaly.values)
            else:
                icore = np.argmax(self.anomaly.values)

        else : 
            icore = np.argmax(np.abs(self.anomaly.values))
        return icore

    @functools.cached_property
    def core_depth(self):
        return np.abs(self.depth_vector[self._icore_depth])

    @functools.cached_property
    def intensity(self):
        return np.abs(self.anomaly[self._icore_depth])
    
    @functools.cached_property
    def signed_intensity(self):
        return self.anomaly[self._icore_depth]
    
    def anomaly_at_depth(self, depth_level, signed = False): 
        if np.sign(depth_level) != np.sign(self.depth_vector[1]): 
            depth_level *=-1 
        
        iref = np.argmin(np.abs(self.depth_vector.values-depth_level)) 
        if signed : 
            return self.anomaly[iref]          
        else : 
            return np.abs(self.anomaly[iref])
        

def compute_anomalies(eddies, dens, nz=100, r_factor=1.2, eddy_type= True):
    """Add anomaly to detected eddies
    Parameters
    ----------
    eddies: Eddies object

    dens: 3D dataarray
        Variable of interest including a depth dimension

    nz: int (optional)
        number of depth layers
    r_factor: float (optional)
        maximum distance to the center to combute background values

    Returns
    -------
     nothing : directly appends anomaly object in each eddies
    """
    for eddy in eddies.eddies:
        eddy.anomaly = Anomaly(eddy, eddies, dens, r_factor=r_factor, nz=nz, eddy_type=eddy_type)
        

    
        

