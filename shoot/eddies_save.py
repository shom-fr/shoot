#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:39:51 2024 by sraynaud
"""
import functools
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import xoa.coords as xcoords
import xoa.geo as xgeo

from . import num as snum
from . import dyn as sdyn
from . import grid as sgrid
from . import fit as sfit
from . import contours as scontours
from . import plot as splot

COLORS = {"anticyclone": "tab:red", "cyclone": "tab:blue", "undefined": "0.5"}


def find_eddy_centers(u, v, window, dx=None, dy=None):
    """Find eddy centers in a velocity field

    Parameters
    ----------
    u: xarray.Dataset
        Velocity along X
    v: xarray.Dataset
        Velocity along X
    window: float
        Window in km
    dx: None, xarray.Dataset
        Resolution along X in m
    dy: None, xarray.Dataset
        Resolution along Y in m

    Returns
    -------
    xarray.Dataset
        With "lon" and "lat" coordinates and "gi" and "gj" grid indices
    """
    assert u.ndim == 2, "It only works for 2d arrays for the moment"
    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
    dxm = np.nanmean(dx)
    dym = np.nanmean(dy)

    # Local angular momentum
    lnam = sdyn.get_lnam(u, v, window, dx=dxm, dy=dym)

    # Mask with positive OW
    ow = sdyn.get_okuboweiss(u, v, dx=dxm, dy=dym)
    lnam = lnam.where(ow < 0)

    # Find local peaks
    wx, wy = sgrid.get_wx_wy(window, dxm, dym)
    minima, maxima = snum.find_signed_peaks_2d(lnam.values, wx, wy)
    extrema = np.vstack((minima, maxima))
    ii = extrema[:, 0]
    jj = extrema[:, 1]

    # Sort cyclones and anti-cyclones
    lat2d, lon2d = xr.broadcast(xcoords.get_lat(u), xcoords.get_lon(u))
    xx, yy = lon2d.values, lat2d.values
    ecorio = sdyn.get_coriolis(yy[jj, ii])
    elons = xx[jj, ii]
    elats = yy[jj, ii]
    elnam = lnam.values[jj, ii]
    eow = ow.values[jj, ii]

    return xr.Dataset(
        {
            "lnam": ("neddies", elnam, {"long_name": "Local normalized angular momentum"}),
            "coriolis": ("neddies", ecorio, {"long_name": "Coriolis parameter"}),
            "ow": ("neddies", eow, {"long_name": "Okubow Weiss"}),
        },
        coords={
            "gi": ("neddies", ii, {"long_name": "Grid indices along X"}),
            "gj": ("neddies", jj, {"long_name": "Grid indices along Y"}),
            "lon": ("neddies", elons, {"long_name": "Longitudes"}),
            "lat": ("neddies", elats, {"long_name": "Latitudes"}),
        },
        attrs={"window": window, "wx": wx, "wy": wy, "dx_mean": dxm, "dy_mean": dym},
    ), lnam, ow, extrema


class Ellipse:
    def __init__(self, lon, lat, a, b, angle, sign=0, fit=None):
        self.lon, self.lat, self.a, self.b, self.angle, self.sign = lon, lat, a, b, angle, sign
        self.fit = fit
        self.fit_error = fit.fun if fit is not None else None
        self.radius = np.sqrt(self.a**2 + self.b**2)

    @classmethod
    def from_coords(cls, lons, lats):
        params, fit = sfit.fit_ellipse_from_coords(lons, lats, get_fit=True)
        return cls(*(list(params.values()) + [0, fit]))

    @property
    def _params_str_(self):
        return f"lon={self.lon}, lat={self.lat}, a={self.a}, b={self.b}, angle={self.angle}, sign={self.sign}"

    def __str__(self):
        return f"Ellipse({self._params_str_})"

    def __repr__(self):
        return f"<{self}>"

    # @staticmethod
    # def get_discretization_error(npts):
    #     """https://www.wolframalpha.com/input?i=integrate+%281+-+%28cos+y%29+%2F+%28cos+x%29%29%5E2+dx"""
    #     alpha = np.pi / npts
    #     a2 = alpha / 2
    #     c2 = np.cos(a2)
    #     s2 = np.sin(a2)
    #     out = np.sin(alpha) * np.cos(alpha)
    #     out += 2 * np.cos(alpha) * (np.log(c2 - s2) - np.log(s2 + c2))
    #     out += alpha
    #     out /= alpha
    #     return out

    @property
    def eddy_type(self):
        if self.sign == 0:
            return "undefined"
        if (self.sign * self.lat) > 0:
            return "cylone"
        return "anticyclone"

    @property
    def color(self):
        return COLORS[self.eddy_type]

    def plot(self, ax=None, color=None, npts=100, **kwargs):
        if color is None:
            color = self.color

        return splot.plot_ellipse(
            self.lon, self.lat, self.a, self.b, self.angle, ax=ax, npts=npts, color=color, **kwargs
        )


class RawEddy2D:
    """A basic eddy attached to a grid point"""

    def __init__(
        self,
        i,
        j,
        u,
        v,
        ssh=None,
        dx=None,
        dy=None,
        uv_error=0.01,
        max_ellipse_error=0.01,
        nlevels=100,
        robust=0.03,
        **attrs,
    ):
        self.i, self.j = i, j
        lat = xcoords.get_lat(u)
        lon = xcoords.get_lon(u)
        if lon.ndim == 1:
            self.glon, self.glat = float(lon[i]), float(lat[j])
        else:
            self.glon, self.glat = float(lon[j, i]), float(lat[j, i])
        self.u, self.v = u, v
        self._ssh = ssh
        self._dx, self._dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)
        self.uv_error = uv_error
        self.max_ellipse_error = max_ellipse_error
        self.nlevels = nlevels
        self.robust = robust
        self.attrs = attrs

    @functools.cached_property
    def ssh(self):
        if self._ssh is not None:
            return self._ssh
        return sfit.fit_ssh_from_uv(
            self.u, self.v, dx=self._dx, dy=self._dy, uv_error=self.uv_error
        )

    @functools.cached_property
    def _uvgeos(self):
        return sdyn.get_geos(self.ssh)

    @property
    def ugeos(self):
        return self._uvgeos[0]

    @property
    def vgeos(self):
        return self._uvgeos[1]

    @functools.cached_property
    def contours(self):
        # Closed contours
        dss = scontours.get_closed_contours(
            self.glon, self.glat, self.ssh, nlevels=self.nlevels, robust=self.robust
        )

        # Fit ellipses, add currents and filter
        valid_contours = []
        for ds in dss:
            ellipse = Ellipse.from_coords(ds.lon, ds.lat)
            if ellipse.fit_error < self.max_ellipse_error:
                ds.attrs["ellipse"] = ellipse
                scontours.add_contour_uv(ds, self.ugeos.values, self.vgeos.values)
                scontours.add_contour_dx_dy(ds)
                valid_contours.append(ds)
                ellipse.sign = np.sign(ds.mean_angular_momentum)
        return valid_contours

    @functools.cached_property
    def ncontours(self):
        return len(self.contours)

    @functools.cached_property
    def boundary_contour(self):
        if not self.ncontours:
            return
        dsb = self.contours[0]
        for ds in self.contours:
            if ds.length > dsb.length:
                dsb = ds
        return dsb

    @property
    def ellipse(self):
        """Ellipse fited from :attr:`boundary_contour` or None"""
        if self.ncontours:
            return self.boundary_contour.ellipse

    @functools.cached_property
    def radius(self):
        """Radius deduced from :attr:`ellipse` or 0"""
        if not self.ncontours:
            return 0.0
        return self.ellipse.radius

    @functools.cached_property
    def elon(self):
        """Longitude of :attr:`ellipse` or None"""
        if self.ellipse:
            return self.ellipse.lon

    @functools.cached_property
    def elat(self):
        """Latitude of :attr:`ellipse` or None"""
        if self.ellipse:
            return self.ellipse.lat

    @functools.cached_property
    def lon(self):
        """Longitude of center either from grid or :attr:`ellipse`"""
        return self.ellipse.lon if self.ellipse else self.glon

    @functools.cached_property
    def lat(self):
        """Latitude of center either from grid or :attr:`ellipse`"""
        return self.ellipse.lat if self.ellipse else self.glat

    @functools.cached_property
    def vmax_contour(self):
        if not self.ncontours:
            return
        dsv = self.contours[0]
        for ds in self.contours:
            if ds.mean_velocity > dsv.mean_velocity:
                dsv = ds
        return dsv

    @property
    def sign(self):
        if not self.ncontours:
            return 0
        return np.sign(self.vmax_contour.mean_angular_momentum)

    @functools.cached_property
    def coriolis(self):
        return sdyn.get_corio(self.lat)

    @functools.cached_property
    def eddy_type(self):
        if not self.ncontours:
            return "invalid"
        if (self.sign * self.lat) > 0:
            return "cyclone"
        return "anticyclone"

    @functools.cached_property
    def color(self):
        return COLORS.get(self.eddy_type, COLORS["undefined"])

    def contains_points(self, lons, lats):
        if not self.is_valid():
            return np.zeros(lons.shape, dtype="?")
        points = np.array([lons, lats]).T
        return snum.points_in_polygon(points, self.boundary_contour)

    def contains_da(self, da):
        if not self.is_valid():
            valid = np.zeros(da.shape, dtype="?")
        else:
            lon = xcoords.get_lon(da)
            lat = xcoords.get_lat(da)
            lats, lons = xr.broadcast(lat, lon)
            points = np.array([lons.values, lats.values]).T
            valid = snum.points_in_polygon(points, self.boundary_contour)
        return xr.DataArray(valid, dims=da.dims, coords=da.coords)

    def plot(self, ax=None, lw=1, color=None, **kwargs):
        """Quickly plot the eddy"""
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self.color
        kw = dict(color=color, **kwargs)
        out = {"center": ax.scatter(self.lon, self.lat, **kw)}
        if self.ncontours:
            out["boundary"] = ax.plot(
                self.boundary_contour.lon, self.boundary_contour.lat, lw=lw, **kw
            )
            out["ellipse"] = self.boundary_contour.ellipse.plot(ax=ax, lw=lw / 2, **kw)
            out["velmax"] = ax.plot(self.vmax_contour.lon, self.vmax_contour.lat, "--", lw=lw, **kw)
        return out


def detect_eddies(
    u, v, window_center, window_fit=None, ssh=None, dx=None, dy=None, min_radius=None, **kwargs
):
    """Detect all eddies in a velocity field"""
    if window_fit is None:
        window_fit = 1.5 * window_center

    if dx is None or dy is None:
        dxdy = sgrid.get_dx_dy(u)
        if dx is None:
            dx = dxdy[0]
        if dy is None:
            dy = dxdy[1]
    dxm = np.nanmean(dx)
    dym = np.nanmean(dy)

    lat = xcoords.get_lat(u)
    lon = xcoords.get_lon(u)
    lat2d, lon2d = xr.broadcast(lat, lon)

    # Find center of cyclones and anticyclones
    centers, lnam, ow, extrema = find_eddy_centers(u, v, window_center, dx=dxm, dy=dym)

    # Fit window
    wx, wy = sgrid.get_wx_wy(window_fit, dxm, dym) #window on which we look for streamlines closed contours 
    wx2 = wx // 2
    wy2 = wy // 2
    xdim = xcoords.get_xdim(u)
    ydim = xcoords.get_ydim(u)
    nx = u.sizes[xdim]
    ny = u.sizes[ydim]

    # Loop on detected centers
    eddies = []
    for ic in range(centers.lon.shape[0]):

        # Local selection
        i = int(centers.gi[ic])
        j = int(centers.gj[ic])
        imin = max(i - wx2, 0)
        imax = min(i + wx2 + 1, nx)
        jmin = max(j - wy2, 0)
        jmax = min(j + wy2 + 1, ny)
        isel = {xdim: slice(imin, imax), ydim: slice(jmin, jmax)}
        ul = u[isel]
        vl = v[isel]
        sshl = ssh[isel] if ssh is not None else None
        if isinstance(dx, xr.DataArray):
            dxl = dx[isel]
            dyl = dy[isel]
        else:
            dxl, dyl = dx, dy

        # Init eddy
        eddy = RawEddy2D(i - imin, j - jmin, ul, vl, ssh=sshl, dx=dxl, dy=dyl, **kwargs)

        # Checks
        if not eddy.ncontours:
            continue
        if eddy.sign != np.sign(centers.lnam.values[ic]):
            warnings.warn("Eddy sign inconsistency. Skipped...")
            raise ("Eddy sign inconsistency. Skipped...")
            continue
        if min_radius and eddy.radius < min_radius:
            continue
        if len(eddy.contours):
            eddies.append(eddy)

        eddy.attrs.update(
            lnam=float(centers.lnam[ic]),
            coriolis=float(centers.coriolis[ic]),
            window_center=window_center,
            window_fit=window_fit,
        )

    return eddies, lnam, ow, extrema
