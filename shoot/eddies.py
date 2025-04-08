# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:39:51 2024 by sraynaud
"""
import functools
import warnings
import numpy as np
from scipy.interpolate import splprep, make_interp_spline, splev
import multiprocessing as mp
import itertools
from itertools import repeat
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

COLORS = {"anticyclone": "tab:red", "cyclone": "tab:blue", "undefined": "0.5"}


def find_eddy_centers(u, v, window, dx=None, dy=None, paral=False):
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
    wx, wy = sgrid.get_wx_wy(window, dxm, dym)  ## WARNING IT HAS BEEN MODIFIED
    minima, maxima = snum.find_signed_peaks_2d(lnam.values, wx, wy, paral=paral)
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

    return (
        xr.Dataset(
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
        ),
        lnam,
        ow,
        extrema,
    )


class Ellipse:
    def __init__(self, lon, lat, a, b, angle, sign=0, fit=None):
        self.lon, self.lat, self.a, self.b, self.angle, self.sign = lon, lat, a, b, angle, sign
        self.fit_error = fit
        self.radius = np.sqrt(self.a**2 + self.b**2)

    @classmethod
    def from_coords(cls, lons, lats):
        params, fit = sfit.fit_ellipse_from_coords(lons, lats, get_fit=True)
        return cls(*(list(params.values()) + [0, fit]))

    @classmethod
    def reconstruct(cls, elon, elat, a, b, angle):
        return cls(elon, elat, a, b, angle)

    @property
    def _params_str_(self):
        return f"lon={self.lon}, lat={self.lat}, a={self.a}, b={self.b}, angle={self.angle}, sign={self.sign}"

    def __str__(self):
        return f"Ellipse({self._params_str_})"

    def __repr__(self):
        return f"<{self}>"

    def to_json(self):
        obj = {
            'lon': self.lon,
            'lat': self.lat,
            'a': self.a,
            'b': self.b,
            'angle': self.angle,
            'eddy_type': self.eddy_type,
        }
        return json.dumps(obj)

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
        max_ellipse_error=0.1,  # 0.1,  # 0.03,  # 0.01,
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
        self.track_id = None
        self.is_parent = False
        self.attrs = attrs

    def dictio(self):
        obj = {
            'lon': str(self.glon),
            'lat': str(self.glat),
            'i': str(self.i),
            'j': str(self.j),
            'ro': str(self.ro),
            'eddy_type': self.ellipse.eddy_type,
            'track_nb': self.track_nb,
            'parent': self.parent,
            'radius': str(self.radius),
            'length': str(self.boundary_contour.length),
            'lons': [str(a) for a in self.boundary_contour.lon.values],
            'lats': [str(a) for a in self.boundary_contour.lat.values],
            'vmax_radius': str(self.vmax_contour.radius),
            'vmax': str(self.vmax_contour.mean_velocity),
            'vmax_lons': [str(a) for a in self.vmax_contour.lon.values],
            'vmax_lats': [str(a) for a in self.vmax_contour.lat.values],
            'elon': str(self.ellipse.lon),
            'elat': str(self.ellipse.lat),
            'a': str(self.ellipse.a),
            'b': str(self.ellipse.b),
            'angle': str(self.ellipse.angle),
        }
        return obj

    @functools.cached_property
    def ssh(self):
        if self._ssh is not None:
            return self._ssh
        return strl.psi(self.u, self.v)

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
            # check if ellipse center fall inside the eddy contour
            if not snum.points_in_polygon(
                np.array([ellipse.lon, ellipse.lat]), np.array([ds.lon, ds.lat]).T
            ):
                continue
            if ellipse.fit_error < self.max_ellipse_error:
                # if True:
                ds.attrs["ellipse"] = ellipse
                scontours.add_contour_uv(ds, self.ugeos.values, self.vgeos.values)
                scontours.add_contour_dx_dy(ds)
                valid_contours.append(ds)
                ellipse.sign = np.sign(ds.mean_angular_momentum)
        return valid_contours

    @functools.cached_property
    def ncontours(self):
        return len(self.contours)

    def is_eddy(self, min_radius):  # a mettre dans RawEddy2D
        # Checks if closed contour exists
        if not self.ncontours:
            # print(self.glon, self.glat, "no contour")
            return False
        if min_radius and self.radius < min_radius:
            # print(self.glon, self.glat, "small radius")
            return False
        if np.isnan(self.vmax_contour.mean_velocity):
            # print(self.glon, self.glat, "nan velocity")
            return False
        # if self.vmax_contour.ellipse.fit_error > self.max_ellipse_error / 2: #ce test est inutile
        #     # print(self.glon, self.glat, "ellipse error")
        #     return False
        # print(self.glon, self.glat, "is eddy")
        return True

    @functools.cached_property
    def boundary_contour(self):
        if not self.ncontours:
            return
        dsb = self.contours[0]
        for ds in self.contours:
            if ds.length > dsb.length:
                dsb = ds
        # interpolation step using splrep
        ok = np.where(np.abs(np.diff(dsb.lon)) + np.abs(np.diff(dsb.lat)) > 1e-10)[0]
        ok = np.concatenate([ok, [len(dsb.lon) - 1]])
        try:
            tck, u = splprep([dsb.lon[ok], dsb.lat[ok]], s=0)  # avoid repeated values
        except ValueError:
            print("certainement un problème de redondance des points")
        xy_int = splev(np.linspace(0, 1, 50), tck)
        dsb['lon_int'] = xy_int[0]
        dsb['lat_int'] = xy_int[1]
        return dsb

    @property
    def ellipse(self):
        """Ellipse fited from :attr:`boundary_contour` or None"""
        if self.ncontours:
            # return self.boundary_contour.ellipse
            return self.vmax_contour.ellipse

    @functools.cached_property
    def radius(self):  # differs from AMEDA which compute from the AREA
        """Radius deduced from :attr:`ellipse` or 0"""
        if not self.ncontours:
            return 0.0
        # return self.ellipse.radius
        return self.boundary_contour.radius / 1000  # in km

    @functools.cached_property
    def ro(self):
        """Rosby number of the eddy"""
        f = 2 * sdyn.OMEGA * np.sin(self.glat)
        return self.vmax_contour.mean_velocity / (f * self.vmax_contour.length)

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
                if Ellipse.from_coords(ds.lon, ds.lat).fit_error > self.max_ellipse_error / 10:
                    continue
                dsv = ds
        ok = np.where(np.abs(np.diff(dsv.lon)) + np.abs(np.diff(dsv.lat)) > 0)[0]
        ok = np.concatenate([ok, [len(dsv.lon) - 1]])
        tck, u = splprep([dsv.lon[ok], dsv.lat[ok]], s=0)  # avoid repeated values
        xy_int = splev(np.linspace(0, 1, 50), tck)
        dsv['lon_int'] = xy_int[0]
        dsv['lat_int'] = xy_int[1]
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

    def contains_eddy(self, eddy):
        points = np.array([eddy.vmax_contour.lon.values, eddy.vmax_contour.lat.values]).T
        valid = snum.points_in_polygon(
            points, np.array([self.vmax_contour.lon, self.vmax_contour.lat]).T
        )
        return valid.all()

    def intersects_eddy(self, eddy):
        points = np.array([eddy.vmax_contour.lon.values, eddy.vmax_contour.lat.values]).T
        valid = snum.points_in_polygon(
            points, np.array([self.vmax_contour.lon, self.vmax_contour.lat]).T
        )
        return valid.any()

    def plot(self, ax=None, lw=1, color=None, **kwargs):
        """Quickly plot the eddy"""
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = self.color
        kw = dict(color=color, **kwargs)
        # out = {"center": ax.scatter(self.lon, self.lat, **kw)}
        out = {"center": ax.scatter(self.glon, self.glat, s=10, **kw)}
        if self.ncontours:
            # out["boundary"] = ax.plot(
            #     self.boundary_contour.lon, self.boundary_contour.lat, lw=lw, **kw
            # )
            out["boundary"] = ax.plot(
                self.boundary_contour.lon_int, self.boundary_contour.lat_int, lw=lw, **kw
            )
            # out["ellipse"] = self.boundary_contour.ellipse.plot(ax=ax, lw=lw / 2, **kw)
            # out["ellipse"] = self.vmax_contour.ellipse.plot(ax=ax, lw=lw / 2, **kw)
            out["ellipse"] = self.ellipse.plot(ax=ax, lw=lw / 2, **kw)
            out["velmax"] = ax.plot(self.vmax_contour.lon, self.vmax_contour.lat, "--", lw=lw, **kw)
        return out


class Eddy:  ##This is a minimal class without computing capabilities
    def __init__(
        self,
        lon,
        lat,
        i,
        j,
        ro,
        eddy_type,
        track_id,
        is_parent,
        radius,
        length,
        vmax_radius,
        vmax_length,
        vmax,
        elon,
        elat,
        a,
        b,
        angle,
    ):
        self.glon = lon
        self.glat = lat
        self.i = i
        self.j = j
        self.ro = ro
        self.radius = radius
        self.length = length
        self.vmax = vmax
        self.vmax_length = vmax_length
        self.vmax_radius = vmax_radius
        self.ellipse = Ellipse.reconstruct(elon, elat, a, b, angle)
        self.eddy_type = eddy_type
        self.track_id = track_id
        self.is_parent = is_parent

    @classmethod
    def reconstruct(cls, ds):
        return cls(
            float(ds.x_cen.values),
            float(ds.y_cen.values),
            int(ds.i_cen.values),
            int(ds.j_cen.values),
            float(ds.Ro.values),
            str(ds.eddy_type[int(ds.track_id.values)].values),
            int(ds.track_id.values),
            bool(ds.is_parent.values),
            float(ds.eff_radius.values),
            float(ds.eff_length.values),
            float(ds.vmax_radius.values),
            float(ds.vmax_length.values),
            float(ds.vmax.values),
            float(ds.x_ell.values),
            float(ds.y_ell.values),
            float(ds.a_ell.values),
            float(ds.b_ell.values),
            float(ds.angle_ell.values),
        )


class Eddies:
    """This class contains a list of detected eddies at one time"""

    def __init__(self, time, eddies, window_center, window_fit, min_radius):
        self.time = time
        self.eddies = eddies  # list of 2DRawEddies or Eddy object
        self.window_center = window_center
        self.window_fit = window_fit
        self.min_radius = min_radius

    @classmethod
    def reconstruct(cls, ds):
        window_center = float(ds.window_center[:-3])
        window_fit = float(ds.window_fit[:-3])
        min_radius = float(ds.min_radius[:-3])
        time = ds.time[0]
        eddies = []
        for i in range(len(ds.obs)):
            eddies.append(Eddy.reconstruct(ds.isel(obs=i)))
        return cls(time, eddies, window_center, window_fit, min_radius)

    def test_eddy(eddy, min_radius):
        if eddy.is_eddy(min_radius):
            return eddy

    @classmethod
    def detect_eddies(
        cls,
        u,
        v,
        window_center,
        window_fit=None,
        ssh=None,
        dx=None,
        dy=None,
        min_radius=None,
        **kwargs,
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
        # import time
        # start_centers = time.time()
        centers, lnam, ow, extrema = find_eddy_centers(
            u, v, window_center, dx=dxm, dy=dym, paral=False
        )
        # end_centers = time.time()
        # print("center research takes %.3fs" % (end_centers - start_centers))

        # Fit window
        wx, wy = sgrid.get_wx_wy(
            window_fit, dxm, dym
        )  # window on which we look for streamlines closed contours
        wx2 = wx // 2
        wy2 = wy // 2
        xdim = xcoords.get_xdim(u)
        ydim = xcoords.get_ydim(u)
        nx = u.sizes[xdim]
        ny = u.sizes[ydim]

        def def_eddy(ic, wx2c, wy2c):
            # Local selection
            i = int(centers.gi[ic])
            j = int(centers.gj[ic])
            imin = max(i - wx2c, 0)
            imax = min(i + wx2c + 1, nx)
            jmin = max(j - wy2c, 0)
            jmax = min(j + wy2c + 1, ny)
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
            eddy = RawEddy2D(
                i - imin,
                j - jmin,
                ul,
                vl,
                ssh=sshl,
                dx=dxl,
                dy=dyl,
                **kwargs,
            )
            eddy.attrs.update(
                lnam=float(centers.lnam[ic]),
                coriolis=float(centers.coriolis[ic]),
                window_center=window_center,
                window_fit=window_fit,
            )
            return eddy

        # start_eddy = time.time()
        eddies = []
        wx2c = wx2
        wy2c = wy2
        print("On travaille sur %i cpus" % mp.cpu_count())
        while (centers.lon.shape[0] > 0) and (wx2c < 2 * wx2):
            eddies_tmp = []
            for ic in range(centers.lon.shape[0]):
                eddies_tmp.append(def_eddy(ic, wx2c, wy2c))

            with mp.Pool(mp.cpu_count()) as p:
                eddies_tmp = p.starmap(Eddies.test_eddy, zip(eddies_tmp, repeat(min_radius)))
                p.close()

            ind_good = []
            for i, eddy in enumerate(eddies_tmp):
                if not eddy is None:
                    if wx2c + int(wx2c / 2) >= 2 * wx2:  # no more chance to be conserved
                        eddies.append(eddy)
                    else:
                        if (
                            len(eddy.vmax_contour.line) == len(eddy.boundary_contour.line)
                            and (eddy.vmax_contour.line == eddy.boundary_contour.line).all()
                        ):
                            ind_good.append(i)
                        else:
                            eddies.append(eddy)

            centers = centers.isel(neddies=ind_good)
            wx2c += int(wx2c / 2)
            wy2c += int(wy2c / 2)

        ## Cheking inclusion step
        ## This step can be modify to account for eddy-eddy interaction
        contain = np.ones(len(eddies)) * True
        for i in range(len(eddies)):
            for j in range(len(eddies)):
                if i == j:
                    continue
                # if eddies[i].contains_eddy(eddies[j]): #avoid full inclusion
                if eddies[i].intersects_eddy(eddies[j]):  # avoid intersection
                    if eddies[i].vmax_contour.mean_velocity > eddies[j].vmax_contour.mean_velocity:
                        contain[j] = False
                    else:
                        contain[i] = False

        eddies = [eddies[i] for i in range(len(eddies)) if contain[i]]
        return cls(u.time.values, eddies, window_center, window_fit, min_radius)

    @property
    def ds(self):
        return xr.Dataset(
            {
                "time": (("obs"), np.repeat(self.time, len(self.eddies))),
                "i_cen": (("obs"), [e.i for e in self.eddies]),
                "j_cen": (("obs"), [e.j for e in self.eddies]),
                "x_cen": (("obs"), [e.glon for e in self.eddies]),
                "y_cen": (("obs"), [e.glat for e in self.eddies]),
                "track_id": (("obs"), [e.track_id for e in self.eddies]),
                "is_parent": (("obs"), [e.is_parent for e in self.eddies]),
                # "eddy_type": (("obs"), [e.eddy_type for e in self.eddies]),
                "eff_radius": (("obs"), [e.radius for e in self.eddies]),
                "eff_length": (("obs"), [e.boundary_contour.length for e in self.eddies]),
                "vmax_radius": (("obs"), [e.vmax_contour.radius for e in self.eddies]),
                "vmax_length": (("obs"), [e.vmax_contour.length for e in self.eddies]),
                "vmax": (("obs"), [e.vmax_contour.mean_velocity for e in self.eddies]),
                "Ro": (("obs"), [e.ro for e in self.eddies]),
                "x_ell": (("obs"), [e.ellipse.lon for e in self.eddies]),
                "y_ell": (("obs"), [e.ellipse.lat for e in self.eddies]),
                "a_ell": (("obs"), [e.ellipse.a for e in self.eddies]),
                "b_ell": (("obs"), [e.ellipse.b for e in self.eddies]),
                "angle_ell": (("obs"), [e.ellipse.angle for e in self.eddies]),
                "x_eff_contour": (
                    ("obs", "nb_sample"),
                    [e.boundary_contour.lon_int for e in self.eddies],
                ),
                "y_eff_contour": (
                    ("obs", "nb_sample"),
                    [e.boundary_contour.lat_int for e in self.eddies],
                ),
                "x_vmax_contour": (
                    ("obs", "nb_sample"),
                    [e.vmax_contour.lon_int for e in self.eddies],
                ),
                "y_vmax_contour": (
                    ("obs", "nb_sample"),
                    [e.vmax_contour.lat_int for e in self.eddies],
                ),
            },
            attrs={
                'window_center': '%i km' % self.window_center,
                'window_fit': '%i km' % self.window_fit,
                'min_radius': '%i km' % self.min_radius,
                "project": "SHOOT",
                "institution": "SHOM",
                "contact": "jean.baptiste.roustan@shom.fr",
            },
        )

    def save(self, path_nc):
        "this save at .nc format"
        self.ds.to_netcdf(path_nc)


class EvolEddies:
    """This class contain a list of Eddies object for following times"""

    def __init__(self, eddies):
        self.eddies = eddies  # list of Eddies object
        if len(eddies) > 1:
            self.dt = (eddies[1].time - eddies[0].time) / np.timedelta64(1, 's')
        else:
            self.dt = None

    @classmethod
    def reconstruct(cls, ds):
        "reconstructs an EvolEddies object from an xarray dataset"
        eddies = []
        for t in np.unique(ds.time):
            eddies.append(Eddies.reconstruct(ds.where(ds.time == t, drop=True)))
        return cls(eddies)

    @classmethod
    def detect_eddies(cls, ds, window_center, window_fit, min_radius, u='ugos', v='vgos', ssh=None):
        "ds is a temporal dataframe"
        eddies = []
        for i in range(len(ds.time)):
            # print(np.datetime_as_string(ds.time[i], unit='D'))
            dss = ds.isel(time=i)
            if not ssh is None:
                eddies_ssh = Eddies.detect_eddies(
                    dss[u],
                    dss[v],
                    window_center,
                    window_fit=window_fit,
                    ssh=dss[ssh],
                    min_radius=min_radius,
                    paral=True,
                )
            else:
                eddies_ssh = Eddies.detect_eddies(
                    dss[u],
                    dss[v],
                    window_center,
                    window_fit=window_fit,
                    min_radius=min_radius,
                    paral=True,
                )
            eddies.append(eddies_ssh)
        return cls(eddies)

    def add(self, eddies):
        """update base on the new Eddies object"""
        self.eddies.append(eddies)

    @property
    def ds(self):
        ds = None
        for eddies in self.eddies:  # warning ifno eddies have been detected it crashes
            if ds is None:
                ds = eddies.ds
            else:
                ds = xr.concat([ds, eddies.ds], dim='obs')
        return ds

    def save(self, path_nc):
        "this save at .nc format"
        self.ds.to_netcdf(path_nc)
