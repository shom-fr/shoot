#!/usr/bin/env python3
"""
Created on Thu Aug  19 10:20:12 2025

@author: jbroust
"""

import functools
import os

import numpy as np
import xarray as xr

from .. import geo as sgeo
from .download import load_from_ds


def _scalar(values):
    """Scalar from a single-element array"""
    values = np.asarray(values)
    return values.item() if values.dtype.kind not in "mM" else values.reshape(-1)[0]


def _interp_to_depths(prf, name, depth):
    """Interpolate a profile variable to depths, using valid levels only

    Levels with a missing pressure or value are dropped and pressures are
    sorted before a linear interpolation. Values outside the measured range
    are NaN.
    """
    pres = prf.PRES.values.reshape(-1)
    values = prf[name].values.reshape(-1)
    valid = ~(np.isnan(pres) | np.isnan(values))
    if valid.sum() < 2:
        return np.full(len(depth), np.nan)
    da = xr.DataArray(values[valid], dims="PRES", coords={"PRES": pres[valid]})
    da = da.sortby("PRES").drop_duplicates("PRES")
    if da.size < 2:
        return np.full(len(depth), np.nan)
    return da.interp(PRES=depth).values


class Profile:
    """Individual in-situ profile with temperature and salinity data

    Parameters
    ----------
    prf : xarray.Dataset
        Single Argo profile as given by argopy, with ``TIME``, ``LATITUDE``,
        ``LONGITUDE``, ``PRES``, ``TEMP``, ``PSAL`` and optionally ``PLATFORM_NUMBER``.
    depth : array-like, optional
        Depths in meters of the interpolated profile. Defaults to 1 to 2000 m.
        Pressure in dbar is used as depth in meters.
    max_nan_fraction : float, default 0.8
        Maximum fraction of NaNs of the interpolated temperature for a valid profile.
    """

    def __init__(self, prf, depth=None, max_nan_fraction=0.8):
        self.time = _scalar(prf.TIME.values)
        self.lat = float(_scalar(prf.LATITUDE.values))
        self.lon = float(_scalar(prf.LONGITUDE.values))
        self.float_id = _scalar(prf.PLATFORM_NUMBER.values) if "PLATFORM_NUMBER" in prf.variables else None
        self.depth = np.arange(1, 2001) if depth is None else np.asarray(depth)
        self.temp = _interp_to_depths(prf, "TEMP", self.depth)
        self.sal = _interp_to_depths(prf, "PSAL", self.depth)
        self.valid = bool(np.sum(np.isnan(self.temp)) < max_nan_fraction * len(self.temp))


class Profiles:
    """Collection of in-situ profiles with temperature and salinity data

    Manages a collection of Profile objects, providing methods to load from datasets,
    convert to xarray format, save to NetCDF, and associate with eddies.

    Parameters
    ----------
    time : xarray.DataArray
        Time coordinate for the profiles.
    root_path : str
        Root directory path for data storage.
    brut_prf : xarray.Dataset
        Raw profile dataset.
    kwargs
        Passed to :class:`Profile`.

    Attributes
    ----------
    profiles : list of Profile
        List of valid Profile objects.
    years : ndarray
        Array of years covered by the profiles.
    ds : xarray.Dataset
        Profiles converted to xarray Dataset format (cached property).
    """

    def __init__(self, time, root_path, brut_prf, **kwargs):
        """Initialize Profiles collection

        Parameters
        ----------
        time : xarray.DataArray
            Time coordinate for the profiles.
        root_path : str
            Root directory path for data storage.
        brut_prf : xarray.Dataset
            Raw profile dataset.
        kwargs
            Passed to :class:`Profile`.
        """
        self.root_path = root_path
        self.time = time
        self.years = np.arange(self.time.min().dt.year.values, self.time.max().dt.year.values + 1)

        self.profiles = []
        self.float_ids = []
        for i in range(brut_prf.sizes["N_PROF"]):
            prf = Profile(brut_prf.isel(N_PROF=i), **kwargs)
            if prf.valid:
                self.profiles.append(prf)
                if prf.float_id not in self.float_ids:
                    self.float_ids.append(prf.float_id)

    @classmethod
    def from_ds(cls, ds, root_path, max_depth=1000, src="erddap", **kwargs):
        """Create Profiles from an xarray Dataset

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with time coordinate.
        root_path : str
            Root directory path for data storage.
        max_depth : float, default 1000
            Maximum depth (m) of the downloaded profiles.
        src : {"erddap", "gdac", "argovis"}, default "erddap"
            Data source of argopy.
        kwargs
            Passed to :class:`Profile`.

        Returns
        -------
        Profiles
            New Profiles instance.
        """
        brut_prf = load_from_ds(ds, root_path, max_depth=max_depth, src=src)
        return cls(ds.time, root_path, brut_prf, **kwargs)

    @functools.cached_property
    def ds(self):
        """create profile array"""
        lats = np.array([prf.lat for prf in self.profiles])
        lons = np.array([prf.lon for prf in self.profiles])
        times = np.array([prf.time for prf in self.profiles])
        temp = np.array([prf.temp for prf in self.profiles])
        sal = np.array([prf.sal for prf in self.profiles])
        float_id = np.array([prf.float_id for prf in self.profiles])
        p_id = np.arange(0, len(lats), dtype="int32")

        ds = xr.Dataset(
            {
                "time": (("profil"), times),
                "p_id": (("profil"), p_id),
                "float_id": (("profil"), float_id),
                "lat": (("profil"), lats),
                "lon": (("profil"), lons),
                "temp": (("profil", "depth"), temp),
                "sal": (("profil", "depth"), sal),
            },
            coords={
                "profil": np.arange(0, len(lats)),
                "depth": self.profiles[0].depth,
            },
        )
        return ds

    def to_netcdf(self, name=None, path=None):
        """Save profiles to NetCDF format"""
        if not path:
            path = self.root_path
        if not name:
            name = f"profil_{self.years[0]}_{self.years[-1]}.nc"
        self.ds.to_netcdf(
            os.path.join(
                path,
                name,
            )
        )

    def associate(self, eddies, nlag=2):
        """Associate profiles with eddies

        Attributes profiles to eddies (EvolEddies2D object) and adds info to profiles
        if they are considered colocalized within a structure.
        """
        eddy_pos = np.ones(len(self.ds.profil)) * -1
        for eddy in eddies.eddies:  # list of Eddies2D object object
            tstart = eddy.time - np.timedelta64(nlag, "D")
            tend = eddy.time + np.timedelta64(nlag, "D")
            ind_prf = np.where((self.ds.time >= tstart) & (self.ds.time <= tend))[0]
            prf = self.ds.isel(profil=ind_prf)

            for e in eddy.eddies:  # list of Raw2dEddies object
                e.p_id = []
                inside = e.contains_points(prf.lon, prf.lat)
                for i, b in enumerate(inside):
                    if b:
                        e.p_id.append(prf.isel(profil=i).p_id.values)
                        eddy_pos[prf.isel(profil=i).p_id.values] = e.track_id
        self.ds = self.ds.assign(eddy_pos=("profil", np.array(eddy_pos, dtype=np.int32)))

    def background(self, eddies, nlag, dlag):
        """Define background profile for each eddies
        profiles are considered when they are in a window of nlag around the detection
        date and at a maximum of dlag.

        eddies is a EvolEddies2D object
        nlag is a number of day
        dlag is a distance expected in kilometers
        """
        assert hasattr(self.ds, "eddy_pos"), (
            "Please associate eddies with profile before computing background"
        )
        ds_bck = self.ds.where(self.ds.eddy_pos == -1, drop=True)
        for eddy in eddies.eddies:  # list of Eddies2D object object
            tstart = eddy.time - np.timedelta64(nlag, "D")
            tend = eddy.time + np.timedelta64(nlag, "D")
            ind_prf = np.where((ds_bck.time >= tstart) & (ds_bck.time <= tend))[0]
            prf_bck = ds_bck.isel(profil=ind_prf)
            for e in eddy.eddies:
                dlon = prf_bck.lon - e.lon
                dlat = prf_bck.lat - e.lat
                dx = sgeo.deg2m(dlon, e.lat)
                dy = sgeo.deg2m(dlat)
                dxy = np.sqrt(dx**2 + dy**2)
                e_prf_bck = prf_bck.where(dxy < 1000 * dlag, drop=True)
                e.bck_id = list(e_prf_bck.p_id.astype("int32").values)
