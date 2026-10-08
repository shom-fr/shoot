#!/usr/bin/env python3
"""
Created on Thu Jan  9 13:24:21 2025

@author: jbroust
"""

import functools
import logging

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment

from ..core import track as ctrack
from . import eddies2d as seddies

logger = logging.getLogger(__name__)


def _vmax_radius(eddy):
    """Radius of the maximal speed contour of a detected or reconstructed eddy"""
    try:
        return eddy.vmax_contour.radius
    except AttributeError:
        return eddy.vmax_radius


def _eddy_type_codes(*eddy_lists):
    """Integer codes of eddy types for each list of eddies"""
    codes = {}
    return [[codes.setdefault(eddy.eddy_type, len(codes)) for eddy in eddies] for eddies in eddy_lists]


def _association_cost(new_eddies, parent_eddies, dist_ref, radius_ref_avg, ro_ref_avg, time_cost_ref):
    """Cost matrix between new and parent eddies with :func:`shoot.core.track.association_cost`"""
    if not len(new_eddies) or not len(parent_eddies):
        return np.zeros((len(new_eddies), len(parent_eddies)))
    type_new, type_ref = _eddy_type_codes(new_eddies, parent_eddies)
    return ctrack.association_cost(
        [eddy.lon for eddy in new_eddies],
        [eddy.lat for eddy in new_eddies],
        [eddy.radius for eddy in new_eddies],
        [eddy.ro for eddy in new_eddies],
        type_new,
        [eddy.lon for eddy in parent_eddies],
        [eddy.lat for eddy in parent_eddies],
        [eddy.radius for eddy in parent_eddies],
        [eddy.ro for eddy in parent_eddies],
        type_ref,
        dist_ref,
        dist_new=[_vmax_radius(eddy) for eddy in new_eddies],
        radius_ref_avg=radius_ref_avg,
        ro_ref_avg=ro_ref_avg,
        time_cost_ref=time_cost_ref,
    )


class Associate:
    """Associate eddies between time steps for tracking

    Uses the Ameda tracking algorithm with cost function
    based on distance, Rossby number, and radius similarity.

    Parameters
    ----------
    track_eddies : list
        List of Track objects containing eddy trajectories.
    parent_eddies : list
        Eddies from the previous time step.
    new_eddies : list
        Eddies from the current time step to associate.
    Dt : float
        Time step (s) between parent and new eddies.
    Tc : float
        Characteristic time scale for tracking.
    C : float, default 6.5
        Characteristic eddy drift speed (km/day).

    Attributes
    ----------
    cost : ndarray
        Cost matrix for eddy association (cached property).
    """

    def __init__(
        self,
        track_eddies,
        parent_eddies,
        new_eddies,
        Dt,
        Tc,
        C=6.5,
    ):
        """Initialize eddy tracking association

        Parameters
        ----------
        track_eddies : list
            List of Track objects containing eddy trajectories.
        parent_eddies : list
            Eddies from the previous time step.
        new_eddies : list
            Eddies from the current time step to associate.
        Dt : float
            Time step (s) between parent and new eddies.
        Tc : float
            Characteristic time scale for tracking.
        C : float, default 6.5
            Characteristic eddy drift speed (km/day).
        """
        self.parent_eddies = parent_eddies  # reference eddies
        self.new_eddies = new_eddies  # next time eddies
        self.track_eddies = track_eddies
        self._Dt = Dt  # actual time step between eddies 1 and eddies 2
        self._Tc = Tc
        self._C = C

    def search_dist_ref(self, eddyj, dt):
        """Parent eddy part of the search distance for eddy association

        The full search distance is this plus the vmax radius of the new eddy.
        `dt` is given in seconds and the result in meters.
        """
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        Ravg = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            Ravg += _vmax_radius(self.track_eddies[eddyj.track_id].eddies[i])
            n += 1
        return self._C * 1e3 * (1 + dt / 86400) / 2 + Ravg / n  # km/day -> m

    def search_dist(self, eddyj, eddyi):
        """Compute search distance for eddy association"""
        return self.search_dist_ref(eddyj, self._Dt) + _vmax_radius(eddyi)

    def ro_avg(self, eddyj):
        """Average Rossby number over last 5 time steps"""
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        ro = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            ro += self.track_eddies[eddyj.track_id].eddies[i].ro
            n += 1
        return ro / n

    def rad_avg(self, eddyj):
        """Average radius over last 5 time steps"""
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        radius = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            radius += self.track_eddies[eddyj.track_id].eddies[i].radius
            n += 1
        return radius / n

    @functools.cached_property
    def cost(self):
        """Cost function between each eddy pair"""
        return _association_cost(
            self.new_eddies,
            self.parent_eddies,
            [self.search_dist_ref(eddy, self._Dt) for eddy in self.parent_eddies],
            [self.rad_avg(eddy) for eddy in self.parent_eddies],
            [self.ro_avg(eddy) for eddy in self.parent_eddies],
            [(0.5 * self._Dt / self._Tc) ** 2] * len(self.parent_eddies),
        )

    def order(self):
        M = self.cost
        idel = []
        for i in range(M.shape[0]):
            if (M[i] > 1e3).all():
                idel.append(i)
        Mclean = np.delete(M, idel, axis=0)  # delete impossible solutions
        raw, col = linear_sum_assignment(Mclean)
        for i, j in zip(raw, col):
            np.delete(self.new_eddies, idel)[i].track_id = self.parent_eddies[j].track_id


class AssociateMulti:
    """Associate eddies with multiple backward time steps for tracking"""

    def __init__(
        self,
        track_eddies,
        parent_eddies,  ## list of backward eddies
        new_eddies,
        Dt,  ##list of dt
        Tc,
        C=6.5,
    ):
        self.parent_eddies = parent_eddies  # reference backward eddies
        self.new_eddies = new_eddies  # next time eddies
        self.track_eddies = track_eddies
        self._Dt = Dt  # time steps
        self._Tc = Tc
        self._C = C

    search_dist_ref = Associate.search_dist_ref

    def search_dist(self, eddyj, eddyi, dt):
        """Compute search distance for eddy association"""
        return self.search_dist_ref(eddyj, dt) + _vmax_radius(eddyi)

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
        """Cost function between each eddy pair"""
        parents = [eddy for eddies in self.parent_eddies for eddy in eddies]
        dts = [dt for eddies, dt in zip(self.parent_eddies, self._Dt) for eddy in eddies]
        return _association_cost(
            self.new_eddies,
            parents,
            [self.search_dist_ref(eddy, dt) for eddy, dt in zip(parents, dts)],
            [self.rad_avg(eddy) for eddy in parents],
            [self.ro_avg(eddy) for eddy in parents],
            [(0.5 * dt / self._Tc) ** 2 for dt in dts],
        )

    def indexmatching(self):
        index = {}
        cmp = 0
        for k in range(len(self.parent_eddies)):
            for kj in range(len(self.parent_eddies[k])):
                index[cmp] = (k, kj)
                cmp += 1
        return index

    def order(self):
        # compute global matrix
        M = self.cost
        index = self.indexmatching()

        jdel = []  # eliminate already parent eddies that can't be assigned again
        for j in range(M.shape[1]):
            if self.parent_eddies[index[j][0]][index[j][1]].is_parent:
                jdel.append(j)
                del index[j]
        Mclean = np.delete(M, jdel, axis=1)
        index = {i: index[k] for i, k in enumerate(index)}

        # Clean this matrix with impossible connections
        idel = []  # new eddies impossible to match
        for i in range(M.shape[0]):
            if (Mclean[i] > 1e3).all():
                idel.append(i)
        Mclean = np.delete(Mclean, idel, axis=0)  # delete impossible solutions
        # Connect rows with columns with Hungarian algorithm
        raw, col = linear_sum_assignment(Mclean)

        for i, j in zip(raw, col):
            k = index[j][0]
            kj = index[j][1]
            if Mclean[i, j] > 1000:
                continue
            np.delete(self.new_eddies, idel)[i].track_id = self.parent_eddies[k][kj].track_id
            self.parent_eddies[k][kj].is_parent = True


class Track:
    """Individual eddy track containing a temporal sequence of eddies

    Represents a single eddy trajectory through time. Stores the sequence
    of associated :class:`~shoot.eddies.eddies2d.GriddedEddy2D` objects
    and their corresponding times.

    Parameters
    ----------
    eddy : GriddedEddy2D or list
        Initial eddy or list of eddies.
    time : numpy.datetime64 or list
        Time(s) of the eddy detection(s).
    number : int
        Unique track identifier.
    dt : float
        Time step between detections in seconds.
    Tc : float
        Characteristic time scale for tracking in seconds.
    C : float, default 6.5
        Characteristic eddy drift speed (km/day).
    """

    def __init__(
        self,
        eddy,
        time,
        number,
        dt,
        Tc,
        C=6.5,
    ):
        self.eddies = [eddy] if not isinstance(eddy, list) else eddy
        self.number = number
        self.active = True
        self.times = [time] if not isinstance(time, list) else time
        self._dt = dt
        self._Tc = Tc
        self._C = C

    @classmethod
    def reconstruct(cls, eddies, times, number, dt, Tc, C=6.5):
        return cls(eddies, times, number, dt, Tc, C)

    def update(self, eddy, time):
        self.eddies.append(eddy)
        self.times.append(time)

    @property
    def ds(self):
        return xr.Dataset(
            {
                "date_first_detection": (("eddies"), [self.times[0]]),
                "date_last_detection": (("eddies"), [self.times[-1]]),
                "life_time": (
                    ("eddies"),
                    [(self.times[-1] - self.times[0]) / np.timedelta64(1, "D")],
                    {"long_name": "Track life time", "units": "day"},
                ),
                "x_start": (("eddies"), [self.eddies[0].lon], {"units": "degrees_east"}),
                "y_start": (("eddies"), [self.eddies[0].lat], {"units": "degrees_north"}),
                "x_end": (("eddies"), [self.eddies[-1].lon], {"units": "degrees_east"}),
                "y_end": (("eddies"), [self.eddies[-1].lat], {"units": "degrees_north"}),
                "track_type": (("eddies"), [self.eddies[0].eddy_type]),
            },
        )


class Tracks:
    """Collection of eddy tracks over time

    Performs eddy tracking on an :class:`~shoot.eddies.eddies2d.EvolEddies2D`
    object using the Chelton et al. (2011) algorithm with backward association.

    Parameters
    ----------
    eddies : EvolEddies2D
        Detected eddies at multiple time steps.
    nback : int
        Number of backward time steps for multi-step association.
    C : float, default 6.5
        Characteristic eddy drift speed (km/day).
    """

    def __init__(
        self,
        eddies,
        nback,
        C=6.5,
        **attrs,
    ):
        self.eddies = eddies  #  EvolEddies object
        self.times = [e.time for e in eddies.eddies]  # corresponding time vector
        self.nback = nback
        self._dt = eddies.dt
        self._Tc = nback * eddies.dt
        self._C = C
        self.nb_step = int(self._Tc / self._dt)  # maximum backward timesteps
        self.nb_tracks = 0
        self.attrs = attrs
        self.track_eddies = {}  # list of tracks

    @classmethod
    def reconstruct(cls, ds, nback, C=6.5):
        """Reconstruct tracks from a tracked xarray dataset"""
        ## Reconstruct eddies
        eddies = seddies.EvolEddies2D.reconstruct(ds)
        ## Reconstruct tracks
        track_eddies = {}  # dictionary of tracks
        for i in np.unique(ds.track_id):
            tmp = ds.where(ds.track_id == i, drop=True)  # select eddies belonging to track i
            trace_times = list(tmp.time.values)
            trace_number = i
            trace_eddies = []
            for j in range(len(tmp.obs)):
                trace_eddies.append(seddies.Eddy.reconstruct(tmp.isel(obs=j), track=True))
            track_eddies[i] = Track.reconstruct(
                trace_eddies,
                trace_times,
                trace_number,
                eddies.dt,
                eddies.dt * nback,
                C,
            )
        my_tracks = cls(eddies, nback, C)
        my_tracks.track_eddies = track_eddies
        my_tracks.nb_tracks = len(track_eddies)
        return my_tracks

    @property
    def ds(self):
        ds = None
        for nb_track in self.track_eddies:
            track = self.track_eddies[nb_track]
            if ds is None:
                ds = track.ds
            else:
                ds = xr.concat([ds, track.ds], dim="eddies")
        ds_eddies = self.eddies.ds
        return xr.merge([ds_eddies, ds], compat="override")

    def to_netcdf(self, path_nc):
        "Save to NetCDF format"
        self.ds.to_netcdf(path_nc)

    def track_init(self):
        logger.info("Initializing tracks from first time step")
        for i, eddy in enumerate(self.eddies.eddies[0].eddies):  # initialized with the first detected eddies
            self.track_eddies[i] = Track(eddy, self.times[0], i, self._dt, self._Tc, self._C)
            eddy.track_id = i  # update eddy track number
            self.nb_tracks += 1

    def track_steps(self):
        logger.info("Processing tracking steps")
        for i in range(1, len(self.times)):  # compute tracking on all following time steps
            t = self.times[i]
            logger.debug("Tracking time step: %s", t)
            new_eddies = self.eddies.eddies[i].eddies
            self.update_multi(
                [self.eddies.eddies[i - k].eddies for k in range(1, min(i, self.nback) + 1)],
                new_eddies,
                [self._dt * k for k in range(1, min(i, self.nback) + 1)],
            )
            for eddy in new_eddies:
                if eddy.track_id is None:  # Create a new track
                    self.track_eddies[self.nb_tracks] = Track(
                        eddy, t, self.nb_tracks, self._dt, self._Tc, self._C
                    )
                    eddy.track_id = self.nb_tracks  # update eddy track number
                    self.nb_tracks += 1

                else:  ## append to existing track
                    self.track_eddies[eddy.track_id].update(eddy, t)

    def track_step(self):
        """
        Track only for the last eddies.
        Supposes the job has been done before for others.
        Useful for update concerns.
        """
        i = len(self.times) - 1
        t = self.times[i]
        new_eddies = self.eddies.eddies[i].eddies
        self.update_multi(
            [self.eddies.eddies[i - k].eddies for k in range(1, min(i, self.nback) + 1)],
            new_eddies,
            [self._dt * k for k in range(1, min(i, self.nback) + 1)],
        )
        for eddy in new_eddies:
            if eddy.track_id is None:  # Create a new track
                self.track_eddies[self.nb_tracks] = Track(
                    eddy, t, self.nb_tracks, self._dt, self._Tc, self._C
                )
                eddy.track_id = self.nb_tracks  # update eddy track number
                self.nb_tracks += 1

            else:  ## append to existing track
                self.track_eddies[eddy.track_id].update(eddy, t)

    def tracking(self):
        """compute the eddy tracking"""
        self.track_init()
        self.track_steps()
        # return self.track_eddies

    def update(self, parent_eddies, new_eddies, Dt):
        """Update based on last detected eddies"""
        Associate(self.track_eddies, parent_eddies, new_eddies, Dt, self._Tc, self._C).order()

    def update_multi(self, parent_eddies, new_eddies, Dt):
        """Update based on several preceding time eddies"""
        AssociateMulti(self.track_eddies, parent_eddies, new_eddies, Dt, self._Tc, self._C).order()

    def refresh(self, new_eddies):
        """refresh a track with a new Eddies object (next time)"""
        self.eddies.add(new_eddies)
        self.times.append(new_eddies.time)
        self.track_step()


def track_eddies(eddies, nback, C=6.5):
    """Track eddies across time steps

    Parameters
    ----------
    eddies : EvolEddies2D
        Detections at several time steps.
        The time interval should preferably be constant.
    nback : int
        Number of backward time steps for eddy tracking.
        Typically around 10 days for satellite altimetry
        and 1 or 2 days for numerical simulations.
    C : float, default 6.5
        Characteristic eddy drift speed (km/day).

    Returns
    -------
    Tracks
        Tracks object with eddy trajectories.

    Example
    -------
    >>> from shoot.eddies.track import track_eddies
    >>> tracks = track_eddies(evol_eddies, nback=10)  # doctest: +SKIP
    >>> tracks.to_netcdf("tracked.nc")  # doctest: +SKIP
    """
    tracks = Tracks(eddies, nback, C=C)
    tracks.tracking()
    return tracks


def update_tracks(ds, new_eddies, nback, C=6.5):
    """Update tracking based on new eddies detection

    Parameters
    ----------
    ds : xarray.Dataset
        Already tracked period.
    new_eddies : Eddies2D
        Detections at the new time step.
    nback : int
        Number of backward time steps for eddy tracking.
        Typically around 10 days for satellite altimetry
        and 1 or 2 days for numerical simulations.
    C : float, default 6.5
        Characteristic eddy drift speed (km/day).

    Returns
    -------
    Tracks
        Updated tracks object.
    """
    tracks = Tracks.reconstruct(ds, nback, C=C)
    tracks.refresh(new_eddies)
    return tracks
