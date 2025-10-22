#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:24:21 2025

@author: jbroust
"""

import functools
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment
import xoa.geo as xgeo

from . import eddies2d as seddies


class Associate:
    def __init__(
        self,
        track_eddies,
        parent_eddies,
        new_eddies,
        Dt,
        Tc,
        C=6.5 * 1e3 / 86400,
    ):
        self.parent_eddies = parent_eddies  # reference eddies
        self.new_eddies = new_eddies  # next time eddies
        self.track_eddies = track_eddies
        self._Dt = Dt  # actual time step between eddies 1 and eddies 2
        self._Tc = Tc
        self._C = C

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
        M = np.zeros((len(self.new_eddies), len(self.parent_eddies)))
        for i in range(len(self.new_eddies)):
            for j in range(len(self.parent_eddies)):
                dlat = self.parent_eddies[j].lat - self.new_eddies[i].lat
                dlon = self.parent_eddies[j].lon - self.new_eddies[i].lon
                x = xgeo.deg2m(dlon, self.parent_eddies[j].lat)
                y = xgeo.deg2m(dlat)

                D_ij = self.search_dist(self.parent_eddies[j], self.new_eddies[i])

                # Distance term
                dxy = np.sqrt(x**2 + y**2)

                M[i, j] = (dxy**2) / (D_ij**2) if dxy < D_ij else 1e6
                # dynamical similarity
                roj = self.ro_avg(self.parent_eddies[j])
                rj = self.rad_avg(self.parent_eddies[j])

                DR = (self.parent_eddies[j].radius - self.new_eddies[i].radius) / (
                    rj + self.new_eddies[i].radius
                )
                DR0 = (self.parent_eddies[j].ro - self.new_eddies[i].ro) / (
                    roj + self.new_eddies[i].ro
                )

                # Warning avoid couple cyclone with anticylone
                M[i, j] += (
                    DR**2 + DR0**2
                    if self.parent_eddies[j].eddy_type == self.new_eddies[i].eddy_type
                    else 1e6
                )

                # temporal proximity
                M[i, j] += (0.5 * self._Dt / self._Tc) ** 2
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
            np.delete(self.new_eddies, idel)[i].track_id = self.parent_eddies[j].track_id


class AssociateMulti:
    def __init__(
        self,
        track_eddies,
        parent_eddies,  ## list of backward eddies
        new_eddies,
        Dt,  ##list of dt
        Tc,
        C=6.5 * 1e3 / 86400,
    ):
        self.parent_eddies = parent_eddies  # reference backward eddies
        self.new_eddies = new_eddies  # next time eddies
        self.track_eddies = track_eddies
        self._Dt = Dt  # time steps
        self._Tc = Tc
        self._C = C

    def search_dist(self, eddyj, eddyi, dt):
        istart = max(0, len(self.track_eddies[eddyj.track_id].eddies) - 5)
        n = 0
        Ravg = 0
        for i in range(istart, len(self.track_eddies[eddyj.track_id].eddies)):
            try:
                Ravg += self.track_eddies[eddyj.track_id].eddies[i].vmax_contour.radius
            except AttributeError:
                Ravg += self.track_eddies[eddyj.track_id].eddies[i].vmax_radius
            n += 1
        try:
            Dij = self._C * (1 + dt) / 2 + Ravg / n + eddyi.vmax_contour.radius
        except AttributeError:
            Dij = self._C * (1 + dt) / 2 + Ravg / n + eddyi.vmax_radius
        return Dij

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
        """cost function between each eddy pairs"""
        nj = np.sum([len(self.parent_eddies[k]) for k in range(len(self.parent_eddies))])
        M = np.zeros((len(self.new_eddies), nj))
        for i in range(len(self.new_eddies)):
            cmp = 0
            for k in range(len(self.parent_eddies)):
                for j in range(len(self.parent_eddies[k])):
                    dlat = self.parent_eddies[k][j].lat - self.new_eddies[i].lat
                    dlon = self.parent_eddies[k][j].lon - self.new_eddies[i].lon
                    x = xgeo.deg2m(dlon, self.parent_eddies[k][j].lat)
                    y = xgeo.deg2m(dlat)

                    D_ij = self.search_dist(
                        self.parent_eddies[k][j],
                        self.new_eddies[i],
                        self._Dt[k],
                    )

                    # Distance term
                    dxy = np.sqrt(x**2 + y**2)

                    M[i, cmp] = (dxy**2) / (D_ij**2) if dxy < D_ij else 1e6

                    # dynamical similarity
                    roj = self.ro_avg(self.parent_eddies[k][j])
                    rj = self.rad_avg(self.parent_eddies[k][j])

                    DR = (self.parent_eddies[k][j].radius - self.new_eddies[i].radius) / (
                        rj + self.new_eddies[i].radius
                    )
                    DR0 = (self.parent_eddies[k][j].ro - self.new_eddies[i].ro) / (
                        roj + self.new_eddies[i].ro
                    )

                    # Warning avoid couple cyclone with anticylone
                    M[i, cmp] += (
                        DR**2 + DR0**2
                        if self.parent_eddies[k][j].eddy_type == self.new_eddies[i].eddy_type
                        else 1e6
                    )

                    # temporal proximity
                    M[i, cmp] += (0.5 * self._Dt[k] / self._Tc) ** 2
                    cmp += 1

        return np.sqrt(M)

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

        # Clean this matrix with unpossible connexions
        idel = []  # new eddies impossible to match
        for i in range(M.shape[0]):
            if (Mclean[i] > 1e3).all():
                idel.append(i)
        Mclean = np.delete(Mclean, idel, axis=0)  # delete impossible solutions
        # Connect raws with column with Hugarian algorithm
        raw, col = linear_sum_assignment(Mclean)

        for i, j in zip(raw, col):
            k = index[j][0]
            kj = index[j][1]
            if Mclean[i, j] > 1000:
                continue
            np.delete(self.new_eddies, idel)[i].track_id = self.parent_eddies[k][kj].track_id
            self.parent_eddies[k][kj].is_parent = True


class Track:
    def __init__(
        self,
        eddy,
        time,
        number,
        dt,
        Tc,
        C=6.5 * 1e3 / 86400,  # 6.5 km.day in m/s
    ):
        self.eddies = [eddy] if not type(eddy) == list else eddy
        self.number = number
        self.active = True
        self.times = [time] if not type(time) == list else time
        self._dt = dt
        self._Tc = Tc
        self._C = C

    @classmethod
    def reconstruct(cls, eddies, times, number, dt, Tc):
        return cls(eddies, times, number, dt, Tc)

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
                ),
                "x_start": (("eddies"), [self.eddies[0].lon]),
                "y_start": (("eddies"), [self.eddies[0].lat]),
                "x_end": (("eddies"), [self.eddies[-1].lon]),
                "y_end": (("eddies"), [self.eddies[-1].lat]),
                "eddy_type": (("eddies"), [self.eddies[0].eddy_type]),
            },
        )


class Tracks:
    """This class represents a list of tracks: track_eddies
    It tracks the EvolEddies object
    """

    def __init__(
        self,
        eddies,
        nback,
        C=6.5 * 1e3 / 86400,  # 6.5 km.day in m/s
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
    def reconstruct(cls, ds, nback):
        """reconstruct the trackings from panda dataframe of eddies"""
        ##reconstruction des eddies
        eddies = seddies.EvolEddies2D.reconstruct(ds)
        ##reconstruction des traces
        track_eddies = {}  # dictionnary of tracks
        for i in np.unique(ds.track_id):
            tmp = ds.where(ds.track_id == i, drop=True)  # selectionne les eddies de la trace  i
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
            )
        my_tracks = cls(eddies, nback)
        my_tracks.track_eddies = track_eddies
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
        return xr.merge([ds_eddies, ds])

    def save(self, path_nc):
        "this save at .nc format"
        self.ds.to_netcdf(path_nc)

    def track_init(self):
        print("initialization")
        for i, eddy in enumerate(
            self.eddies.eddies[0].eddies
        ):  # initialized with the the first detected eddies
            self.track_eddies[i] = Track(eddy, self.times[0], i, self._dt, self._Tc)
            eddy.track_id = i  # actualise eddy track number
            self.nb_tracks += 1

    def track_steps(self):
        print("tracking steps")
        for i in range(1, len(self.times)):  # compute tracking on all following time steps
            t = self.times[i]
            print(t)
            new_eddies = self.eddies.eddies[i].eddies
            self.update_multi(
                [self.eddies.eddies[i - k].eddies for k in range(1, min(i, self.nback) + 1)],
                new_eddies,
                [self._dt * k for k in range(1, min(i, self.nback) + 1)],
            )
            for eddy in new_eddies:
                if eddy.track_id is None:  # Create a new track
                    self.track_eddies[self.nb_tracks] = Track(
                        eddy, t, self.nb_tracks, self._dt, self._Tc
                    )
                    eddy.track_id = self.nb_tracks  # actualise eddy track number
                    self.nb_tracks += 1

                else:  ##æppend to existing track
                    self.track_eddies[eddy.track_id].update(eddy, t)

    def track_step(self):
        """
        track only for the last eddies
        suppose the job has been done before for others
        usefull for update concerns
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
                    eddy, t, self.nb_tracks, self._dt, self._Tc
                )
                eddy.track_id = self.nb_tracks  # actualise eddy track number
                self.nb_tracks += 1

            else:  ##æppend to existing track
                self.track_eddies[eddy.track_id].update(eddy, t)

    def tracking(self):
        """compute the eddy tracking"""
        self.track_init()
        self.track_steps()
        # return self.track_eddies

    def update(self, parent_eddies, new_eddies, Dt):
        """update based on last detected eddies"""
        Associate(self.track_eddies, parent_eddies, new_eddies, Dt, self._Tc).order()

    def update_multi(self, parent_eddies, new_eddies, Dt):
        """update based on sevral precdeding time eddies"""
        AssociateMulti(self.track_eddies, parent_eddies, new_eddies, Dt, self._Tc).order()

    def refresh(self, new_eddies):
        """refresh a track with a new Eddies object (next time)"""
        self.eddies.add(new_eddies)
        self.times.append(new_eddies.time)
        self.track_step()


def track_eddies(eddies, nback):
    """Add anomaly to detected eddies
    Parameters
    ----------
    eddies: EvolEddies object
        represent detections at several timestep.
        Mind that the time interval is preferably constant

    nback: int
        number of backward time step for eddy tracking.
        Typically around 10 days for satelitte altimetry
        and 1 or 2 days for numerical simulations

    Returns
    -------
     tracks: Tracks object
         directly appends anomaly object in each eddies
    """
    tracks = Tracks(eddies, nback)
    tracks.tracking()
    return tracks


def update_tracks(ds, new_eddies, nback):
    """update tracking based on new eddies detection
    Parameters
    ----------
    ds: xarray dataset
      already tracked perdiod

    new_eddies: Eddies object
        represent detections at the new time step.

    nback: int
        number of backward time step for eddy tracking.
        Typically around 10 days for satelitte altimetry
        and 1 or 2 days for numerical simulations

    Returns
    -------
     tracks: Tracks object
         updated track object
    """
    tracks = Tracks.reconstruct(ds, nback)
    tracks.refresh(new_eddies)
    return tracks
