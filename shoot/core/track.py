#!/usr/bin/env python3
"""
Eddy association numeric routines

Cost matrices used to associate eddies between two sets, as in the
Chelton et al. (2011) tracking algorithm.
"""

import numba
import numpy as np

from .geo import EARTH_RADIUS

#: Cost of an impossible association
IMPOSSIBLE = 1e6


@numba.njit(cache=True)
def _association_cost_(
    lon_new,
    lat_new,
    radius_new,
    ro_new,
    type_new,
    dist_new,
    lon_ref,
    lat_ref,
    radius_ref,
    ro_ref,
    type_ref,
    dist_ref,
    radius_ref_avg,
    ro_ref_avg,
    time_cost_ref,
):
    nnew = lon_new.size
    nref = lon_ref.size
    cost = np.empty((nnew, nref))
    for i in range(nnew):
        for j in range(nref):
            # Distance term
            x = lon_ref[j] - lon_new[i]
            x = x * np.pi * EARTH_RADIUS / 180.0
            x *= np.cos(np.radians(lat_ref[j]))
            y = (lat_ref[j] - lat_new[i]) * np.pi * EARTH_RADIUS / 180.0
            dmax = dist_ref[j] + dist_new[i]
            dxy = np.sqrt(x**2 + y**2)
            cost[i, j] = (dxy**2) / (dmax**2) if dxy < dmax else IMPOSSIBLE

            # Dynamical similarity, avoiding cyclone/anticyclone coupling
            if type_ref[j] == type_new[i]:
                dr = (radius_ref[j] - radius_new[i]) / (radius_ref_avg[j] + radius_new[i])
                dro = (ro_ref[j] - ro_new[i]) / (ro_ref_avg[j] + ro_new[i])
                cost[i, j] += dr**2 + dro**2
            else:
                cost[i, j] += IMPOSSIBLE

            # Temporal proximity
            cost[i, j] += time_cost_ref[j]
    return np.sqrt(cost)


def association_cost(
    lon_new,
    lat_new,
    radius_new,
    ro_new,
    type_new,
    lon_ref,
    lat_ref,
    radius_ref,
    ro_ref,
    type_ref,
    dist_ref,
    dist_new=None,
    radius_ref_avg=None,
    ro_ref_avg=None,
    time_cost_ref=None,
):
    """Cost of associating new eddies with reference eddies

    For a new eddy `i` and a reference eddy `j`, the squared cost is the sum of:

    - the distance term ``(d_ij / D_ij)**2``, where ``D_ij = dist_ref[j] + dist_new[i]``,
      or :data:`IMPOSSIBLE` when ``d_ij >= D_ij``;
    - the dynamical similarity term ``DR**2 + DRo**2`` with
      ``DR = (R_j - R_i) / (Ravg_j + R_i)`` and ``DRo = (Ro_j - Ro_i) / (Roavg_j + Ro_i)``,
      or :data:`IMPOSSIBLE` when the eddy types differ;
    - the temporal term ``time_cost_ref[j]``.

    Parameters
    ----------
    lon_new, lat_new, radius_new, ro_new : array-like
        Positions (degrees), radii and Rossby numbers of the new eddies.
    type_new : array-like of int
        Eddy type codes of the new eddies.
    lon_ref, lat_ref, radius_ref, ro_ref : array-like
        Positions (degrees), radii and Rossby numbers of the reference eddies.
    type_ref : array-like of int
        Eddy type codes of the reference eddies.
    dist_ref : array-like
        Reference part of the maximal distance in meters.
    dist_new : array-like, optional
        New eddy part of the maximal distance in meters. Defaults to 0.
    radius_ref_avg, ro_ref_avg : array-like, optional
        Radii and Rossby numbers of the reference eddies used in the
        similarity denominators, typically averaged along tracks.
        Default to `radius_ref` and `ro_ref`.
    time_cost_ref : array-like, optional
        Temporal cost of the reference eddies. Defaults to 0.

    Returns
    -------
    ndarray
        Cost matrix of shape (nnew, nref).
    """

    def as1d(values, default=0.0, like=None):
        if values is None:
            if isinstance(default, np.ndarray):
                return default
            return np.full(len(like), default)
        return np.asarray(values, dtype="d").reshape(-1)

    lon_new = as1d(lon_new)
    lon_ref = as1d(lon_ref)
    radius_ref = as1d(radius_ref)
    ro_ref = as1d(ro_ref)
    return _association_cost_(
        lon_new,
        as1d(lat_new),
        as1d(radius_new),
        as1d(ro_new),
        np.asarray(type_new, dtype=np.int64).reshape(-1),
        as1d(dist_new, like=lon_new),
        lon_ref,
        as1d(lat_ref),
        radius_ref,
        ro_ref,
        np.asarray(type_ref, dtype=np.int64).reshape(-1),
        as1d(dist_ref),
        as1d(radius_ref_avg, radius_ref),
        as1d(ro_ref_avg, ro_ref),
        as1d(time_cost_ref, like=lon_ref),
    )
