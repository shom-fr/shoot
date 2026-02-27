"""
Streamline computation for 2D vector fields

Functions for computing streamfunctions from velocity fields.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import xarray as xr

from . import geo as sgeo
from . import meta as smeta


def psi(u, v):
    """Compute streamfunction from velocity field

    Integrates the velocity field to obtain the streamfunction using
    a four-quadrant integration scheme centered on the domain center.

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component (2D).
    v : xarray.DataArray
        Meridional velocity component (2D).

    Returns
    -------
    xarray.DataArray
        Streamfunction field in km²/s.

    Notes
    -----
    Assumes domain is centered on the feature of interest.

    Example
    -------
    >>> import xarray as xr, numpy as np
    >>> from shoot.streamline import psi
    >>> lon = xr.DataArray(np.linspace(0, 1, 30), dims="lon",
    ...     attrs={"standard_name": "longitude"})
    >>> lat = xr.DataArray(np.linspace(43, 44, 20), dims="lat",
    ...     attrs={"standard_name": "latitude"})
    >>> u = xr.DataArray(np.random.rand(20, 30), dims=("lat", "lon"),
    ...     coords={"lon": lon, "lat": lat})
    >>> v = xr.DataArray(np.random.rand(20, 30), dims=("lat", "lon"),
    ...     coords={"lon": lon, "lat": lat})
    >>> sf = psi(u, v)  # doctest: +SKIP
    """
    # define center position assume that we provide a domain centered on the peak
    ci = u.shape[1] // 2
    cj = u.shape[0] // 2

    # get lat, lon
    lat, lon = smeta.get_lat(u), smeta.get_lon(u)
    lat2d, lon2d = xr.broadcast(lat, lon)

    lon_ref = lon.mean()
    lat_ref = lat.mean()

    dlon2d = lon2d - lon_ref
    dlat2d = lat2d - lat_ref

    # convert lon, lat into kilometer distance matrix
    x = sgeo.deg2m(dlon2d, lat2d.values) / 1e3
    y = sgeo.deg2m(dlat2d) / 1e3

    # create 4 domains for the integration
    ly1 = u[cj:].shape[0]
    ly2 = u[:cj].shape[0]
    lx1 = u[:, ci:].shape[1]
    lx2 = u[:, :ci].shape[1]

    ### ---------- xy integration ------------- ##

    # integrate in the four domains
    cx1 = cumulative_trapezoid(v[cj, ci:], x[cj, ci:], initial=0)
    cx2 = cumulative_trapezoid(v[cj, ci::-1], x[cj, ci::-1])

    # expand vector to matrix size
    mcx11 = np.tile(cx1, (ly1, 1))
    mcx12 = np.tile(cx1, (ly2, 1))
    mcx21 = np.tile(cx2, (ly1, 1))
    mcx22 = np.tile(cx2, (ly2, 1))

    # integrate psi
    psi_xy11 = mcx11 - cumulative_trapezoid(u[cj:, ci:], y[cj:, ci:], initial=0, axis=0)
    psi_xy12 = mcx12 - cumulative_trapezoid(u[cj::-1, ci:], y[cj::-1, ci:], axis=0)
    psi_xy21 = mcx21 - cumulative_trapezoid(u[cj:, ci - 1 :: -1], y[cj:, ci - 1 :: -1], initial=0, axis=0)
    psi_xy22 = mcx22 - cumulative_trapezoid(u[cj::-1, ci - 1 :: -1], y[cj::-1, ci - 1 :: -1], axis=0)

    # Concatenate the 4 parts (NE, SE, NO, SO)
    psi_xy = np.block(
        [
            [psi_xy22[::-1, ::-1], psi_xy12[::-1, :]],
            [psi_xy21[:, ::-1], psi_xy11],
        ]
    )

    ### ---------- yx integration ------------- ## il faudra inverser les signes sur les integration en variable neg

    cy1 = -cumulative_trapezoid(u[cj:, ci], y[cj:, ci], initial=0)
    cy2 = -cumulative_trapezoid(u[cj::-1, ci], y[cj::-1, ci])

    mcy11 = np.tile(cy1, (lx1, 1)).T
    mcy12 = np.tile(cy2, (lx1, 1)).T
    mcy21 = np.tile(cy1, (lx2, 1)).T
    mcy22 = np.tile(cy2, (lx2, 1)).T

    # PSI from integrating u first and then v (4 parts of eq. A2)
    psi_yx11 = mcy11 + cumulative_trapezoid(v[cj:, ci:], x[cj:, ci:], initial=0, axis=-1)
    psi_yx21 = mcy21 + cumulative_trapezoid(v[cj:, ci::-1], x[cj:, ci::-1], axis=-1)
    psi_yx12 = mcy12 + cumulative_trapezoid(v[cj - 1 :: -1, ci:], x[cj - 1 :: -1, ci:], initial=0, axis=-1)
    psi_yx22 = mcy22 + cumulative_trapezoid(v[cj - 1 :: -1, ci::-1], x[cj - 1 :: -1, ci::-1], axis=-1)

    # Concatenate the 4 parts (NE, SE, NO, SO)
    psi_yx = np.block(
        [
            [psi_yx22[::-1, ::-1], psi_yx12[::-1, :]],
            [psi_yx21[:, ::-1], psi_yx11],
        ]
    )

    # Compute PSI as the average between the two (eq. A = (A1 + A2) / 2)
    psi = (psi_xy + psi_yx) / 2

    # Format
    psi = u.copy(data=psi)
    psi.attrs.clear()
    psi.name = "psi"
    psi.attrs.update(long_name="Streamfunction")
    return psi
