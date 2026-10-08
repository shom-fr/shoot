"""
Streamline computation for 2D vector fields

Functions for computing streamfunctions from velocity fields.
"""

import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from scipy.integrate import cumulative_trapezoid

from . import geo as sgeo
from . import meta as smeta


def psi(u, v, ci=None, cj=None):
    """Compute streamfunction from velocity field

    Integrates the velocity field to obtain the streamfunction using
    a four-quadrant integration scheme centered on a grid point,
    as in AMEDA (Le Vu et al., 2018).

    Parameters
    ----------
    u : xarray.DataArray
        Zonal velocity component (2D) in m/s.
    v : xarray.DataArray
        Meridional velocity component (2D) in m/s.
    ci : int, optional
        Grid index along X of the integration origin.
        Defaults to the center of the domain.
    cj : int, optional
        Grid index along Y of the integration origin.
        Defaults to the center of the domain.

    Returns
    -------
    xarray.DataArray
        Streamfunction field in 1e3 m²/s (velocities in m/s times distances in km),
        null at the integration origin.

    Notes
    -----
    Missing velocities (land points) are set to zero for the integration,
    so that they do not propagate along the integration paths.
    The streamfunction is then masked on land and extended by one pixel
    into land with the average of its ocean neighbours, as in AMEDA.

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
    # integration origin, by default the center of the domain
    if ci is None:
        ci = u.shape[1] // 2
    if cj is None:
        cj = u.shape[0] // 2

    # backward slices from the origin, empty when the origin is on the first index
    rci = slice(ci - 1, None, -1) if ci > 0 else slice(0, 0)
    rcj = slice(cj - 1, None, -1) if cj > 0 else slice(0, 0)

    # null velocities on land for the integration
    ocean = np.isfinite(u.values) & np.isfinite(v.values)
    u_ = np.nan_to_num(u.values)
    v_ = np.nan_to_num(v.values)

    # get lat, lon
    lat, lon = smeta.get_lat(u), smeta.get_lon(u)
    lat2d, lon2d = xr.broadcast(lat, lon)

    lon_ref = lon.mean()
    lat_ref = lat.mean()

    dlon2d = lon2d - lon_ref
    dlat2d = lat2d - lat_ref

    # convert lon, lat into kilometer distance matrix
    x = np.asarray(sgeo.deg2m(dlon2d, lat2d.values) / 1e3)
    y = np.asarray(sgeo.deg2m(dlat2d) / 1e3)

    # create 4 domains for the integration
    ly1 = u_[cj:].shape[0]
    ly2 = u_[:cj].shape[0]
    lx1 = u_[:, ci:].shape[1]
    lx2 = u_[:, :ci].shape[1]

    ### ---------- xy integration ------------- ##

    # integrate in the four domains
    cx1 = cumulative_trapezoid(v_[cj, ci:], x[cj, ci:], initial=0)
    cx2 = cumulative_trapezoid(v_[cj, ci::-1], x[cj, ci::-1])

    # expand vector to matrix size
    mcx11 = np.tile(cx1, (ly1, 1))
    mcx12 = np.tile(cx1, (ly2, 1))
    mcx21 = np.tile(cx2, (ly1, 1))
    mcx22 = np.tile(cx2, (ly2, 1))

    # integrate psi
    psi_xy11 = mcx11 - cumulative_trapezoid(u_[cj:, ci:], y[cj:, ci:], initial=0, axis=0)
    psi_xy12 = mcx12 - cumulative_trapezoid(u_[cj::-1, ci:], y[cj::-1, ci:], axis=0)
    psi_xy21 = mcx21 - cumulative_trapezoid(u_[cj:, rci], y[cj:, rci], initial=0, axis=0)
    psi_xy22 = mcx22 - cumulative_trapezoid(u_[cj::-1, rci], y[cj::-1, rci], axis=0)

    # Concatenate the 4 parts (NE, SE, NO, SO)
    psi_xy = np.block(
        [
            [psi_xy22[::-1, ::-1], psi_xy12[::-1, :]],
            [psi_xy21[:, ::-1], psi_xy11],
        ]
    )

    ### ---------- yx integration ------------- # TODO: inverser les signes (integration en variable neg)

    cy1 = -cumulative_trapezoid(u_[cj:, ci], y[cj:, ci], initial=0)
    cy2 = -cumulative_trapezoid(u_[cj::-1, ci], y[cj::-1, ci])

    mcy11 = np.tile(cy1, (lx1, 1)).T
    mcy12 = np.tile(cy2, (lx1, 1)).T
    mcy21 = np.tile(cy1, (lx2, 1)).T
    mcy22 = np.tile(cy2, (lx2, 1)).T

    # PSI from integrating u first and then v (4 parts of eq. A2)
    psi_yx11 = mcy11 + cumulative_trapezoid(v_[cj:, ci:], x[cj:, ci:], initial=0, axis=-1)
    psi_yx21 = mcy21 + cumulative_trapezoid(v_[cj:, ci::-1], x[cj:, ci::-1], axis=-1)
    psi_yx12 = mcy12 + cumulative_trapezoid(v_[rcj, ci:], x[rcj, ci:], initial=0, axis=-1)
    psi_yx22 = mcy22 + cumulative_trapezoid(v_[rcj, ci::-1], x[rcj, ci::-1], axis=-1)

    # Concatenate the 4 parts (NE, SE, NO, SO)
    psi_yx = np.block(
        [
            [psi_yx22[::-1, ::-1], psi_yx12[::-1, :]],
            [psi_yx21[:, ::-1], psi_yx11],
        ]
    )

    # Compute PSI as the average between the two (eq. A = (A1 + A2) / 2)
    psi = (psi_xy + psi_yx) / 2

    # Mask land and extend psi by one pixel into land with the average of its ocean neighbours
    if not ocean.all():
        psi[~ocean] = np.nan
        kernel = np.ones((3, 3))
        coast = ndi.binary_dilation(ocean, structure=kernel) & ~ocean
        psi_sum = ndi.convolve(np.where(ocean, psi, 0.0), kernel, mode="constant")
        nocean = ndi.convolve(ocean.astype("d"), kernel, mode="constant")
        psi[coast] = psi_sum[coast] / nocean[coast]

    # Format
    psi = u.copy(data=psi)
    psi.attrs.clear()
    psi.name = "psi"
    psi.attrs.update(long_name="Streamfunction", units="1e3 m2 s-1")
    return psi
