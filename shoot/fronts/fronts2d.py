#!/usr/bin/env python3
"""
2D front detection

Detection of fronts in a 2D field, like sea surface temperature, with the
algorithms of :mod:`shoot.core.fronts`.
"""

import xarray as xr

from .. import meta as smeta
from ..core import fronts as cfronts
from ..core import image as cimage

#: Available detection methods
METHODS = ("cca", "cca_sliding", "boa", "canny")


def _get_yx_field(da):
    """2D field with dimensions ordered as (Y, X), and its coordinates if any"""
    if da.ndim != 2:
        raise ValueError(f"A 2D field is expected, got dimensions {da.dims}")
    da = da.transpose(smeta.get_ydim(da), smeta.get_xdim(da))
    return da, smeta.get_lon(da, errors="ignore"), smeta.get_lat(da, errors="ignore")


def detect_fronts(da, method="cca", **kwargs):
    """Detect fronts in a 2D field

    Parameters
    ----------
    da : xarray.DataArray
        2D field, like sea surface temperature.
    method : {"cca", "cca_sliding", "boa", "canny"}, default "cca"
        Detection method:

        - "cca": Cayula and Cornillon (1992) single image edge detector
          (:func:`shoot.core.fronts.cca_sied`), on grids with 1D coordinates;
        - "cca_sliding": Cayula-Cornillon criteria on sliding windows
          (:func:`shoot.core.fronts.cca_sliding`), on grids with 1D coordinates;
        - "boa": Belkin and O'Reilly (2009) gradient magnitude
          (:func:`shoot.core.fronts.boa`) greater than `threshold`;
        - "canny": Canny edge detector (:func:`shoot.core.image.canny`).
    kwargs
        Parameters of the method:

        - "cca" and "cca_sliding": criteria of :func:`shoot.core.fronts.cca_window`
          (`min_theta`, `min_pop_prop`, `min_pop_mean_diff`, `min_single_cohesion`,
          `min_global_cohesion`, `bin_width`, `min_threshold`), plus `step`
          for "cca_sliding";
        - "boa": `threshold` (default 0.3) on the normalized gradient magnitude;
        - "canny": `low`, `high`, `sigma` (default 5) and `aperture_size`
          (default 5) of :func:`shoot.core.image.canny`.

    Returns
    -------
    xarray.DataArray
        Boolean front mask, with the dimensions ordered as (Y, X).

    Example
    -------
    >>> from shoot.fronts.fronts2d import detect_fronts
    >>> fronts = detect_fronts(ds.thetao, method="boa", threshold=0.2)  # doctest: +SKIP
    """
    if method not in METHODS:
        raise ValueError(f"Invalid method {method!r}, choose one of {METHODS}")
    da, lon, lat = _get_yx_field(da)
    z = da.values.copy()

    if method in ("cca", "cca_sliding"):
        if lon is None or lat is None or lon.ndim != 1 or lat.ndim != 1:
            raise ValueError(f"The {method!r} method requires 1D longitude and latitude coordinates")
        cca_method = "sied" if method == "cca" else "sliding"
        mask, _, _ = cfronts.cca(z, lon.values, lat.values, method=cca_method, **kwargs)
        mask = mask != 0
    elif method == "boa":
        threshold = kwargs.pop("threshold", 0.3)
        mask = cfronts.boa(z, **kwargs) >= threshold
    else:
        kwargs = {"sigma": 5, "aperture_size": 5, **kwargs}
        mask = cimage.canny(z, **kwargs) != 0

    return xr.DataArray(
        mask,
        dims=da.dims,
        coords=da.coords,
        name="fronts",
        attrs={"long_name": "Front mask", "method": method},
    )
