"""Tests for shoot.fronts"""

import numpy as np
import pytest
import xarray as xr

from shoot.fronts.fronts2d import METHODS, detect_fronts


@pytest.fixture(scope="module")
def sst():
    """Sea surface temperature with a front along a meridian"""
    lon = xr.DataArray(np.linspace(10.0, 15.3, 128), dims="lon", attrs={"standard_name": "longitude"})
    lat = xr.DataArray(np.linspace(35.0, 39.0, 96), dims="lat", attrs={"standard_name": "latitude"})
    values = 20 + 2 * np.tanh((lon.values[None, :] - 12.5) / 0.1) + 0 * lat.values[:, None]
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name="sst")


@pytest.mark.parametrize("method", METHODS)
def test_detect_fronts(sst, method):
    """A boolean mask with fronts near the front longitude"""
    fronts = detect_fronts(sst.copy(), method=method)
    assert fronts.dtype == bool and fronts.dims == ("lat", "lon")
    np.testing.assert_array_equal(fronts.lon, sst.lon)
    lons = np.broadcast_to(sst.lon.values, sst.shape)[fronts.values]
    assert len(lons) > 0 and abs(np.median(lons) - 12.5) < 0.1


def test_detect_fronts_invalid_method(sst):
    with pytest.raises(ValueError):
        detect_fronts(sst, method="unknown")
