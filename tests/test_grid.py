#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for grid utilities
"""

import pytest
import numpy as np
import xarray as xr
from shoot.grid import get_dx_dy, get_wx_wy


class TestGetDxDy:
    """Test grid resolution calculations"""

    def test_get_dx_dy_with_provided_values(self):
        """Test that provided dx/dy values are returned as-is"""
        da = xr.DataArray(
            np.zeros((10, 10)),
            coords={"lat": np.linspace(40, 50, 10), "lon": np.linspace(0, 10, 10)},
            dims=["lat", "lon"],
        )
        da.lat.attrs["standard_name"] = "latitude"
        da.lon.attrs["standard_name"] = "longitude"

        dx_in = 1000.0
        dy_in = 1000.0

        dx, dy = get_dx_dy(da, dx=dx_in, dy=dy_in)

        assert dx == dx_in
        assert dy == dy_in

    def test_get_dx_dy_computed(self):
        """Test that dx/dy are computed from coordinates"""
        da = xr.DataArray(
            np.zeros((10, 10)),
            coords={"lat": np.linspace(40, 50, 10), "lon": np.linspace(0, 10, 10)},
            dims=["lat", "lon"],
        )
        da.lat.attrs["standard_name"] = "latitude"
        da.lon.attrs["standard_name"] = "longitude"

        dx, dy = get_dx_dy(da)

        assert isinstance(dx, xr.DataArray)
        assert isinstance(dy, xr.DataArray)
        assert dx.shape == (10, 10)
        assert dy.shape == (10, 10)
        assert dx.attrs["units"] == "m"
        assert dy.attrs["units"] == "m"

    def test_get_dx_dy_reasonable_values(self):
        """Test that computed dx/dy have reasonable values"""
        # 1 degree at ~45N should be roughly 70-110 km
        da = xr.DataArray(
            np.zeros((10, 10)),
            coords={"lat": np.linspace(44, 46, 10), "lon": np.linspace(0, 2, 10)},
            dims=["lat", "lon"],
        )
        da.lat.attrs["standard_name"] = "latitude"
        da.lon.attrs["standard_name"] = "longitude"

        dx, dy = get_dx_dy(da)

        # Should be on the order of tens of kilometers
        assert np.nanmean(dx.values) > 10000  # More than 10 km
        assert np.nanmean(dx.values) < 200000  # Less than 200 km
        assert np.nanmean(dy.values) > 10000
        assert np.nanmean(dy.values) < 200000


class TestGetWxWy:
    """Test window size calculations"""

    def test_get_wx_wy_basic(self):
        """Test window size calculation"""
        window = 50  # km
        dx = xr.DataArray(np.ones((10, 10)) * 5000)  # 5 km resolution
        dy = xr.DataArray(np.ones((10, 10)) * 5000)

        wx, wy = get_wx_wy(window, dx, dy)

        # 50 km / 5 km = 10, rounded to odd number should be 11
        assert wx % 2 == 1  # Must be odd
        assert wy % 2 == 1
        assert wx >= 9  # Should be around 11
        assert wy >= 9

    def test_get_wx_wy_returns_odd(self):
        """Test that window sizes are always odd numbers"""
        window = 100
        dx = xr.DataArray(np.ones((10, 10)) * 10000)
        dy = xr.DataArray(np.ones((10, 10)) * 10000)

        wx, wy = get_wx_wy(window, dx, dy)

        assert wx % 2 == 1
        assert wy % 2 == 1

    def test_get_wx_wy_different_resolutions(self):
        """Test with different dx and dy resolutions"""
        window = 50
        dx = xr.DataArray(np.ones((10, 10)) * 5000)  # 5 km
        dy = xr.DataArray(np.ones((10, 10)) * 10000)  # 10 km

        wx, wy = get_wx_wy(window, dx, dy)

        # wx should be larger than wy since dx is smaller
        assert wx > wy

    def test_get_wx_wy_large_window(self):
        """Test with large window size"""
        window = 200
        dx = xr.DataArray(np.ones((10, 10)) * 5000)
        dy = xr.DataArray(np.ones((10, 10)) * 5000)

        wx, wy = get_wx_wy(window, dx, dy)

        assert wx > 20  # Should be reasonably large
        assert wy > 20
        assert wx % 2 == 1
        assert wy % 2 == 1
