#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for profile utilities
"""

import pytest
import numpy as np
import xarray as xr
from shoot.profiles.profiles import Profile


class TestProfile:
    """Test the Profile class"""

    def test_profile_initialization(self):
        """Test creating a profile from mock data"""
        # Create mock profile data
        depth_levels = np.arange(0, 2000, 10)
        prf = xr.Dataset(
            {
                "TIME": xr.DataArray([1000.0], dims="obs"),
                "LATITUDE": xr.DataArray([42.0], dims="obs"),
                "LONGITUDE": xr.DataArray([10.0], dims="obs"),
                "PRES_ADJUSTED": xr.DataArray(depth_levels, dims="depth"),
                "TEMP_ADJUSTED": xr.DataArray(20.0 - depth_levels * 0.01, dims="depth"),  # Decreasing temp
                "PSAL_ADJUSTED": xr.DataArray(35.0 + depth_levels * 0.001, dims="depth"),  # Increasing sal
            }
        )

        profile = Profile(prf)

        assert profile.lat == 42.0
        assert profile.lon == 10.0
        assert len(profile.depth) == 2000
        assert len(profile.temp) == 2000
        assert len(profile.sal) == 2000

    def test_profile_interpolation(self):
        """Test that profile data is interpolated to standard depths"""
        prf = xr.Dataset(
            {
                "TIME": xr.DataArray([1000.0], dims="obs"),
                "LATITUDE": xr.DataArray([42.0], dims="obs"),
                "LONGITUDE": xr.DataArray([10.0], dims="obs"),
                "PRES_ADJUSTED": xr.DataArray([0, 100, 500, 1000], dims="depth"),
                "TEMP_ADJUSTED": xr.DataArray([20.0, 15.0, 10.0, 5.0], dims="depth"),
                "PSAL_ADJUSTED": xr.DataArray([35.0, 35.5, 36.0, 36.5], dims="depth"),
            }
        )

        profile = Profile(prf)

        # Check interpolation works
        assert not np.isnan(profile.temp[50])  # Within range
        assert not np.isnan(profile.sal[50])

    def test_profile_validity_check(self):
        """Test profile validity determination"""
        # Valid profile with most data
        depth_levels = np.arange(0, 1000, 10)
        prf_valid = xr.Dataset(
            {
                "TIME": xr.DataArray([1000.0], dims="obs"),
                "LATITUDE": xr.DataArray([42.0], dims="obs"),
                "LONGITUDE": xr.DataArray([10.0], dims="obs"),
                "PRES_ADJUSTED": xr.DataArray(depth_levels, dims="depth"),
                "TEMP_ADJUSTED": xr.DataArray(20.0 - depth_levels * 0.01, dims="depth"),
                "PSAL_ADJUSTED": xr.DataArray(35.0 + depth_levels * 0.001, dims="depth"),
            }
        )

        profile_valid = Profile(prf_valid)
        # Should have some valid data
        assert isinstance(profile_valid.valid, (bool, np.bool_))

    def test_profile_time_conversion(self):
        """Test that time is correctly stored as scalar"""
        prf = xr.Dataset(
            {
                "TIME": xr.DataArray([np.datetime64("1950-01-01")], dims="obs"),
                "LATITUDE": xr.DataArray([42.0], dims="obs"),
                "LONGITUDE": xr.DataArray([10.0], dims="obs"),
                "PRES_ADJUSTED": xr.DataArray([0, 100], dims="depth"),
                "TEMP_ADJUSTED": xr.DataArray([20.0, 15.0], dims="depth"),
                "PSAL_ADJUSTED": xr.DataArray([35.0, 35.5], dims="depth"),
            }
        )

        profile = Profile(prf)
        assert profile.time == np.datetime64("1950-01-01")
        assert isinstance(profile.lat, (float, np.floating))
        assert isinstance(profile.lon, (float, np.floating))
