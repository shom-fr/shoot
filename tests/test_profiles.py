#!/usr/bin/env python3
"""
Tests for profile utilities
"""

import numpy as np
import pytest
import xarray as xr

from shoot.profiles.profiles import Profile, Profiles


@pytest.fixture
def argo_profiles():
    """Three Argo profiles as given by argopy (standard mode), with NaN padding"""
    nan = np.nan
    pres = [
        [0.0, 100.0, 500.0, 1000.0, nan],  # padded
        [500.0, 0.0, 100.0, 200.0, 300.0],  # unsorted
        [0.0, 5.0, nan, nan, nan],  # too short to be valid
    ]
    temp = [
        [20.0, 15.0, 10.0, 5.0, nan],
        [10.0, 20.0, 15.0, nan, 13.0],  # missing value
        [20.0, 19.0, nan, nan, nan],
    ]
    sal = [
        [38.0, 38.5, 38.7, 38.8, nan],
        [38.7, 38.0, 38.5, 38.6, 38.6],
        [38.0, 38.1, nan, nan, nan],
    ]
    dims = ("N_PROF", "N_LEVELS")
    return xr.Dataset(
        {
            "PRES": (dims, pres),
            "TEMP": (dims, temp),
            "PSAL": (dims, sal),
            "PLATFORM_NUMBER": ("N_PROF", [6903000, 6903000, 6903001]),
        },
        coords={
            "TIME": ("N_PROF", np.array(["2024-01-01", "2024-01-05", "2024-01-09"], dtype="M8[ns]")),
            "LATITUDE": ("N_PROF", [35.0, 35.2, 35.4]),
            "LONGITUDE": ("N_PROF", [15.0, 15.1, 15.2]),
        },
    )


def test_profile(argo_profiles):
    """Scalars and interpolation to depths, ignoring NaN levels"""
    profile = Profile(argo_profiles.isel(N_PROF=0))
    assert profile.time == np.datetime64("2024-01-01")
    assert profile.lat == 35.0 and isinstance(profile.lat, float)
    assert profile.lon == 15.0 and isinstance(profile.lon, float)
    assert profile.float_id == 6903000
    assert len(profile.depth) == len(profile.temp) == len(profile.sal) == 2000
    np.testing.assert_allclose(profile.temp[[0, 49, 299, 999]], [19.95, 17.5, 12.5, 5.0])
    assert np.isnan(profile.temp[1000:]).all()
    assert profile.valid


def test_profile_unsorted_with_missing_values(argo_profiles):
    """Levels are sorted and missing values are skipped"""
    profile = Profile(argo_profiles.isel(N_PROF=1), depth=[50, 250, 400])
    np.testing.assert_allclose(profile.temp, [17.5, 13.5, 11.5])
    np.testing.assert_allclose(profile.sal, [38.25, 38.6, 38.65])


def test_profile_invalid_and_without_platform(argo_profiles):
    """Short profiles are invalid and the platform number is optional"""
    profile = Profile(argo_profiles.isel(N_PROF=2).drop_vars("PLATFORM_NUMBER"))
    assert profile.float_id is None
    assert not profile.valid


def test_profiles(argo_profiles):
    """Collection of valid profiles"""
    profiles = Profiles(argo_profiles.TIME, "/tmp", argo_profiles)
    assert len(profiles.profiles) == 2
    assert profiles.float_ids == [6903000]
    ds = profiles.ds
    assert ds.temp.shape == (2, 2000)
    np.testing.assert_allclose(ds.lat, [35.0, 35.2])
