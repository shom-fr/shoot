#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for streamline calculations
"""

import pytest
import numpy as np
import xarray as xr
from shoot.streamline import psi


class TestStreamfunction:
    """Test stream function calculation"""

    def test_psi_shape(self):
        """Test that psi returns same shape as input"""
        ny, nx = 20, 20
        u = xr.DataArray(
            np.random.randn(ny, nx),
            coords={
                "lat": (("y", "x"), np.linspace(40, 45, ny)[:, None] * np.ones((ny, nx))),
                "lon": (("y", "x"), np.ones((ny, 1)) * np.linspace(0, 5, nx)),
            },
            dims=["y", "x"],
        )
        u.lat.attrs["standard_name"] = "latitude"
        u.lon.attrs["standard_name"] = "longitude"

        v = xr.DataArray(
            np.random.randn(ny, nx),
            coords={"lat": u.lat, "lon": u.lon},
            dims=["y", "x"],
        )
        v.lat.attrs["standard_name"] = "latitude"
        v.lon.attrs["standard_name"] = "longitude"

        result = psi(u, v)

        assert result.shape == u.shape
        assert result.dims == u.dims

    def test_psi_solid_body_rotation(self):
        """Test psi for solid body rotation"""
        ny, nx = 20, 20
        lat_center = 42.5
        lon_center = 2.5

        lats = np.linspace(40, 45, ny)
        lons = np.linspace(0, 5, nx)
        lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")

        # Simple solid body rotation (not perfectly realistic but testable)
        omega = 1e-5
        dlat = lat2d - lat_center
        dlon = lon2d - lon_center

        # Approximate tangential velocities
        u = -omega * dlat * 10  # Scaled for reasonable values
        v = omega * dlon * 10

        u_da = xr.DataArray(
            u,
            coords={
                "lat": (("y", "x"), lat2d),
                "lon": (("y", "x"), lon2d),
            },
            dims=["y", "x"],
        )
        u_da.lat.attrs["standard_name"] = "latitude"
        u_da.lon.attrs["standard_name"] = "longitude"

        v_da = xr.DataArray(
            v,
            coords={"lat": u_da.lat, "lon": u_da.lon},
            dims=["y", "x"],
        )
        v_da.lat.attrs["standard_name"] = "latitude"
        v_da.lon.attrs["standard_name"] = "longitude"

        result = psi(u_da, v_da)

        # For a vortex, psi should have extremum near center
        assert result.shape == u.shape
        assert not np.isnan(result.values).all()

    def test_psi_attributes(self):
        """Test that psi has correct attributes"""
        ny, nx = 10, 10
        u = xr.DataArray(
            np.ones((ny, nx)),
            coords={
                "lat": (("y", "x"), np.linspace(40, 45, ny)[:, None] * np.ones((ny, nx))),
                "lon": (("y", "x"), np.ones((ny, 1)) * np.linspace(0, 5, nx)),
            },
            dims=["y", "x"],
        )
        u.lat.attrs["standard_name"] = "latitude"
        u.lon.attrs["standard_name"] = "longitude"

        v = xr.DataArray(
            np.zeros((ny, nx)),
            coords={"lat": u.lat, "lon": u.lon},
            dims=["y", "x"],
        )
        v.lat.attrs["standard_name"] = "latitude"
        v.lon.attrs["standard_name"] = "longitude"

        result = psi(u, v)

        assert result.name == "psi"
        assert "long_name" in result.attrs
        assert result.attrs["long_name"] == "Streamfunction"

    def test_psi_zero_velocity(self):
        """Test psi with zero velocity field"""
        ny, nx = 10, 10
        u = xr.DataArray(
            np.zeros((ny, nx)),
            coords={
                "lat": (("y", "x"), np.linspace(40, 45, ny)[:, None] * np.ones((ny, nx))),
                "lon": (("y", "x"), np.ones((ny, 1)) * np.linspace(0, 5, nx)),
            },
            dims=["y", "x"],
        )
        u.lat.attrs["standard_name"] = "latitude"
        u.lon.attrs["standard_name"] = "longitude"

        v = xr.DataArray(
            np.zeros((ny, nx)),
            coords={"lat": u.lat, "lon": u.lon},
            dims=["y", "x"],
        )
        v.lat.attrs["standard_name"] = "latitude"
        v.lon.attrs["standard_name"] = "longitude"

        result = psi(u, v)

        # Zero velocity should give nearly zero stream function
        assert result.shape == u.shape
        assert np.allclose(result.values, 0.0)
