#!/usr/bin/env python3
"""
Tests for streamline calculations
"""

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


def _geostrophic_eddy(n=41, amplitude=0.1, lat0=43.0, land=None):
    """Velocities of a gaussian geostrophic eddy, with NaNs where `land` is True"""
    lat = np.linspace(lat0 - 1, lat0 + 1, n)
    lon = np.linspace(4, 6, n)
    lon2d, lat2d = np.meshgrid(lon, lat)
    x = (lon2d - lon.mean()) * np.pi * 6371e3 / 180 * np.cos(np.radians(lat2d))
    y = (lat2d - lat0) * np.pi * 6371e3 / 180
    eta = amplitude * np.exp(-(x**2 + y**2) / (2 * 30e3**2))
    f = 2 * 7.2921e-5 * np.sin(np.radians(lat2d))
    u = -9.81 / f * np.gradient(eta, axis=0) / np.gradient(y, axis=0)
    v = 9.81 / f * np.gradient(eta, axis=1) / np.gradient(x, axis=1)
    if land is not None:
        u[land] = np.nan
        v[land] = np.nan
    coords = {
        "lat": ("lat", lat, {"standard_name": "latitude"}),
        "lon": ("lon", lon, {"standard_name": "longitude"}),
    }
    u = xr.DataArray(u, dims=("lat", "lon"), coords=coords)
    v = xr.DataArray(v, dims=("lat", "lon"), coords=coords)
    return u, v, eta


class TestStreamfunctionEddy:
    """Test the stream function of a geostrophic eddy"""

    def test_psi_scale(self):
        """Psi in 1e3 m2/s is g / (1e3 f) times the sea level"""
        u, v, eta = _geostrophic_eddy()
        result = psi(u, v)
        f0 = 2 * 7.2921e-5 * np.sin(np.radians(43.0))
        ratio = np.ptp(result.values) / np.ptp(eta)
        np.testing.assert_allclose(ratio, 9.81 / (1e3 * f0), rtol=0.03)
        assert result.attrs["units"] == "1e3 m2 s-1"

    def test_psi_origin(self):
        """Psi is null at the integration origin"""
        u, v, _ = _geostrophic_eddy()
        result = psi(u, v, ci=12, cj=27)
        assert result.values[27, 12] == 0.0

    def test_psi_origin_on_first_index(self):
        """The integration origin can be on the first indices"""
        u, v, _ = _geostrophic_eddy()
        result = psi(u, v, ci=0, cj=0)
        assert result.shape == u.shape
        assert result.values[0, 0] == 0.0
        assert np.isfinite(result.values).all()

    def test_psi_land(self):
        """Land does not propagate NaNs and psi is extended by one pixel into land"""
        n = 41
        land = np.zeros((n, n), dtype=bool)
        land[30:, 32:] = True
        u, v, _ = _geostrophic_eddy(n=n, land=land)
        result = psi(u, v).values

        # ocean points are all valid
        assert np.isfinite(result[~land]).all()
        # first land pixel is filled, farther land points are masked
        assert np.isfinite(result[30, 32]) and np.isfinite(result[35, 32])
        assert np.isnan(result[31:, 33:]).all()

        # ocean values are close to those without land
        u0, v0, _ = _geostrophic_eddy(n=n)
        ref = psi(u0, v0).values
        assert np.abs(result - ref)[~land].max() < 0.05 * np.ptp(ref)
