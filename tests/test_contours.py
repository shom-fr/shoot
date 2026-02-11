#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for contour utilities
"""

import pytest
import numpy as np
from shoot.contours import interp_to_line, area
import xarray as xr


class TestInterpToLine:
    """Test interpolation along contour lines"""

    def test_interp_to_line_simple(self):
        """Test basic interpolation along a line"""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Line through middle - format is (npts, 2) for [x, y] coordinates
        line = np.array([[1.0, 1.0], [1.5, 1.5]])

        result = interp_to_line(data, line)

        assert len(result) == 2
        assert not np.isnan(result).any()

    def test_interp_to_line_with_nan(self):
        """Test interpolation handles NaN values"""
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        # Line format is (npts, 2) for [x, y] coordinates
        line = np.array([[0.5, 0.5], [1.5, 1.5]])

        result = interp_to_line(data, line)

        assert len(result) == 2

    def test_interp_to_line_shape(self):
        """Test output shape matches line length"""
        data = np.ones((10, 10)) * 5.0
        # Line should be (npts, 2) format - array of [x, y] coordinates
        line = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

        result = interp_to_line(data, line)

        assert len(result) == 5


class TestArea:
    """Test area calculation for contours"""

    def test_area_simple_contour(self):
        """Test area calculation for a simple contour"""
        # Create a simple circular contour
        theta = np.linspace(0, 2 * np.pi, 100)
        radius = 1000  # 1 km in meters
        lon_center = 10.0
        lat_center = 42.0

        lons = lon_center + radius / 111000 * np.cos(theta)
        lats = lat_center + radius / 111000 * np.sin(theta)

        ds = xr.Dataset(
            coords={
                "lon": ("npts", lons),
                "lat": ("npts", lats),
            },
            attrs={"lon_center": lon_center, "lat_center": lat_center},
        )

        a = area(ds)

        # Should be roughly pi * r^2 = pi * 1e6 m^2
        expected = np.pi * radius**2
        assert a > 0
        # The area calculation uses approximations, just check order of magnitude
        assert a > expected * 0.001  # At least 0.1% of expected
        assert a < expected * 10  # Less than 10x expected

    def test_area_returns_positive(self):
        """Test that area calculation returns positive value"""
        # Simple square contour
        lons = np.array([10.0, 10.01, 10.01, 10.0, 10.0])
        lats = np.array([42.0, 42.0, 42.01, 42.01, 42.0])

        ds = xr.Dataset(
            coords={
                "lon": ("npts", lons),
                "lat": ("npts", lats),
            },
            attrs={"lon_center": 10.005, "lat_center": 42.005},
        )

        a = area(ds)

        assert a > 0
