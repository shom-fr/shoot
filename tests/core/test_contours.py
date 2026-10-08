"""Tests for shoot.core.contours"""

import numpy as np

import shoot.core.contours as scontours


def _gaussian(ny=40, nx=50):
    yy, xx = np.mgrid[:ny, :nx]
    return np.exp(-((xx - 25) ** 2 + (yy - 20) ** 2) / 50.0)


def test_find_closed_contours():
    """Nested closed contours around the peak, none around another point"""
    z = _gaussian()
    contours = scontours.find_closed_contours(z, 25, 20, nlevels=10)
    assert len(contours) > 3
    for level, line in contours:
        assert (line[0] == line[-1]).all()
    assert not scontours.find_closed_contours(z, 3, 3, nlevels=10)


def test_find_closed_contours_land():
    """Contours enclosing land points are rejected"""
    z = _gaussian()
    z[20, 28] = np.nan
    contours = scontours.find_closed_contours(z, 25, 20, robust=0.0)
    assert contours
    assert max(np.abs(line - [25, 20]).max() for _, line in contours) < 3


def test_contour_steps_and_smooth():
    """Length of a circle and its periodic resampling"""
    theta = np.linspace(0, 2 * np.pi, 200)
    lon, lat = 0.1 * np.cos(theta), 0.1 * np.sin(theta)
    dx, dy, length = scontours.contour_steps(lon, lat)
    np.testing.assert_allclose(length, 2 * np.pi * 0.1 * 111195, rtol=1e-2)
    lon_int, lat_int = scontours.smooth_contour(lon, lat, npts=50)
    assert lon_int.shape == (50,)
    np.testing.assert_allclose(np.hypot(lon_int, lat_int), 0.1, rtol=1e-3)
