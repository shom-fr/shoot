"""Tests for shoot.core.fronts"""

import numpy as np
import pytest

import shoot.core.fronts as sfronts


@pytest.fixture(scope="module")
def front_field():
    """Temperature field with a meandering front, and its coordinates"""
    ny, nx = 96, 128
    x = np.linspace(10.0, 15.3, nx)
    y = np.linspace(35.0, 39.0, ny)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    xfront = 12.5 + 0.3 * np.sin(2 * np.pi * (yy - 35) / 4)
    z = 20 + 2 * np.tanh((xx - xfront) / 0.1)
    return z, x, y, xfront[:, 0]


def _mean_distance_to_front(xpts, ypts, y, xfront):
    return np.mean(np.abs(xpts - np.interp(ypts, y, xfront)))


@pytest.mark.parametrize("method", ["sied", "sliding"])
def test_cca(front_field, method):
    """Front points are found along the front"""
    z, x, y, xfront = front_field
    zin = z.copy()
    mask, xpts, ypts = sfronts.cca(z, x, y, method=method)
    np.testing.assert_array_equal(z, zin)  # input unchanged
    assert mask.shape == z.shape and mask.sum() > 0
    assert _mean_distance_to_front(xpts, ypts, y, xfront) < 0.05


def test_cca_window_uniform():
    """A uniform window has no front"""
    x, y, _, status = sfronts.cca_window(np.full((33, 33), 20.0), [0, 1, 0, 1])
    assert x is None and status != sfronts.CCA_FRONT


def test_boa(front_field):
    """The gradient magnitude is maximal at the front"""
    z, x, y, xfront = front_field
    magnitude, direction = sfronts.boa(z, direction=True)
    j = z.shape[0] // 2
    assert abs(x[np.nanargmax(magnitude[j])] - xfront[j]) < 0.1
    assert np.nanmedian(direction[magnitude > 0.5]) == pytest.approx(90, abs=30)  # eastward gradient
