"""Tests for shoot.core.image"""

import numpy as np

import shoot.core.image as simage


def test_sobel_gradients_ramp():
    """Constant gradients of a linear ramp"""
    yy, xx = np.mgrid[:20, :30].astype("d")
    gx, gy = simage.sobel_gradients(2 * xx + 3 * yy)
    np.testing.assert_allclose(gx[1:-1, 1:-1], 2 * 8)
    np.testing.assert_allclose(gy[1:-1, 1:-1], 3 * 8)


def test_hysteresis_chain():
    """Weak edges connected to a strong one through other weak edges are kept"""
    edges = np.zeros((5, 8), dtype=np.uint8)
    edges[2, 1] = 255
    edges[2, 2:6] = 50  # chain of weak edges
    edges[0, 7] = 50  # isolated weak edge
    out = simage.hysteresis(edges)
    assert (out[2, 1:6] == 255).all()
    assert out[0, 7] == 0


def test_fill_nans_nearest():
    """NaNs are replaced by the nearest valid value"""
    z = np.array([[1.0, np.nan, np.nan, 4.0]])
    np.testing.assert_array_equal(simage.fill_nans_nearest(z), [[1.0, 1.0, 4.0, 4.0]])


def test_canny():
    """Edges along a meandering front, not on NaNs"""
    yy, xx = np.mgrid[:96, :128].astype("d")
    xfront = 60 + 8 * np.sin(2 * np.pi * yy[:, 0] / 96)
    z = 20 + 2 * np.tanh((xx - xfront[:, None]) / 2)
    z[:10, :10] = np.nan
    edges = simage.canny(z, sigma=1)
    jj, ii = np.nonzero(edges)
    assert len(jj) > 0 and not edges[:10, :10].any()
    assert np.mean(np.abs(ii - xfront[jj])) < 2
