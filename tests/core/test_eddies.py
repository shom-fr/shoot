"""Tests for shoot.core.eddies"""

import numpy as np

import shoot.core.eddies as seddies


def _square(x0, y0, size):
    return np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]], "d")


def test_argmax_first():
    """First maximum, NaNs ignored after the first position"""
    assert seddies.argmax_first([1.0, 3.0, 3.0, np.nan]) == 1


def test_filter_intersecting():
    """The slowest of two intersecting eddies is removed"""
    polys = [_square(0, 0, 2), _square(1, 1, 2), _square(10, 10, 1)]
    keep = seddies.filter_intersecting(polys, [1.0, 2.0, 0.5])
    np.testing.assert_array_equal(keep, [False, True, True])
