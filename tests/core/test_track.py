"""Tests for shoot.core.track"""

import numpy as np

import shoot.core.track as strack


def test_association_cost():
    """Close eddies of the same type are cheap, other pairs are impossible"""
    cost = strack.association_cost(
        [5.0, 5.0, 8.0],  # lon_new
        [43.0, 43.0, 43.0],  # lat_new
        [20.0, 20.0, 20.0],  # radius_new
        [0.1, 0.1, 0.1],  # ro_new
        [0, 1, 0],  # type_new
        [5.05],  # lon_ref
        [43.0],  # lat_ref
        [20.0],  # radius_ref
        [0.1],  # ro_ref
        [0],  # type_ref
        [50e3],  # dist_ref
    )
    assert cost.shape == (3, 1)
    np.testing.assert_allclose(cost[0, 0], 4067.0 / 50e3, rtol=1e-3)
    assert cost[1, 0] >= 1e3  # different type
    assert cost[2, 0] >= 1e3  # too far
