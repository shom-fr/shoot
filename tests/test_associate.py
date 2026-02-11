#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for eddy association
"""

import pytest
import numpy as np
from shoot.eddies.associate import Associate


class MockEddy:
    """Mock eddy for testing"""

    def __init__(self, lat, lon, radius, ro, eddy_type):
        self.lat = lat
        self.lon = lon
        self.radius = radius
        self.ro = ro
        self.eddy_type = eddy_type
        self.id = None


class TestAssociate:
    """Test the Associate class for eddy matching"""

    def test_associate_close_eddies(self):
        """Test that close eddies with similar properties have low cost"""
        ref_eddies = [MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic")]
        eddies = [MockEddy(42.01, 10.01, 52000, 0.11, "anticyclonic")]

        assoc = Associate(eddies, ref_eddies, dmax=50)
        cost = assoc.cost

        # Cost should be small for close, similar eddies
        assert cost[0, 0] < 10

    def test_associate_far_eddies(self):
        """Test that distant eddies have high cost"""
        ref_eddies = [MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic")]
        eddies = [MockEddy(45.0, 15.0, 52000, 0.11, "anticyclonic")]

        assoc = Associate(eddies, ref_eddies, dmax=50)
        cost = assoc.cost

        # Cost should be very high (>1000) for far eddies
        assert cost[0, 0] > 1000

    def test_associate_different_types(self):
        """Test that cyclonic and anticyclonic eddies have high cost"""
        ref_eddies = [MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic")]
        eddies = [MockEddy(42.01, 10.01, 52000, 0.11, "cyclonic")]

        assoc = Associate(eddies, ref_eddies, dmax=50)
        cost = assoc.cost

        # Different types should have very high cost even if close
        assert cost[0, 0] > 1000

    def test_cost_matrix_shape(self):
        """Test cost matrix has correct shape"""
        ref_eddies = [
            MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic"),
            MockEddy(43.0, 11.0, 60000, 0.15, "cyclonic"),
        ]
        eddies = [
            MockEddy(42.1, 10.1, 51000, 0.11, "anticyclonic"),
            MockEddy(43.1, 11.1, 61000, 0.16, "cyclonic"),
            MockEddy(44.0, 12.0, 55000, 0.12, "anticyclonic"),
        ]

        assoc = Associate(eddies, ref_eddies, dmax=100)
        cost = assoc.cost

        assert cost.shape == (3, 2)  # 3 eddies x 2 ref_eddies

    def test_cost_increases_with_distance(self):
        """Test that cost increases with distance"""
        ref_eddy = MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic")
        eddy_close = MockEddy(42.01, 10.01, 50000, 0.1, "anticyclonic")
        eddy_far = MockEddy(42.1, 10.1, 50000, 0.1, "anticyclonic")

        assoc_close = Associate([eddy_close], [ref_eddy], dmax=100)
        assoc_far = Associate([eddy_far], [ref_eddy], dmax=100)

        cost_close = assoc_close.cost[0, 0]
        cost_far = assoc_far.cost[0, 0]

        assert cost_far > cost_close

    def test_multiple_associations(self):
        """Test cost matrix for multiple eddies"""
        ref_eddies = [
            MockEddy(42.0, 10.0, 50000, 0.1, "anticyclonic"),
            MockEddy(43.0, 11.0, 60000, 0.15, "cyclonic"),
        ]
        eddies = [
            MockEddy(42.02, 10.02, 51000, 0.11, "anticyclonic"),
            MockEddy(43.02, 11.02, 61000, 0.16, "cyclonic"),
        ]

        assoc = Associate(eddies, ref_eddies, dmax=50)
        cost = assoc.cost

        # Diagonal should have lower costs (matching eddies)
        assert cost[0, 0] < cost[0, 1]  # First eddy closer to first ref
        assert cost[1, 1] < cost[1, 0]  # Second eddy closer to second ref
