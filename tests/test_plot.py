#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for plotting utilities
"""

import pytest
import numpy as np
from shoot.plot import plot_ellipse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class TestPlotEllipse:
    """Test ellipse plotting"""

    def test_plot_ellipse_returns_artist(self):
        """Test that plot_ellipse returns a matplotlib artist"""
        fig, ax = plt.subplots()

        lon, lat = 10.0, 42.0
        a, b = 50, 30  # km
        angle = 45  # degrees

        result = plot_ellipse(lon, lat, a, b, angle, ax=ax)

        assert result is not None
        assert len(result) > 0
        plt.close(fig)

    def test_plot_ellipse_with_gca(self):
        """Test plot_ellipse without explicit axis"""
        fig, ax = plt.subplots()
        plt.sca(ax)

        result = plot_ellipse(10.0, 42.0, 50, 30, 45)

        assert result is not None
        plt.close(fig)

    def test_plot_ellipse_different_angles(self):
        """Test plotting ellipses with different angles"""
        fig, ax = plt.subplots()

        for angle in [0, 45, 90, 180]:
            result = plot_ellipse(10.0, 42.0, 50, 30, angle, ax=ax)
            assert result is not None

        plt.close(fig)

    def test_plot_ellipse_circle(self):
        """Test plotting a circle (a == b)"""
        fig, ax = plt.subplots()

        result = plot_ellipse(10.0, 42.0, 50, 50, 0, ax=ax)

        assert result is not None
        plt.close(fig)

    def test_plot_ellipse_custom_npts(self):
        """Test plotting with custom number of points"""
        fig, ax = plt.subplots()

        result = plot_ellipse(10.0, 42.0, 50, 30, 45, ax=ax, npts=50)

        assert result is not None
        plt.close(fig)

    def test_plot_ellipse_kwargs(self):
        """Test that kwargs are passed to plot"""
        fig, ax = plt.subplots()

        result = plot_ellipse(
            10.0, 42.0, 50, 30, 45,
            ax=ax,
            color='red',
            linewidth=2,
            linestyle='--'
        )

        assert result is not None
        assert result[0].get_color() == 'red'
        assert result[0].get_linewidth() == 2
        assert result[0].get_linestyle() == '--'
        plt.close(fig)
