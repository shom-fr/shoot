#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for eddy tracking
"""

import pytest
import numpy as np
from shoot.eddies.track import Track


class MockEddy:
    """Mock eddy for testing"""

    def __init__(self, lon, lat, eddy_type="anticyclonic"):
        self.lon = lon
        self.lat = lat
        self.eddy_type = eddy_type
        self.track_id = None


class TestTrack:
    """Test the Track class"""

    def test_track_initialization(self):
        """Test creating a track"""
        eddy = MockEddy(10.0, 42.0)
        time = np.datetime64("2024-01-01")

        track = Track(eddy, time, number=0, dt=1.0, Tc=10.0)

        assert len(track.eddies) == 1
        assert len(track.times) == 1
        assert track.eddies[0] == eddy
        assert track.times[0] == time
        assert track.number == 0
        assert track.active is True

    def test_track_initialization_with_list(self):
        """Test creating a track with list of eddies"""
        eddies = [MockEddy(10.0, 42.0), MockEddy(10.1, 42.1)]
        times = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]

        track = Track(eddies, times, number=0, dt=1.0, Tc=10.0)

        assert len(track.eddies) == 2
        assert len(track.times) == 2

    def test_track_update(self):
        """Test adding an eddy to a track"""
        eddy1 = MockEddy(10.0, 42.0)
        time1 = np.datetime64("2024-01-01")

        track = Track(eddy1, time1, number=0, dt=1.0, Tc=10.0)

        eddy2 = MockEddy(10.1, 42.1)
        time2 = np.datetime64("2024-01-02")
        track.update(eddy2, time2)

        assert len(track.eddies) == 2
        assert len(track.times) == 2
        assert track.eddies[1] == eddy2
        assert track.times[1] == time2

    def test_track_reconstruct(self):
        """Test reconstructing a track from eddies and times"""
        eddies = [MockEddy(10.0, 42.0), MockEddy(10.1, 42.1)]
        times = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]

        track = Track.reconstruct(eddies, times, number=5, dt=1.0, Tc=10.0)

        assert len(track.eddies) == 2
        assert len(track.times) == 2
        assert track.number == 5

    def test_track_ds_property(self):
        """Test that ds property creates a dataset"""
        eddy1 = MockEddy(10.0, 42.0)
        eddy2 = MockEddy(10.1, 42.1)
        time1 = np.datetime64("2024-01-01")
        time2 = np.datetime64("2024-01-05")

        track = Track(eddy1, time1, number=0, dt=1.0, Tc=10.0)
        track.update(eddy2, time2)

        ds = track.ds

        assert "date_first_detection" in ds
        assert "date_last_detection" in ds
        assert "life_time" in ds
        assert "x_start" in ds
        assert "y_start" in ds

        # Check values
        assert ds.date_first_detection.values[0] == time1
        assert ds.date_last_detection.values[0] == time2
        assert ds.life_time.values[0] == 4.0  # 4 days
        assert ds.x_start.values[0] == 10.0
        assert ds.y_start.values[0] == 42.0

    def test_track_multiple_updates(self):
        """Test adding multiple eddies sequentially"""
        eddy = MockEddy(10.0, 42.0)
        time = np.datetime64("2024-01-01")

        track = Track(eddy, time, number=0, dt=1.0, Tc=10.0)

        for i in range(1, 5):
            new_eddy = MockEddy(10.0 + i * 0.1, 42.0 + i * 0.1)
            new_time = np.datetime64(f"2024-01-0{i+1}")
            track.update(new_eddy, new_time)

        assert len(track.eddies) == 5
        assert len(track.times) == 5
        assert track.eddies[-1].lon == 10.4
        assert track.times[-1] == np.datetime64("2024-01-05")
