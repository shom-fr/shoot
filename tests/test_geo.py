"""Tests for shoot.geo module"""
import pytest
import numpy as np
import shoot.geo as sgeo


class TestGeoUtilities:
    """Test geographic utility functions"""
    
    def test_deg2m_zonal_distance_equator(self):
        """Test degree to meter conversion for zonal distance at equator"""
        # At equator (lat=0), 1 degree longitude ≈ 111.32 km
        result = sgeo.deg2m(1.0, lat=0.0)
        expected = np.pi * sgeo.EARTH_RADIUS / 180.0  # ≈ 111320 m
        assert abs(result - expected) < 1e-6
        
    def test_deg2m_meridional_distance(self):
        """Test degree to meter conversion for meridional distance"""
        # 1 degree latitude ≈ 111.32 km everywhere
        result = sgeo.deg2m(1.0)
        expected = np.pi * sgeo.EARTH_RADIUS / 180.0
        assert abs(result - expected) < 1e-6
        
    def test_deg2m_at_60_degrees(self):
        """Test degree to meter conversion at 60°N latitude"""
        # At 60°N, zonal distance should be half of equatorial value
        result = sgeo.deg2m(1.0, lat=60.0)
        expected = np.pi * sgeo.EARTH_RADIUS / 180.0 * np.cos(np.radians(60.0))
        assert abs(result - expected) < 1e-6
        
    def test_deg2m_at_poles(self):
        """Test degree to meter conversion at poles"""
        # At 90°N, zonal distance should be 0
        result = sgeo.deg2m(1.0, lat=90.0)
        expected = 0.0
        assert abs(result - expected) < 1e-10
        
    def test_m2deg_conversion_equator(self):
        """Test meter to degree conversion at equator"""
        # Convert 111320 m to degrees at equator
        result = sgeo.m2deg(111320, lat=0.0)
        expected = 1.0
        assert abs(result - expected) < 1e-3
        
    def test_m2deg_meridional(self):
        """Test meter to degree conversion for meridional distance"""
        result = sgeo.m2deg(111320)
        expected = 1.0
        assert abs(result - expected) < 1e-3
        
    def test_deg2m_m2deg_roundtrip_equator(self):
        """Test that deg2m and m2deg are inverse operations at equator"""
        deg_input = 2.5
        lat = 0.0
        
        meters = sgeo.deg2m(deg_input, lat=lat)
        deg_output = sgeo.m2deg(meters, lat=lat)
        
        assert abs(deg_input - deg_output) < 1e-10
        
    def test_deg2m_m2deg_roundtrip_45N(self):
        """Test that deg2m and m2deg are inverse operations at 45°N"""
        deg_input = 2.5
        lat = 45.0
        
        meters = sgeo.deg2m(deg_input, lat=lat)
        deg_output = sgeo.m2deg(meters, lat=lat)
        
        assert abs(deg_input - deg_output) < 1e-10
        
    def test_earth_radius_constant(self):
        """Test that EARTH_RADIUS has expected value"""
        assert sgeo.EARTH_RADIUS == 6371e3


@pytest.mark.parametrize("lat,expected_ratio", [
    (0.0, 1.0),      # Equator
    (30.0, np.sqrt(3)/2),  # 30°N
    (60.0, 0.5),     # 60°N
    (90.0, 0.0),     # North pole
])
def test_deg2m_latitude_scaling(lat, expected_ratio):
    """Test that zonal degree-to-meter conversion scales correctly with latitude"""
    equatorial_distance = sgeo.deg2m(1.0, lat=0.0)
    lat_distance = sgeo.deg2m(1.0, lat=lat)
    
    ratio = lat_distance / equatorial_distance
    assert abs(ratio - expected_ratio) < 1e-10


@pytest.mark.parametrize("deg_input", [0.1, 0.5, 1.0, 2.5, 10.0])
def test_various_degree_inputs(deg_input):
    """Test conversion with various degree inputs"""
    lat = 45.0
    
    meters = sgeo.deg2m(deg_input, lat=lat)
    deg_output = sgeo.m2deg(meters, lat=lat)
    
    assert abs(deg_input - deg_output) < 1e-10


def test_negative_degrees():
    """Test conversion with negative degree values"""
    result_pos = sgeo.deg2m(1.0, lat=45.0)
    result_neg = sgeo.deg2m(-1.0, lat=45.0)
    
    assert abs(result_pos + result_neg) < 1e-10


def test_array_input():
    """Test conversion with numpy array input"""
    degrees = np.array([0.5, 1.0, 2.0])
    lat = 45.0
    
    result = sgeo.deg2m(degrees, lat=lat)
    
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    
    # Check each element
    for i, deg in enumerate(degrees):
        expected = sgeo.deg2m(deg, lat=lat)
        assert abs(result[i] - expected) < 1e-10
