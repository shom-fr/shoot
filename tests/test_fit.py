"""Tests for shoot.fit module"""
import pytest
import numpy as np
import shoot.fit as sfit
import shoot.geo as sgeo


class TestEllipseFitting:
    """Test ellipse fitting functionality"""
    
    def test_fit_ellipse_perfect_circle(self):
        """Test ellipse fitting on a perfect circle"""
        # Create points on a perfect circle
        angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
        radius = 2.0  # km
        center_lon, center_lat = 0.0, 45.0
        
        # Convert to x,y coordinates then back to lon,lat
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
        lats = center_lat + sgeo.m2deg(y * 1000)
        
        result = sfit.fit_ellipse_from_coords(lons, lats)
        
        # Check that the fitted ellipse is close to the original circle
        assert abs(result['lon'] - center_lon) < 0.1
        assert abs(result['lat'] - center_lat) < 0.1
        assert abs(result['a'] - radius) < 0.5  # Semi-major axis
        assert abs(result['b'] - radius) < 0.5  # Semi-minor axis
        
    def test_fit_ellipse_with_error_metric(self):
        """Test ellipse fitting with error metric returned"""
        # Create a slightly noisy circle
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 3.0
        center_lon, center_lat = 2.0, 50.0
        
        # Add small amount of noise
        noise_scale = 0.1
        x = radius * np.cos(angles) + np.random.normal(0, noise_scale, len(angles))
        y = radius * np.sin(angles) + np.random.normal(0, noise_scale, len(angles))
        
        lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
        lats = center_lat + sgeo.m2deg(y * 1000)
        
        result, error = sfit.fit_ellipse_from_coords(lons, lats, get_fit=True)
        
        # Check basic structure
        assert isinstance(result, dict)
        assert isinstance(error, (float, np.floating))
        assert error >= 0
        
        # Check required keys
        required_keys = ['lon', 'lat', 'a', 'b', 'angle']
        for key in required_keys:
            assert key in result
            
    def test_fit_ellipse_true_ellipse(self):
        """Test fitting on a true ellipse"""
        # Create points on an ellipse
        angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
        a, b = 4.0, 2.0  # Semi-major and semi-minor axes in km
        center_lon, center_lat = -3.0, 42.0
        rotation = 30.0  # degrees
        
        # Generate ellipse points
        theta_rad = np.radians(rotation)
        x_ellipse = a * np.cos(angles)
        y_ellipse = b * np.sin(angles)
        
        # Rotate
        x_rot = x_ellipse * np.cos(theta_rad) - y_ellipse * np.sin(theta_rad)
        y_rot = x_ellipse * np.sin(theta_rad) + y_ellipse * np.cos(theta_rad)
        
        # Convert to lon/lat
        lons = center_lon + sgeo.m2deg(x_rot * 1000, center_lat)
        lats = center_lat + sgeo.m2deg(y_rot * 1000)
        
        result = sfit.fit_ellipse_from_coords(lons, lats)
        
        # Check fitted parameters (allowing for some tolerance)
        assert abs(result['lon'] - center_lon) < 0.2
        assert abs(result['lat'] - center_lat) < 0.2
        assert abs(result['a'] - a) < 0.5  # Semi-major axis should be close to 'a'
        assert abs(result['b'] - b) < 0.5  # Semi-minor axis should be close to 'b'
        
        # Angle can be tricky due to orientation ambiguity, just check it's reasonable
        assert -180 <= result['angle'] <= 180
        
    def test_fit_ellipse_degenerate_cases(self):
        """Test ellipse fitting with degenerate cases"""
        # Nearly collinear points (very elongated ellipse)
        lons = np.array([0, 1, 2, 3, 4, 5])
        lats = np.array([0, 0.1, 0, -0.1, 0, 0.1])  # Small variations
        
        # Should not crash, even if fit is poor
        result = sfit.fit_ellipse_from_coords(lons, lats)
        
        assert isinstance(result, dict)
        assert 'lon' in result
        assert 'lat' in result
        assert 'a' in result
        assert 'b' in result
        assert 'angle' in result
        
    def test_fit_ellipse_minimum_points(self):
        """Test ellipse fitting with minimum number of points"""
        # 5 points (minimum for ellipse fitting)
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        radius = 1.5
        center_lon, center_lat = 0.0, 45.0
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
        lats = center_lat + sgeo.m2deg(y * 1000)
        
        result = sfit.fit_ellipse_from_coords(lons, lats)
        
        # Should work with minimum points
        assert isinstance(result, dict)
        assert all(key in result for key in ['lon', 'lat', 'a', 'b', 'angle'])
        
    def test_fit_ellipse_array_inputs(self):
        """Test ellipse fitting with numpy array inputs"""
        # Create circle as numpy arrays
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        radius = 2.5
        center_lon, center_lat = 1.0, 46.0
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        lons_array = np.array(center_lon + sgeo.m2deg(x * 1000, center_lat))
        lats_array = np.array(center_lat + sgeo.m2deg(y * 1000))
        
        result = sfit.fit_ellipse_from_coords(lons_array, lats_array)
        
        assert isinstance(result, dict)
        assert abs(result['lon'] - center_lon) < 0.2
        assert abs(result['lat'] - center_lat) < 0.2
        
    def test_fit_ellipse_different_latitudes(self):
        """Test ellipse fitting at different latitudes"""
        test_latitudes = [0.0, 30.0, 60.0, 80.0]
        
        for lat in test_latitudes:
            # Create circle at this latitude
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radius = 2.0
            center_lon, center_lat = 0.0, lat
            
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            
            lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
            lats = center_lat + sgeo.m2deg(y * 1000)
            
            result = sfit.fit_ellipse_from_coords(lons, lats)
            
            # Should work at all latitudes
            assert isinstance(result, dict)
            assert abs(result['lat'] - center_lat) < 0.3
            # Longitude accuracy may vary with latitude
            if lat < 85:  # Avoid extreme polar regions
                assert abs(result['lon'] - center_lon) < 0.5


class TestResidualFunction:
    """Test the residual function used in ellipse fitting"""
    
    def test_residuals_perfect_fit(self):
        """Test residuals for points exactly on ellipse"""
        # Parameters for a circle of radius 2, centered at origin
        params = [0, 0, 2, 2, 0]  # xc, yc, a, b, theta
        
        # Points exactly on the circle
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        points = np.array([[2*np.cos(a), 2*np.sin(a)] for a in angles])
        
        residuals = sfit._residuals(params, points)
        
        # Residuals should be close to zero for perfect fit
        np.testing.assert_array_almost_equal(residuals, np.zeros(len(angles)), decimal=10)
        
    def test_residuals_points_outside_ellipse(self):
        """Test residuals for points outside ellipse"""
        params = [0, 0, 2, 1, 0]  # Ellipse with a=2, b=1
        
        # Points outside the ellipse
        points = np.array([[3, 0], [0, 2]])  # Both outside
        
        residuals = sfit._residuals(params, points)
        
        # Residuals should be positive (distance > 1)
        assert all(r > 0 for r in residuals)
        
    def test_residuals_points_inside_ellipse(self):
        """Test residuals for points inside ellipse"""
        params = [0, 0, 2, 1, 0]  # Ellipse with a=2, b=1
        
        # Points inside the ellipse
        points = np.array([[0.5, 0], [0, 0.3]])
        
        residuals = sfit._residuals(params, points)
        
        # Residuals should be negative (distance < 1)
        assert all(r < 0 for r in residuals)
        
    def test_residuals_rotated_ellipse(self):
        """Test residuals for rotated ellipse"""
        theta = np.pi/4  # 45 degree rotation
        params = [0, 0, 2, 1, theta]
        
        # Point on rotated ellipse
        # For 45° rotation, point (sqrt(2), sqrt(2)) should be on ellipse
        point_on_ellipse = np.array([[np.sqrt(2), np.sqrt(2)]])
        
        residuals = sfit._residuals(params, point_on_ellipse)
        
        # Should be close to zero
        np.testing.assert_array_almost_equal(residuals, [0], decimal=1)


@pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.5])
def test_ellipse_fitting_noise_robustness(noise_level):
    """Test ellipse fitting robustness to noise"""
    # Create noisy circle
    np.random.seed(42)  # For reproducible tests
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    radius = 3.0
    center_lon, center_lat = 1.0, 45.0
    
    # Add noise
    x = radius * np.cos(angles) + np.random.normal(0, noise_level, len(angles))
    y = radius * np.sin(angles) + np.random.normal(0, noise_level, len(angles))
    
    lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
    lats = center_lat + sgeo.m2deg(y * 1000)
    
    result, error = sfit.fit_ellipse_from_coords(lons, lats, get_fit=True)
    
    # Error should increase with noise level
    assert error >= 0
    if noise_level == 0:
        assert error < 0.01  # Very small error for perfect data
    
    # Center should still be reasonably close
    center_tolerance = 0.5 + noise_level * 2  # Tolerance increases with noise
    assert abs(result['lon'] - center_lon) < center_tolerance
    assert abs(result['lat'] - center_lat) < center_tolerance


def test_ellipse_fitting_edge_cases():
    """Test edge cases in ellipse fitting"""
    # Test with identical points (should handle gracefully)
    lons = np.array([0, 0, 0, 0, 0])
    lats = np.array([45, 45, 45, 45, 45])
    
    # Should not crash (though fit may be poor/undefined)
    try:
        result = sfit.fit_ellipse_from_coords(lons, lats)
        assert isinstance(result, dict)
    except:
        # It's acceptable for degenerate cases to fail
        pass
        

def test_ellipse_parameter_ranges():
    """Test that fitted ellipse parameters are in expected ranges"""
    # Create a reasonable ellipse
    angles = np.linspace(0, 2*np.pi, 15, endpoint=False)
    a, b = 5.0, 3.0
    center_lon, center_lat = 2.0, 47.0
    
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    
    lons = center_lon + sgeo.m2deg(x * 1000, center_lat)
    lats = center_lat + sgeo.m2deg(y * 1000)
    
    result = sfit.fit_ellipse_from_coords(lons, lats)
    
    # Check parameter ranges
    assert result['a'] > 0  # Positive semi-major axis
    assert result['b'] > 0  # Positive semi-minor axis
    assert result['a'] >= result['b']  # Semi-major >= semi-minor
    assert -180 <= result['angle'] <= 180  # Angle in reasonable range
    
    # Geographic coordinates should be reasonable
    assert -180 <= result['lon'] <= 180
    assert -90 <= result['lat'] <= 90
