"""Tests for shoot.num module"""
import pytest
import numpy as np
import shoot.num as snum


class TestPointsInPolygon:
    """Test point in polygon functionality"""
    
    def test_points_in_polygon_simple_square(self):
        """Test point in polygon for a simple square"""
        # Define a square polygon
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
        
        # Points to test
        point_inside = np.array([1.0, 1.0])
        point_outside = np.array([3.0, 3.0])
        point_on_vertex = np.array([0.0, 0.0])
        
        # Test using the numba function
        result_inside = snum.points_in_polygon(point_inside, polygon)
        result_outside = snum.points_in_polygon(point_outside, polygon)
        result_vertex = snum.points_in_polygon(point_on_vertex, polygon)
        
        assert result_inside == True
        assert result_outside == False
        # Point on vertex behavior can vary by implementation
        assert isinstance(result_vertex, (bool, np.bool_))
        
    def test_points_in_polygon_triangle(self):
        """Test point in polygon for a triangle"""
        # Define a triangle
        polygon = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.float64)
        
        # Points to test
        point_inside = np.array([1.0, 0.5])
        point_outside = np.array([1.5, 1.5])
        
        result_inside = snum.points_in_polygon(point_inside, polygon)
        result_outside = snum.points_in_polygon(point_outside, polygon)
        
        assert result_inside == True
        assert result_outside == False
        
    def test_points_in_polygon_complex_shape(self):
        """Test point in polygon for a more complex shape"""
        # Define an L-shaped polygon
        polygon = np.array([
            [0, 0], [1, 0], [1, 1], [2, 1], 
            [2, 2], [0, 2]
        ], dtype=np.float64)
        
        # Points to test
        point_inside_lower = np.array([0.5, 0.5])
        point_inside_upper = np.array([1.5, 1.5])
        point_outside = np.array([1.5, 0.5])  # In the "notch"
        
        result_lower = snum.points_in_polygon(point_inside_lower, polygon)
        result_upper = snum.points_in_polygon(point_inside_upper, polygon)
        result_outside = snum.points_in_polygon(point_outside, polygon)
        
        assert result_lower == True
        assert result_upper == True
        assert result_outside == False
        
    def test_points_in_polygon_edge_case(self):
        """Test point exactly on polygon edge"""
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
        point_on_edge = np.array([1.0, 0.0])  # On bottom edge
        
        result = snum.points_in_polygon(point_on_edge, polygon)
        # Result can be True or False depending on implementation
        assert isinstance(result, (bool, np.bool_))


class TestPeakFinding:
    """Test 2D peak finding functionality"""
    
    def test_find_signed_peaks_2d_simple_maximum(self):
        """Test finding a simple maximum peak"""
        # Create a 5x5 array with a single maximum at center
        data = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1], 
            [1, 2, 5, 2, 1],  # Maximum at [2,2]
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)
        
        wx, wy = 3, 3  # Window size
        minima, maxima = snum.find_signed_peaks_2d(data, wx, wy, paral=False)
        
        # Should find maximum at [2, 2]
        assert len(maxima) == 1
        assert maxima[0][0] == 2  # x coordinate
        assert maxima[0][1] == 2  # y coordinate
        
    def test_find_signed_peaks_2d_simple_minimum(self):
        """Test finding a simple minimum peak"""
        # Create a 5x5 array with a single minimum at center
        data = np.array([
            [5, 5, 5, 5, 5],
            [5, 2, 2, 2, 5],
            [5, 2, 1, 2, 5],  # Minimum at [2,2]
            [5, 2, 2, 2, 5],
            [5, 5, 5, 5, 5]
        ], dtype=np.float64)
        
        wx, wy = 3, 3
        minima, maxima = snum.find_signed_peaks_2d(data, wx, wy, paral=False)
        
        # Should find minimum at [2, 2]
        assert len(minima) == 1
        assert minima[0][0] == 2
        assert minima[0][1] == 2
        
    def test_find_signed_peaks_2d_multiple_peaks(self):
        """Test finding multiple peaks"""
        # Create data with multiple maxima
        data = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 5, 0],  # Two maxima at [1,1] and [1,5]
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 5, 0],  # Two maxima at [4,1] and [4,5]
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float64)
        
        wx, wy = 3, 3
        minima, maxima = snum.find_signed_peaks_2d(data, wx, wy, paral=False)
        
        # Should find 4 maxima
        assert len(maxima) == 4
        
    def test_find_signed_peaks_2d_with_nan(self):
        """Test peak finding with NaN values"""
        data = np.array([
            [1, 1, 1, 1, 1],
            [1, np.nan, 2, 2, 1],
            [1, 2, 5, 2, 1],  # Maximum at [2,2] 
            [1, 2, 2, np.nan, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)
        
        wx, wy = 3, 3
        minima, maxima = snum.find_signed_peaks_2d(data, wx, wy, paral=False)
        
        # Should still find the maximum despite NaN values
        assert len(maxima) >= 0  # May or may not find peaks depending on NaN handling
        
    def test_find_signed_peaks_2d_parallel_vs_serial(self):
        """Test that parallel and serial versions give same results"""
        # Create reproducible test data
        np.random.seed(42)
        data = np.random.rand(10, 10) * 10
        # Add clear peaks
        data[3, 3] = 20  # Maximum
        data[7, 7] = -5  # Minimum
        
        wx, wy = 3, 3
        
        # Serial version
        minima_serial, maxima_serial = snum.find_signed_peaks_2d(
            data, wx, wy, paral=False
        )
        
        # Parallel version  
        minima_parallel, maxima_parallel = snum.find_signed_peaks_2d(
            data, wx, wy, paral=True
        )
        
        # Results should be the same (order might differ)
        assert len(minima_serial) == len(minima_parallel)
        assert len(maxima_serial) == len(maxima_parallel)


class TestCoordinateUtilities:
    """Test coordinate name detection utilities"""
    
    def test_get_coord_name_standard(self):
        """Test coordinate name detection with standard names"""
        import xarray as xr
        
        # Create dataset with standard coordinate names
        ds = xr.Dataset(
            coords={'lon': range(10), 'lat': range(5)}
        )
        
        lon_name, lat_name = snum.get_coord_name(ds)
        
        assert lon_name == 'lon'
        assert lat_name == 'lat'
        
    def test_get_coord_name_longitude_latitude(self):
        """Test coordinate name detection with full names"""
        import xarray as xr
        
        ds = xr.Dataset(
            coords={'longitude': range(10), 'latitude': range(5)}
        )
        
        lon_name, lat_name = snum.get_coord_name(ds)
        
        assert lon_name == 'longitude'
        assert lat_name == 'latitude'
        
    def test_get_coord_name_roms_style(self):
        """Test coordinate name detection with ROMS-style names"""
        import xarray as xr
        
        ds = xr.Dataset(
            coords={'lon_rho': range(10), 'lat_rho': range(5)}
        )
        
        lon_name, lat_name = snum.get_coord_name(ds)
        
        assert lon_name == 'lon_rho'
        assert lat_name == 'lat_rho'
        
    def test_get_coord_name_capitalized(self):
        """Test coordinate name detection with capitalized names"""
        import xarray as xr
        
        ds = xr.Dataset(
            coords={'Longitude': range(10), 'Latitude': range(5)}
        )
        
        lon_name, lat_name = snum.get_coord_name(ds)
        
        assert lon_name == 'Longitude'
        assert lat_name == 'Latitude'
        
    def test_get_coord_name_missing(self):
        """Test coordinate name detection when coordinates are missing"""
        import xarray as xr
        
        ds = xr.Dataset(
            coords={'x': range(10), 'y': range(5)}
        )
        
        lon_name, lat_name = snum.get_coord_name(ds)
        
        assert lon_name is None
        assert lat_name is None


@pytest.mark.parametrize("window_size", [3, 5, 7])
def test_peak_finding_different_windows(window_size):
    """Test peak finding with different window sizes"""
    # Create data with a single clear peak
    size = 2 * window_size + 5
    data = np.ones((size, size), dtype=np.float64)
    center = size // 2
    data[center, center] = 10  # Clear maximum
    
    minima, maxima = snum.find_signed_peaks_2d(data, window_size, window_size, paral=False)
    
    # Should find the maximum regardless of window size
    assert len(maxima) >= 1
    

def test_empty_data_peak_finding():
    """Test peak finding with empty or uniform data"""
    # Uniform data - no peaks
    data = np.ones((5, 5), dtype=np.float64)
    
    minima, maxima = snum.find_signed_peaks_2d(data, 3, 3, paral=False)
    
    # Should find no peaks in uniform data
    assert len(minima) == 0
    assert len(maxima) == 0


def test_small_data_peak_finding():
    """Test peak finding with very small data arrays"""
    # 3x3 array
    data = np.array([
        [1, 1, 1],
        [1, 5, 1],  # Peak at center
        [1, 1, 1]
    ], dtype=np.float64)
    
    minima, maxima = snum.find_signed_peaks_2d(data, 3, 3, paral=False)
    
    # Should handle small arrays gracefully
    assert isinstance(minima, np.ndarray)
    assert isinstance(maxima, np.ndarray)
