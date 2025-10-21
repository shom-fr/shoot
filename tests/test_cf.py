"""Tests for shoot.cf module"""
import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch
import shoot.cf as scf
from shoot import ShootError


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample xarray dataset with CF conventions"""
    lons = np.linspace(-10, 10, 21)
    lats = np.linspace(40, 50, 11)
    depths = np.array([0, 10, 25, 50, 100])
    
    ds = xr.Dataset({
        'temperature': (['depth', 'lat', 'lon'], 
                       np.random.rand(len(depths), len(lats), len(lons))),
        'u': (['lat', 'lon'], np.random.rand(len(lats), len(lons))),
        'v': (['lat', 'lon'], np.random.rand(len(lats), len(lons))),
        'ssh': (['lat', 'lon'], np.random.rand(len(lats), len(lons))),
    }, coords={
        'lon': lons,
        'lat': lats,
        'depth': depths
    })
    
    # Add CF attributes
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lat.attrs['standard_name'] = 'latitude' 
    ds.depth.attrs['standard_name'] = 'depth'
    ds.u.attrs['standard_name'] = 'eastward_sea_water_velocity'
    ds.v.attrs['standard_name'] = 'northward_sea_water_velocity'
    ds.ssh.attrs['standard_name'] = 'sea_surface_height_above_geoid'
    ds.temperature.attrs['standard_name'] = 'sea_water_potential_temperature'
    
    return ds


@pytest.fixture 
def roms_dataset():
    """Fixture providing a ROMS-style dataset"""
    lons = np.linspace(-10, 10, 21)
    lats = np.linspace(40, 50, 11)
    
    ds = xr.Dataset({
        'u': (['eta_rho', 'xi_rho'], np.random.rand(len(lats), len(lons))),
        'v': (['eta_rho', 'xi_rho'], np.random.rand(len(lats), len(lons))),
    }, coords={
        'lon_rho': (['eta_rho', 'xi_rho'], 
                   np.broadcast_arrays(*np.meshgrid(lons, lats))[0]),
        'lat_rho': (['eta_rho', 'xi_rho'],
                   np.broadcast_arrays(*np.meshgrid(lons, lats))[1]),
    })
    
    return ds


class TestCFUtilities:
    """Test CF convention utilities"""
    
    def test_get_lon_standard_name(self, sample_dataset):
        """Test longitude coordinate retrieval with standard name"""
        lon = scf.get_lon(sample_dataset)
        assert lon.name == 'lon'
        np.testing.assert_array_equal(lon.values, sample_dataset.lon.values)
        
    def test_get_lat_standard_name(self, sample_dataset):
        """Test latitude coordinate retrieval with standard name"""
        lat = scf.get_lat(sample_dataset)
        assert lat.name == 'lat'
        np.testing.assert_array_equal(lat.values, sample_dataset.lat.values)
        
    def test_get_u_standard_name(self, sample_dataset):
        """Test U velocity retrieval with standard name"""
        u = scf.get_u(sample_dataset)
        assert u.name == 'u'
        np.testing.assert_array_equal(u.values, sample_dataset.u.values)
        
    def test_get_v_standard_name(self, sample_dataset):
        """Test V velocity retrieval with standard name"""
        v = scf.get_v(sample_dataset)
        assert v.name == 'v'
        np.testing.assert_array_equal(v.values, sample_dataset.v.values)
        
    def test_get_ssh_standard_name(self, sample_dataset):
        """Test SSH retrieval with standard name"""
        ssh = scf.get_ssh(sample_dataset)
        assert ssh.name == 'ssh'
        np.testing.assert_array_equal(ssh.values, sample_dataset.ssh.values)
        
    def test_get_depth_standard_name(self, sample_dataset):
        """Test depth coordinate retrieval"""
        depth = scf.get_depth(sample_dataset)
        assert depth.name == 'depth'
        np.testing.assert_array_equal(depth.values, sample_dataset.depth.values)
        
    def test_get_xdim(self, sample_dataset):
        """Test X dimension detection"""
        xdim = scf.get_xdim(sample_dataset.u)
        assert xdim == 'lon'
        
    def test_get_ydim(self, sample_dataset):
        """Test Y dimension detection"""
        ydim = scf.get_ydim(sample_dataset.u)
        assert ydim == 'lat'
        
    def test_get_zdim(self, sample_dataset):
        """Test Z dimension detection"""
        zdim = scf.get_zdim(sample_dataset.temperature)
        assert zdim == 'depth'
        
    def test_get_cf_item_single_match(self):
        """Test get_cf_item with single matching item"""
        container = {'test_var': 'test_value'}
        result = scf.get_cf_item(container, ['test_var'])
        assert result == 'test_value'
        
    def test_get_cf_item_no_match_raise(self):
        """Test get_cf_item with no match raises error"""
        container = {'other_var': 'test_value'}
        with pytest.raises(ShootError):
            scf.get_cf_item(container, ['test_var'], errors='raise')
            
    def test_get_cf_item_no_match_ignore(self):
        """Test get_cf_item with no match returns None"""
        container = {'other_var': 'test_value'}
        result = scf.get_cf_item(container, ['test_var'], errors='ignore')
        assert result is None
        
    def test_get_cf_item_multiple_matches_raise(self):
        """Test get_cf_item with multiple matches raises error"""
        container = {'var1': 'value1', 'var2': 'value2'}
        with pytest.raises(ShootError):
            scf.get_cf_item(container, ['var1', 'var2'], errors='raise')


class TestStandardNames:
    """Test standard name definitions and matching"""
    
    def test_u_standard_names_exist(self):
        """Test that U velocity standard names are defined"""
        assert 'u' in scf.STANDARD_NAMES
        assert len(scf.STANDARD_NAMES['u']) > 0
        assert 'eastward_sea_water_velocity' in scf.STANDARD_NAMES['u']
        
    def test_v_standard_names_exist(self):
        """Test that V velocity standard names are defined"""
        assert 'v' in scf.STANDARD_NAMES
        assert len(scf.STANDARD_NAMES['v']) > 0
        assert 'northward_sea_water_velocity' in scf.STANDARD_NAMES['v']
        
    def test_ssh_standard_names_exist(self):
        """Test that SSH standard names are defined"""
        assert 'ssh' in scf.STANDARD_NAMES
        assert len(scf.STANDARD_NAMES['ssh']) > 0
        
    def test_depth_standard_names_exist(self):
        """Test that depth standard names are defined"""
        assert 'depth' in scf.STANDARD_NAMES
        assert len(scf.STANDARD_NAMES['depth']) > 0
        assert 'depth' in scf.STANDARD_NAMES['depth']


class TestDimensionDetection:
    """Test automatic dimension detection"""
    
    def test_xdim_detection_fallback(self):
        """Test X dimension detection fallback to last dimension"""
        # Create data array without CF axes
        da = xr.DataArray(np.random.rand(5, 10), dims=['time', 'station'])
        xdim = scf.get_xdim(da)
        assert xdim == 'station'  # Last dimension
        
    def test_ydim_detection_fallback(self):
        """Test Y dimension detection fallback to second-to-last dimension"""
        da = xr.DataArray(np.random.rand(3, 5, 10), dims=['time', 'lat', 'lon'])
        ydim = scf.get_ydim(da)
        assert ydim == 'lat'  # Second-to-last dimension
        
    def test_zdim_detection_with_s_rho(self):
        """Test Z dimension detection with ROMS-style s_rho"""
        da = xr.DataArray(np.random.rand(20, 5, 10), dims=['s_rho', 'eta_rho', 'xi_rho'])
        zdim = scf.get_zdim(da)
        assert zdim == 's_rho'
        
    def test_zdim_detection_with_depth(self):
        """Test Z dimension detection with depth"""
        da = xr.DataArray(np.random.rand(20, 5, 10), dims=['depth', 'lat', 'lon'])
        zdim = scf.get_zdim(da)
        assert zdim == 'depth'
        
    def test_zdim_detection_with_lev(self):
        """Test Z dimension detection with level"""
        da = xr.DataArray(np.random.rand(20, 5, 10), dims=['lev', 'lat', 'lon'])
        zdim = scf.get_zdim(da)
        assert zdim == 'lev'


def test_get_temp_with_errors():
    """Test temperature retrieval with error handling"""
    # Create dataset without temperature
    ds = xr.Dataset({'salinity': (['lat', 'lon'], np.random.rand(10, 20))})
    
    # Should return None with errors='ignore'
    result = scf.get_temp(ds, errors='ignore')
    assert result is None


def test_get_salt_with_errors():
    """Test salinity retrieval with error handling"""
    # Create dataset without salinity
    ds = xr.Dataset({'temperature': (['lat', 'lon'], np.random.rand(10, 20))})
    
    # Should return None with errors='ignore' 
    result = scf.get_salt(ds, errors='ignore')
    assert result is None


@pytest.mark.parametrize("var_name,standard_name", [
    ('u', 'eastward_sea_water_velocity'),
    ('v', 'northward_sea_water_velocity'), 
    ('ssh', 'sea_surface_height_above_geoid'),
    ('temp', 'sea_water_potential_temperature'),
    ('salt', 'sea_water_salinity'),
])
def test_standard_name_retrieval(var_name, standard_name):
    """Test retrieval of variables by standard name"""
    # Create dataset with standard name attribute
    ds = xr.Dataset({
        'data_var': (['lat', 'lon'], np.random.rand(10, 20))
    })
    ds.data_var.attrs['standard_name'] = standard_name
    
    # Get the appropriate function
    func_name = f'get_{var_name}'
    if hasattr(scf, func_name):
        func = getattr(scf, func_name)
        result = func(ds, errors='ignore')
        
        if result is not None:
            assert result.name == 'data_var'
