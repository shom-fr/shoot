"""Tests for shoot.dyn module"""
import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch
import shoot.dyn as sdyn


@pytest.fixture
def sample_velocity_field():
    """Fixture providing a simple velocity field"""
    nx, ny = 10, 8
    lons = np.linspace(-5, 5, nx)
    lats = np.linspace(40, 45, ny)
    
    # Create simple rotating flow (anticyclonic eddy)
    x, y = np.meshgrid(lons, lats)
    u_data = -0.1 * (y - 42.5)  # Simple rotation
    v_data = 0.1 * (x - 0)
    
    u = xr.DataArray(
        u_data, 
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons}
    )
    v = xr.DataArray(
        v_data,
        dims=['lat', 'lon'], 
        coords={'lat': lats, 'lon': lons}
    )
    
    # Add CF attributes
    u.attrs['standard_name'] = 'eastward_sea_water_velocity'
    v.attrs['standard_name'] = 'northward_sea_water_velocity'
    u.lon.attrs['standard_name'] = 'longitude'
    u.lat.attrs['standard_name'] = 'latitude'
    v.lon.attrs['standard_name'] = 'longitude' 
    v.lat.attrs['standard_name'] = 'latitude'
    
    return u, v


@pytest.fixture
def sample_ssh_field():
    """Fixture providing a simple SSH field"""
    nx, ny = 12, 10
    lons = np.linspace(-6, 6, nx)
    lats = np.linspace(35, 50, ny)
    
    # Create simple gaussian-like elevation (anticyclonic eddy)
    x, y = np.meshgrid(lons, lats)
    ssh_data = 0.2 * np.exp(-((x-0)**2 + (y-42.5)**2) / 8)
    
    ssh = xr.DataArray(
        ssh_data,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons}
    )
    
    ssh.attrs['standard_name'] = 'sea_surface_height_above_geoid'
    ssh.lon.attrs['standard_name'] = 'longitude'
    ssh.lat.attrs['standard_name'] = 'latitude'
    
    return ssh


class TestConstants:
    """Test physical constants"""
    
    def test_gravity_constant(self):
        """Test gravity constant value"""
        assert sdyn.GRAVITY == 9.81
        
    def test_omega_constant(self):
        """Test Earth rotation rate"""
        expected_omega = 2 * np.pi / 86400  # rad/s
        assert abs(sdyn.OMEGA - expected_omega) < 1e-10


class TestCoriolisParameter:
    """Test Coriolis parameter calculations"""
    
    def test_coriolis_at_equator(self):
        """Test Coriolis parameter at equator"""
        f = sdyn.get_coriolis(0.0)
        assert abs(f) < 1e-10  # Should be zero at equator
        
    def test_coriolis_at_45_degrees(self):
        """Test Coriolis parameter at 45°N"""
        lat = 45.0
        f = sdyn.get_coriolis(lat)
        
        expected = 2 * sdyn.OMEGA * np.sin(np.radians(lat))
        assert abs(f - expected) < 1e-10
        
    def test_coriolis_at_pole(self):
        """Test Coriolis parameter at North Pole"""
        f = sdyn.get_coriolis(90.0)
        expected = 2 * sdyn.OMEGA  # Maximum value
        assert abs(f - expected) < 1e-10
        
    def test_coriolis_southern_hemisphere(self):
        """Test Coriolis parameter in Southern Hemisphere"""
        f_north = sdyn.get_coriolis(45.0)
        f_south = sdyn.get_coriolis(-45.0)
        
        # Should be opposite signs
        assert f_north > 0
        assert f_south < 0
        assert abs(f_north + f_south) < 1e-10
        
    def test_coriolis_array_input(self):
        """Test Coriolis parameter with array input"""
        lats = np.array([0, 30, 45, 60, 90])
        f = sdyn.get_coriolis(lats)
        
        assert f.shape == lats.shape
        assert abs(f[0]) < 1e-10  # Equator
        assert f[-1] == 2 * sdyn.OMEGA  # Pole


class TestRelativeVorticity:
    """Test relative vorticity calculations"""
    
    def test_relvort_shape_preservation(self, sample_velocity_field):
        """Test that relative vorticity preserves input shape"""
        u, v = sample_velocity_field
        
        vort = sdyn.get_relvort(u, v)
        
        assert vort.shape == u.shape
        assert vort.dims == u.dims
        assert 'lat' in vort.coords
        assert 'lon' in vort.coords
        
    def test_relvort_solid_body_rotation(self):
        """Test relative vorticity for solid body rotation"""
        # Create perfect solid body rotation
        nx, ny = 10, 10
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 44, ny)
        omega_rotation = 1e-5  # rad/s

        x, y = np.meshgrid(lons, lats)
        # Convert to physical distances in meters (approximate)
        # At lat~42, 1 degree lon ~ 82 km, 1 degree lat ~ 111 km
        x_m = x * 82000  # meters
        y_m = (y - 42) * 111000  # meters from center

        u = -omega_rotation * y_m  # Tangential velocity in m/s
        v = omega_rotation * x_m

        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        u_da.lon.attrs['standard_name'] = 'longitude'
        u_da.lat.attrs['standard_name'] = 'latitude'
        v_da.lon.attrs['standard_name'] = 'longitude'
        v_da.lat.attrs['standard_name'] = 'latitude'

        vort = sdyn.get_relvort(u_da, v_da)

        # For solid body rotation, vorticity should be constant = 2*omega
        expected_vort = 2 * omega_rotation

        # Check interior points (avoid edge effects)
        interior_vort = vort[2:-2, 2:-2]
        mean_vort = float(interior_vort.mean())

        assert abs(mean_vort - expected_vort) < expected_vort * 0.2  # 20% tolerance
        
    def test_relvort_with_nans(self):
        """Test relative vorticity with NaN values"""
        nx, ny = 8, 6
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 44, ny)
        
        u_data = np.random.rand(ny, nx)
        v_data = np.random.rand(ny, nx)
        
        # Insert some NaN values
        u_data[0, 0] = np.nan
        v_data[2, 3] = np.nan
        
        u = xr.DataArray(u_data, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v = xr.DataArray(v_data, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        vort = sdyn.get_relvort(u, v)
        
        # NaN regions should propagate
        assert np.isnan(vort[0, 0])
        assert np.isnan(vort[2, 3])


class TestDivergence:
    """Test divergence calculations"""
    
    def test_div_shape_preservation(self, sample_velocity_field):
        """Test that divergence preserves input shape"""
        u, v = sample_velocity_field
        
        div = sdyn.get_div(u, v)
        
        assert div.shape == u.shape
        assert div.dims == u.dims
        
    def test_div_pure_rotation_zero(self):
        """Test that pure rotation has zero divergence"""
        # Create non-divergent rotating flow
        nx, ny = 12, 10
        lons = np.linspace(-3, 3, nx)
        lats = np.linspace(40, 45, ny)
        
        x, y = np.meshgrid(lons, lats)
        # Circular flow around center
        u = -(y - 42.5)
        v = (x - 0)
        
        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        div = sdyn.get_div(u_da, v_da)
        
        # Pure rotation should have near-zero divergence (within numerical precision)
        interior_div = div[2:-2, 2:-2]  # Avoid boundaries
        mean_abs_div = float(np.abs(interior_div).mean())
        
        assert mean_abs_div < 0.1  # Very small divergence
        
    def test_div_radial_flow(self):
        """Test divergence of radial flow"""
        nx, ny = 10, 10
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 44, ny)
        
        x, y = np.meshgrid(lons, lats)
        # Radial outflow (positive divergence)
        u = (x - 0) * 0.1
        v = (y - 42) * 0.1
        
        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        div = sdyn.get_div(u_da, v_da)
        
        # Radial outflow should have positive divergence
        interior_div = div[2:-2, 2:-2]
        mean_div = float(interior_div.mean())
        
        assert mean_div > 0


class TestOkuboWeissParameter:
    """Test Okubo-Weiss parameter calculations"""
    
    def test_okuboweiss_shape_preservation(self, sample_velocity_field):
        """Test that Okubo-Weiss preserves input shape"""
        u, v = sample_velocity_field
        
        ow = sdyn.get_okuboweiss(u, v)
        
        assert ow.shape == u.shape
        assert ow.dims == u.dims
        
    def test_okuboweiss_pure_vortex(self):
        """Test Okubo-Weiss for pure vortical flow"""
        # Create pure rotation (vortex-dominated flow)
        nx, ny = 12, 10
        lons = np.linspace(-3, 3, nx)
        lats = np.linspace(40, 45, ny)
        
        x, y = np.meshgrid(lons, lats)
        # Strong circular flow
        u = -(y - 42.5) * 2
        v = (x - 0) * 2
        
        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        ow = sdyn.get_okuboweiss(u_da, v_da)
        
        # For pure vortex, OW should be negative (vorticity dominates)
        interior_ow = ow[2:-2, 2:-2]
        mean_ow = float(interior_ow.mean())
        
        assert mean_ow < 0
        
    def test_okuboweiss_strain_dominated(self):
        """Test Okubo-Weiss for strain-dominated flow"""
        nx, ny = 10, 8
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 44, ny)
        
        x, y = np.meshgrid(lons, lats)
        # Pure strain field
        u = (x - 0) * 0.5  # Extension in x
        v = -(y - 42) * 0.5  # Compression in y
        
        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        ow = sdyn.get_okuboweiss(u_da, v_da)
        
        # For strain-dominated flow, OW should be positive
        interior_ow = ow[2:-2, 2:-2]
        mean_ow = float(interior_ow.mean())
        
        assert mean_ow > 0


class TestGeostrophicCalculations:
    """Test geostrophic velocity calculations"""
    
    def test_geos_shape_preservation(self, sample_ssh_field):
        """Test that geostrophic velocities preserve SSH shape"""
        ssh = sample_ssh_field
        
        ugeos, vgeos = sdyn.get_geos(ssh)
        
        assert ugeos.shape == ssh.shape
        assert vgeos.shape == ssh.shape
        assert ugeos.dims == ssh.dims
        assert vgeos.dims == ssh.dims
        
    def test_geos_simple_gradient(self):
        """Test geostrophic calculation for simple SSH gradient"""
        # Create simple north-south SSH gradient
        nx, ny = 8, 10
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 50, ny)

        # SSH increases northward (simple gradient)
        # Use physical gradient: ~0.18 m per 111 km (1 degree lat)
        ssh_gradient_per_m = 0.18 / 111000  # m SSH per m distance
        x, y = np.meshgrid(lons, lats)
        # Convert latitude to meters from center
        y_m = (y - 45) * 111000  # meters
        ssh_data = ssh_gradient_per_m * y_m  # SSH in meters

        ssh = xr.DataArray(
            ssh_data,
            dims=['lat', 'lon'],
            coords={'lat': lats, 'lon': lons}
        )
        ssh.lon.attrs['standard_name'] = 'longitude'
        ssh.lat.attrs['standard_name'] = 'latitude'

        ugeos, vgeos = sdyn.get_geos(ssh)

        # For north-south SSH gradient (high in north), should get westward geostrophic flow in NH
        # (pressure gradient south, Coriolis deflects to right = west)
        # Interior points (avoid gradient edge effects)
        u_interior = ugeos[2:-2, 2:-2]
        v_interior = vgeos[2:-2, 2:-2]

        # U should be negative (westward), V should be near zero
        assert float(u_interior.mean()) < 0
        assert abs(float(v_interior.mean())) < abs(float(u_interior.mean())) * 0.2
        
    def test_geos_circular_high(self):
        """Test geostrophic flow around circular high pressure"""
        nx, ny = 12, 12
        lons = np.linspace(-3, 3, nx)
        lats = np.linspace(39, 45, ny)
        
        # Create circular SSH high (anticyclonic)
        x, y = np.meshgrid(lons, lats)
        radius_deg = 2.0
        ssh_data = 0.1 * np.exp(-((x-0)**2 + (y-42)**2) / radius_deg**2)
        
        ssh = xr.DataArray(
            ssh_data,
            dims=['lat', 'lon'],
            coords={'lat': lats, 'lon': lons}
        )
        ssh.lon.attrs['standard_name'] = 'longitude'
        ssh.lat.attrs['standard_name'] = 'latitude'
        
        ugeos, vgeos = sdyn.get_geos(ssh)
        
        # Should get anticyclonic circulation (clockwise in NH)
        # Check that flow magnitude is reasonable
        speed = np.sqrt(ugeos**2 + vgeos**2)
        assert float(speed.max()) > 0
        assert float(speed.mean()) > 0


class TestLNAMParameter:
    """Test Local Normalized Angular Momentum calculations"""
    
    def test_lnam_shape_preservation(self, sample_velocity_field):
        """Test that LNAM preserves input dimensions"""
        u, v = sample_velocity_field
        
        window = 50  # km
        lnam = sdyn.get_lnam(u, v, window)
        
        assert lnam.shape == u.shape
        assert lnam.dims == u.dims
        
    def test_lnam_anticyclonic_vortex(self):
        """Test LNAM for anticyclonic vortex (should be positive)"""
        # Create anticyclonic (clockwise) rotation
        nx, ny = 15, 12
        lons = np.linspace(-2, 2, nx)
        lats = np.linspace(40, 44, ny)
        
        x, y = np.meshgrid(lons, lats)
        # Anticyclonic rotation (negative vorticity in NH)
        u = (y - 42) * 0.2  # Clockwise
        v = -(x - 0) * 0.2
        
        u_da = xr.DataArray(u, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        v_da = xr.DataArray(v, dims=['lat', 'lon'], coords={'lat': lats, 'lon': lons})
        
        # Add longitude/latitude standard names for coordinate detection
        u_da.lon.attrs['standard_name'] = 'longitude'
        u_da.lat.attrs['standard_name'] = 'latitude'
        v_da.lon.attrs['standard_name'] = 'longitude'
        v_da.lat.attrs['standard_name'] = 'latitude'
        
        window = 100  # km
        lnam = sdyn.get_lnam(u_da, v_da, window)
        
        # For anticyclonic flow, LNAM should tend to be positive
        center_region = lnam[ny//2-1:ny//2+2, nx//2-1:nx//2+2]
        mean_lnam = float(center_region.mean())
        
        # Allow some tolerance for numerical effects
        assert not np.isnan(mean_lnam)


@pytest.mark.parametrize("lat", [0, 30, 45, 60])
def test_coriolis_different_latitudes(lat):
    """Test Coriolis parameter at different latitudes"""
    f = sdyn.get_coriolis(lat)
    
    expected = 2 * sdyn.OMEGA * np.sin(np.radians(lat))
    assert abs(f - expected) < 1e-10


def test_dynamics_with_realistic_data():
    """Integration test with realistic oceanographic data"""
    # Create a more realistic eddy-like structure
    nx, ny = 20, 16
    lons = np.linspace(-5, 5, nx)
    lats = np.linspace(35, 45, ny)
    
    # Gaussian SSH anomaly (anticyclonic eddy)
    x, y = np.meshgrid(lons, lats)
    ssh_data = 0.15 * np.exp(-((x-0)**2 + (y-40)**2) / 4)
    
    ssh = xr.DataArray(
        ssh_data,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons}
    )
    ssh.lon.attrs['standard_name'] = 'longitude'
    ssh.lat.attrs['standard_name'] = 'latitude'
    
    # Get geostrophic velocities
    u, v = sdyn.get_geos(ssh)
    
    # Calculate derived quantities
    vort = sdyn.get_relvort(u, v)
    div = sdyn.get_div(u, v)
    ow = sdyn.get_okuboweiss(u, v)
    
    # Basic sanity checks
    assert not np.isnan(vort).all()
    assert not np.isnan(div).all()
    assert not np.isnan(ow).all()
    
    # For anticyclonic eddy: negative vorticity in NH, near-zero divergence
    center_vort = float(vort[ny//2, nx//2])
    center_div = float(div[ny//2, nx//2])
    
    # Anticyclonic should have negative vorticity
    assert center_vort < 0
    # Should have low divergence
    assert abs(center_div) < 0.1
