"""Tests for shoot.acoustic module"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock
import shoot.acoustic as sacoustic


@pytest.fixture
def typical_sound_speed_profile():
    """Fixture providing a typical ocean sound speed profile"""
    depths = np.array([0, 10, 20, 50, 75, 100, 200, 500, 1000, 2000], dtype=float)
    # Typical profile: surface channel, deep sound channel minimum, then increase
    sound_speeds = np.array([1510, 1515, 1518, 1510, 1505, 1480, 1475, 1490, 1510, 1530], dtype=float)
    return sound_speeds, depths


@pytest.fixture
def sample_cs_dataset():
    """Fixture providing a 3D sound speed dataset"""
    lons = np.linspace(-5, 5, 10)
    lats = np.linspace(40, 45, 8)
    depths = np.array([0, 10, 25, 50, 100, 200, 500, 1000], dtype=float)

    # Create realistic sound speed data
    cs_data = np.zeros((len(depths), len(lats), len(lons)))
    for k, depth in enumerate(depths):
        # Surface maximum, decrease to minimum around 100m, then increase
        if depth < 50:
            cs_data[k, :, :] = 1520 - depth * 0.2
        elif depth < 200:
            cs_data[k, :, :] = 1510 - (depth - 50) * 0.15
        else:
            cs_data[k, :, :] = 1487.5 + (depth - 200) * 0.03

    ds = xr.Dataset(
        {'cs': (['s_rho', 'eta_rho', 'xi_rho'], cs_data)},
        coords={
            'lon_rho': (['eta_rho', 'xi_rho'], np.broadcast_arrays(*np.meshgrid(lons, lats))[0]),
            'lat_rho': (['eta_rho', 'xi_rho'], np.broadcast_arrays(*np.meshgrid(lons, lats))[1]),
            'depth': (
                ['s_rho', 'eta_rho', 'xi_rho'],
                np.broadcast_arrays(depths[:, None, None], np.ones((len(lats), len(lons))))[0],
            ),
        },
    )

    return ds


class TestBasicAcousticFunctions:
    """Test basic acoustic analysis functions"""

    def test_ilmax_detection(self):
        """Test local maxima detection"""
        # Profile with clear maxima at indices 1 and 3
        profile = np.array([1, 3, 2, 5, 1])
        maxima = sacoustic._ilmax(profile)

        expected_maxima = np.array([1, 3])
        np.testing.assert_array_equal(maxima, expected_maxima)

    def test_ilmax_single_maximum(self):
        """Test single maximum detection"""
        profile = np.array([1, 2, 5, 3, 1])
        maxima = sacoustic._ilmax(profile)

        expected_maxima = np.array([2])
        np.testing.assert_array_equal(maxima, expected_maxima)

    def test_ilmax_no_maxima(self):
        """Test when no local maxima exist"""
        profile = np.array([5, 4, 3, 2, 1])  # Monotonically decreasing
        maxima = sacoustic._ilmax(profile)

        assert len(maxima) == 0

    def test_ilmin_detection(self):
        """Test local minima detection"""
        # Profile with clear minima at indices 1 and 3
        profile = np.array([5, 2, 4, 1, 3])
        minima = sacoustic._ilmin(profile)

        expected_minima = np.array([1, 3])
        np.testing.assert_array_equal(minima, expected_minima)

    def test_ilmin_single_minimum(self):
        """Test single minimum detection"""
        profile = np.array([5, 4, 1, 3, 4])
        minima = sacoustic._ilmin(profile)

        expected_minima = np.array([2])
        np.testing.assert_array_equal(minima, expected_minima)

    def test_ilmin_no_minima(self):
        """Test when no local minima exist"""
        profile = np.array([1, 2, 3, 4, 5])  # Monotonically increasing
        minima = sacoustic._ilmin(profile)

        # Should return empty array or NaN, depending on implementation
        assert len(minima) == 0 or np.isnan(minima).all()


class TestAcousticParameters:
    """Test acoustic parameter calculations"""

    def test_ecs_calculation(self, typical_sound_speed_profile):
        """Test ECS (surface channel thickness) calculation"""
        sound_speeds, depths = typical_sound_speed_profile

        result = sacoustic._ecs(sound_speeds, depths)

        # Should return a depth value or 0
        assert result >= 0
        # For typical profile, should be positive (surface channel exists)
        assert result > 0

    def test_ecs_no_surface_channel(self):
        """Test ECS when no surface channel exists"""
        # Monotonically decreasing profile (no surface maximum)
        sound_speeds = np.array([1520, 1510, 1500, 1490, 1480])
        depths = np.array([0, 50, 100, 200, 500])

        result = sacoustic._ecs(sound_speeds, depths)

        # Should return 0 when no surface channel
        assert result == 0 or np.isnan(result)

    def test_mcp_calculation(self, typical_sound_speed_profile):
        """Test MCP (deep sound channel minimum) calculation"""
        sound_speeds, depths = typical_sound_speed_profile

        result = sacoustic._mcp(sound_speeds, depths)

        # Should return a depth value
        # assert isinstance(result, (float, np.floating))
        assert result >= 0
        # For typical profile with deep minimum, should be around 200m
        assert 100 < result < 300

    def test_iminc_calculation(self, typical_sound_speed_profile):
        """Test IMINC (incomplete channel) calculation"""
        sound_speeds, depths = typical_sound_speed_profile

        result = sacoustic._iminc(sound_speeds, depths)

        # Should return a depth value or NaN
        assert isinstance(result, (float, np.floating)) or np.isnan(result)


class TestProfileAcous:
    """Test ProfileAcous class"""

    def test_profile_acous_initialization(self, typical_sound_speed_profile):
        """Test ProfileAcous class initialization"""
        sound_speeds, depths = typical_sound_speed_profile

        # Convert to xarray
        profile = xr.DataArray(sound_speeds, dims=['depth'])
        depth = xr.DataArray(depths, dims=['depth'])

        prof_acous = sacoustic.ProfileAcous(profile, depth)

        assert prof_acous.profile is not None
        assert prof_acous.depth is not None
        assert prof_acous.depth_dim == 'depth'

    def test_profile_acous_properties(self, typical_sound_speed_profile):
        """Test ProfileAcous cached properties"""
        sound_speeds, depths = typical_sound_speed_profile

        profile = xr.DataArray(sound_speeds, dims=['depth'])
        depth = xr.DataArray(depths, dims=['depth'])

        prof_acous = sacoustic.ProfileAcous(profile, depth)

        # Test that properties can be accessed without errors
        ilmax = prof_acous.ilmax
        assert isinstance(ilmax, np.ndarray)

        ecs = prof_acous.ecs
        assert isinstance(ecs, (float, np.floating)) or np.isnan(ecs)

        mcp = prof_acous.mcp
        assert isinstance(mcp, (float, np.floating)) or np.isnan(mcp)


class TestAcousEddyClass:
    """Test AcousEddy class functionality"""

    def setup_method(self):
        """Set up mock anomaly for testing"""
        self.mock_anomaly = MagicMock()

        # Mock depth vector
        self.mock_anomaly.depth_vector = xr.DataArray(np.linspace(0, 1000, 50), dims=['depth'])

        # Mock mean profiles with realistic sound speed values
        depths = np.linspace(0, 1000, 50)
        # Typical surface channel profile
        cs_inside = 1520 - depths * 0.3 + (depths > 100) * (depths - 100) * 0.1
        cs_outside = 1518 - depths * 0.28 + (depths > 120) * (depths - 120) * 0.12

        self.mock_anomaly.mean_profil_inside = xr.DataArray(cs_inside, dims=['depth'])
        self.mock_anomaly.mean_profil_outside = xr.DataArray(cs_outside, dims=['depth'])

        # Mock individual profiles
        n_profiles = 5
        inside_profiles = np.tile(cs_inside, (n_profiles, 1))
        outside_profiles = np.tile(cs_outside, (n_profiles, 1))

        # Add some noise
        inside_profiles += np.random.normal(0, 1, inside_profiles.shape)
        outside_profiles += np.random.normal(0, 1, outside_profiles.shape)

        self.mock_anomaly.profils_inside = xr.DataArray(inside_profiles, dims=['nb_profil', 'depth'])
        self.mock_anomaly.profils_outside = xr.DataArray(outside_profiles, dims=['nb_profil', 'depth'])

        # Add coordinate arrays for lon/lat
        lons = np.random.uniform(-5, 5, n_profiles)
        lats = np.random.uniform(40, 45, n_profiles)

        self.mock_anomaly.profils_inside = self.mock_anomaly.profils_inside.assign_coords(
            lon_rho=(['nb_profil'], lons), lat_rho=(['nb_profil'], lats)
        )
        self.mock_anomaly.profils_inside.lon_rho.attrs['standard_name'] = 'longitude'
        self.mock_anomaly.profils_inside.lat_rho.attrs['standard_name'] = 'latitude'

        self.mock_anomaly.profils_outside = self.mock_anomaly.profils_outside.assign_coords(
            lon_rho=(['nb_profil'], lons + 1),  # Slightly different positions
            lat_rho=(['nb_profil'], lats + 1),
        )
        self.mock_anomaly.profils_outside.lon_rho.attrs['standard_name'] = 'longitude'
        self.mock_anomaly.profils_outside.lat_rho.attrs['standard_name'] = 'latitude'

    def test_acous_eddy_initialization(self):
        """Test AcousEddy class initialization"""
        acous_eddy = sacoustic.AcousEddy(self.mock_anomaly)

        assert acous_eddy.anomaly is not None

    def test_acous_eddy_acoustic_properties(self):
        """Test acoustic properties calculation"""
        acous_eddy = sacoustic.AcousEddy(self.mock_anomaly)

        # Test individual profile properties
        ecs_inside = acous_eddy.ecs_inside
        assert isinstance(ecs_inside, (float, np.floating)) or np.isnan(ecs_inside)

        mcp_inside = acous_eddy.mcp_inside
        assert isinstance(mcp_inside, (float, np.floating)) or np.isnan(mcp_inside)

        # Test that outside properties also work
        ecs_outside = acous_eddy.ecs_outside
        assert isinstance(ecs_outside, (float, np.floating)) or np.isnan(ecs_outside)

    def test_acous_eddy_distance_calculation(self):
        """Test acoustic distance calculation between profiles"""
        # Test the static distance method
        e1, e2 = 100.0, 120.0
        distance = sacoustic.AcousEddy._distance(e1, e2)

        assert isinstance(distance, (float, np.floating))
        assert distance >= 0

        # Test with NaN values
        distance_nan = sacoustic.AcousEddy._distance(np.nan, np.nan)
        assert distance_nan == 0

        distance_one_nan = sacoustic.AcousEddy._distance(100.0, np.nan)
        assert distance_one_nan == 1

    def test_acous_eddy_acoustic_impact(self):
        """Test acoustic impact calculation"""
        acous_eddy = sacoustic.AcousEddy(self.mock_anomaly)

        impact = acous_eddy.acoustic_impact
        assert isinstance(impact, (int, float, np.floating, np.integer))
        assert impact >= 0

    def test_acous_eddy_profile_arrays(self):
        """Test that profile arrays are correctly constructed"""
        acous_eddy = sacoustic.AcousEddy(self.mock_anomaly)

        # Test inside profiles arrays
        ecs_insides = acous_eddy.ecs_insides
        assert isinstance(ecs_insides, xr.DataArray)
        assert 'nb_profil' in ecs_insides.dims
        assert 'lon_rho' in ecs_insides.coords
        assert 'lat_rho' in ecs_insides.coords

        # Test outside profiles arrays
        mcp_outsides = acous_eddy.mcp_outsides
        assert isinstance(mcp_outsides, xr.DataArray)
        assert 'nb_profil' in mcp_outsides.dims


@pytest.mark.parametrize("profile_type", ["surface_channel", "deep_channel", "monotonic"])
def test_different_profile_types(profile_type):
    """Test acoustic functions with different profile types"""
    depths = np.array([0, 50, 100, 200, 500, 1000])

    if profile_type == "surface_channel":
        # Strong surface channel
        sound_speeds = np.array([1525, 1520, 1515, 1480, 1490, 1510])
    elif profile_type == "deep_channel":
        # Deep sound channel
        sound_speeds = np.array([1515, 1510, 1505, 1475, 1485, 1500])
    else:  # monotonic
        # No clear channel structure
        sound_speeds = np.array([1520, 1515, 1510, 1505, 1500, 1495])

    ecs = sacoustic._ecs(sound_speeds, depths)
    mcp = sacoustic._mcp(sound_speeds, depths)

    # All should return valid numbers or NaN
    assert isinstance(ecs, (int, float, np.floating, np.integer)) or np.isnan(ecs)
    assert isinstance(mcp, (int, float, np.floating, np.integer)) or np.isnan(mcp)
