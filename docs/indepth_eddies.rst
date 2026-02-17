.. _indepth_eddies:

Eddies
======

This section covers the detection, tracking, and analysis of mesoscale ocean eddies using shoot. Eddies are rotating coherent structures that play a crucial role in ocean circulation and transport.

.. toctree::
   :maxdepth: 2

   indepth_eddies_detection
   indepth_eddies_tracking

What are Ocean Eddies?
-----------------------

Ocean eddies are circular currents of water that rotate around a center:

**Cyclonic eddies**:
   - Rotate counterclockwise in Northern Hemisphere
   - Clockwise in Southern Hemisphere
   - Associated with upwelling and cooler water
   - Negative sea surface height anomaly
   - Positive LNAM signature

**Anticyclonic eddies**:
   - Rotate clockwise in Northern Hemisphere
   - Counterclockwise in Southern Hemisphere
   - Associated with downwelling and warmer water
   - Positive sea surface height anomaly
   - Negative LNAM signature

Characteristics
~~~~~~~~~~~~~~~

**Spatial scales**:
   - Diameter: 50-500 km (mesoscale)
   - Typical: 100-200 km at mid-latitudes
   - Scale with Rossby radius of deformation

**Temporal scales**:
   - Lifetime: Weeks to months
   - Can persist for over a year
   - Rotation period: Days to weeks

**Physical properties**:
   - Swirl velocity: 0.1-1 m/s
   - Rossby number: 0.1-0.5
   - Vertical extent: 100-1000+ m

Importance
~~~~~~~~~~

Eddies are important because they:

- Transport heat, salt, and nutrients
- Influence biological productivity
- Impact fisheries and marine ecosystems
- Affect acoustic propagation
- Contribute to ocean mixing
- Play a role in climate

Detection in shoot
------------------

shoot detects eddies using the **LNAM method** (Local Normalized Angular Momentum):

1. **Compute LNAM field** from velocity components (u, v)
2. **Apply Okubo-Weiss filter** to identify vortex-dominated regions
3. **Find local extrema** as potential eddy centers
4. **Extract SSH contours** around each center
5. **Identify maximum velocity contour** as eddy boundary
6. **Compute properties** (radius, intensity, shape)
7. **Apply quality filters** (size, ellipticity, validity)

The method is detailed in :ref:`indepth_eddies_detection`.

Key Parameters
~~~~~~~~~~~~~~

**window_center** (km):
   - Scale for LNAM computation
   - Typically 1-2 × Rossby radius
   - ~50 km for mid-latitude altimetry

**window_fit** (km):
   - Search domain for SSH contours
   - Typically 5-10 × Rossby radius
   - ~120 km for mid-latitude altimetry

**min_radius** (km):
   - Minimum eddy size to retain
   - Often set to Rossby radius
   - ~15 km for mid-latitudes

**ellipse_error** (fraction):
   - Maximum deviation from perfect ellipse
   - 0.05 (5%) for clean satellite data
   - 0.10 (10%) for noisy model output

Tracking in shoot
-----------------

shoot tracks eddies through time using the **Chelton et al. (2011)** algorithm:

1. **Detect eddies** at each time step
2. **Compute cost matrix** based on:

   - Spatial distance between centers
   - Rossby number similarity
   - Radius similarity
   - Type matching (cyclone/anticyclone)

3. **Find optimal matching** using Hungarian algorithm
4. **Update existing tracks** or create new ones
5. **Handle births and deaths** of eddies

The tracking system is detailed in :ref:`indepth_eddies_tracking`.

Key Parameters
~~~~~~~~~~~~~~

**max_distance** (km):
   - Maximum search radius for matching
   - Based on typical advection distance
   - ~50 km for daily data

**Dt** (days):
   - Actual time interval between detections
   - Affects cost function weighting

**Tc** (days):
   - Characteristic time scale
   - Typically 10-20 days for mesoscale

Quick Example
-------------

Basic Detection
~~~~~~~~~~~~~~~

.. code-block:: python

    from shoot.eddies.eddies2d import Eddies2D
    import xarray as xr

    # Load data
    ds = xr.open_dataset("velocity_field.nc")

    # Detect eddies
    eddies = Eddies2D.detect_eddies(
        ds.u,              # Zonal velocity
        ds.v,              # Meridional velocity
        window_center=50,  # Detection window (km)
        window_fit=120,    # Fitting window (km)
        ssh=ds.ssh,        # Optional SSH field
        min_radius=15      # Minimum radius (km)
    )

    # Results
    print(f"Detected {len(eddies.eddies)} eddies")
    for eddy in eddies.eddies:
        print(f"  {eddy.eddy_type} at ({eddy.lon:.1f}, {eddy.lat:.1f})")
        print(f"    Radius: {eddy.radius:.1f} km")

Basic Tracking
~~~~~~~~~~~~~~

.. code-block:: python

    # Detect at multiple times
    eddies_list = []
    for t in range(len(ds.time)):
        eddies_t = Eddies2D.detect_eddies(
            ds.u.isel(time=t),
            ds.v.isel(time=t),
            50, 120, min_radius=15
        )
        eddies_list.append(eddies_t)

    # Track (see tracking guide for full implementation)
    # tracks = track_eddies(eddies_list, max_distance=50)

Advanced Topics
---------------

3D Detection
~~~~~~~~~~~~

Detect eddies across multiple depth levels and associate vertically:

.. code-block:: python

    from shoot.eddies.eddies3d import EddiesByDepth

    # 3D velocity field
    eddies_3d = EddiesByDepth.detect_eddies_3d(
        ds.u,              # 3D zonal velocity
        ds.v,              # 3D meridional velocity
        window_center=50,
        window_fit=120,
        max_distance=10    # Max vertical shift (km)
    )

    # Access by depth
    for depth in eddies_3d.eddies3d:
        print(f"Depth {depth}: {len(eddies_3d.eddies3d[depth].eddies)} eddies")

Profile Association
~~~~~~~~~~~~~~~~~~~

Link in-situ observations with detected eddies:

.. code-block:: python

    from shoot.profiles.profiles import Profiles

    # Load Argo profiles
    profiles = Profiles.from_ds(ds, root_path="./argo_data")

    # Associate with eddies
    profiles.associate(eddies, nlag=2)  # ±2 days

    # Profiles inside vs outside
    inside = profiles.ds.where(profiles.ds.eddy_pos >= 0, drop=True)
    outside = profiles.ds.where(profiles.ds.eddy_pos == -1, drop=True)

See :ref:`indepth_profiles` for details.

Acoustic Analysis
~~~~~~~~~~~~~~~~~

Assess how eddies affect sound propagation:

.. code-block:: python

    from shoot.hydrology import Anomaly
    from shoot.acoustic import AcousEddy

    # Create anomaly from eddy and profiles
    anomaly = Anomaly(eddy, profiles.ds, nlag=2)

    if anomaly.is_valid():
        # Analyze acoustic impact
        acous = AcousEddy(anomaly)
        impact = acous.acoustic_impact

        print(f"Acoustic impact: {impact:.3f}")
        print(f"Surface duct: {acous.ecs_inside:.1f} m inside")
        print(f"              {acous.ecs_outside:.1f} m outside")

Data Sources
------------

Satellite Altimetry
~~~~~~~~~~~~~~~~~~~

Best for surface eddy detection:

- **Products**: DUACS, AVISO, CMEMS
- **Resolution**: 1/4° (~25 km)
- **Temporal**: Daily
- **Variables**: SSH, geostrophic velocities
- **Coverage**: Global oceans

Ocean Models
~~~~~~~~~~~~

Provide 3D structure:

- **Models**: CROCO, NEMO, MOM
- **Resolution**: 1/12° to 1/36° (~2-8 km)
- **Temporal**: Hourly to daily outputs
- **Variables**: U, V, SSH, T, S at multiple depths
- **Coverage**: Regional or global

Reanalysis
~~~~~~~~~~

Combines models and observations:

- **Products**: GLORYS, ORAS
- **Resolution**: 1/12° (~8 km)
- **Temporal**: Daily
- **Variables**: 3D fields (U, V, T, S)
- **Coverage**: Global, multi-decadal

Best Practices
--------------

Parameter Selection
~~~~~~~~~~~~~~~~~~~

1. Start with standard values based on latitude
2. Adjust based on data resolution
3. Validate with known eddies if available
4. Check sensitivity to parameters
5. Document chosen values

Quality Control
~~~~~~~~~~~~~~~

1. Visually inspect detected eddies
2. Check size distribution is reasonable
3. Verify tracking produces smooth trajectories
4. Remove spurious short-lived features
5. Validate against independent data

Performance
~~~~~~~~~~~

For large datasets:

- Process subregions separately
- Use parallel detection (with caution)
- Save intermediate results
- Consider coarser temporal sampling

Common Issues
-------------

No Eddies Detected
~~~~~~~~~~~~~~~~~~

**Possible causes**:
   - Insufficient velocity magnitude
   - Grid resolution too coarse
   - Parameters too restrictive
   - Data quality issues

**Solutions**:
   - Check velocity field statistics
   - Reduce min_radius
   - Adjust window sizes
   - Verify data integrity

Too Many False Positives
~~~~~~~~~~~~~~~~~~~~~~~~~

**Possible causes**:
   - Noisy data
   - Parameters too permissive
   - Insufficient quality filtering

**Solutions**:
   - Increase min_radius
   - Decrease ellipse_error
   - Add coastal masking
   - Filter by duration in tracking

Tracking Failures
~~~~~~~~~~~~~~~~~

**Possible causes**:
   - max_distance too small
   - Detection inconsistency
   - Large time gaps
   - Rapid eddy evolution

**Solutions**:
   - Increase search distance
   - Improve detection parameters
   - Interpolate missing times
   - Adjust cost function weights

Further Reading
---------------

**Comprehensive guides**:
   - :ref:`indepth_eddies_detection` - Complete detection guide
   - :ref:`indepth_eddies_tracking` - Complete tracking guide

**Algorithm details**:
   - :ref:`algos.eddies` - Mathematical description
   - :ref:`algos.track` - Tracking algorithm

**Related topics**:
   - :ref:`indepth_metadata` - Working with different data formats
   - :ref:`indepth_profiles` - Analyzing eddy water properties
   - :ref:`quickstart` - Quick start examples

**Scientific references**:
   - Chelton et al. (2011) - Tracking algorithm
   - Faghmous et al. (2015) - Detection methods
   - Nencioli et al. (2010) - Eddy characterization
