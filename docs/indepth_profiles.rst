.. _indepth_profiles:

Working with Profiles
=====================

This guide explains how to integrate in-situ profile data (like Argo floats) with detected eddies to analyze 3D structure and compute anomalies.

Overview
--------

shoot can:

- **Download Argo profiles** for your study domain
- **Associate profiles with eddies** based on spatiotemporal proximity
- **Compute anomalies** comparing inside vs outside eddy
- **Analyze acoustic impacts** of eddy-induced changes

This enables:

- Validation of surface detection with subsurface data
- 3D characterization of eddy water masses
- Study of eddy impacts on temperature/salinity
- Assessment of effects on sound propagation

Profile Data Structure
----------------------

Argo Profile Format
~~~~~~~~~~~~~~~~~~~

Argo profiles contain:

- **Position**: Latitude, longitude
- **Time**: Observation datetime
- **Pressure**: Measurement depths
- **Temperature**: In-situ temperature
- **Salinity**: Practical salinity

Example profile structure:

.. code-block:: python

    from shoot.profiles.profiles import Profile
    import numpy as np
    import xarray as xr

    # Typical Argo profile
    prf = xr.Dataset({
        'TIME': (['obs'], [np.datetime64('2024-01-15')]),
        'LATITUDE': (['obs'], [42.5]),
        'LONGITUDE': (['obs'], [10.2]),
        'PRES_ADJUSTED': (['depth'], np.arange(0, 2000, 10)),
        'TEMP_ADJUSTED': (['depth'], 20 - np.arange(0, 2000, 10) * 0.01),
        'PSAL_ADJUSTED': (['depth'], 35 + np.arange(0, 2000, 10) * 0.001),
    })

Profile Class
~~~~~~~~~~~~~

Individual profiles are represented by ``Profile`` objects:

.. code-block:: python

    profile = Profile(prf)

    print(f"Location: ({profile.lon}, {profile.lat})")
    print(f"Time: {profile.time}")
    print(f"Valid: {profile.valid}")
    print(f"Depth range: {profile.depth[0]}-{profile.depth[-1]} m")
    print(f"Temperature: {profile.temp[0]:.2f}°C at surface")

Profile properties:

- ``time``: Observation datetime
- ``lat, lon``: Position
- ``depth``: Standard depth array (1-2000 m)
- ``temp``: Temperature interpolated to standard depths
- ``sal``: Salinity interpolated to standard depths
- ``valid``: Quality flag (sufficient non-NaN data)

Downloading Profiles
--------------------

Automatic Download
~~~~~~~~~~~~~~~~~~

Download profiles for a dataset's domain:

.. code-block:: python

    from shoot.profiles.download import load_from_ds
    import xarray as xr

    # Load your ocean data
    ds = xr.open_dataset("velocity_field.nc")

    # Download Argo profiles matching spatiotemporal extent
    profiles_raw = load_from_ds(
        ds,
        root_path="./argo_data",  # Cache directory
        max_depth=1000             # Maximum depth (m)
    )

    print(f"Downloaded {len(profiles_raw.N_PROF)} profiles")

This:

1. Determines bounding box from ds.lon/lat
2. Determines time range from ds.time
3. Downloads Argo data from ERDDAP
4. Caches by year in root_path
5. Returns combined xarray Dataset

Manual Download
~~~~~~~~~~~~~~~

For more control:

.. code-block:: python

    from shoot.profiles.download import Download
    import pandas as pd

    # Define domain explicitly
    time = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    time_da = xr.DataArray(time, dims='time')

    downloader = Download(
        time=time_da,
        lat_min=40.0,
        lat_max=45.0,
        lon_min=5.0,
        lon_max=15.0,
        root_path="./argo_data",
        max_depth=1500
    )

    profiles_raw = downloader.profiles

Creating Profile Collections
-----------------------------

From Downloaded Data
~~~~~~~~~~~~~~~~~~~~

Convert to Profiles collection:

.. code-block:: python

    from shoot.profiles.profiles import Profiles

    # Create from dataset
    profiles = Profiles.from_ds(ds, root_path="./argo_data")

    print(f"Created {len(profiles.profiles)} valid profiles")
    print(f"Years covered: {profiles.years}")

Profile collections filter invalid profiles (too many NaNs).

Accessing Profile Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Loop through individual profiles
    for profile in profiles.profiles:
        print(f"Profile at ({profile.lon:.2f}, {profile.lat:.2f})")
        print(f"  Surface temp: {profile.temp[0]:.2f}°C")
        print(f"  Surface sal: {profile.sal[0]:.2f}")

    # Convert to xarray Dataset
    ds_profiles = profiles.ds

    print(ds_profiles)
    # Contains:
    #   - time(profil): Profile observation times
    #   - lat(profil): Latitudes
    #   - lon(profil): Longitudes
    #   - temp(profil, depth): Temperature profiles
    #   - sal(profil, depth): Salinity profiles

Associating with Eddies
------------------------

Spatial-Temporal Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~

Associate profiles with detected eddies:

.. code-block:: python

    from shoot.eddies.eddies2d import Eddies2D

    # Detect eddies
    eddies = Eddies2D.detect_eddies(
        ds.u, ds.v, 50, 120, ssh=ds.ssh
    )

    # Associate profiles with eddies
    profiles.associate(
        eddies,
        nlag=2  # Match within ±2 days
    )

    # Now profiles.ds contains 'eddy_pos' variable
    print(profiles.ds.eddy_pos)
    # -1 = not in any eddy
    # >= 0 = track_id of associated eddy

Selecting Inside/Outside
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Profiles inside any eddy
    inside = profiles.ds.where(profiles.ds.eddy_pos >= 0, drop=True)
    print(f"{len(inside.profil)} profiles inside eddies")

    # Profiles outside all eddies
    outside = profiles.ds.where(profiles.ds.eddy_pos == -1, drop=True)
    print(f"{len(outside.profil)} profiles outside eddies")

    # Profiles in specific eddy
    track_id = 5
    in_eddy_5 = profiles.ds.where(profiles.ds.eddy_pos == track_id, drop=True)

Computing Anomalies
-------------------

For Eddy Objects
~~~~~~~~~~~~~~~~

Use the ``Anomaly`` class from hydrology module:

.. code-block:: python

    from shoot.hydrology import Anomaly

    # For a specific eddy
    eddy = eddies.eddies[0]

    # Create anomaly object
    anomaly = Anomaly(
        eddy=eddy,
        profiles_ds=profiles.ds,
        nlag=2  # Time window (days)
    )

    # Check if enough data
    if anomaly.is_valid():
        # Mean profiles inside vs outside
        temp_in = anomaly.mean_profil_inside
        temp_out = anomaly.mean_profil_outside

        # Compute anomaly
        temp_anom = anomaly.anomaly_at_depth(depth_level=100)
        print(f"Temperature anomaly at 100m: {temp_anom:.2f}°C")

Mean Profile Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    if anomaly.is_valid():
        depths = anomaly.depth_vector

        # Plot mean profiles
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # Temperature
        axes[0].plot(anomaly.mean_profil_inside.temp, depths, 'r-',
                     label='Inside')
        axes[0].plot(anomaly.mean_profil_outside.temp, depths, 'b-',
                     label='Outside')
        axes[0].set_xlabel('Temperature (°C)')
        axes[0].set_ylabel('Depth (m)')
        axes[0].invert_yaxis()
        axes[0].legend()
        axes[0].grid(True)

        # Salinity
        axes[1].plot(anomaly.mean_profil_inside.sal, depths, 'r-')
        axes[1].plot(anomaly.mean_profil_outside.sal, depths, 'b-')
        axes[1].set_xlabel('Salinity')
        axes[1].invert_yaxis()
        axes[1].grid(True)

        plt.suptitle(f"Eddy at ({eddy.lon:.1f}, {eddy.lat:.1f})")
        plt.tight_layout()
        plt.show()

Anomaly Statistics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute anomalies at multiple depths
    depths_interest = [50, 100, 200, 500, 1000]

    for depth in depths_interest:
        temp_anom = anomaly.anomaly_at_depth(depth, variable='temp')
        sal_anom = anomaly.anomaly_at_depth(depth, variable='sal')

        if temp_anom is not None:
            print(f"Depth {depth}m:")
            print(f"  ΔT = {temp_anom:.3f}°C")
            print(f"  ΔS = {sal_anom:.3f}")

Acoustic Analysis
-----------------

Sound Speed Computation
~~~~~~~~~~~~~~~~~~~~~~~

Compute sound speed from T/S profiles:

.. code-block:: python

    import gsw

    # For a profile
    profile = profiles.profiles[0]

    # Compute sound speed using Gibbs SeaWater toolbox
    c = gsw.sound_speed(
        SA=profile.sal,          # Absolute salinity
        CT=profile.temp,         # Conservative temperature
        p=profile.depth          # Pressure (≈ depth)
    )

    print(f"Sound speed range: {c.min():.1f} - {c.max():.1f} m/s")

Acoustic Parameters
~~~~~~~~~~~~~~~~~~~

Compute acoustic properties of eddies:

.. code-block:: python

    from shoot.acoustic import AcousEddy

    # Create acoustic eddy analyzer
    acous_eddy = AcousEddy(anomaly)

    # Acoustic parameters inside eddy
    ecs_in = acous_eddy.ecs_inside      # Surface duct thickness
    mcp_in = acous_eddy.mcp_inside      # Deep sound speed minimum
    iminc_in = acous_eddy.iminc_inside  # Intermediate minimum

    # Outside eddy
    ecs_out = acous_eddy.ecs_outside
    mcp_out = acous_eddy.mcp_outside
    iminc_out = acous_eddy.iminc_outside

    # Overall impact
    impact = acous_eddy.acoustic_impact
    print(f"Acoustic impact: {impact:.3f}")

The acoustic impact quantifies how much the eddy alters sound propagation by comparing inside and outside acoustic parameters.

Acoustic Impact Maps
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # For all eddies with sufficient profiles
    impacts = []
    eddy_lons = []
    eddy_lats = []

    for eddy in eddies.eddies:
        anomaly = Anomaly(eddy, profiles.ds, nlag=2)

        if anomaly.is_valid():
            acous = AcousEddy(anomaly)
            impacts.append(acous.acoustic_impact)
            eddy_lons.append(eddy.lon)
            eddy_lats.append(eddy.lat)

    # Plot impact map
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(eddy_lons, eddy_lats, c=impacts,
                     s=100, cmap='RdBu_r', vmin=0, vmax=2)
    plt.colorbar(sc, label='Acoustic Impact')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Eddy Acoustic Impact')
    plt.show()

Best Practices
--------------

Data Requirements
~~~~~~~~~~~~~~~~~

For reliable anomaly computation:

- **Minimum profiles inside**: 3-5 profiles
- **Minimum profiles outside**: 10-20 profiles (for stable mean)
- **Temporal window**: ±2 to ±7 days depending on eddy evolution
- **Spatial extent**: Profiles well distributed inside/outside

Quality Control
~~~~~~~~~~~~~~~

.. code-block:: python

    # Check profile quality
    valid_profiles = [p for p in profiles.profiles if p.valid]
    print(f"{len(valid_profiles)}/{len(profiles.profiles)} profiles valid")

    # Check anomaly validity
    if anomaly.is_valid():
        n_in = len(anomaly.profils_inside)
        n_out = len(anomaly.profils_outside)
        print(f"Profiles: {n_in} inside, {n_out} outside")
    else:
        print("Insufficient profiles for anomaly")

Interpolation
~~~~~~~~~~~~~

Profile class interpolates to standard depths (1-2000 m by 1 m). This:

- Enables easy comparison between profiles
- Handles varying native sampling
- Uses NaN for extrapolation beyond data range

Saving Results
~~~~~~~~~~~~~~

.. code-block:: python

    # Save profile dataset with eddy associations
    profiles.ds.to_netcdf("profiles_with_eddies.nc")

    # Save anomaly statistics
    anomaly_data = {
        'eddy_id': [],
        'n_inside': [],
        'n_outside': [],
        'temp_anom_100m': [],
        'sal_anom_100m': [],
    }

    for eddy in eddies.eddies:
        anomaly = Anomaly(eddy, profiles.ds)
        if anomaly.is_valid():
            anomaly_data['eddy_id'].append(eddy.track_id)
            anomaly_data['n_inside'].append(len(anomaly.profils_inside))
            anomaly_data['n_outside'].append(len(anomaly.profils_outside))
            anomaly_data['temp_anom_100m'].append(
                anomaly.anomaly_at_depth(100, 'temp')
            )
            anomaly_data['sal_anom_100m'].append(
                anomaly.anomaly_at_depth(100, 'sal')
            )

    import pandas as pd
    df = pd.DataFrame(anomaly_data)
    df.to_csv("eddy_anomalies.csv", index=False)

Advanced Topics
---------------

Profile Filtering
~~~~~~~~~~~~~~~~~

Apply custom filters:

.. code-block:: python

    # Filter by depth range
    deep_profiles = [p for p in profiles.profiles
                     if np.nanmax(p.depth) > 1500]

    # Filter by data quality
    high_quality = [p for p in profiles.profiles
                    if np.sum(~np.isnan(p.temp)) > 150]

Multiple Eddies
~~~~~~~~~~~~~~~

Composite analysis across eddies:

.. code-block:: python

    # Collect anomalies from all eddies
    temp_anoms = []
    sal_anoms = []

    for eddy in eddies.eddies:
        if eddy.eddy_type == 'anticyclone':  # Focus on anticyclones
            anomaly = Anomaly(eddy, profiles.ds)
            if anomaly.is_valid():
                temp_anoms.append(anomaly.mean_profil_inside.temp -
                                anomaly.mean_profil_outside.temp)
                sal_anoms.append(anomaly.mean_profil_inside.sal -
                               anomaly.mean_profil_outside.sal)

    # Average anomaly
    mean_temp_anom = np.nanmean(temp_anoms, axis=0)
    mean_sal_anom = np.nanmean(sal_anoms, axis=0)

Regional Studies
~~~~~~~~~~~~~~~~

Analyze by region:

.. code-block:: python

    # Western vs Eastern basin
    west_profiles = profiles.ds.where(profiles.ds.lon < 10, drop=True)
    east_profiles = profiles.ds.where(profiles.ds.lon >= 10, drop=True)

    # Compute anomalies separately
    # ...

Troubleshooting
---------------

No Profiles Found
~~~~~~~~~~~~~~~~~

If download returns no profiles:

1. Check internet connection
2. Verify lat/lon/time ranges are valid
3. Ensure date range has Argo coverage
4. Try expanding spatial domain
5. Check ERDDAP server availability

No Inside Profiles
~~~~~~~~~~~~~~~~~~

If eddies have no associated profiles:

1. Increase ``nlag`` temporal window
2. Check eddy detection quality
3. Verify profile time range overlaps eddies
4. Increase spatial domain for downloads

Invalid Anomalies
~~~~~~~~~~~~~~~~~

If anomalies are invalid:

1. Increase temporal window (``nlag``)
2. Ensure enough profiles overall
3. Check profile depth coverage
4. Verify eddy size vs profile spacing

Further Reading
---------------

- :ref:`indepth_eddies_detection` - Detecting eddies for association
- :ref:`quickstart` - Basic usage examples
- :ref:`lib` - API documentation for Profile, Profiles, Anomaly
- Argo documentation - Understanding Argo data structure
