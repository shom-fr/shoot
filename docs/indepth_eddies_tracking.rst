.. _indepth_eddies_tracking:

Eddy Tracking
=============

This guide explains how to track eddies through time using shoot's tracking framework, based on the Chelton et al. (2011) algorithm.

Tracking Overview
-----------------

Eddy tracking associates detected eddies between consecutive time steps to create trajectories. The process:

1. Detect eddies at each time step
2. Compute cost matrix between time steps
3. Find optimal matching using Hungarian algorithm
4. Update track objects with new observations
5. Handle eddy births and deaths

Cost Function
~~~~~~~~~~~~~

The matching cost combines:

- **Spatial distance** - How far eddy centers moved
- **Rossby number similarity** - Intensity consistency
- **Radius similarity** - Size consistency
- **Type matching** - Cyclones only match cyclones

Eddies that match poorly get high cost and may start new tracks.

Basic Tracking
--------------

Track from Pre-detected Eddies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have eddies detected at multiple times:

.. code-block:: python

    from shoot.eddies import eddies2d
    from shoot.eddies.track import Track
    import xarray as xr

    # Load detected eddies from NetCDF
    ds_eddies = xr.open_dataset("eddies_detected_timeseries.nc")

    # Convert back to Eddies2D objects per time
    eddies_list = []
    for t in range(len(ds_eddies.time)):
        eddies_t = eddies2d.Eddies2D.from_dataset(
            ds_eddies.isel(time=t)
        )
        eddies_list.append(eddies_t)

    # Track using CLI or Python API (see below)

Detect and Track in One Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For time-series data:

.. code-block:: python

    ds = xr.open_dataset("velocity_timeseries.nc")

    eddies_list = []

    # Detect at each time
    for t in range(len(ds.time)):
        print(f"Detecting at time {t+1}/{len(ds.time)}")

        ds_t = ds.isel(time=t)
        eddies_t = eddies2d.Eddies2D.detect_eddies(
            ds_t.u, ds_t.v,
            window_center=50,
            window_fit=120,
            min_radius=15
        )

        eddies_list.append(eddies_t)

    print(f"Detected {sum(len(e.eddies) for e in eddies_list)} total eddies")

Now ready for tracking.

Tracking Implementation
-----------------------

Manual Tracking
~~~~~~~~~~~~~~~

Implement tracking with Track objects:

.. code-block:: python

    from shoot.eddies.track import Track, Associate
    import numpy as np

    # Initialize tracks from first time step
    tracks = []
    track_id = 0

    for eddy in eddies_list[0].eddies:
        eddy.track_id = track_id
        tracks.append(Track(eddy, ds.time[0].values))
        track_id += 1

    print(f"Initialized {len(tracks)} tracks")

    # Process subsequent time steps
    for t in range(1, len(eddies_list)):
        # Get eddies at current and previous time
        new_eddies = eddies_list[t].eddies
        prev_eddies = eddies_list[t-1].eddies

        # Time difference (days)
        dt = (ds.time[t] - ds.time[t-1]).values / np.timedelta64(1, 'D')

        # Associate eddies
        associator = Associate(
            tracks,
            prev_eddies,
            new_eddies,
            Dt=dt,
            Tc=10,                    # Characteristic time (days)
            C=6.5e3 / 86400          # Characteristic velocity (m/s)
        )

        # Get cost matrix and find matches
        cost = associator.cost
        # ... apply Hungarian algorithm ...
        # ... update tracks or create new ones ...

Using CLI
~~~~~~~~~

The command-line interface simplifies tracking:

.. code-block:: bash

    # Track detected eddies
    shoot eddies track eddies_detected_*.nc \\
        --max-distance 50 \\
        --output eddies_tracked.nc

    # View tracking statistics
    shoot eddies diags eddies_tracked.nc

Tracking Parameters
-------------------

Maximum Distance
~~~~~~~~~~~~~~~~

**max_distance** sets search radius (km):

- Eddies beyond this distance won't be matched
- Based on typical advection: speed × time interval
- Example: 10 km/day × 7 days = 70 km

.. code-block:: python

    # Daily data
    max_distance = 50  # km

    # Weekly data
    max_distance = 100  # km

Time Scales
~~~~~~~~~~~

**Tc** (characteristic time) affects cost weighting:

- Typically 10-20 days for mesoscale eddies
- Shorter: favors spatial proximity
- Longer: allows more property evolution

.. code-block:: python

    Associate(
        tracks, prev_eddies, new_eddies,
        Dt=7,      # Actual interval (days)
        Tc=10      # Characteristic time (days)
    )

Velocity Scale
~~~~~~~~~~~~~~

**C** (characteristic velocity) for search distance:

- Default: 6.5 km/day = 0.075 m/s
- Adjust for energetic regions

.. code-block:: python

    # Standard mesoscale
    C = 6.5e3 / 86400  # m/s

    # Energetic western boundary
    C = 10e3 / 86400   # m/s

Working with Tracks
-------------------

Track Objects
~~~~~~~~~~~~~

Each Track contains:

.. code-block:: python

    for track in tracks:
        # Track identity
        print(f"Track ID: {track.id}")

        # Eddy observations
        print(f"Length: {len(track.eddies)} observations")

        # Times
        print(f"Times: {track.times}")

        # Access properties over time
        lons = [e.lon for e in track.eddies]
        lats = [e.lat for e in track.eddies]
        radii = [e.radius for e in track.eddies]

Track Duration
~~~~~~~~~~~~~~

Filter by lifetime:

.. code-block:: python

    # Keep tracks lasting > 30 days
    long_tracks = [t for t in tracks if t.duration_days > 30]

    # Compute duration
    for track in tracks:
        dt_days = (track.times[-1] - track.times[0]) / np.timedelta64(1, 'D')
        track.duration_days = dt_days

Track Displacement
~~~~~~~~~~~~~~~~~~

Compute total displacement:

.. code-block:: python

    from shoot import geo as sgeo

    for track in tracks:
        if len(track.eddies) < 2:
            continue

        # First and last positions
        lon0, lat0 = track.eddies[0].lon, track.eddies[0].lat
        lon1, lat1 = track.eddies[-1].lon, track.eddies[-1].lat

        # Distance in km
        dx = sgeo.deg2m(lon1 - lon0, lat0) / 1000
        dy = sgeo.deg2m(lat1 - lat0) / 1000
        distance = np.sqrt(dx**2 + dy**2)

        print(f"Track {track.id} traveled {distance:.1f} km")

Converting to Dataset
~~~~~~~~~~~~~~~~~~~~~

Export tracks to xarray:

.. code-block:: python

    # Create dataset from all tracks
    track_data = []

    for track in tracks:
        for i, (eddy, time) in enumerate(zip(track.eddies, track.times)):
            track_data.append({
                'track_id': track.id,
                'time': time,
                'lon': eddy.lon,
                'lat': eddy.lat,
                'radius': eddy.radius,
                'ro': eddy.ro,
                'eddy_type': eddy.eddy_type
            })

    import pandas as pd
    df = pd.DataFrame(track_data)
    ds_tracks = xr.Dataset.from_dataframe(df)
    ds_tracks.to_netcdf("tracks.nc")

Visualization
-------------

Plotting Individual Tracks
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    from shoot.plot import create_map

    # Create map
    fig, ax = create_map(ds)

    # Plot tracks
    colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))

    for track, color in zip(tracks, colors):
        lons = [e.lon for e in track.eddies]
        lats = [e.lat for e in track.eddies]

        # Plot trajectory
        ax.plot(lons, lats, 'o-', color=color, linewidth=2)

        # Mark start/end
        ax.plot(lons[0], lats[0], 'go', markersize=10)  # Start
        ax.plot(lons[-1], lats[-1], 'ro', markersize=10)  # End

    plt.title(f"{len(tracks)} eddy tracks")
    plt.show()

Track Evolution
~~~~~~~~~~~~~~~

Plot property evolution:

.. code-block:: python

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for track in tracks:
        times = track.times
        radii = [e.radius for e in track.eddies]
        ros = [e.ro for e in track.eddies]

        axes[0].plot(times, radii, 'o-')
        axes[1].plot(times, ros, 'o-')

    axes[0].set_ylabel("Radius (km)")
    axes[1].set_ylabel("Rossby number")
    axes[1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()

Quality Control
---------------

Track Validation
~~~~~~~~~~~~~~~~

Check track quality:

.. code-block:: python

    def validate_track(track, max_speed=30):
        """Check track has realistic properties"""
        if len(track.eddies) < 3:
            return False  # Too short

        # Check displacement between steps
        for i in range(len(track.eddies)-1):
            e1, e2 = track.eddies[i], track.eddies[i+1]
            dt = (track.times[i+1] - track.times[i]) / np.timedelta64(1, 'D')

            dx = sgeo.deg2m(e2.lon - e1.lon, e1.lat) / 1000
            dy = sgeo.deg2m(e2.lat - e1.lat) / 1000
            distance = np.sqrt(dx**2 + dy**2)

            speed = distance / dt  # km/day

            if speed > max_speed:
                return False  # Unrealistic jump

        return True

    valid_tracks = [t for t in tracks if validate_track(t)]
    print(f"{len(valid_tracks)}/{len(tracks)} tracks valid")

Property Jumps
~~~~~~~~~~~~~~

Detect suspicious property changes:

.. code-block:: python

    for track in tracks:
        radii = np.array([e.radius for e in track.eddies])

        # Check for large jumps
        dradii = np.abs(np.diff(radii))

        if np.any(dradii > 20):  # More than 20 km jump
            print(f"Track {track.id} has suspicious radius changes")

Merging/Splitting
~~~~~~~~~~~~~~~~~

Eddies can merge or split. To detect:

.. code-block:: python

    # Check for multiple eddies matching one eddy
    # (indicates potential merging or overly permissive matching)

    # This requires examining the cost matrix and assignments
    # from the Hungarian algorithm output

Advanced Tracking
-----------------

Re-tracking with Different Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # First attempt - conservative
    tracks1 = track_eddies(eddies_list, max_distance=30)
    print(f"Conservative: {len(tracks1)} tracks")

    # Second attempt - permissive
    tracks2 = track_eddies(eddies_list, max_distance=70)
    print(f"Permissive: {len(tracks2)} tracks")

    # Compare statistics
    avg_len1 = np.mean([len(t.eddies) for t in tracks1])
    avg_len2 = np.mean([len(t.eddies) for t in tracks2])
    print(f"Average length: {avg_len1:.1f} vs {avg_len2:.1f}")

Gap Filling
~~~~~~~~~~~

Allow eddies to disappear and reappear:

.. code-block:: python

    # Track with gap tolerance (not yet implemented in shoot)
    # Would allow matching across 1-2 missing observations
    # Useful for clouds in satellite data

Region-specific Tracking
~~~~~~~~~~~~~~~~~~~~~~~~

Track separately in different regions:

.. code-block:: python

    # Western vs eastern basin
    west_tracks = track_region(eddies_list, lon_range=(-180, -100))
    east_tracks = track_region(eddies_list, lon_range=(-100, 0))

Troubleshooting
---------------

Short Tracks
~~~~~~~~~~~~

If most tracks are very short:

1. Increase ``max_distance``
2. Check time resolution is adequate
3. Verify detection is consistent across times
4. Check for data gaps

Many Unmatched Eddies
~~~~~~~~~~~~~~~~~~~~~~

If many eddies don't get matched:

1. Examine cost matrix values
2. Verify eddies aren't changing properties too rapidly
3. Consider increasing search distance
4. Check for detection issues (false positives)

Spurious Jumps
~~~~~~~~~~~~~~

If tracks show unrealistic motion:

1. Decrease ``max_distance``
2. Strengthen property similarity weighting
3. Check for detection duplicates
4. Apply post-processing smoothing

Further Reading
---------------

- :ref:`algos.track` - Tracking algorithm details
- :ref:`indepth_eddies_detection` - Ensuring quality detection
- :ref:`quickstart` - Quick tracking example
- Chelton et al. (2011) - Original tracking algorithm paper
