.. _quickstart:

Quick Start Guide
#################

Welcome to shoot! This guide will help you get started with detecting and tracking ocean objects like eddies, fronts, and other mesoscale features.

What is shoot?
==============

**SHom Ocean Objects Tracker** (shoot) is a Python package for detecting and tracking ocean mesoscale features from gridded velocity and SSH data. It can:

* Detect eddies using the Local Normalized Angular Momentum (LNAM) method
* Track eddies through time using optimal matching algorithms
* Analyze 3D eddy structure across depth levels
* Compute acoustic impacts of eddies on sound propagation
* Associate in-situ profiles with detected eddies

Basic Concepts
==============

Eddies
------

Eddies are rotating water masses that can be detected from velocity fields or sea surface height (SSH) anomalies. shoot identifies:

* **Cyclonic eddies** - rotating counterclockwise in Northern Hemisphere (positive LNAM)
* **Anticyclonic eddies** - rotating clockwise in Northern Hemisphere (negative LNAM)

Each detected eddy includes:

* Center position (lon, lat)
* Radius (at maximum velocity contour)
* Rossby number (intensity measure)
* Ellipse fit parameters
* Speed contours and enclosed areas

Detection Method
----------------

shoot uses the **LNAM method** (Local Normalized Angular Momentum):

1. Compute angular momentum from u/v velocity fields
2. Normalize by distance from potential center
3. Find local extrema as eddy centers
4. Extract contours around centers
5. Fit ellipses and compute properties

Tracking
--------

Eddy tracking associates eddies between consecutive time steps:

* Uses cost function based on distance and similarity
* Employs Hungarian algorithm for optimal matching
* Creates trajectories over time
* Tracks eddy evolution (radius, intensity)

Installation
============

From the repository::

    cd shoot/
    pip install -e .

This installs shoot and its dependencies including xarray, numpy, scipy, and xoa.

First Detection Example
=======================

Let's detect eddies from satellite altimetry data.

Load Data
---------

.. code-block:: python

    import xarray as xr
    from shoot.eddies.eddies2d import Eddies2D
    from shoot.samples import get_sample_file

    # Load sample satellite data
    path = get_sample_file("OBS/SATELLITE/jan2024_ionian_sea_duacs.nc")
    ds = xr.open_dataset(path).isel(time=0)

    # Data contains ugos (zonal velocity) and vgos (meridional velocity)
    print(ds)

Configure Detection
-------------------

Set detection parameters:

.. code-block:: python

    # Window size (km) for computing LNAM and finding centers
    window_center = 50  # Typically ~Rossby radius

    # Window size (km) for fitting SSH and diagnostics
    window_fit = 120  # Typically ~10 * Rossby radius

    # Minimum eddy radius (km) to retain
    min_radius = 10  # Around Rossby radius of deformation

    # Maximum ellipse fitting error (fraction)
    ellipse_error = 0.05  # 5% error tolerance

Run Detection
-------------

.. code-block:: python

    # Detect eddies
    eddies = Eddies2D.detect_eddies(
        ds.ugos,          # Zonal velocity
        ds.vgos,          # Meridional velocity
        window_center,    # Center detection window
        window_fit=window_fit,
        ssh=ds.adt,       # Absolute dynamic topography (optional)
        min_radius=min_radius,
        ellipse_error=ellipse_error,
        paral=False       # Set True for parallel processing
    )

    print(f"Detected {len(eddies.eddies)} eddies")

Access Results
--------------

.. code-block:: python

    # Loop through detected eddies
    for eddy in eddies.eddies:
        print(f"Eddy at ({eddy.lon:.2f}, {eddy.lat:.2f})")
        print(f"  Type: {eddy.eddy_type}")  # "cyclone" or "anticyclone"
        print(f"  Radius: {eddy.radius:.1f} km")
        print(f"  Rossby number: {eddy.ro:.3f}")

Visualize Results
-----------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from shoot.plot import create_map, pcarr

    # Create map
    fig, ax = create_map(ds)

    # Plot SSH
    pcarr(ds.adt, ax=ax, cmap="RdBu_r", vmin=-0.3, vmax=0.3)

    # Overlay detected eddies
    for eddy in eddies.eddies:
        # Plot eddy center
        ax.plot(eddy.lon, eddy.lat, 'ko', markersize=8)

        # Plot maximum velocity contour
        if hasattr(eddy, 'vmax_contour'):
            ax.plot(eddy.vmax_contour.lon,
                   eddy.vmax_contour.lat, 'k-', linewidth=2)

    plt.title(f"Detected {len(eddies.eddies)} eddies")
    plt.show()

First Tracking Example
======================

Track eddies through multiple time steps.

Load Time Series
----------------

.. code-block:: python

    from shoot.eddies.track import Track

    # Load data with time dimension
    path = get_sample_file("OBS/SATELLITE/jan2024_ionian_sea_duacs.nc")
    ds = xr.open_dataset(path)

Detect at Each Time Step
-------------------------

.. code-block:: python

    # Store detected eddies for each time
    eddies_list = []

    for t in range(len(ds.time)):
        ds_t = ds.isel(time=t)

        eddies_t = Eddies2D.detect_eddies(
            ds_t.ugos, ds_t.vgos,
            window_center=50,
            window_fit=120,
            min_radius=10
        )

        eddies_list.append(eddies_t)

    print(f"Detected eddies at {len(eddies_list)} time steps")

Track Eddies
------------

.. code-block:: python

    # Initialize tracking
    tracks = []

    # First time step - create new tracks
    for eddy in eddies_list[0].eddies:
        eddy.track_id = len(tracks)
        tracks.append(Track(eddy, ds.time[0].values))

    # Subsequent time steps - associate and update
    for t in range(1, len(eddies_list)):
        # Association happens here (see in-depth guide)
        # Updates existing tracks or creates new ones
        pass  # See indepth_tracking for full implementation

Using the Command Line
======================

shoot provides a CLI for common tasks.

Detect Eddies
-------------

.. code-block:: bash

    # Detect eddies from a NetCDF file
    shoot eddies detect input.nc \
        --window-center 50 \
        --window-fit 120 \
        --min-radius 10 \
        -o eddies_detected.nc

Track Eddies
------------

.. code-block:: bash

    # Track detected eddies over time
    shoot eddies track eddies_detected.nc \
        --max-distance 50 \
        -o eddies_tracked.nc

Show Diagnostics
----------------

.. code-block:: bash

    # Display eddy statistics
    shoot eddies diags eddies_tracked.nc

Working with xoa
================

shoot uses **xoa** for metadata handling and CF convention support.

The xoa package (located in ``xoa/`` directory) provides:

* CF-compliant coordinate detection (lon, lat, depth, time)
* Standard name searches (SSH, velocities, temperature, salinity)
* ROMS/CROCO model support
* Coordinate transformations

Example using shoot's metadata wrappers:

.. code-block:: python

    from shoot import meta as smeta

    # Automatically find coordinates
    lon = smeta.get_lon(ds)
    lat = smeta.get_lat(ds)
    depth = smeta.get_depth(ds, errors='ignore')

    # Find variables by standard name
    u = smeta.get_u(ds)
    v = smeta.get_v(ds)
    ssh = smeta.get_ssh(ds)

Next Steps
==========

Now that you've run your first detection and tracking:

1. Read the :ref:`indepth` guide for detailed explanations
2. Explore the :ref:`examples <examples>` gallery
3. Check the :ref:`lib` for API documentation
4. See :ref:`cli` for all command-line options

Common Questions
================

**How do I choose window sizes?**
    Start with ``window_center`` around the Rossby radius of deformation
    (~50 km mid-latitudes, ~20 km tropics) and ``window_fit`` around 10 times that.

**Why aren't eddies detected?**
    Check that your velocity fields have sufficient resolution and magnitude.
    Try reducing ``min_radius`` or adjusting ``window_center``.

**Can I use model output?**
    Yes! shoot works with any gridded velocity data. It supports ROMS/CROCO
    native grids through xoa.

**How accurate is the tracking?**
    Tracking accuracy depends on time resolution. For daily data, use
    ``max_distance`` around 50-100 km. For weekly data, increase accordingly.

**What about 3D eddies?**
    See :ref:`indepth_eddies_detection` for 3D detection across depth levels.
