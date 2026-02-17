.. _indepth_eddies_detection:

Eddy Detection
==============

This guide explains shoot's eddy detection system in detail, covering the LNAM method, parameter selection, and advanced detection scenarios.

Detection Method
----------------

shoot uses the **Local Normalized Angular Momentum (LNAM)** method to identify rotating structures. The complete workflow is:

1. Compute LNAM field from velocity components
2. Apply Okubo-Weiss filtering to identify vortex regions
3. Find local extrema as eddy centers
4. Extract SSH contours around centers
5. Identify maximum velocity contour
6. Fit ellipse and compute properties
7. Apply quality filters

See :ref:`algos.eddies` for mathematical details.

Basic Detection
---------------

The simplest detection requires only velocity fields:

.. code-block:: python

    from shoot.eddies.eddies2d import Eddies2D
    import xarray as xr

    # Load data with u, v components
    ds = xr.open_dataset("velocity_data.nc")

    # Detect eddies
    eddies = Eddies2D.detect_eddies(
        ds.u,                 # Zonal velocity (m/s)
        ds.v,                 # Meridional velocity (m/s)
        window_center=50,     # Detection window (km)
        window_fit=120,       # Fitting window (km)
        min_radius=15         # Minimum radius (km)
    )

    print(f"Found {len(eddies.eddies)} eddies")

With SSH Field
--------------

Providing SSH improves boundary detection:

.. code-block:: python

    eddies = Eddies2D.detect_eddies(
        ds.u, ds.v,
        window_center=50,
        window_fit=120,
        ssh=ds.ssh,           # Sea surface height (m)
        min_radius=15
    )

Without SSH, shoot estimates it from the streamfunction or by geostrophic inversion.

Parameter Selection
-------------------

Window Sizes
~~~~~~~~~~~~

**window_center** controls LNAM computation:

- Sets the spatial scale for detecting rotation
- Should be ~1-2 times the Rossby radius
- Smaller: detects smaller eddies, more noise
- Larger: misses small eddies, smoother fields

**window_fit** controls contour extraction:

- Sets the search domain for SSH contours
- Should be ~5-10 times the Rossby radius
- Too small: misses eddy boundaries
- Too large: includes multiple eddies

Practical guidelines:

.. code-block:: python

    # Typical values by latitude
    if lat > 40:  # High latitudes
        window_center = 30   # Smaller Rossby radius
        window_fit = 80
    elif lat > 20:  # Mid-latitudes
        window_center = 50
        window_fit = 120
    else:  # Tropics/subtropics
        window_center = 70   # Larger Rossby radius
        window_fit = 180

Minimum Radius
~~~~~~~~~~~~~~

**min_radius** filters sub-mesoscale features:

- Typically set to ~Rossby radius of deformation
- Too small: many false positives from noise
- Too large: misses genuine small eddies

.. code-block:: python

    # Conservative (fewer false positives)
    min_radius = 20  # km

    # Permissive (captures smaller features)
    min_radius = 10  # km

Ellipse Error
~~~~~~~~~~~~~

**ellipse_error** controls shape validation:

- Maximum deviation from perfect ellipse
- Default 0.05 (5%) for satellite data
- Increase for noisier model output

.. code-block:: python

    # Satellite altimetry (clean)
    ellipse_error = 0.05

    # High-resolution model (noisier)
    ellipse_error = 0.10

Quality Filtering
-----------------

shoot applies several quality checks:

Contour Validity
~~~~~~~~~~~~~~~~

- Requires at least one closed SSH contour
- Contour must enclose the detected center
- Maximum velocity contour must exist

Spatial Validity
~~~~~~~~~~~~~~~~

- Minimum radius threshold (``min_radius``)
- Maximum radius check (avoids unrealistic giants)
- Coastal proximity filter (if land mask available)

Shape Validity
~~~~~~~~~~~~~~

- Ellipse fit quality (``ellipse_error``)
- Aspect ratio constraints (avoids elongated features)
- Circularity check on velocity contour

Uniqueness
~~~~~~~~~~

- No overlapping eddies of same type
- If overlap exists, keeps stronger LNAM
- Prevents double-counting

Advanced Detection
------------------

3D Detection
~~~~~~~~~~~~

Detect eddies across multiple depth levels:

.. code-block:: python

    from shoot.eddies.eddies3d import EddiesByDepth

    # Data with depth dimension
    ds = xr.open_dataset("3d_velocity.nc")

    # Detect at all depths and associate vertically
    eddies_3d = EddiesByDepth.detect_eddies_3d(
        ds.u,                    # 3D zonal velocity
        ds.v,                    # 3D meridional velocity
        window_center=50,
        window_fit=120,
        max_distance=10          # Max vertical center shift (km)
    )

    # Access eddies by depth
    for depth, eddies_z in eddies_3d.eddies3d.items():
        print(f"Depth {depth} m: {len(eddies_z.eddies)} eddies")

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Enable parallel detection for large grids:

.. code-block:: python

    eddies = Eddies2D.detect_eddies(
        ds.u, ds.v,
        window_center=50,
        window_fit=120,
        paral=True              # Enable parallelization
    )

.. warning::
    Parallel mode uses numba parallel features. Test on small domains first
    as it may miss eddies near domain boundaries.

Custom Grid Spacing
~~~~~~~~~~~~~~~~~~~

Explicitly provide grid spacing:

.. code-block:: python

    eddies = Eddies2D.detect_eddies(
        ds.u, ds.v,
        window_center=50,
        window_fit=120,
        dx=10000,               # Grid spacing in x (m)
        dy=10000                # Grid spacing in y (m)
    )

If not provided, spacing is computed from lat/lon coordinates.

Working with Model Output
--------------------------

CROCO Models
~~~~~~~~~~~~~~~~~

shoot supports native CROCO grids through xoa:

.. code-block:: python

    from shoot import meta as smeta

    # Load CROCO output
    ds = xr.open_dataset("croco_output.nc")

    # xoa automatically handles rho-point grids
    u = smeta.get_u(ds)
    v = smeta.get_v(ds)
    ssh = smeta.get_ssh(ds)

    # Detection works normally
    eddies = Eddies2D.detect_eddies(u, v, 50, 120, ssh=ssh)

NEMO Models
~~~~~~~~~~~

NEMO output works directly with CF conventions:

.. code-block:: python

    ds = xr.open_dataset("nemo_output.nc")

    # Variables follow CF standard names
    eddies = Eddies2D.detect_eddies(
        ds.uo,      # zonal velocity
        ds.vo,      # meridional velocity
        50, 120,
        ssh=ds.zos  # sea surface height
    )

Accessing Results
-----------------

Eddy Properties
~~~~~~~~~~~~~~~

Each detected eddy has:

.. code-block:: python

    for eddy in eddies.eddies:
        # Position
        print(f"Center: ({eddy.lon}, {eddy.lat})")

        # Classification
        print(f"Type: {eddy.eddy_type}")  # 'cyclone' or 'anticyclone'

        # Size/intensity
        print(f"Radius: {eddy.radius} km")
        print(f"Rossby number: {eddy.ro}")

        # Shape
        print(f"Ellipse: a={eddy.ellipse.a}, b={eddy.ellipse.b}")
        print(f"Orientation: {eddy.ellipse.theta}°")

        # Contours
        vmax_lon = eddy.vmax_contour.lon
        vmax_lat = eddy.vmax_contour.lat
        print(f"Max velocity contour: {len(vmax_lon)} points")

Converting to Dataset
~~~~~~~~~~~~~~~~~~~~~

Export to xarray Dataset:

.. code-block:: python

    # Convert all eddies to dataset
    ds_eddies = eddies.ds

    # Contains
    print(ds_eddies.lon)      # Eddy centers
    print(ds_eddies.radius)   # Radii
    print(ds_eddies.ro)       # Rossby numbers

    # Save to NetCDF
    ds_eddies.to_netcdf("detected_eddies.nc")

Troubleshooting
---------------

No Eddies Detected
~~~~~~~~~~~~~~~~~~

If no eddies are found:

1. Check velocity magnitudes are sufficient (> 0.01 m/s typically)
2. Verify grid resolution can resolve eddies
3. Try reducing ``min_radius``
4. Check data has valid coordinates

Too Many False Positives
~~~~~~~~~~~~~~~~~~~~~~~~~

If detecting noise:

1. Increase ``min_radius``
2. Decrease ``ellipse_error``
3. Check ``window_center`` isn't too small
4. Verify SSH field quality if provided

Coastal Contamination
~~~~~~~~~~~~~~~~~~~~~

To avoid coastal eddies:

.. code-block:: python

    # Provide land mask
    eddies = Eddies2D.detect_eddies(
        ds.u, ds.v,
        window_center=50,
        window_fit=120,
        mask=ds.mask    # 1=ocean, 0=land
    )

Or filter by distance from coast in post-processing.

Performance Issues
~~~~~~~~~~~~~~~~~~

For large domains:

- Process subregions separately
- Use coarser resolution for initial detection
- Enable ``paral=True`` (with caution)
- Consider data chunking with dask

Further Reading
---------------

- :ref:`algos.eddies` - Algorithm mathematical details
- :ref:`indepth_eddies_tracking` - Tracking detected eddies
- :ref:`quickstart` - Quick detection examples
- :ref:`lib` - API documentation
