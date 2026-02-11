.. _indepth_fronts:

Fronts (Coming Soon)
====================

.. note::
   This section is under development. Front detection and tracking capabilities
   are planned for future releases of shoot.

Overview
--------

Ocean fronts are sharp transitions in water properties (temperature, salinity, density) that occur at various scales throughout the ocean. shoot will provide tools to:

- Detect fronts from temperature/salinity fields
- Track front evolution through time
- Characterize front strength and structure
- Analyze frontal impacts on ecosystems

What are Ocean Fronts?
-----------------------

Ocean fronts are regions of enhanced horizontal gradients in:

- **Temperature** - Thermal fronts
- **Salinity** - Haline fronts
- **Density** - Density fronts (combination of T and S)

Types of Fronts
~~~~~~~~~~~~~~~

**Boundary Current Fronts**:
   - Associated with major currents (Gulf Stream, Kuroshio)
   - Strong, persistent features
   - Width: 10-50 km
   - Temperature contrast: 5-10°C

**Upwelling Fronts**:
   - Coastal upwelling regions
   - Separate cold upwelled from warm offshore water
   - Width: 5-20 km
   - Seasonal variability

**Tidal Fronts**:
   - Form due to tidal mixing
   - Separate stratified from mixed water
   - Width: 1-10 km
   - Associated with shelf seas

**Frontal Eddies**:
   - Meanders and eddies along fronts
   - Width: 10-100 km
   - Dynamic, evolving features

Characteristics
~~~~~~~~~~~~~~~

**Spatial scales**:
   - Width: 1-50 km typically
   - Length: Can extend 100s-1000s km
   - Vertical extent: Surface to thermocline

**Temporal scales**:
   - Persistence: Days to months
   - Evolution: Can meander and generate eddies
   - Seasonal cycles common

**Physical properties**:
   - Horizontal gradient: >0.1°C/km for temperature
   - Cross-front flow: Convergent or divergent
   - Along-front jets: Often present

Importance
~~~~~~~~~~

Fronts are important because they:

- Concentrate nutrients and biology
- Support enhanced productivity
- Affect fisheries (aggregation zones)
- Influence air-sea interaction
- Generate submesoscale features
- Impact navigation and marine operations

Planned Detection Methods
--------------------------

shoot will implement multiple front detection techniques:

Gradient-Based Detection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Planned API (not yet implemented)
    from shoot.fronts import detect_fronts

    # Detect from temperature field
    fronts = detect_fronts(
        ds.temp,
        method='gradient',
        threshold=0.1,        # °C/km
        min_length=20         # km
    )

Edge Detection
~~~~~~~~~~~~~~

Using Canny or similar algorithms:

.. code-block:: python

    # Planned API
    fronts = detect_fronts(
        ds.temp,
        method='canny',
        sigma=2,              # Smoothing scale
        min_length=20
    )

Contour-Based
~~~~~~~~~~~~~

Following specific isotherms/isohalines:

.. code-block:: python

    # Planned API
    fronts = detect_fronts(
        ds.temp,
        method='contour',
        values=[15, 20],      # Isotherms to track
        gradient_threshold=0.1
    )

Planned Tracking
----------------

Track front positions through time:

.. code-block:: python

    # Planned API
    from shoot.fronts import track_fronts

    # Track detected fronts
    tracks = track_fronts(
        fronts_list,
        max_distance=30,      # km
        max_angle_change=45   # degrees
    )

Data Requirements
-----------------

For front detection, you will need:

**Temperature and/or Salinity**:
   - High-resolution fields (< 10 km)
   - SST from satellites (1-4 km)
   - Model output with fine resolution

**Spatial Coverage**:
   - Large enough to capture front extent
   - Avoid domain edge artifacts

**Temporal Resolution**:
   - Sub-daily to daily for tracking
   - Sufficient to capture evolution

Interim Solutions
-----------------

While native front detection is under development, you can:

Use Custom Gradient Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import xarray as xr
    import numpy as np
    from shoot import meta as smeta

    # Load data
    ds = xr.open_dataset("sst_data.nc")

    # Get coordinates
    lon = smeta.get_lon(ds)
    lat = smeta.get_lat(ds)

    # Compute gradients
    dT_dx = ds.temp.differentiate('lon')
    dT_dy = ds.temp.differentiate('lat')

    # Gradient magnitude
    grad_mag = np.sqrt(dT_dx**2 + dT_dy**2)

    # Threshold for fronts
    front_mask = grad_mag > 0.1  # Adjust threshold

Use External Tools
~~~~~~~~~~~~~~~~~~

Consider other packages for front detection:

- **oceanspy** - Has front detection utilities
- **xgcm** - Grid operations for gradient computation
- **scikit-image** - Edge detection algorithms
- **OpenCV** - Computer vision techniques

Then use shoot's tracking framework to follow detected features.

Contributing
------------

If you're interested in front detection capabilities:

- Check the shoot repository for development status
- Open an issue to discuss requirements
- Contribute code following the :ref:`contributing` guidelines

We welcome contributions for:

- Detection algorithms
- Validation datasets
- Test cases
- Documentation

Stay Tuned
----------

Front detection is a planned feature for shoot. Check:

- Project repository for updates
- Release notes for new versions
- Issue tracker for development progress

For now, focus on:

- :ref:`indepth_eddies` - Eddy detection and tracking
- :ref:`indepth_metadata` - Data handling
- :ref:`quickstart` - Getting started with shoot

Questions?
----------

If you have specific needs for front detection:

- Open an issue on the repository
- Describe your use case
- Share example data if possible
- Suggest algorithms or references

Your input helps shape future development!
