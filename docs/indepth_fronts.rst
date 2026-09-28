.. _indepth_fronts:

Fronts
======

.. note::
   Front detection is available. Front tracking is planned for future releases of shoot.

Overview
--------

Ocean fronts are sharp transitions in water properties (temperature, salinity, density) that occur at various scales throughout the ocean. shoot provides or will provide tools to:

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

Detection Methods
-----------------

Fronts are detected in a 2D field, like sea surface temperature, with
:func:`shoot.fronts.fronts2d.detect_fronts`, which returns a boolean front mask:

.. code-block:: python

    from shoot.fronts.fronts2d import detect_fronts

    fronts = detect_fronts(ds.thetao, method="cca", min_pop_mean_diff=0.4)

Four methods are available:

``"cca"``
    Cayula and Cornillon (1992) single image edge detector: the histogram of
    overlapping windows is split into two populations, and a front is the
    contour between them when they are large, distinct and spatially cohesive
    enough (:func:`shoot.core.fronts.cca_window`). Requires 1D coordinates.

``"cca_sliding"``
    The same criteria on windows centered every ``step`` grid points.

``"boa"``
    Belkin and O'Reilly (2009): gradient magnitude after a contextual median
    filter that preserves the extrema, greater than a ``threshold``.

``"canny"``
    Canny edge detector with a Gaussian smoothing of standard deviation ``sigma``.

The numeric algorithms are in :mod:`shoot.core.fronts` and generic image
processing routines in :mod:`shoot.core.image`.
See the example :ref:`sphx_glr_examples_plot_cmcc_front.py`.

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

Contributing
------------

If you're interested in front detection and tracking capabilities:

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

Front tracking is a planned feature for shoot. Check:

- Project repository for updates
- Release notes for new versions
- Issue tracker for development progress

For now, focus on:

- :ref:`indepth_eddies` - Eddy detection and tracking
- :ref:`indepth_metadata` - Data handling
- :ref:`quickstart` - Getting started with shoot

Questions?
----------

If you have specific needs for front detection or tracking:

- Open an issue on the repository
- Describe your use case
- Share example data if possible
- Suggest algorithms or references

Your input helps shape future development!
