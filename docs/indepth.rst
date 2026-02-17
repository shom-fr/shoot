.. _indepth:

In-Depth Guide
##############

This section provides comprehensive explanations of shoot's core features for detecting and tracking various ocean objects. Each guide includes conceptual background, best practices, and advanced usage patterns.

.. toctree::
   :maxdepth: 2

   indepth_eddies
   indepth_fronts
   indepth_metadata
   indepth_profiles

Overview
========

shoot (**SHom Ocean Objects Tracker**) is designed for detecting and tracking mesoscale and submesoscale ocean features from gridded datasets. It provides tools for:

**Ocean Objects**:

1. **Eddies** - Rotating coherent structures (cyclones and anticyclones)
2. **Fronts** - Sharp gradients in temperature, salinity, or density
3. **River plumes** - Fresh water intrusions from river outflows
4. **Solitary waves** - Nonlinear internal waves

**Analysis Capabilities**:

- Detection from velocity, SSH, and scalar fields
- Tracking through time with optimal matching
- 3D analysis across depth levels
- Association with in-situ observations
- Property anomaly computation

Core Concepts
=============

Object Detection
----------------

shoot provides different detection methods optimized for each object type:

**Rotating structures (eddies)**:
   - LNAM (Local Normalized Angular Momentum) method
   - Okubo-Weiss parameter for vortex validation
   - SSH contour extraction
   - Ellipse fitting for shape characterization

**Gradient features (fronts)**:
   - Temperature/salinity gradient computation
   - Canny edge detection
   - Contour following algorithms
   - Front strength characterization

**Custom methods**:
   - Extensible framework for new object types
   - Modular detection pipeline

See object-specific guides for detailed methodologies.

Object Tracking
---------------

The tracking framework works across all object types:

- **Cost function** - Spatial and property similarity
- **Hungarian algorithm** - Optimal matching between time steps
- **Track objects** - Maintain trajectories over time
- **Evolution metrics** - Track property changes

Key features:

- Handles object birth, death, merging, and splitting
- Configurable similarity metrics per object type
- Multi-scale tracking (surface and subsurface)

Metadata System
---------------

shoot uses **xoa** for CF-compliant metadata handling:

- Automatic coordinate detection (lon, lat, depth, time)
- Standard name search (velocities, SSH, temperature, salinity)
- Model-specific support (CROCO grids)
- Dimension inference for flexible data structures

This ensures compatibility with:

- Satellite observations (altimetry, SST)
- Ocean models (CROCO, NEMO, MOM)
- Reanalysis products (GLORYS, ORAS)
- In-situ data (Argo, moorings, gliders)

See :ref:`indepth_metadata` for complete details.

Profile Association
-------------------

For coupling with in-situ observations:

- **Profile loading** - Download Argo data for spatiotemporal domain
- **Object association** - Identify profiles inside/outside features
- **Anomaly computation** - Calculate property differences
- **Impact analysis** - Assess effects (e.g., acoustic, biogeochemical)

See :ref:`indepth_profiles` for complete details.

Key Principles
==============

Coordinate Systems
------------------

shoot works with:

- **Geographic coordinates** - Longitude/latitude in degrees
- **Metric distances** - Conversions using Earth radius and latitude
- **Grid-relative coordinates** - For model native grids
- **Curvilinear grids** - Full support through xoa

Spatial Scales
--------------

Detection parameters depend on target feature:

**Mesoscale eddies**:
   - Typical size: 50-200 km diameter
   - Detection window: ~Rossby radius (20-50 km)
   - Minimum size: ~Rossby radius

**Fronts**:
   - Typical width: 1-20 km
   - Detection scale: Sub-mesoscale resolution
   - Minimum gradient threshold

**Submesoscale features**:
   - Typical size: 1-10 km
   - Requires high-resolution data (~1 km)
   - Short lifetimes (hours to days)

Temporal Scales
---------------

Tracking considerations:

**Mesoscale eddies**:
   - Lifetime: Weeks to months
   - Advection: ~5-10 km/day
   - Sampling: Daily to weekly

**Fronts**:
   - Lifetime: Days to weeks
   - Propagation: Variable (0-50 km/day)
   - Sampling: Sub-daily to daily

**Submesoscale**:
   - Lifetime: Hours to days
   - Evolution: Rapid
   - Sampling: Hourly to daily

Data Requirements
-----------------

Minimum requirements vary by object type:

**For eddies**:
   - Velocity fields (U, V) or SSH
   - Resolution: ~5-25 km
   - Temporal: Daily to weekly

**For fronts**:
   - Temperature and/or salinity fields
   - Resolution: ~1-10 km
   - Temporal: Sub-daily to daily

**General**:
   - CF-compliant coordinate information
   - Sufficient spatial coverage
   - Consistent temporal sampling

Quality Control
===============

Detection Quality
-----------------

Object-specific quality filters:

**Eddies**:
   1. Minimum radius threshold
   2. Ellipticity constraints
   3. Contour closure requirement
   4. Coastal masking
   5. Overlap prevention

**Fronts**:
   1. Minimum gradient threshold
   2. Continuity requirements
   3. Length constraints
   4. Orientation consistency

Tracking Quality
----------------

Quality indicators for tracks:

- **Match distance** - Realistic displacement
- **Property consistency** - Smooth evolution
- **Track duration** - Minimum lifetime
- **Trajectory smoothness** - Physical propagation

Validation
----------

Recommended validation steps:

1. Visual inspection of detected objects
2. Size/intensity distribution analysis
3. Tracking statistics (lifetime, displacement)
4. Comparison with known features
5. Sensitivity analysis to parameters

Getting Started
===============

For comprehensive guides on specific ocean objects:

1. :ref:`indepth_eddies` - Mesoscale eddy detection and tracking
2. :ref:`indepth_fronts` - Oceanic front identification (coming soon)

For supporting systems:

3. :ref:`indepth_metadata` - Working with xoa and CF conventions
4. :ref:`indepth_profiles` - In-situ data integration

For algorithm details:

- :ref:`algos.eddies` - Mathematical description of eddy detection and tracking

For practical examples:

- :ref:`quickstart` - Getting started quickly
- :ref:`examples <examples>` - Gallery of use cases

For API reference:

- :ref:`lib` - Complete function and class documentation
