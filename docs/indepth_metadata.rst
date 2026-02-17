.. _indepth_metadata:

Metadata and CF Conventions
============================

This guide explains how shoot uses **xoa** for metadata handling and CF convention support, enabling seamless work with diverse ocean datasets.

Overview
--------

shoot uses the xoa package (located in ``xoa/`` directory) for:

- **Coordinate detection** - Automatically find lon, lat, depth, time
- **Variable search** - Locate variables by CF standard names
- **Model support** - Handle CROCO and other model grids
- **Flexible data** - Work with various naming conventions

This approach allows shoot to work with:

- Satellite altimetry (DUACS, AVISO)
- Ocean models (CROCO, NEMO, MOM)
- Reanalysis products (GLORYS, ORAS)
- In-situ data (Argo, moorings)

Using shoot.meta
----------------

shoot provides a simplified interface to xoa in ``shoot.meta``:

Basic Coordinate Access
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from shoot import meta as smeta
    import xarray as xr

    ds = xr.open_dataset("ocean_data.nc")

    # Get coordinates
    lon = smeta.get_lon(ds)
    lat = smeta.get_lat(ds)
    depth = smeta.get_depth(ds, errors='ignore')  # May not exist
    time = smeta.get_time(ds, errors='ignore')

    print(f"Longitude: {lon.name}")
    print(f"Latitude: {lat.name}")

Variable Access
~~~~~~~~~~~~~~~

.. code-block:: python

    # Find variables by function
    u = smeta.get_u(ds)        # Zonal velocity
    v = smeta.get_v(ds)        # Meridional velocity
    ssh = smeta.get_ssh(ds)    # Sea surface height
    temp = smeta.get_temp(ds, errors='ignore')
    salt = smeta.get_salt(ds, errors='ignore')

    # These work regardless of actual variable names in file

Dimension Names
~~~~~~~~~~~~~~~

.. code-block:: python

    # Get dimension names from a data array
    xdim = smeta.get_xdim(ds.u)  # e.g., 'lon' or 'xi_rho'
    ydim = smeta.get_ydim(ds.u)  # e.g., 'lat' or 'eta_rho'
    zdim = smeta.get_zdim(ds.temp, errors='ignore')

    print(f"Data dimensions: {ydim}, {xdim}")

How xoa Works
-------------

CF Standard Names
~~~~~~~~~~~~~~~~~

xoa searches for variables using CF standard names:

.. code-block:: python

    # Velocity standard names
    # - eastward_sea_water_velocity (u)
    # - northward_sea_water_velocity (v)

    # SSH standard names
    # - sea_surface_height_above_geoid (ssh)
    # - sea_surface_height_above_sea_level (adt, sla)

    # Temperature/salinity
    # - sea_water_potential_temperature
    # - sea_water_salinity

When a variable has the correct standard name, xoa finds it automatically.

Coordinate Detection
~~~~~~~~~~~~~~~~~~~~

xoa detects coordinates through:

1. **Standard names** - e.g., longitude, latitude
2. **Axis attributes** - e.g., axis='X', axis='Y'
3. **Units** - e.g., degrees_east, degrees_north
4. **Name patterns** - e.g., lon, longitude, nav_lon
5. **Position** - Last dimension often X, second-to-last often Y

Positional Fallback
~~~~~~~~~~~~~~~~~~~

If CF metadata is missing:

.. code-block:: python

    # Fallback to positional
    xdim = smeta.get_xdim(da, allow_positional=True)
    # Returns last dimension if no CF axis found

Custom Specifications
~~~~~~~~~~~~~~~~~~~~~

xoa uses configuration files to define additional patterns.

**Internal Specifications**

shoot includes its own metadata specifications in ``shoot/meta.cfg`` that are automatically loaded on import. These specifications define extended lists of CF standard names for ocean variables (velocities, SSH, depth, etc.).

See :ref:`metaspec` for complete details on the internal specifications and how to create your own.

**Using Specifications**

.. code-block:: python

    # shoot's internal specs are automatically loaded
    # when you import shoot

    # Or set specific specs for your data
    smeta.set_meta_specs("croco")  # Use CROCO conventions
    smeta.set_meta_specs("mydata.cfg")  # Use custom file

Working with Different Data Sources
------------------------------------

Satellite Altimetry
~~~~~~~~~~~~~~~~~~~

AVISO/DUACS data typically has:

.. code-block:: python

    ds = xr.open_dataset("duacs_product.nc")

    # Standard CF names work directly
    u = smeta.get_u(ds)        # ugos or u
    v = smeta.get_v(ds)        # vgos or v
    ssh = smeta.get_ssh(ds)    # adt or sla

    # Coordinates auto-detected
    lon = smeta.get_lon(ds)    # longitude
    lat = smeta.get_lat(ds)    # latitude
    time = smeta.get_time(ds)  # time

CROCO Models
~~~~~~~~~~~~~~~~~

Native CROCO grids use staggered coordinates:

.. code-block:: python

    ds = xr.open_dataset("croco_his.nc")

    # xoa handles rho/u/v/psi points
    u = smeta.get_u(ds)        # u on u-points (xi_u, eta_u)
    v = smeta.get_v(ds)        # v on v-points (xi_v, eta_v)
    temp = smeta.get_temp(ds)  # temp on rho-points (xi_rho, eta_rho)

    # Gets 2D coordinate arrays
    lon = smeta.get_lon(ds)    # lon_rho(eta_rho, xi_rho)
    lat = smeta.get_lat(ds)    # lat_rho(eta_rho, xi_rho)

    # Vertical coordinate
    s_rho = smeta.get_depth(ds)  # Or compute z from s-levels

NEMO Models
~~~~~~~~~~~

NEMO follows CF closely:

.. code-block:: python

    ds = xr.open_dataset("nemo_output.nc")

    # Direct variable access
    u = smeta.get_u(ds)        # uo or vozocrtx
    v = smeta.get_v(ds)        # vo or vomecrty
    ssh = smeta.get_ssh(ds)    # zos or sossheig
    temp = smeta.get_temp(ds)  # thetao or votemper

    # Standard coordinates
    lon = smeta.get_lon(ds)    # nav_lon or glamt
    lat = smeta.get_lat(ds)    # nav_lat or gphit
    depth = smeta.get_depth(ds)  # depth or deptht

Reanalysis Products
~~~~~~~~~~~~~~~~~~~

GLORYS, ORAS, etc.:

.. code-block:: python

    ds = xr.open_dataset("glorys_subset.nc")

    # Usually CF-compliant
    u = smeta.get_u(ds)        # uo
    v = smeta.get_v(ds)        # vo
    temp = smeta.get_temp(ds)  # thetao
    salt = smeta.get_salt(ds)  # so

Error Handling
--------------

Graceful Failures
~~~~~~~~~~~~~~~~~

Control behavior when variables/coordinates not found:

.. code-block:: python

    # Raise exception (default)
    try:
        depth = smeta.get_depth(ds)
    except Exception as e:
        print(f"No depth found: {e}")

    # Ignore and return None
    depth = smeta.get_depth(ds, errors='ignore')
    if depth is None:
        print("No depth coordinate")

    # Warn but continue
    depth = smeta.get_depth(ds, errors='warn')

Checking Availability
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Check if 3D data available
    has_depth = smeta.get_depth(ds, errors='ignore') is not None

    # Check if SSH available
    has_ssh = smeta.get_ssh(ds, errors='ignore') is not None

    if has_ssh:
        print("Using SSH for detection")
        eddies = Eddies2D.detect_eddies(u, v, 50, 120, ssh=ssh)
    else:
        print("Using streamfunction")
        eddies = Eddies2D.detect_eddies(u, v, 50, 120)

Advanced Usage
--------------

Custom Variable Names
~~~~~~~~~~~~~~~~~~~~~

If your data uses non-standard names:

.. code-block:: python

    # Direct access if you know the names
    u = ds['my_u_variable']
    v = ds['my_v_variable']

    # Or rename to standard names
    ds = ds.rename({
        'my_u_variable': 'u',
        'my_v_variable': 'v',
        'my_lon': 'longitude',
        'my_lat': 'latitude'
    })

    # Then use shoot.meta
    u = smeta.get_u(ds)

Adding Standard Names
~~~~~~~~~~~~~~~~~~~~~

Enhance your dataset:

.. code-block:: python

    # Add CF standard names
    ds.u.attrs['standard_name'] = 'eastward_sea_water_velocity'
    ds.v.attrs['standard_name'] = 'northward_sea_water_velocity'
    ds.ssh.attrs['standard_name'] = 'sea_surface_height_above_geoid'

    # Add axis attributes
    ds.lon.attrs['axis'] = 'X'
    ds.lat.attrs['axis'] = 'Y'
    ds.depth.attrs['axis'] = 'Z'
    ds.time.attrs['axis'] = 'T'

    # Now xoa detection works better
    u = smeta.get_u(ds)

Multi-Grid Models
~~~~~~~~~~~~~~~~~

For models with multiple grids (like CROCO):

.. code-block:: python

    # U and V on different grids
    u_ugrid = ds.u  # On u-points
    v_vgrid = ds.v  # On v-points

    # May need interpolation to common grid
    # (shoot handles this internally for detection)

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # For sigma-coordinate models
    # Compute z-levels from s-levels and SSH
    import xoa.sigma as xsigma

    # Get 3D depth field
    z = xsigma.get_sigma_depths(
        ds,
        ssh=ds.zeta,
        hc=ds.hc,
        s_rho=ds.s_rho,
        Cs_r=ds.Cs_r
    )

Best Practices
--------------

Always Use shoot.meta
~~~~~~~~~~~~~~~~~~~~~

Instead of direct access:

.. code-block:: python

    # Bad - assumes specific names
    lon = ds.longitude
    u = ds.ugos

    # Good - works with any naming
    lon = smeta.get_lon(ds)
    u = smeta.get_u(ds)

Check Data First
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Verify data structure
    print(ds)
    print(ds.attrs)

    # Check coordinate detection
    lon = smeta.get_lon(ds)
    print(f"Found longitude: {lon.name}")
    print(f"  Dims: {lon.dims}")
    print(f"  Shape: {lon.shape}")

Document Assumptions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # If you make assumptions, document them
    # E.g., "This code assumes CF-compliant NetCDF with SSH"

    ssh = smeta.get_ssh(ds)
    if ssh is None:
        raise ValueError("This workflow requires SSH data")

Provide Metadata
~~~~~~~~~~~~~~~~

When creating output files:

.. code-block:: python

    # Add proper metadata
    ds_out = xr.Dataset({
        'eddy_lon': (['eddy'], lons),
        'eddy_lat': (['eddy'], lats),
    })

    # Add standard names
    ds_out.eddy_lon.attrs['standard_name'] = 'longitude'
    ds_out.eddy_lon.attrs['units'] = 'degrees_east'
    ds_out.eddy_lat.attrs['standard_name'] = 'latitude'
    ds_out.eddy_lat.attrs['units'] = 'degrees_north'

    ds_out.to_netcdf("eddies.nc")

Troubleshooting
---------------

"Cannot find longitude"
~~~~~~~~~~~~~~~~~~~~~~~

If coordinate detection fails:

1. Check variable has longitude data
2. Verify dims/coords structure
3. Add axis or standard_name attributes
4. Use direct access as fallback

"Variable not found"
~~~~~~~~~~~~~~~~~~~~

If variable search fails:

1. Print dataset to see actual names
2. Check standard_name attributes
3. Register custom patterns
4. Access variable directly by name

CROCO Grid Issues
~~~~~~~~~~~~~~~~

For CROCO:

1. Ensure xoa is properly installed
2. Check that grid variables (lon_rho, lat_rho, etc.) exist
3. Verify s-coordinate parameters if using 3D
4. May need to specify grid type explicitly

Further Reading
---------------

- :ref:`metaspec` - Internal metadata specification details
- xoa documentation (in ``xoa/`` directory)
- `CF Conventions <http://cfconventions.org/>`_
- :ref:`indepth_eddies_detection` - Using metadata in detection
- :ref:`quickstart` - Basic usage examples
