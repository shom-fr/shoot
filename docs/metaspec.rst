.. _metaspec:

Metadata Specification
======================

shoot includes internal metadata specifications that extend xoa's capability to recognize ocean variables and coordinates. These specifications are defined in the configuration file ``shoot/meta.cfg``.

Configuration File Location
---------------------------

The internal metadata specifications are located at::

    shoot/meta.cfg

This file is automatically loaded when you import shoot, so you don't need to manually register it.

File Structure
--------------

The configuration file follows the ConfigObj/INI format with nested sections:

.. code-block:: ini

    [register]
    name=shoot

    [data_vars]

        [[variable_name]]
            [[[attrs]]]
            standard_name = comma,separated,standard,names
            long_name = Variable Long Name
            units = variable_units

What It Defines
---------------

The ``shoot/meta.cfg`` file specifies:

Standard Names
~~~~~~~~~~~~~~

Extended lists of CF standard names that xoa should recognize for each variable type:

**Zonal velocity (u)**:
    - ``sea_water_x_velocity``
    - ``eastward_sea_water_velocity``
    - ``surface_sea_water_x_velocity``
    - ``geostrophic_eastward_sea_water_velocity``
    - ``surface_geostrophic_eastward_sea_water_velocity``
    - ``surface_geostrophic_sea_water_x_velocity``
    - ``surface_geostrophic_eastward_sea_water_velocity_assuming_mean_sea_level_for_geoid``
    - ``sea_water_x_velocity_at_u_location``

**Meridional velocity (v)**:
    - ``sea_water_y_velocity``
    - ``northward_sea_water_velocity``
    - ``surface_sea_water_y_velocity``
    - ``geostrophic_northward_sea_water_velocity``
    - ``surface_geostrophic_northward_sea_water_velocity``
    - ``surface_geostrophic_sea_water_y_velocity``
    - ``surface_geostrophic_northward_sea_water_velocity_assuming_mean_sea_level_for_geoid``

**Depth**:
    - ``ocean_layer_depth``
    - ``depth_below_geoid``
    - ``altitude``

Alternative Names
~~~~~~~~~~~~~~~~~

Some variables have alternative short names:

**Depth**: ``dep`` (in addition to ``depth``)

Visualization Settings
~~~~~~~~~~~~~~~~~~~~~~

Default colormaps for visualization:

**Depth**: ``cmo.deep`` (from cmocean)

How It Works
------------

Automatic Registration
~~~~~~~~~~~~~~~~~~~~~~

When you import shoot, the metadata specifications are automatically registered with xoa:

.. code-block:: python

    import shoot
    # shoot/meta.cfg is automatically loaded

This means all the standard names defined in ``meta.cfg`` are immediately available for variable detection.

Using the Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~

The specifications work seamlessly with shoot's metadata functions:

.. code-block:: python

    from shoot import meta as smeta
    import xarray as xr

    ds = xr.open_dataset("ocean_data.nc")

    # These functions use the specifications from meta.cfg
    u = smeta.get_u(ds)  # Recognizes all u standard names
    v = smeta.get_v(ds)  # Recognizes all v standard names
    depth = smeta.get_depth(ds)  # Recognizes depth standard names

Variable Recognition Priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xoa searches for variables in this order:

1. Variables with matching **CF standard_name** attribute (from meta.cfg)
2. Variables with matching **name** (including alt_names from meta.cfg)
3. Variables with matching **long_name** patterns
4. Fallback to positional detection if enabled

Extending the Specifications
-----------------------------

Creating Custom Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create your own metadata specification file for specific datasets or models:

.. code-block:: ini

    # mydata.cfg
    [register]
    name=mydata

    [data_vars]

        [[u]]
            alt_names = u_vel, uvel, U
            [[[attrs]]]
            standard_name = my_custom_eastward_velocity

        [[ssh]]
            alt_names = sea_level, zeta, eta
            [[[attrs]]]
            standard_name = my_ssh_name

Loading Custom Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From Python:

.. code-block:: python

    from shoot import meta as smeta

    # Load your custom specifications
    smeta.set_meta_specs("mydata.cfg")

    # Now xoa recognizes your custom names
    u = smeta.get_u(ds)  # Finds 'u_vel', 'uvel', or 'U'

From Command Line:

.. code-block:: bash

    shoot eddies detect input.nc \
        --meta-file mydata.cfg \
        -o output.nc

Combining Specifications
~~~~~~~~~~~~~~~~~~~~~~~~

You can combine multiple specification files:

.. code-block:: python

    # Use both shoot and custom specifications
    smeta.register_meta_specs()  # Load shoot/meta.cfg
    smeta.set_meta_specs("mydata.cfg")  # Add custom specs

    # xoa now recognizes both sets of names

Full File Content
-----------------

The complete ``shoot/meta.cfg`` file:

.. literalinclude:: ../shoot/meta.cfg
   :language: ini
   :linenos:

Best Practices
--------------

1. **Use standard CF names** when possible - they're widely recognized
2. **Add alt_names** for common non-standard abbreviations in your data
3. **Document custom specs** - include comments explaining unusual names
4. **Test recognition** - verify xoa finds your variables:

   .. code-block:: python

       # Check if variable is found
       u = smeta.get_u(ds, errors='ignore')
       if u is None:
           print(f"Available variables: {list(ds.data_vars)}")
           print("Update meta.cfg or use --meta-file")

5. **Share specifications** - if you create specs for a common model/dataset, contribute them back

Related Documentation
---------------------

- :ref:`indepth_metadata` - Complete metadata handling guide
- :ref:`quickstart` - Using metadata in practice
- xoa documentation - Underlying metadata framework

See Also
--------

- `CF Standard Names <http://cfconventions.org/standard-names.html>`_
- `CF Conventions <http://cfconventions.org/>`_
- ConfigObj documentation for file format details
