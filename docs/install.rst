Installation
============

.. highlight:: bash

Dependencies
------------

shoot requires ``python>=3.10`` and depends on the following packages:

.. list-table::
   :widths: 10 90

   * - `numpy <https://numpy.org/>`_
     - Numpy is a comprehensive library for scientific computation
   * - `numba <https://numba.pydata.org/>`_
     - A high performance python compiler.
   * - `scipy <https://www.scipy.org/scipylib/index.html>`_
     - Scipy provides many user-friendly and efficient numerical routines,
       such as routines for numerical integration, interpolation,
       optimization, linear algebra, and statistics.
   * - `xarray <http://xarray.pydata.org/en/stable/>`_
     - xarray is an open source project and Python package that makes working
       with labelled multi-dimensional arrays simple, efficient, and fun!
   * - `xoa <https://xoa.readthedocs.io>`_
     - xoa finds coordinates and variables from their metadata.
   * - `contourpy <https://pypi.org/project/contourpy/>`_
     - Contourpy is a library for computing contours on grids
   * - `matplotlib <https://matplotlib.org/>`_
     - Matplotlib is a comprehensive library for creating static, animated,
       and interactive visualizations in Python.
   * - `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_
     - Cartopy is a package for geospatial data processing
   * - `argopy <https://argopy.readthedocs.io>`_
     - Argopy downloads Argo profiles (with ``erddapy<3``).
   * - `netCDF4 <https://unidata.github.io/netcdf4-python/>`_, `psutil <https://psutil.readthedocs.io>`_,
       `threadpoolctl <https://github.com/joblib/threadpoolctl>`_
     - Input/output, memory monitoring and thread control of parallel workers.

Optional dependencies are grouped as follows:

.. list-table::
   :widths: 15 35 50

   * - ``diags``
     - cmocean, dask, gsw
     - Diagnostics of the command line interface
   * - ``samples``
     - pooch
     - Sample data of the examples
   * - ``emodnet``
     - owslib
     - EMODnet bathymetry background of maps
   * - ``colors``
     - colorlog
     - Colored logs
   * - ``all``
     -
     - All of them

A conda environment with all the dependencies is given in ``env/environment.yml``.

From sources
------------

Clone the repository::

    $ git clone https://github.com/shom-fr/shoot.git

Run the installation command from the root directory::

    $ cd shoot
    $ pip install .

or with all the optional dependencies::

    $ pip install ".[all]"
