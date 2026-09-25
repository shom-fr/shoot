.. _lib:
    
Library
=======

Xarray interface
----------------

These modules accept and return :mod:`xarray` objects, and find
coordinates and dimensions from metadata with :mod:`xoa`.

.. autosummary::
    :toctree: api

    shoot
    shoot.acoustic
    shoot.contours
    shoot.dyn
    shoot.eddies
    shoot.eddies.associate
    shoot.eddies.eddies2d
    shoot.eddies.eddies3d
    shoot.eddies.track
    shoot.hydrology
    shoot.fit
    shoot.grid
    shoot.meta
    shoot.num
    shoot.plot
    shoot.profiles
    shoot.profiles.download
    shoot.profiles.profiles

Numeric core
------------

The :mod:`shoot.core` subpackage contains the pure numeric routines on which
the xarray interface relies. They only work with :mod:`numpy` arrays and scalars,
and never import :mod:`xarray` or :mod:`xoa`.
Arrays are of shape ``(ny, nx)``, grid indices are ``(i, j)``,
units are SI and contour lines are ``(n, 2)`` arrays of fractional ``(i, j)`` indices.

.. autosummary::
    :toctree: api

    shoot.core
    shoot.core.contours
    shoot.core.dyn
    shoot.core.eddies
    shoot.core.fit
    shoot.core.geo
    shoot.core.num
    shoot.core.track
