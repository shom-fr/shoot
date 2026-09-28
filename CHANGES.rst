What's new
##########

Develop
=======

New features
------------
- New :mod:`shoot.core` subpackage of pure numeric routines, on which the other
  modules are xarray interfaces.
- Eddy detection is 10 to 30 times faster: numba ellipse fit, contour tests
  in grid-index space and less xarray overhead.
- Eddy tracking is about 30 times faster with a numba cost matrix.
- Effective parallel detection over time steps or eddy centers, with the new
  :mod:`shoot.paral` module; ``paral=None`` (default) chooses automatically.

Breaking changes
----------------
- The ellipse fit converges better, so detected eddies may slightly differ.
- ``shoot.num.get_coord_name`` is replaced by :func:`shoot.meta.get_lon_lat_names`.
- The private functions ``shoot.dyn._get_*_`` are replaced by public ones
  in :mod:`shoot.core.dyn`.

Deprecations
------------
- ``shoot.fit._residuals`` is an alias of :func:`shoot.core.fit.ellipse_residuals`.

Bug fixes
---------
- The ellipse fit no longer fails on contours reduced to a point.
- ``numba`` and ``threadpoolctl`` are declared as dependencies.

Documentation
-------------
- The library reference separates the xarray interface from the numeric core.


YYYY-0M-MICRO
=============

New features
------------

Breaking changes
----------------

Deprecations
------------

Bug fixes
---------

Documentation
-------------

