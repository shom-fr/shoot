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
- Front detection with :func:`shoot.fronts.fronts2d.detect_fronts` (Cayula-Cornillon,
  Belkin-O'Reilly and Canny methods), based on the numeric :mod:`shoot.core.fronts`
  and :mod:`shoot.core.image`, 25 to 60 times faster and without OpenCV.

Breaking changes
----------------
- The ellipse fit converges better, so detected eddies may slightly differ.
- ``shoot.num.get_coord_name`` is replaced by :func:`shoot.meta.get_lon_lat_names`.
- The private functions ``shoot.dyn._get_*_`` are replaced by public ones
  in :mod:`shoot.core.dyn`.
- ``shoot.front.algos`` is replaced by :mod:`shoot.fronts` and :mod:`shoot.core.fronts`,
  with snake_case names.

Deprecations
------------
- ``shoot.fit._residuals`` is an alias of :func:`shoot.core.fit.ellipse_residuals`.

Bug fixes
---------
- The ellipse fit no longer fails on contours reduced to a point.
- ``numba`` and ``threadpoolctl`` are declared as dependencies.
- Front detection: the Cayula-Cornillon histogram includes the maximum values,
  all the front lines of a window are kept, input fields are no longer modified,
  Canny ignores land, and hysteresis follows chains of weak edges.
- Argo profiles: downloaded with argopy's standard mode (adjusted, quality-controlled
  values), and interpolated to depths from valid and sorted levels only.

Documentation
-------------
- The library reference separates the xarray interface from the numeric core.
- Front detection documentation and example.


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
