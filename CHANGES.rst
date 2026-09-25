What's new
##########

Develop
=======

New features
------------
- 2D eddy detection is about 15 times faster on large fields:

  - :func:`shoot.fit.fit_ellipse_from_coords` uses a numba
    Levenberg-Marquardt solver with an analytical Jacobian
    instead of :func:`scipy.optimize.least_squares`;
  - :func:`shoot.contours.core_find_closed_contours` performs the closed,
    center-inclusion and land-inclusion tests in grid-index space,
    so that only the retained contours are interpolated to lon/lat
    by :func:`shoot.contours.contour_to_dataset`;
  - :class:`shoot.eddies.eddies2d.GriddedEddy2D` accepts ``lon2d`` and ``lat2d``
    to avoid inferring coordinates for each sub-window;
  - :func:`shoot.contours.add_contour_uv`, :func:`shoot.contours.add_contour_dx_dy`
    and ``GriddedEddy2D.intersects_eddy`` avoid xarray overhead;
  - new numba functions :func:`shoot.num.point_in_polygon` and
    :func:`shoot.num.any_points_in_polygon`.

Breaking changes
----------------
- The ellipse fit now converges on nearly circular contours where
  scipy used to stop early, so fit errors are lower or equal and a few
  more contours pass the ``ellipse_error`` threshold: detected eddies
  may slightly differ.
- ``shoot.num.get_coord_name`` is replaced by :func:`shoot.meta.get_lon_lat_names`,
  which relies on :func:`shoot.meta.get_lon` and :func:`shoot.meta.get_lat`
  (xoa) instead of a name-prefix heuristic.

Deprecations
------------

Bug fixes
---------
- The ellipse fit no longer fails on degenerate contours reduced to a point:
  it returns NaN parameters with an infinite error so that they are rejected.

Documentation
-------------


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

