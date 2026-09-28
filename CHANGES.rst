What's new
##########

Develop
=======

New features
------------
- New :mod:`shoot.core` subpackage of pure numeric routines (numpy arrays only,
  no xarray nor xoa), on which the other modules are xarray interfaces
  (see ``mds/core-subpackage-plan.md``):
  :mod:`~shoot.core.contours`, :mod:`~shoot.core.dyn`, :mod:`~shoot.core.eddies`,
  :mod:`~shoot.core.fit`, :mod:`~shoot.core.geo`, :mod:`~shoot.core.num`
  and :mod:`~shoot.core.track`.
  :mod:`shoot.geo`, :mod:`shoot.num`, :mod:`shoot.fit` and :mod:`shoot.dyn`
  re-export their former content from it.
- 2D eddy detection is about 20 times faster on large fields
  (see ``mds/speedup-eddy-detection.md``):

  - the ellipse fit (:func:`shoot.core.fit.fit_ellipse`) uses a numba
    Levenberg-Marquardt solver with an analytical Jacobian
    instead of :func:`scipy.optimize.least_squares`;
  - closed contours are found by :func:`shoot.core.contours.find_closed_contours`,
    which performs the closed, center-inclusion and land-inclusion tests in
    grid-index space, so that only the retained contours are interpolated to lon/lat;
  - the per-center contour processing (:func:`shoot.core.eddies.find_eddy_contours`)
    and the removal of intersecting eddies (:func:`shoot.core.eddies.filter_intersecting`)
    are pure numeric, and contour datasets are built once;
  - :class:`shoot.eddies.eddies2d.GriddedEddy2D` accepts ``lon2d`` and ``lat2d``
    to avoid inferring coordinates for each sub-window.
- Eddy tracking and association are about 25 times faster thanks to
  the numba cost matrix :func:`shoot.core.track.association_cost`.
- Faster parallel eddy detection (see ``mds/tutorials-speedup.md``):

  - :meth:`~shoot.eddies.eddies2d.EvolEddies2D.detect_eddies` processes the
    time steps in parallel (about 4 times faster than sequentially with 8 cores);
  - :meth:`~shoot.eddies.eddies2d.Eddies2D.detect_eddies` uses a single pool of
    workers per detection instead of one per window iteration, or a shared one
    given with the new ``pool`` argument;
  - new generic :mod:`shoot.paral` module: :func:`~shoot.paral.create_pool` creates
    workers limited to one BLAS and numba thread to avoid oversubscribing the cores,
    after compiling numba kernels with one or several warmup functions
    like :func:`shoot.core.eddies.warmup`;
  - ``paral=None`` (new default) automatically switches parallelism on with the
    "fork" start method: over time steps when there are at least
    :data:`~shoot.eddies.eddies2d.PARAL_MIN_TIMES` of them, and over centers
    when the sequential time estimated from a sample of candidates exceeds
    :data:`~shoot.eddies.eddies2d.PARAL_MIN_SECONDS`.

Breaking changes
----------------
- The ellipse fit now converges on nearly circular contours where
  scipy used to stop early, so fit errors are lower or equal and a few
  more contours pass the ``ellipse_error`` threshold: detected eddies
  may slightly differ.
- ``shoot.num.get_coord_name`` is replaced by :func:`shoot.meta.get_lon_lat_names`,
  which relies on :func:`shoot.meta.get_lon` and :func:`shoot.meta.get_lat`
  (xoa) instead of a name-prefix heuristic.
- The private numeric functions of :mod:`shoot.dyn` (``_get_lnam_``, ``_get_div_``…)
  are replaced by public ones in :mod:`shoot.core.dyn` (:func:`~shoot.core.dyn.lnam`,
  :func:`~shoot.core.dyn.div`…).

Deprecations
------------
- ``shoot.fit._residuals`` is an alias of :func:`shoot.core.fit.ellipse_residuals`.

Bug fixes
---------
- ``numba`` and ``threadpoolctl`` are declared as dependencies.
- The ellipse fit no longer fails on degenerate contours reduced to a point:
  it returns NaN parameters with an infinite error so that they are rejected.

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

