#!/usr/bin/env python3
"""
Optimization and fitting routines

The routines live in :mod:`shoot.core.fit` and are re-exported here.
"""

from .core.dyn import GRAVITY, OMEGA  # noqa: F401
from .core.fit import ellipse_residuals, fit_ellipse, fit_ellipse_from_coords  # noqa: F401

#: Backward compatible alias of :func:`shoot.core.fit.ellipse_residuals`
_residuals = ellipse_residuals
