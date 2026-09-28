#!/usr/bin/env python3
"""
Optimization and fitting routines

Functions for fitting geometric shapes to spatial data.
"""

import numba
import numpy as np

from . import geo as sgeo

# %%
# Ellipse Mean Square fit
# -----------------------


def ellipse_residuals(params, points):
    """Compute residuals between points and ellipse

    Parameters
    ----------
    params : array-like
        Ellipse parameters [xc, yc, a, b, theta].
    points : ndarray
        Point coordinates of shape (n, 2).

    Returns
    -------
    ndarray
        Normalized distances minus 1.
    """
    xc, yc, a, b, theta = params
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shifted = points - [xc, yc]
    rotated = shifted @ R
    normed = rotated / [a, b]
    distances = np.linalg.norm(normed, axis=1)
    return distances - 1


@numba.njit(cache=True)
def _ellipse_residuals_jac(params, x, y, res, jac):
    """Residuals of :func:`ellipse_residuals` and their analytical Jacobian"""
    xc, yc, a, b, theta = params
    c = np.cos(theta)
    s = np.sin(theta)
    for k in range(x.size):
        dx = x[k] - xc
        dy = y[k] - yc
        u = (dx * c + dy * s) / a
        v = (-dx * s + dy * c) / b
        d = np.sqrt(u * u + v * v)
        res[k] = d - 1.0
        if d == 0.0:
            d = 1e-300
        jac[k, 0] = (-u * c / a + v * s / b) / d
        jac[k, 1] = (-u * s / a - v * c / b) / d
        jac[k, 2] = -u * u / a / d
        jac[k, 3] = -v * v / b / d
        jac[k, 4] = u * v * (b / a - a / b) / d
    return 0.5 * np.dot(res, res)


@numba.njit(cache=True)
def fit_ellipse(x, y, maxiter=200, ftol=1e-8, xtol=1e-8, gtol=1e-8):
    """Geometric ellipse fit with a Levenberg-Marquardt algorithm

    Parameters
    ----------
    x, y : ndarray
        Point coordinates in meters.

    Returns
    -------
    params : ndarray
        Ellipse parameters [xc, yc, a, b, theta].
        They are NaN when the points are degenerate (all aligned
        along X or Y, or no point).
    error : float
        Mean squared residual, infinite for degenerate points.
    """
    n = x.size
    p = np.full(5, np.nan)
    if n == 0:
        return p, np.inf
    p[0] = x.mean()
    p[1] = y.mean()
    p[2] = 0.5 * (x.max() - x.min())
    p[3] = 0.5 * (y.max() - y.min())
    p[4] = 0.0
    if not (p[2] > 0 and p[3] > 0):
        p[:] = np.nan
        return p, np.inf
    amax = 5.0 * max(p[2], p[3])  # avoid degenerate "infinite" ellipses
    res = np.empty(n)
    jac = np.empty((n, 5))
    resn = np.empty(n)
    jacn = np.empty((n, 5))
    pn = np.empty(5)
    cost = _ellipse_residuals_jac(p, x, y, res, jac)
    lam = 1e-3
    for it in range(maxiter):
        A = jac.T @ jac
        g = jac.T @ res
        if np.abs(g).max() < gtol:
            break
        dA = np.diag(A).copy()
        for i in range(5):
            if dA[i] <= 0:
                dA[i] = 1e-12
        accepted = False
        for _ in range(30):
            M = A.copy()
            for i in range(5):
                M[i, i] += lam * dA[i]
            delta = np.linalg.solve(M, -g)
            pn[:] = p + delta
            if pn[2] <= 0 or pn[3] <= 0 or pn[2] > amax or pn[3] > amax:
                lam *= 4.0
                continue
            costn = _ellipse_residuals_jac(pn, x, y, resn, jacn)
            if np.isfinite(costn) and costn < cost:
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            break
        dcost = cost - costn
        p[:] = pn
        res[:] = resn
        jac[:] = jacn
        cost = costn
        lam = max(lam / 3.0, 1e-12)
        if dcost < ftol * (cost + dcost):
            break
        if np.sqrt(np.dot(delta, delta)) < xtol * (xtol + np.sqrt(np.dot(p, p))):
            break
    return p, 2.0 * cost / n


def fit_ellipse_from_coords(lons, lats, get_fit=False):
    """Fit ellipse to geographic coordinates

    Uses a numba Levenberg-Marquardt least-squares optimization of
    :func:`ellipse_residuals` to fit an ellipse to a set of points
    given in geographic coordinates.

    Parameters
    ----------
    lons : array-like
        Longitude coordinates in degrees.
    lats : array-like
        Latitude coordinates in degrees.
    get_fit : bool, default False
        If True, return fit error along with parameters.

    Returns
    -------
    dict or tuple
        Dictionary with keys:
        - lon : Center longitude in degrees
        - lat : Center latitude in degrees
        - a : Semi-major axis in kilometers
        - b : Semi-minor axis in kilometers
        - angle : Orientation angle in degrees

        If get_fit=True, returns (dict, error) tuple.

    Example
    -------
    >>> import numpy as np
    >>> from shoot.core.fit import fit_ellipse_from_coords
    >>> theta = np.linspace(0, 2 * np.pi, 50)
    >>> lons = 5.0 + 0.5 * np.cos(theta)
    >>> lats = 43.0 + 0.3 * np.sin(theta)
    >>> params = fit_ellipse_from_coords(lons, lats)  # doctest: +SKIP
    >>> print(f"center: ({params['lon']:.1f}, {params['lat']:.1f})")  # doctest: +SKIP
    """

    lons = np.asarray(lons, dtype="d")
    lats = np.asarray(lats, dtype="d")

    lon0 = lons.mean()
    lat0 = lats.mean()

    x = np.ascontiguousarray(sgeo.deg2m(lons - lon0, lat0))
    y = np.ascontiguousarray(sgeo.deg2m(lats - lat0))

    (xc, yc, a, b, theta), error = fit_ellipse(x, y)

    lat = lat0 + sgeo.m2deg(yc)
    lon = lon0 + sgeo.m2deg(xc, lat=lat0)

    if b > a:
        a, b = b, a
        theta += np.pi / 2

    out = dict(lon=lon, lat=lat, a=a / 1e3, b=b / 1e3, angle=np.degrees(theta))
    if get_fit:
        out = out, error
    return out
