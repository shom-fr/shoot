#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization routines
"""
import numpy as np
import scipy.optimize as scio
import jax
import jax.numpy as jnp
import xoa.coords as xcoords
import xoa.cf as xcf
import xoa.geo as xgeo
import matplotlib.pyplot as plt

from . import grid as sgrid

jax.config.update("jax_enable_x64", True)

GRAVITY = 9.81
OMEGA = 2 * np.pi / 86400
EARTH_RADIUS = 6371e3


# %% SSH


def _func_ssh_from_uv_(ff, uo, vo, fmask, mul2uv, dh2u, dh2v, f0, f1, f2):
    ssh = jnp.array(ff.reshape(uo.shape))

    # SSH to velocity
    va = jnp.gradient(ssh, axis=1) * dh2v
    ua = jnp.gradient(ssh, axis=0) * dh2u

    # Obs part
    du = (ua - uo) * fmask
    dv = (va - vo) * fmask
    fun = (mul2uv * du**2).sum()
    fun += (mul2uv * dv**2).sum()

    # Zeroth order at the edges only
    fun += f0 * (ssh[0] ** 2).sum()
    fun += f0 * (ssh[-1] ** 2).sum()
    fun += f0 * (ssh[:, 0] ** 2).sum()
    fun += f0 * (ssh[:, -1] ** 2).sum()

    # First order at the edges only
    # fun += f1 * 0.25 * (((ssh[:, 2:] - ssh[:, :-2]) ** 2)).sum()
    # fun += f1 * 0.25 * (((ssh[2:, :] - ssh[:-2, :]) ** 2)).sum()
    # - d/dx
    fun += f1 * 0.25 * (((ssh[:2, 2:] - ssh[:2, :-2]) ** 2)).sum()  # south
    fun += f1 * 0.25 * (((ssh[-2:, 2:] - ssh[-2:, :-2]) ** 2)).sum()  # north
    # - d/dy
    fun += f1 * 0.25 * (((ssh[2:, :2] - ssh[:-2, :2]) ** 2)).sum()  # west
    fun += f1 * 0.25 * (((ssh[2:, -2:] - ssh[:-2, -2:]) ** 2)).sum()  # east

    # Second order
    fun += f2 * ((ssh[1:-1, 2:] + ssh[1:-1, :-2] - 2 * ssh[1:-1, 1:-1]) ** 2).sum()
    fun += f2 * ((ssh[2:, 1:-1] + ssh[:-2, 1:-1] - 2 * ssh[1:-1, 1:-1]) ** 2).sum()
    fun += (
        f2
        * (
            0.125
            * (  #  2 * ((1/2)*(1/2))**2 = 2 * 0.5**4
                ssh[2:, 2:] + ssh[:-2, :-2] - ssh[2:, :-2] - ssh[:-2, 2:]
            )
            ** 2
        ).sum()
    )

    return fun


_grad_ssh_from_uv_ = jax.grad(_func_ssh_from_uv_)
_func_ssh_from_uv_ = jax.jit(_func_ssh_from_uv_)
_grad_ssh_from_uv_ = jax.jit(_grad_ssh_from_uv_)


def _nfunc_ssh_from_uv_(*args, **kwargs):
    return np.array(_func_ssh_from_uv_(*args, **kwargs))


def _ngrad_ssh_from_uv_(*args, **kwargs):
    return np.array(_grad_ssh_from_uv_(*args, **kwargs))


def fit_ssh_from_uv(u, v, uv_error=0.01, f0=1.0, f1=1.0, f2=1.0, dx=None, dy=None):
    """Fit an SSH to geostrophic currents

    Parameters
    ----------
    u: xarray.DataArray
        Current along X in m/s
    v: xarray.DataArray
        Current along Y
    uv_error: float
        Quadratic velocity error in m2/s2
    f0: float
        Field penalisation factor
    f1: float
        Gradients penalisation factor
    f2: float
        Second derivative penalisation factor

    Return
    ------
    xarray.DataArray
        SSH array
    """
    invalid = np.isnan(u.values)
    valid = ~invalid
    fmask = valid.astype("d")

    uo = np.where(valid, u.values, 0)
    vo = np.where(valid, v.values, 0)

    # ff = (ssh*0).values[jjvalid, iivalid]
    ff = np.zeros(uo.size)

    dx, dy = sgrid.get_dx_dy(u, dx=dx, dy=dy)

    lat = xcoords.get_lat(u)
    corio = 2 * OMEGA * np.sin(np.radians(lat))
    dh2u = -(GRAVITY / (corio * dy)).values
    dh2v = (GRAVITY / (corio * dx)).values

    # Signal to noise
    sn = (uo**2 + vo**2).mean()  # mean signal
    sn /= np.where(uv_error == 0, 1, uv_error)
    mul2uv = 4 * np.pi * sn

    # Optmize
    args = (uo, vo, fmask, mul2uv, dh2u, dh2v, f0, f1, f2)
    res = scio.minimize(_nfunc_ssh_from_uv_, ff, args, jac=_ngrad_ssh_from_uv_, method="L-BFGS-B")

    # Format
    ssh = u.copy(data=res.x.reshape(u.shape))
    ssh.attrs.clear()
    ssh.name = None
    ssh = xcf.get_cf_specs().format_data_var(ssh, "ssh", format_coords=False, rename_dims=False)
    return ssh


# %% Ellipse


def _func_ellipse_error_jax_(params, lons, lats, radius):
    """Mean squared distance to a equavalent unit circle"""
    "https://chatgpt.com/share/eaa4b938-56b1-406d-b275-b6d68baaafe2"
    lon, lat, ax, ay, b = params
    alpha = jnp.arctan2(ay, ax)
    a = jnp.sqrt(ax**2 + ay**2)
    x = (lons - lon) * np.pi * radius / 180.0 * jnp.cos(jnp.radians(lat))
    y = (lats - lat) * np.pi * radius / 180.0

    ca = jnp.cos(alpha)
    sa = jnp.sin(alpha)

    X = x * ca + y * sa
    Y = -x * sa + y * ca

    d = jnp.sqrt(X**2 / a**2 + Y**2 / b**2)

    # d2 = (ca**2 / a**2 + sa**2 / b**2) * dxx**2
    # d2 += 2 * ca * sa * (1 / a**2 + b**2) * dxx * dyy
    # d2 += (sa**2 / a**2 + ca**2 / b**2) * dyy**2
    return ((d - 1) ** 2).mean()


_grad_ellipse_error_ = jax.grad(_func_ellipse_error_jax_)
_func_ellipse_error_ = jax.jit(_func_ellipse_error_jax_)
_grad_ellipse_error_ = jax.jit(_grad_ellipse_error_)


def _nfunc_ellipse_error_(*args, **kwargs):
    return np.array(_func_ellipse_error_(*args, **kwargs))


def _ngrad_ellipse_error_(*args, **kwargs):
    return np.array(_grad_ellipse_error_(*args, **kwargs))


# def _ellipse_error_(params, lons, lats):
#     """Mean squared distance to a equavalent unit circle"""
#     lon, lat, ax, ay, b = params
#     alpha = np.arctan2(ay, ax)
#     a = np.sqrt(ax**2 + ay**2) * 1e3
#     b *= 1e3
#     xx = xgeo.deg2m(lons - lon, lats.mean())
#     yy = xgeo.deg2m(lats - lat)

#     ca = np.cos(alpha)
#     sa = np.sin(alpha)

#     X = xx * ca + yy * sa
#     Y = -xx * sa + yy * ca

#     d = np.sqrt(X**2 / a**2 + Y**2 / b**2)

#     # d2 = (ca**2 / a**2 + sa**2 / b**2) * xx**2
#     # d2 += 2 * ca * sa * (1 / a**2 - 1 / b**2) * xx * yy
#     # d2 += (sa**2 / a**2 + ca**2 / b**2) * yy**2

#     return ((d - 1) ** 2).mean()


def fit_ellipse_from_coords(lons, lats, get_fit=False):
    """Fit an allipse to a contour line

    Parameters
    ----------
    lons: array(npts)
        Longitudes in degrees
    lats: array(npts)
        Latitudes in degrees

    Returns
    -------
    dict:
        lon: center lon in degrees
        lat: center lat in degrees
        a: semi-major axis in km
        b: semi-minor axis in km
        angle: angle in degrees
    """
    lons = np.array(lons)
    lats = np.array(lats)
    lon0 = lons.mean()
    lat0 = lats.mean()
    xx = xgeo.deg2m(lons - lon0, lat0)
    yy = xgeo.deg2m(lats - lat0)
    ax0 = np.sqrt(xx**2 + yy**2).mean() * 1e-3
    ay0 = 0.1
    b0 = 0.5 * ax0
    params0 = np.array([lon0, lat0, ax0, ay0, b0])

    res = scio.minimize(
        # _ellipse_error_,
        _nfunc_ellipse_error_,
        params0,
        args=(lons, lats, xgeo.EARTH_RADIUS * 1e-3),
        jac=_ngrad_ellipse_error_,
        method="L-BFGS-B",
        # method="Newton-CG",
        bounds=[
            (float(lons.min()), float(lons.max())),
            (float(lats.min()), float(lats.max())),
            (0.1, 200),
            (0.1, 200),
            (0.1, 200),
        ],
    )
    if lon0 > 24:
        pass
    lon, lat, ax, ay, b = res.x
    angle = np.degrees(np.arctan2(ay, ax))
    a = np.sqrt(ax**2 + ay**2)
    if b > a:
        a, b = b, a
        angle += 90.0

    out = dict(lon=lon, lat=lat, a=a, b=b, angle=angle)
    if get_fit:
        out = out, res
    return out
