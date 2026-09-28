#!/usr/bin/env python3
"""
Generic image processing routines

Filters, gradients and edge detection on 2D arrays of shape (ny, nx),
with X along the last axis.
"""

import math

import numba
import numpy as np
import scipy.fft
import scipy.ndimage as ndi

#: Derivative and smoothing kernels of the Sobel operator per aperture size
SOBEL_KERNELS = {
    3: ([-1, 0, 1], [1, 2, 1]),
    5: ([-1, -2, 0, 2, 1], [1, 4, 6, 4, 1]),
    7: ([-1, -4, -5, 0, 5, 4, 1], [1, 6, 15, 20, 15, 6, 1]),
}


def sobel_gradients(z, size=3, mode="mirror"):
    """Sobel derivatives along X and Y

    Same conventions as :func:`cv2.Sobel`, including for NaNs: X is the last
    axis and is filtered first, zero coefficients are skipped along Y only,
    and the default ``mode="mirror"`` corresponds to OpenCV's default border.

    Parameters
    ----------
    z : ndarray
        2D field.
    size : {3, 5, 7}, default 3
        Aperture size.
    mode : {"mirror", "nearest", "reflect", "wrap"}, default "mirror"
        Border mode, as in :mod:`scipy.ndimage`
        ("nearest" is OpenCV's replicate border).

    Returns
    -------
    gx, gy : ndarray
    """
    deriv, smooth = SOBEL_KERNELS[size]
    z = np.asarray(z, dtype="d")
    gx = _correlate1d(_correlate1d(z, deriv, 1, mode), smooth, 0, mode, skip_zeros=True)
    gy = _correlate1d(_correlate1d(z, smooth, 1, mode), deriv, 0, mode, skip_zeros=True)
    return gx, gy


_NUMPY_PAD_MODES = {"mirror": "reflect", "nearest": "edge", "reflect": "symmetric", "wrap": "wrap"}


def _correlate1d(z, kernel, axis, mode, skip_zeros=False):
    """1D correlation, possibly skipping zero coefficients so that NaNs do not spread through them"""
    half = len(kernel) // 2
    pad = [(0, 0)] * z.ndim
    pad[axis] = (half, half)
    zpad = np.pad(z, pad, mode=_NUMPY_PAD_MODES[mode])
    n = z.shape[axis]
    out = None
    for k, coef in enumerate(kernel):
        if coef == 0 and skip_zeros:
            continue
        term = coef * np.take(zpad, np.arange(k, k + n), axis=axis)
        out = term if out is None else out + term
    return out


def gradient_direction(gx, gy):
    """Direction of a gradient in degrees clockwise from the Y axis

    The Y axis is the north when Y increases northward, and X the east.
    """
    direction = np.degrees(np.arctan2(gy, gx))
    direction = np.where(direction < 0, 360 + direction, direction)
    return (360 - direction + 90) % 360


def fill_nans_nearest(z):
    """Fill NaNs with the nearest valid values

    Parameters
    ----------
    z : ndarray
        Field with NaNs.

    Returns
    -------
    ndarray
        Filled copy of `z`.
    """
    z = np.asarray(z)
    nans = np.isnan(z)
    if not nans.any() or nans.all():
        return z.copy()
    indices = ndi.distance_transform_edt(nans, return_distances=False, return_indices=True)
    return z[tuple(indices)]


def fft_filter(z, kernel):
    """Circular filtering by FFT, as ``filter2`` of the EBImage R package

    Parameters
    ----------
    z : ndarray
        2D field.
    kernel : ndarray
        2D kernel with odd dimensions, smaller than `z`.

    Returns
    -------
    ndarray
    """
    kernel = np.asarray(kernel)
    dx = z.shape
    df = kernel.shape
    if df[0] % 2 == 0 or df[1] % 2 == 0:
        raise ValueError("dimensions of the kernel must be odd")
    if dx[0] < df[0] or dx[1] < df[1]:
        raise ValueError("dimensions of the field must be greater than those of the kernel")

    cx = tuple(elem // 2 for elem in dx)
    cf = tuple(elem // 2 for elem in df)
    wf = np.zeros(shape=dx)
    wf[cx[0] - cf[0] - 1 : cx[0] + cf[0], cx[1] - cf[1] - 1 : cx[1] + cf[1]] = kernel
    wf = np.fft.fft2(wf)
    index1 = np.concatenate((np.arange(cx[0], dx[0] + 1), np.arange(1, cx[0]))) - 1
    index2 = np.concatenate((np.arange(cx[1], dx[1] + 1), np.arange(1, cx[1]))) - 1
    y = (scipy.fft.ifft2(scipy.fft.fft2(z) * wf)).real
    return y[np.ix_(index1, index2)]


@numba.njit(cache=True)
def _local_extrema_(z, size):
    ny, nx = z.shape
    half = size // 2
    center = half * size + half
    out = np.zeros((ny, nx))
    for j in range(half, ny - half):
        for i in range(half, nx - half):
            window = z[j - half : j + half + 1, i - half : i + half + 1].copy().ravel()
            if np.isnan(window).all():
                continue
            if np.argmax(window) == center or np.argmin(window) == center:
                out[j, i] = 1
    return out


def find_local_extrema(z, size=5):
    """Flag the points that are the extremum of the window centered on them

    As with :func:`numpy.argmax`, a window containing NaNs has its first NaN
    as extremum.

    Parameters
    ----------
    z : ndarray
        2D field.
    size : int, default 5
        Odd window size.

    Returns
    -------
    ndarray
        1 at extrema, 0 elsewhere and within ``size // 2`` of the borders.
    """
    return _local_extrema_(np.asarray(z, dtype="d"), size)


@numba.njit(cache=True)
def _contextual_median_filter_(z, extrema, size, margin):
    ny, nx = z.shape
    half = size // 2
    center = half * size + half
    out = z * 0
    for j in range(margin, ny - margin):
        for i in range(margin, nx - margin):
            if extrema[j, i] != 0:
                out[j, i] = z[j, i]
                continue
            window = z[j - half : j + half + 1, i - half : i + half + 1].copy().ravel()
            if np.isnan(window).all():
                out[j, i] = z[j, i]
            elif np.argmax(window) == center or np.argmin(window) == center:
                out[j, i] = np.nanmedian(window)
            else:
                out[j, i] = z[j, i]
    return out


def contextual_median_filter(z, extrema, size=3, margin=None):
    """Median filter applied only to the local extrema of a small window

    The contextual median filter of Belkin and O'Reilly (2009): a point that
    is the extremum of its ``size x size`` window, and is not flagged in
    `extrema` (typically extrema of a larger window), is replaced by the
    median of the window.

    Parameters
    ----------
    z : ndarray
        2D field.
    extrema : ndarray
        Points not to filter (non-zero), like :func:`find_local_extrema` of a larger window.
    size : int, default 3
        Odd window size.
    margin : int, optional
        Points within this distance of the borders are set to ``z * 0``.
        Defaults to ``size // 2``.

    Returns
    -------
    ndarray
        Same type as `z`.
    """
    if margin is None:
        margin = size // 2
    return _contextual_median_filter_(np.asarray(z), np.asarray(extrema, dtype="d"), size, margin)


@numba.njit(cache=True)
def non_max_suppression(mag, angle):
    """Keep the gradient magnitude only where it is maximal across the gradient direction

    Parameters
    ----------
    mag : ndarray
        Gradient magnitude.
    angle : ndarray
        Gradient direction in radians.

    Returns
    -------
    ndarray
        Magnitude at the local maxima, 0 elsewhere and on borders.
    """
    ny, nx = mag.shape
    out = np.zeros((ny, nx))
    angle = angle * 180.0 / np.pi
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            a = angle[j, i]
            if a < 0:
                a += 180
            q = 255.0
            r = 255.0
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = mag[j, i + 1]
                r = mag[j, i - 1]
            elif 22.5 <= a < 67.5:
                q = mag[j + 1, i - 1]
                r = mag[j - 1, i + 1]
            elif 67.5 <= a < 112.5:
                q = mag[j + 1, i]
                r = mag[j - 1, i]
            elif 112.5 <= a < 157.5:
                q = mag[j - 1, i - 1]
                r = mag[j + 1, i + 1]
            if mag[j, i] >= q and mag[j, i] >= r:
                out[j, i] = mag[j, i]
    return out


def double_threshold(z, low_ratio=0.05, high_ratio=0.15, weak=50, strong=255):
    """Classify values as strong or weak edges

    Parameters
    ----------
    z : ndarray
        Edge strength, like the output of :func:`non_max_suppression`.
    low_ratio : float, default 0.05
        Low threshold as a fraction of the high threshold.
    high_ratio : float, default 0.15
        High threshold as a fraction of the maximum of `z`.
    weak, strong : int
        Output values of the weak and strong edges.

    Returns
    -------
    ndarray of uint8
        `weak` where ``low <= z <= high``, `strong` where ``z > high``, 0 elsewhere.
    """
    high = z.max() * high_ratio
    low = high * low_ratio
    out = np.zeros(z.shape, dtype=np.uint8)
    out[z >= high] = strong
    out[(z <= high) & (z >= low)] = weak
    return out


@numba.njit(cache=True)
def _hysteresis_(edges, weak, strong):
    ny, nx = edges.shape
    out = np.zeros_like(edges)
    stack = np.empty((ny * nx, 2), dtype=np.int64)
    nstack = 0
    for j in range(ny):
        for i in range(nx):
            if edges[j, i] == strong:
                out[j, i] = strong
                stack[nstack, 0] = j
                stack[nstack, 1] = i
                nstack += 1
    while nstack > 0:
        nstack -= 1
        j = stack[nstack, 0]
        i = stack[nstack, 1]
        for jj in range(max(j - 1, 0), min(j + 2, ny)):
            for ii in range(max(i - 1, 0), min(i + 2, nx)):
                if edges[jj, ii] == weak and out[jj, ii] == 0:
                    out[jj, ii] = strong
                    stack[nstack, 0] = jj
                    stack[nstack, 1] = ii
                    nstack += 1
    return out


def hysteresis(edges, weak=50, strong=255):
    """Keep the weak edges connected to a strong edge

    Weak edges are retained when they are connected to a strong edge,
    directly or through other weak edges (8-connectivity).

    Parameters
    ----------
    edges : ndarray
        Output of :func:`double_threshold`.
    weak, strong : int
        Values of the weak and strong edges.

    Returns
    -------
    ndarray
        `strong` for retained edges, 0 elsewhere.
    """
    return _hysteresis_(np.asarray(edges), weak, strong)


def canny_from_gradients(gx, gy, low_ratio=0.05, high_ratio=0.15):
    """Canny edges from gradients

    Non-maximum suppression, double threshold and hysteresis.

    Parameters
    ----------
    gx, gy : ndarray
        Gradients along X and Y.
    low_ratio, high_ratio : float
        See :func:`double_threshold`.

    Returns
    -------
    ndarray of uint8
        255 on edges, 0 elsewhere.
    """
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    nms = non_max_suppression(mag, angle)
    return hysteresis(double_threshold(nms, low_ratio, high_ratio))


# Tangent of 22.5° in 15-bit fixed point, as in OpenCV
_TG22 = int(math.tan(math.radians(22.5)) * (1 << 15) + 0.5)


@numba.njit(cache=True)
def _canny_(dx, dy, low, high):
    ny, nx = dx.shape
    mag = np.zeros((ny + 2, nx + 2), dtype=np.int64)
    mag[1:-1, 1:-1] = np.abs(dx) + np.abs(dy)
    # 0: weak candidate, 1: not an edge, 2: edge
    state = np.ones((ny + 2, nx + 2), dtype=np.int8)
    stack = np.empty((ny * nx, 2), dtype=np.int64)
    nstack = 0
    for j in range(ny):
        for i in range(nx):
            m = mag[j + 1, i + 1]
            if m <= low:
                continue
            xs = dx[j, i]
            ys = dy[j, i]
            x = abs(xs)
            y = abs(ys) << 15
            tg22x = x * _TG22
            if y < tg22x:
                ok = m > mag[j + 1, i] and m >= mag[j + 1, i + 2]
            else:
                tg67x = tg22x + (x << 16)
                if y > tg67x:
                    ok = m > mag[j, i + 1] and m >= mag[j + 2, i + 1]
                else:
                    s = -1 if (xs ^ ys) < 0 else 1
                    ok = m > mag[j, i + 1 - s] and m > mag[j + 2, i + 1 + s]
            if not ok:
                continue
            if m > high:
                state[j + 1, i + 1] = 2
                stack[nstack, 0] = j + 1
                stack[nstack, 1] = i + 1
                nstack += 1
            else:
                state[j + 1, i + 1] = 0
    while nstack > 0:
        nstack -= 1
        j = stack[nstack, 0]
        i = stack[nstack, 1]
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if state[j + dj, i + di] == 0:
                    state[j + dj, i + di] = 2
                    stack[nstack, 0] = j + dj
                    stack[nstack, 1] = i + di
                    nstack += 1
    out = np.zeros((ny, nx), dtype=np.uint8)
    for j in range(ny):
        for i in range(nx):
            if state[j + 1, i + 1] == 2:
                out[j, i] = 255
    return out


def canny(z, low=None, high=None, sigma=0.0, aperture_size=3):
    """Canny edge detector

    Same algorithm as :func:`cv2.Canny` with the L1 gradient norm.
    A field that is not of type uint8 is first scaled to [0, 255] bytes,
    after filling its NaNs with the nearest valid values.

    Parameters
    ----------
    z : ndarray
        2D field, possibly with NaNs.
    low, high : float, optional
        Hysteresis thresholds on the gradient norm of the byte field
        (floored to integers). By default, `high` is the 95th percentile
        of the Sobel gradient norm, and `low` is 40 % of `high`.
    sigma : float, default 0
        Standard deviation in grid points of a Gaussian smoothing of the byte
        field, if strictly positive.
    aperture_size : {3, 5, 7}, default 3
        Aperture size of the Sobel operator.

    Returns
    -------
    ndarray of uint8
        255 on edges, 0 elsewhere and on NaNs.
    """
    z = np.asarray(z)
    nans = None
    if z.dtype != np.uint8:
        z = np.asarray(z, dtype="d")
        nans = np.isnan(z)
        if nans.any():
            z = fill_nans_nearest(z)
        z = ((z - z.min()) * (1 / (z.max() - z.min()) * 255)).astype("uint8")
    if low is None or high is None:
        gx, gy = sobel_gradients(z)
        high = np.percentile(np.sqrt(gx**2 + gy**2), 95)
        low = 0.4 * high
    if sigma > 0:
        z = ndi.gaussian_filter(z, sigma=sigma)
    if low > high:
        low, high = high, low
    dx, dy = sobel_gradients(z, aperture_size, mode="nearest")
    if aperture_size == 7:  # derivatives and thresholds scaled by 1/16 as in OpenCV
        dx, dy = (np.clip(np.rint(d / 16), -32768, 32767) for d in (dx, dy))
        low, high = low / 16, high / 16
    edges = _canny_(dx.astype(np.int64), dy.astype(np.int64), math.floor(low), math.floor(high))
    if nans is not None:
        edges[nans] = 0
    return edges
