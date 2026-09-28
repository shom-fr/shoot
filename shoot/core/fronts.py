#!/usr/bin/env python3
"""
Front detection numeric routines

Front detection algorithms on 2D arrays of shape (ny, nx), X along the last
axis, with regular 1D coordinates `x` and `y` when needed:

- Cayula and Cornillon (1992) single image edge detector (CCA):
  :func:`cca_window`, :func:`cca_sied`, :func:`cca_sliding`, :func:`cca`;
- Belkin and O'Reilly (2009) gradient algorithm (BOA): :func:`boa`;
- Canny edge detector: :func:`canny`.
"""

import math

import numba
import numpy as np
import scipy.ndimage as ndi
import scipy.signal

from . import contours as scontours
from . import image as simage

#: CCA window status: a front is found
CCA_FRONT = 0
#: CCA window status: too many NaNs or threshold too low
CCA_INVALID = -1
#: CCA window status: a population is too small
CCA_SMALL_POPULATION = 1
#: CCA window status: population means are too close
CCA_CLOSE_MEANS = 2
#: CCA window status: the histogram is not bimodal enough
CCA_LOW_THETA = 3
#: CCA window status: populations are not cohesive enough
CCA_LOW_COHESION = 4
#: CCA window status: no front line
CCA_NO_LINE = 5

#: Sobel kernels of the BOA algorithm
BOA_KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
BOA_KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


@numba.njit(cache=True)
def _sequential_sum_(values):
    """Sum from left to right in the type of the values"""
    total = np.zeros(1, dtype=values.dtype)[0]
    for value in values:
        total += value
    return total


@numba.njit(cache=True, error_model="numpy")
def _cca_best_split_(counts, xout, nbins):
    """Histogram bin that best separates two populations

    Returns the bin index (-1 if none), the separation, and the count, mean
    of the lower population and mean of the upper population.
    """
    ncounts = counts.size
    best_k = -1
    best_separation = -1.0
    best_a_count = 0
    best_a_mean = best_b_mean = 0.0
    for k in range(1, nbins - 1):
        a_sum = b_sum = 0.0
        a_count = b_count = 0
        for i in range(min(k + 1, ncounts)):
            a_sum += counts[i] * xout[i]
            a_count += counts[i]
        for i in range(k + 1, ncounts):
            b_sum += counts[i] * xout[i]
            b_count += counts[i]
        a_mean = a_sum / a_count
        b_mean = 0.0 if k + 1 >= ncounts else b_sum / b_count
        separation = a_count * b_count * (a_mean - b_mean) * (a_mean - b_mean)
        if separation > best_separation:
            best_k = k
            best_separation = separation
            best_a_count = a_count
            best_a_mean = a_mean
            best_b_mean = b_mean
    return best_k, best_separation, best_a_count, best_a_mean, best_b_mean


@numba.njit(cache=True)
def _cca_cohesion_counts_(w, mask, have_nans, threshold):
    """Counts of neighbors of the same and of any population (bottom and right)"""
    nrows, ncols = w.shape
    a_next_to_a = b_next_to_b = a_next_to_any = b_next_to_any = 0
    for col in range(ncols - 1):
        for row in range(nrows - 1):
            if have_nans and (mask[row, col] or mask[row + 1, col] or mask[row, col + 1]):
                continue
            for nrow, ncol in ((row + 1, col), (row, col + 1)):
                if w[row, col] <= threshold:
                    a_next_to_any += 1
                    if w[nrow, ncol] <= threshold:
                        a_next_to_a += 1
                else:
                    b_next_to_any += 1
                    if w[nrow, ncol] > threshold:
                        b_next_to_b += 1
    return a_next_to_a, b_next_to_b, a_next_to_any, b_next_to_any


def cca_window(
    w,
    bounds,
    min_theta=0.7,
    min_pop_prop=0.2,
    min_pop_mean_diff=0.4,
    min_single_cohesion=0.9,
    min_global_cohesion=0.7,
    corners=None,
    bin_width=0.02,
    min_threshold=0.1,
):
    """Detect a front in a window with the Cayula-Cornillon algorithm

    The histogram of the window is split in two populations (the threshold
    maximizes their separation), and a front is retained if the populations
    are large enough, far enough, well separated and spatially cohesive.
    The front is then the contour of the threshold.

    Parameters
    ----------
    w : ndarray
        2D window. Warning: its NaNs are set to 0 in place.
    bounds : array-like
        Coordinates of the window as ``[xmin, xmax, ymin, ymax]``.
    min_theta : float, default 0.7
        Minimum criterion function θ (bimodality of the histogram).
    min_pop_prop : float, default 0.2
        Minimum proportion of each population.
    min_pop_mean_diff : float, default 0.4
        Minimum difference between the means of the populations.
    min_single_cohesion : float, default 0.9
        Minimum cohesion of each population.
    min_global_cohesion : float, default 0.7
        Minimum global cohesion.
    corners : array-like, optional
        1-based ``[row0, row1, col0, col1]`` limits of the sub-window where
        the front line is computed. Defaults to the whole window.
    bin_width : float, default 0.02
        Width of the histogram bins, in data units.
    min_threshold : float, default 0.1
        Windows whose threshold is lower are rejected.

    Returns
    -------
    x, y : ndarray or None
        Coordinates of the front points.
    threshold : float or None
        Value of the front.
    status : int
        :data:`CCA_FRONT` if a front is found, else the reason of the rejection
        (:data:`CCA_INVALID`, :data:`CCA_SMALL_POPULATION`...).
    """
    xdata, ydata = np.array([]), np.array([])

    mask = np.isnan(w)
    have_nans = bool(mask.any())
    n_nans = int(mask.sum())
    if have_nans and n_nans / w.size > 0.5:
        return None, None, None, CCA_INVALID

    # Histogram and best split into two populations
    wmin, wmax = np.nanmin(w), np.nanmax(w)
    nbins = math.ceil((wmax - wmin) / bin_width)
    bins = np.arange(wmin, wmax, bin_width)
    counts, xout = np.histogram(w[:], bins, [wmin, wmax])
    xout = np.mean(np.vstack([xout[0:-1], xout[1:]]), axis=0)
    threshold = xout[0] if len(xout) else 0

    total_count = w.size - n_nans
    w[mask] = 0
    flat = w.flatten()
    total_sum = _sequential_sum_(flat)
    total_sum_squares = _sequential_sum_(flat * flat)

    k, thresh_separation, thresh_a_count, thresh_a_mean, thresh_b_mean = _cca_best_split_(counts, xout, nbins)
    if k >= 0:
        threshold = xout[k]

    if threshold < min_threshold:
        return None, None, None, CCA_INVALID
    if (thresh_a_count / total_count < min_pop_prop) or (1.0 - thresh_a_count / total_count < min_pop_prop):
        return None, None, None, CCA_SMALL_POPULATION
    if thresh_b_mean - thresh_a_mean < min_pop_mean_diff:
        return None, None, None, CCA_CLOSE_MEANS

    # Criterion function θ (Cayula and Cornillon, 1992, p. 72)
    total_mean = total_sum / total_count
    variance = total_sum_squares - (total_mean * total_mean * total_count)
    theta = thresh_separation / (variance * total_count)
    if theta < min_theta:
        return None, None, None, CCA_LOW_THETA

    # Cohesion: neighbors (bottom and right) of the same population
    a_next_to_a, b_next_to_b, a_next_to_any, b_next_to_any = _cca_cohesion_counts_(
        w, mask, have_nans, threshold
    )
    a_cohesion = a_next_to_a / a_next_to_any
    b_cohesion = b_next_to_b / b_next_to_any
    global_cohesion = (a_next_to_a + b_next_to_b) / (a_next_to_any + b_next_to_any)
    if (
        a_cohesion < min_single_cohesion
        or b_cohesion < min_single_cohesion
        or global_cohesion < min_global_cohesion
    ):
        return None, None, None, CCA_LOW_COHESION

    # Front line
    n_rows, n_cols = w.shape
    x = np.linspace(bounds[0], bounds[1], n_cols)
    y = np.linspace(bounds[2], bounds[3], n_rows)
    if corners is not None:
        x = x[corners[2] - 1 : corners[3]]
        y = y[corners[0] - 1 : corners[1]]
        w = w[corners[0] - 1 : corners[1], corners[2] - 1 : corners[3]]
    w = w.astype("d")
    if have_nans:
        w[w == 0] = np.nan  # restore the NaNs
    lines = [] if np.isnan(w).all() else scontours.contour_lines(w, threshold, x, y)

    # The first line is retained if there are several lines or if it is long and open
    if len(lines) > 1 or (len(lines) == 1 and len(lines[0]) >= 7 and not (lines[0][0] == lines[0][-1]).all()):
        xdata = lines[0][:, 0].round(4)
        ydata = lines[0][:, 1].round(4)

    if xdata.size == 0:
        return xdata, ydata, threshold, CCA_NO_LINE
    return xdata, ydata, threshold, CCA_FRONT


def cca_sied(z, x, y, **kwargs):
    """Cayula-Cornillon single image edge detector

    The image is scanned with 48x48 windows moving by 16 points, each one
    being analysed through four overlapping 32x32 sub-windows with
    :func:`cca_window`.

    Parameters
    ----------
    z : ndarray
        2D field.
    x, y : ndarray
        Regular 1D coordinates.
    kwargs
        Criteria passed to :func:`cca_window`.

    Returns
    -------
    x, y : ndarray
        Coordinates of the front points.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ny, nx = z.shape
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    win16, win48 = 16, 48
    xdata_final, ydata_final = np.array([]), np.array([])
    x_side16, y_side16 = win16 * dx, win16 * dy
    x_side32, y_side32 = 32 * dx, 32 * dy
    sub_rows = np.array([1, 1, 2, 2])
    sub_cols = np.array([1, 2, 2, 1])
    corners = np.array([[17, 33, 17, 33], [17, 33, 1, 17], [1, 17, 1, 17], [1, 17, 17, 33]])

    for wrow in range(1, math.floor(ny / win16) - 1):
        r1 = (wrow - 1) * win16 + 1
        r2 = r1 + win48
        y0 = ymin + (wrow - 1) * y_side16
        for wcol in range(1, math.floor(nx / win16) - 1):
            c1 = (wcol - 1) * win16 + 1
            c2 = c1 + win48
            x0 = xmin + (wcol - 1) * x_side16
            wpad = z[r1 - 1 : r2, c1 - 1 : c2]
            for k in range(4):  # the 4 sliding 33x33 sub-windows
                m1 = (sub_rows[k] - 1) * win16 + 1
                n1 = (sub_cols[k] - 1) * win16 + 1
                w = wpad[m1 - 1 : m1 + 2 * win16, n1 - 1 : n1 + 2 * win16].astype("d")
                sx0 = x0 + (sub_cols[k] - 1) * x_side16
                sy0 = y0 + (sub_rows[k] - 1) * y_side16
                bounds = np.array([sx0, sx0 + x_side32, sy0, sy0 + y_side32])
                xdata, ydata, _, status = cca_window(w, bounds, corners=corners[k, :], **kwargs)
                if status == CCA_FRONT:
                    xdata_final = np.append(xdata_final, xdata)
                    ydata_final = np.append(ydata_final, ydata)
    return xdata_final, ydata_final


def cca_sliding(z, x, y, step=10, size=32, max_nan_fraction=None, **kwargs):
    """Cayula-Cornillon edge detector on sliding windows

    Windows of `size` points centered every `step` points are analysed
    with :func:`cca_window`.

    Parameters
    ----------
    z : ndarray
        2D field. Warning: NaNs are set to 0 in place.
    x, y : ndarray
        1D coordinates.
    step : int, default 10
        Distance between window centers in grid points.
    size : int, default 32
        Window size in grid points.
    max_nan_fraction : float, optional
        Maximum fraction of NaNs of a window. Defaults to `min_pop_prop`.
    kwargs
        Criteria passed to :func:`cca_window`.

    Returns
    -------
    x, y : ndarray
        Coordinates of the front points.
    """
    if max_nan_fraction is None:
        max_nan_fraction = kwargs.get("min_pop_prop", 0.2)
    xdata_final, ydata_final = np.array([]), np.array([])
    ny, nx = z.shape
    mask = np.isnan(z)
    half = size // 2
    for j in range(1, ny - 1, step):
        for i in range(1, nx - 1, step):
            if mask[j, i]:
                continue
            i0, i1 = max(0, i - half), min(nx, i + half + 1)
            j0, j1 = max(0, j - half), min(ny, j + half + 1)
            if mask[j - 1 : j + 2, i - 1 : i + 2].all():
                continue
            w = z[j0:j1, i0:i1]
            if np.sum(np.isnan(w)) / w.size > max_nan_fraction:
                continue
            bounds = np.array([x[i0:i1].min(), x[i0:i1].max(), y[j0:j1].min(), y[j0:j1].max()])
            xdata, ydata, _, status = cca_window(w, bounds, **kwargs)
            if status == CCA_FRONT:
                xdata_final = np.append(xdata_final, xdata)
                ydata_final = np.append(ydata_final, ydata)
    return xdata_final, ydata_final


def points_to_mask(xpts, ypts, x, y):
    """Mask of the grid points nearest to front points

    Parameters
    ----------
    xpts, ypts : ndarray
        Coordinates of the front points.
    x, y : ndarray
        Regular 1D coordinates of the grid.

    Returns
    -------
    ndarray
        1 at the front points, 0 elsewhere.
    """
    ny, nx = len(y), len(x)
    dy = (y.max() - y.min()) / (ny - 1)
    dx = (x.max() - x.min()) / (nx - 1)
    mask = np.zeros((ny, nx))
    cols = np.round((-x.min() + np.asarray(xpts)) / dx).astype(int)
    rows = np.round((y.max() - np.asarray(ypts)) / dy).astype(int)
    mask[rows, cols] = 1
    return np.flipud(mask)


def cca(z, x, y, method="sied", step=10, **kwargs):
    """Front mask and points with the Cayula-Cornillon algorithm

    Parameters
    ----------
    z : ndarray
        2D field.
    x, y : ndarray
        Regular 1D coordinates.
    method : {"sied", "sliding"}, default "sied"
        :func:`cca_sied` or :func:`cca_sliding`.
    step : int, default 10
        Step of :func:`cca_sliding`.
    kwargs
        Criteria passed to :func:`cca_window`.

    Returns
    -------
    mask : ndarray
        1 at fronts, 0 elsewhere.
    xpts, ypts : ndarray
        Coordinates of the front points.
    """
    if method == "sied":
        xpts, ypts = cca_sied(z, x, y, **kwargs)
    else:
        xpts, ypts = cca_sliding(z, x, y, step=step, **kwargs)
    return points_to_mask(xpts, ypts, x, y), xpts, ypts


def boa(z, direction=False):
    """Gradient magnitude with the Belkin-O'Reilly algorithm

    The field is smoothed by a contextual median filter
    (:func:`~shoot.core.image.contextual_median_filter`) that preserves the
    extrema of 5x5 windows, then Sobel gradients are computed. Points near
    land (NaNs) and borders are masked.

    Parameters
    ----------
    z : ndarray
        2D field.
    direction : bool, default False
        Also return the gradient direction in degrees clockwise from north.

    Returns
    -------
    magnitude : ndarray
        Normalized gradient magnitude.
    direction : ndarray
        Only if `direction` is True.
    """
    ny, nx = z.shape
    z = np.asarray(z)

    # Contextual median filter
    peaks5 = simage.find_local_extrema(z, size=5)
    peaks5_shifted = np.ones_like(peaks5)
    peaks5_shifted[:-1, :-1] = peaks5[1:, 1:]
    filtered = simage.contextual_median_filter(z, peaks5_shifted, size=3, margin=2)
    filtered = filtered.astype("d")

    # Sobel gradients
    land = np.isnan(filtered)
    filtered[land] = 0
    tgx = simage.fft_filter(filtered, BOA_KERNEL_X)
    tgy = simage.fft_filter(filtered, BOA_KERNEL_Y)
    tx = tgx / np.abs(BOA_KERNEL_X).sum()
    ty = tgy / np.abs(BOA_KERNEL_Y).sum()
    magnitude = np.sqrt(tx**2 + ty**2)

    # Mask land, its neighbors and borders
    valid = np.where(land, np.nan, 1.0)
    inner = np.full((ny, nx), np.nan)
    inner[5 : ny - 2, 5 : nx - 2] = 1
    valid = np.flip((valid * inner).T, 0)
    kernel = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(3, 3)
    valid = scipy.signal.convolve2d(valid, kernel, boundary="symm", mode="same")
    valid = np.flip(valid, 0).T
    magnitude = valid * magnitude
    if not direction:
        return magnitude
    return magnitude, valid * simage.gradient_direction(tgx, tgy)


def canny(z, low=None, high=None, sigma=5, aperture_size=5):
    """Canny edges of a field scaled to bytes

    Parameters
    ----------
    z : ndarray
        2D field.
    low, high : float, optional
        Hysteresis thresholds on the gradient norm of the field scaled to
        [0, 255]. By default, `high` is the 95th percentile of the Sobel
        gradient norm and `low` is 40 % of it.
    sigma : float, default 5
        Standard deviation of the Gaussian smoothing.
    aperture_size : {3, 5, 7}, default 5
        Aperture size of the Sobel operator.

    Returns
    -------
    ndarray of uint8
        255 on edges, 0 elsewhere.
    """
    z = np.asarray(z, dtype="d")
    zbyte = ((z - np.nanmin(z)) * (1 / (np.nanmax(z) - np.nanmin(z)) * 255)).astype("uint8")
    if not low:
        gx, gy = simage.sobel_gradients(zbyte)
        high = np.nanpercentile(np.sqrt(gx**2 + gy**2), 95)
        low = 0.4 * high
    zbyte = ndi.gaussian_filter(np.flipud(zbyte), sigma=sigma)
    return np.flipud(simage.canny(zbyte, low, high, aperture_size=aperture_size))
