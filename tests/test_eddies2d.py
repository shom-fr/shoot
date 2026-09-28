"""Tests for shoot.eddies.eddies2d"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from shoot.dyn import get_geos
from shoot.eddies.eddies2d import Eddies2D, EvolEddies2D


@pytest.fixture(scope="module")
def eddy_series():
    """Two time steps of a synthetic field with a cyclone and an anticyclone"""
    lon = xr.DataArray(np.arange(80) / 12.0, dims="lon", attrs={"standard_name": "longitude"})
    lat = xr.DataArray(35 + np.arange(60) / 12.0, dims="lat", attrs={"standard_name": "latitude"})
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    ssh = []
    for shift in (0.0, 0.1):
        field = np.zeros(lat2d.shape)
        for x0, y0, amp in ((1.8 + shift, 37.5, 0.2), (4.7 + shift, 37.4, -0.15)):
            field += amp * np.exp(-((lon2d - x0) ** 2 + (lat2d - y0) ** 2) / (2 * 0.3**2))
        ssh.append(field)
    time = xr.DataArray(pd.date_range("2024-01-01", periods=2), dims="time", attrs={"standard_name": "time"})
    ssh = xr.DataArray(
        np.array(ssh), dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}
    )
    u, v = get_geos(ssh)
    return xr.Dataset({"ssh": ssh, "u": u, "v": v})


def _summary(eddies):
    return sorted((e.glon, e.glat, e.radius, e.vmax_contour.mean_velocity) for e in eddies.eddies)


def test_detect_eddies_parallel_modes(eddy_series):
    """Parallel processing over time steps or centers gives the sequential eddies"""
    kw = dict(window_fit=120, min_radius=10, ellipse_error=0.05)
    seq = EvolEddies2D.detect_eddies(eddy_series, 50, u="u", v="v", ssh="ssh", paral=False, **kw)
    assert [len(e.eddies) for e in seq.eddies] == [2, 2]

    times = EvolEddies2D.detect_eddies(eddy_series, 50, u="u", v="v", ssh="ssh", paral=True, nb_procs=2, **kw)
    assert [_summary(e) for e in times.eddies] == [_summary(e) for e in seq.eddies]

    ds0 = eddy_series.isel(time=0)
    centers = Eddies2D.detect_eddies(ds0.u, ds0.v, 50, ssh=ds0.ssh, paral=True, nb_procs=2, **kw)
    assert _summary(centers) == _summary(seq.eddies[0])
