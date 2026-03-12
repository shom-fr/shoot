#!/usr/bin/env python3
"""
Detect CROCO-MED1.8 eddies at different depths and link the detections
======================================================================

In this example, 3D eddies are detected from CROCO model currents interpolated at several depths
and collocated at RHO points.
"""

# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import cmocean  # noqa
import matplotlib.pyplot as plt
import xarray as xr

from shoot.dyn import get_relvort
from shoot.eddies.eddies3d import Eddies3D
from shoot.meta import set_meta_specs
from shoot.plot import create_map, pcarr
from shoot.samples import get_sample_file

xr.set_options(display_style="text")

# %%
# Load croco-specific naming conventions to find dims, coords and variables
set_meta_specs("croco")

# %%
# Read data

root_path = "MODELS/CROCO/MED/pelops_3d_interp.nc"
path = get_sample_file(root_path)
ds_3d = xr.open_dataset(path)

# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers
window_center = 50  # 50

window_fit = 120

min_radius = 10


# %%  Eddies3D object

eddies3d = Eddies3D.detect_eddies_3d(
    ds_3d.u,
    ds_3d.v,
    window_center,
    window_fit=window_fit,
    min_radius=min_radius,
    paral=True,
)


# %%
eddies3d.plot2d(depth=10)


# %% Mapping at each depth

# Plots eddies
# -----
#
# We plot eddies with the relative vorticity as background.
#

eddies3d_slice = eddies3d.eddies_byslice

for i in range(len(ds_3d.depth)):
    fig, ax = create_map(
        ds_3d.isel(depth=i).lon_rho,
        ds_3d.isel(depth=i).lat_rho,
        figsize=(8, 5),
    )
    cb = get_relvort(ds_3d.isel(depth=i).u, ds_3d.isel(depth=i).v).plot(
        x="lon_rho",
        y="lat_rho",
        cmap="cmo.curl",
        ax=ax,
        add_colorbar=False,
        transform=pcarr,
    )
    plt.colorbar(cb, label=r"$\zeta$")

    plt.quiver(
        ds_3d.lon_rho[::5, ::5].values,
        ds_3d.lat_rho[::5, ::5].values,
        ds_3d.isel(depth=i).u[::5, ::5].values,
        ds_3d.isel(depth=i).v[::5, ::5].values,
        color="k",
        transform=pcarr,
    )

    for eddy in eddies3d_slice.eddies3d[i].eddies:
        eddy.plot(transform=pcarr, lw=1)
        # plt.text(eddy.glon, eddy.glat, eddy.z_id, transform=pcarr)
        plt.text(
            eddy.glon,
            eddy.glat,
            f"{eddy.z_id} - {eddy.vmax_contour.mean_velocity:.2f} m/s",
            transform=pcarr,
        )

    plt.title(f"Relative vorticity at {ds_3d.isel(depth=i).depth} m depth")
    plt.tight_layout()
