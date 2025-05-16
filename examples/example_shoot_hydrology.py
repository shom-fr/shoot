#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect CROCO-MED1.8 eddies at surface and look at profile anomalies
====================================

In this example, eddies are detected at surface of a CROCO model currents interpolated up to 1000 m and collocated at RHO points.
It has to be adaptated to you own CROCO 3D model.
The dataset used as input need to be interpolated in z before using this functionality
"""
# %%
# Initialisations
# -----------------
#
# Import needed stuff.
import os, sys
import cmocean
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import gsw
import xoa.cf as xcf
import xoa.sigma as xsig
import cmocean as cm

sys.path.append('/home/shom/jbroust/Documents/CODE/SHOOT_LIB/')
from shoot.eddies import Eddies
from shoot.hydrology import Anomaly, compute_anomalies
from shoot.acoustique import (
    ProfileAcous,
    AcousEddy,
    Acous2D,
    acoustic_points,
    get_ecs,
    get_iminc,
    get_mcp,
)
from shoot.plot import create_map, pcarr
from shoot.dyn import get_relvort
from shoot.contours import get_lnam_peaks
from shoot.rossby import Rossby
from shoot.grid import get_dx_dy


xr.set_options(display_style="text")

# %%
# Load croco-specific naming conventions to find dims, coords and variables
xcf.set_cf_specs("croco.cfg")

# %%
# Read data
root_path = '/local/tmp/jbroust/DATA/CROCO/MED1.8'
path = os.path.join(root_path, 'ionian_shoot_3d_10042024.nc')

ds_3d = xr.open_dataset(path)
## Pelops area
# ds_3d = ds_3d.isel(x_rho=slice(350, 550), y_rho=slice(50, 250))

ds_2d = ds_3d.isel(s_rho=len(ds_3d.s_rho) - 1)

# Compute the density
ct = gsw.conversions.CT_from_pt(ds_3d.salt, ds_3d.temp)
pres = gsw.conversions.p_from_z(ds_3d.depth, ds_3d.lat_rho)
ds_3d['sig0'] = gsw.density.sigma0(ds_3d.salt, ct)

# %% D
# Detect eddies
# -------------
# Parameters
# ~~~~~~~~~~
#
# Window size in km to compute the LNAM and find eddy centers
window_center = 50  # 50

window_fit = 120

min_radius = 20

# %%
# Detection
# ~~~~~~~~~
import time

start = time.time()

eddies = Eddies.detect_eddies(
    ds_2d.u,
    ds_2d.v,
    window_center,
    window_fit=window_fit,
    ssh=ds_2d.zeta,
    min_radius=min_radius,
)
end = time.time()
print('it takes %.1f s' % (end - start))

# %%
# Detect anomaly exemple on a particular eddy
eddy = eddies.eddies[0]
# here you can choose the desire variable : density, salinity, temp, celerity
anomaly = Anomaly(eddy, eddies, ds_3d.sig0, r_factor=1.2)

# %%
# Plots eddies
# -----
#
# We plot eddies with the relative vorticity as background.
#
fig, ax = create_map(ds_2d.lon_rho, ds_2d.lat_rho, figsize=(8, 5))
get_relvort(ds_2d.u, ds_2d.v).plot(
    x="lon_rho", y="lat_rho", cmap="cmo.curl", ax=ax, add_colorbar=False, transform=pcarr
)

for eddy in eddies.eddies:
    eddy.plot(transform=pcarr, lw=1)
ax.scatter(
    anomaly._profils_outside.lon_rho,
    anomaly._profils_outside.lat_rho,
    s=10,
    marker='o',
    # c = ecs_outside,
    # c=mcp_outside,
    c=anomaly._profils_outside.max(dim='s_rho'),
    cmap='nipy_spectral',
    vmin=28,
    vmax=30,
    transform=pcarr,
)

cmb = ax.scatter(
    anomaly._profils_inside.lon_rho,
    anomaly._profils_inside.lat_rho,
    s=10,
    marker='*',
    # c=ecs_inside,
    # c=mcp_inside,
    c=anomaly._profils_inside.max(dim='s_rho'),
    cmap='nipy_spectral',
    vmin=28,
    vmax=30,
    transform=pcarr,
)

plt.colorbar(cmb, label='Max density [kg/m3]')
plt.title("Eddy detection")
plt.tight_layout()

# %%
# Plot anomaly

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(anomaly.anomaly, anomaly.depth_vector, label='averaged profile')
plt.plot(anomaly.center_anomaly, anomaly.depth_vector, label="center profile")
# plt.xlabel(r'$\Delta\sigma_0$ [kg/m3]')
plt.xlabel(r'$\Delta cs$ [m/s]')
plt.ylabel('Depth [m]')
plt.ylim(-2000, 0)
plt.legend()
ax = plt.subplot(122)
ax.plot(anomaly.mean_profil_inside, anomaly.depth_vector, c='b', label='inside')
ax.fill_betweenx(
    anomaly.depth_vector,
    anomaly.mean_profil_inside - anomaly.std_profil_inside,
    anomaly.mean_profil_inside + anomaly.std_profil_inside,
    alpha=0.2,
    color='b',
)
ax.plot(anomaly.mean_profil_outside, anomaly.depth_vector, c='k', label='outside')
ax.fill_betweenx(
    anomaly.depth_vector,
    anomaly.mean_profil_outside - anomaly.std_profil_outside,
    anomaly.mean_profil_outside + anomaly.std_profil_outside,
    alpha=0.2,
    color='k',
)
plt.legend()
# plt.xlabel(r'$\sigma_0$ [kg/m3]')
plt.xlabel(r'$cs$ [m/s]')
# plt.ylabel('Depth [m]')
plt.yticks([], [])
plt.ylim(-2000, 0)

# %%
# Compute anomalies of all eddies

# compute_anomalies(eddies, ds_3d.sig0, r_factor=1.05)
compute_anomalies(eddies, ds_3d.cs, r_factor=1.05)

# %%
# Compute anomalies of all eddies

acoustic_points(eddies)

# %%
# Plots eddies colored by density anomaly intensity
# -----
#
# We plot eddies with the relative vorticity as background.
#

fig, ax = create_map(ds_2d.lon_rho, ds_2d.lat_rho, figsize=(8, 5))
get_relvort(ds_2d.u, ds_2d.v).plot(
    x="lon_rho", y="lat_rho", cmap="cmo.curl", ax=ax, add_colorbar=False, transform=pcarr
)
for eddy in eddies.eddies:
    eddy.plot(transform=pcarr, lw=1)
    ax.scatter(
        eddy.anomaly._profils_outside.lon_rho,
        eddy.anomaly._profils_outside.lat_rho,
        s=10,
        marker='o',
        c=eddy.ecs_outside,
        # c=eddy.mcp_outside,
        # c=eddy.iminc_outside,
        cmap='nipy_spectral',
        vmin=-400,
        vmax=0,
        transform=pcarr,
    )
    cmb = ax.scatter(
        eddy.anomaly._profils_inside.lon_rho,
        eddy.anomaly._profils_inside.lat_rho,
        s=10,
        marker='*',
        c=eddy.ecs_inside,
        # c=eddy.mcp_inside,
        # c=eddy.iminc_inside,
        cmap='nipy_spectral',
        vmin=-400,
        vmax=0,
        transform=pcarr,
    )

    # cb = plt.scatter(
    #     eddy.glon,
    #     eddy.glat,
    #     c=eddy.anomaly.anomaly[np.argmax(np.abs(eddy.anomaly.anomaly))],
    #     cmap=cm.cm.balance_r,
    #     transform=pcarr,
    #     vmin=-0.5,
    #     vmax=0.5,
    # )
    # cb = plt.scatter(eddy.glon, eddy.glat, c=-eddy.anomaly.depth_vector[np.argmax(np.abs(eddy.anomaly.anomaly))], cmap = 'nipy_spectral', transform=pcarr, vmin = 0, vmax = 500)

# plt.colorbar(cb, label=r'$\Delta\sigma$ [kg/m3]')
plt.colorbar(cmb, label='ECS [m]')
# plt.colorbar(cb, label = 'Maximum anomaly depth [m]' )
plt.title("Relative vorticity")
plt.tight_layout()

# %%
# Print anomalies for all eddies

z_max = 1500
for eddy in eddies.eddies:
    anomaly = eddy.anomaly

    # Plot pos points
    plt.figure()
    plt.scatter(anomaly._profils_outside.lon_rho, anomaly._profils_outside.lat_rho)
    plt.scatter(eddy.glon, eddy.glat)
    plt.plot(eddy.boundary_contour.lon, eddy.boundary_contour.lat)
    plt.plot(eddy.vmax_contour.lon, eddy.vmax_contour.lat)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(anomaly.anomaly, anomaly.depth_vector)
    plt.ylim(-600, 0)
    ax = plt.subplot(122)
    ax.plot(anomaly.profil_inside, anomaly.depth_vector, label='inside')
    ax.plot(anomaly.mean_profil_outside, anomaly.depth_vector, c='k', label='outside')
    ax.fill_betweenx(
        anomaly.depth_vector,
        anomaly.mean_profil_outside - anomaly.std_profil_outside,
        anomaly.mean_profil_outside + anomaly.std_profil_outside,
        alpha=0.2,
        color='k',
    )
    plt.ylim(-z_max, 0)

# %%
