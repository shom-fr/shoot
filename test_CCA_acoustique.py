#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:10:08 2025

@author: jbroust
"""

import os, time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import gsw 
import contourpy
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d
from shoot.front.algos import canny_front, compute_gradients, my_canny_from_gradients, boa_wrapper, wrapper_cca
from shoot.acoustic import get_ecs, get_iminc, get_mcp


# %%
# Import data
#path = "/local/tmp/jbroust/DATA/CMCC_MED"
path = "/usr/site/tmp/DOPS/STM/REC/DEV/SHOOT/Data"
ds = xr.open_dataset(os.path.join(path, "hydro_cur_zos_hourly_3D_cmcc_test.nc"))
#"temp_cur_zos_cmcc_ana_ionian_proteion2024.nc"))
ds
#dss = ds.isel(time=0).sel(latitude=slice(36, 40), longitude=slice(1, 8))
#dss = ds.isel(time=0).sel(latitude=slice(36, 38), longitude=slice(3.5, 5.5))
#dss = ds.isel(time=0).sel(latitude=slice(34, 36), longitude=slice(14, 17))
dss = ds.isel(time=0).sel(latitude=slice(32, 38), longitude=slice(10, 18))

# Compute the density
ct = gsw.conversions.CT_from_pt(dss.so, dss.thetao)
pres = gsw.conversions.p_from_z(-dss.depth, dss.latitude.mean())
dss["sig0"] = gsw.density.sigma0(dss.so, ct)

# Compute the sound celerity
dss["cs"] = gsw.density.sound_speed(dss.so, ct, pres)
s_ecs = get_ecs(dss.cs)
s_mcp = get_mcp(dss.cs)
dss = dss.isel(depth=0)

# dpc = xr.open_dataset(os.path.join(path, "hydro_cur_zos_hourly_3D_cmcc_test_lon(14,17)_lat(34,36)_dpc_output.nc"))
# ecs = dpc.z_sld.isel(time=0)
# mcp = dpc.z_iminc.isel(time=0)

## SST field 
sst = xr.open_dataset(os.path.join(path, "SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_a_sea_surface_temperature_18.12W-36.25E_30.25N-46.00N_2025-10-28-2025-10-30.nc"))
sst = sst.isel(time=0).sel(latitude=slice(32, 38), longitude=slice(10, 18))


# %% CCA detections
# this algo is already based on line detection (converted to matrix front) 
# thus it does not require any filling 

## Juno param 
## On peut jouer sur la diff de température --> regarder par rapport au gradient moyen/std 
start = time.time()
cca, x_cca, y_cca = wrapper_cca(dss.thetao.copy(), 
                                minPopProp=0.2, 
                                minPopMeanDiff=1, 
                                minTheta=0.7, 
                                minSinglePopCohesion=0.9, 
                                minGlobalPopCohesion =0.7, 
                                algo ='sied',
                                njump=10) #used only for bf algo

end = time.time()
print("duaration for sied algo : %.2f min"%((end-start)/60))


nb = 2
plt.figure(figsize=(12,10))
plt.subplot(221)
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
#plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
#plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
plt.scatter(x_cca, y_cca, c='gray', alpha=0.3, s = 10)


plt.subplot(222)
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.zos, cmap="nipy_spectral")
plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
#plt.colorbar(cb)

# plt.figure(figsize=(12,5))
plt.subplot(223)
plt.title("ECS")
cb = plt.pcolormesh(dss.longitude, dss.latitude, s_ecs, cmap="Spectral_r", vmin = 00, vmax = 50)
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.5)
#plt.colorbar(cb)


plt.subplot(224)
plt.title("MCP")
cb = plt.pcolormesh(dss.longitude, dss.latitude, s_mcp, cmap="Spectral_r", vmin = 100, vmax = 400)
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.5)
#plt.colorbar(cb)


# %% Acoustic analysis 

gx_ecs, gy_ecs, gxy_ecs, gdir_ecs = compute_gradients(s_ecs.values)
gx_mcp, gy_mcp, gxy_mcp, gdir_mcp = compute_gradients(s_mcp.values)

"gxy_ecs_filt = uniform_filter(gxy_ecs, size=5)"
k = 10
kernel = np.ones((k, k)) / (k * k)
gxy_ecs_filt = convolve2d(gxy_ecs, kernel, mode="same", boundary="symm")

plt.figure(figsize=(12,10))
plt.subplot(221)
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
#plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
#plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
plt.scatter(x_cca, y_cca, c='gray', alpha=0.3, s = 10)


plt.subplot(222)
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.zos, cmap="nipy_spectral")
plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
#plt.colorbar(cb)


plt.subplot(223)
plt.title(r"$\nabla$ECS")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_ecs_filt, cmap="Spectral_r", vmin = 0., vmax = 100)
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.5)
#plt.colorbar(cb)


plt.subplot(224)
plt.title(r"$\nabla$MCP")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_mcp, cmap="Spectral_r", vmin = 100, vmax = 400)
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.5)
#plt.colorbar(cb)

# %% Comparaison to SST L3 field 


start = time.time()
cca, x_cca, y_cca = wrapper_cca(dss.thetao.copy(), 
                                minPopProp=0.2, 
                                minPopMeanDiff=1, 
                                minTheta=0.7, 
                                minSinglePopCohesion=0.9, 
                                minGlobalPopCohesion =0.7, 
                                algo ='sied',
                                njump=10) #used only for bf algo

end = time.time()
print("duaration for sied algo : %.2f min"%((end-start)/60))

start = time.time()
cca_sst, x_cca_sst, y_cca_sst = wrapper_cca(sst.sea_surface_temperature.copy(), 
                                minPopProp=0.2, 
                                minPopMeanDiff=1, 
                                minTheta=0.7, 
                                minSinglePopCohesion=0.9, 
                                minGlobalPopCohesion =0.7, 
                                algo ='sied',
                                njump=10) #used only for bf algo

end = time.time()
print("duration for sied algo : %.2f min"%((end-start)/60))


plt.figure(figsize=(11,10))
plt.subplot(211)
plt.title("CMCC - %s"%dss.time.dt.strftime("%Y-%m-%d").values)
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal", vmin = 18, vmax = 28)
#plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
#plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
plt.scatter(x_cca, y_cca, c='gray', alpha=0.3, s = 10)
plt.colorbar(cb)

plt.subplot(212)
plt.title("SST L3 - %s"%dss.time.dt.strftime("%Y-%m-%d").values)
cb = plt.pcolormesh(sst.longitude, sst.latitude, sst.sea_surface_temperature-273.16, cmap="cmo.thermal", vmin = 18, vmax = 28)
# plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")
# plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
plt.scatter(x_cca, y_cca, c='blue', alpha=0.3, s = 10, label = "Model fronts")
plt.scatter(x_cca_sst, y_cca_sst, c='gray', alpha=0.3, s = 10, label = "SST fronts")
plt.legend()
plt.colorbar(cb)


