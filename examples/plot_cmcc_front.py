#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:45:52 2025

@author: jbroust
"""
import os, time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import gsw 
import contourpy
from shoot.front.algos import canny_front, compute_gradients, my_canny_from_gradients, boa_wrapper, wrapper_cca
from shoot.acoustic import get_ecs, get_iminc, get_mcp

"""
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


#%% Plot 
# plt.figure(figsize=(12,5))
# plt.subplot(121)
# plt.title("ECS")
# cb = plt.pcolormesh(dss.longitude, dss.latitude, s_ecs, cmap="Spectral_r", vmin = 0, vmax = 50)
# plt.colorbar()

# plt.subplot(122)
# plt.title("MCP")
# cb = plt.pcolormesh(dss.longitude, dss.latitude, s_mcp, cmap="Spectral_r", vmin = 50, vmax = 500)
# plt.colorbar()

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.title("ECS")
cb = plt.pcolormesh(dpc.lon, dpc.lat, ecs, cmap="Spectral_r", vmin = 00, vmax = 50)
plt.colorbar()

plt.subplot(122)
plt.title("MCP")
cb = plt.pcolormesh(dpc.lon, dpc.lat, mcp, cmap="Spectral_r", vmin = 50, vmax = 500)
plt.colorbar()



#%% Plot situation 

gx_t, gy_t, gxy_t, gdir_t = compute_gradients(dss.thetao.values)
gx_s, gy_s, gxy_s, gdir_s = compute_gradients(dss.so.values)
gx_z, gy_z, gxy_z, gdir_z = compute_gradients(dss.zos.values)
gx_ecs, gy_ecs, gxy_ecs, gdir_ecs = compute_gradients(ecs.values)
gx_mcp, gy_mcp, gxy_mcp, gdir_mcp = compute_gradients(mcp.values)

mgxy_s = np.nanmean(gxy_s)
mgxy_t = np.nanmean(gxy_t)
mgxy_z = np.nanmean(gxy_z) 
mgxy_ecs = np.nanmean(gxy_ecs) 
mgxy_mcp = np.nanmean(gxy_mcp) 

gxy = gxy_t/mgxy_t + gxy_s/mgxy_s  + gxy_z/mgxy_z 

nb = 3 

plt.figure(figsize=(12,10))

plt.subplot(221)
plt.title("temp gradient")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_t, cmap="Spectral_r")
plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')
#plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")

plt.subplot(222)
plt.title("ssh gradient")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_z, cmap="Spectral_r")
plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')
#plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")

plt.subplot(223)
plt.title("ECS gradient")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_ecs, cmap="Spectral_r")
plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')

plt.subplot(224)
plt.title("MCP gradient")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_mcp, cmap="Spectral_r")
plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')



# plt.subplot(223)
# plt.title("salinity gradient")
# cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy_s, cmap="Spectral_r")
# plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')
# #plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")

# plt.subplot(224)
# plt.title("merged gradient")
# cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy, cmap="Spectral_r")
# plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')
# #plt.quiver(dss.longitude[::nb], dss.latitude[::nb], dss.uo[::nb, ::nb], dss.vo[::nb, ::nb], color = "gray")


plt.figure(figsize=(10,10))
plt.subplot(221)
plt.scatter(gxy_t.flatten(), gxy_ecs.flatten())
plt.subplot(222)
plt.scatter(gxy_z.flatten(), gxy_ecs.flatten())
plt.subplot(223)
plt.scatter(gxy_t.flatten(), gxy_mcp.flatten())
plt.subplot(224)
plt.scatter(gxy_z.flatten(), gxy_mcp.flatten())


#%% Canny multifield

gx = gx_t/mgxy_t + gx_s/mgxy_s  + gx_z/mgxy_z 
gy = gy_t/mgxy_t + gy_s/mgxy_s  + gy_z/mgxy_z 

canny = my_canny_from_gradients(gx_t, gy_t)

plt.figure()
plt.title("merged gradient")
cb = plt.pcolormesh(dss.longitude, dss.latitude, gxy, cmap="Spectral_r")
#plt.contour(dss.longitude, dss.latitude, dss.zos, colors = 'k')
plt.pcolormesh(dss.longitude, dss.latitude, canny, cmap='gray_r', alpha=0.5)

# %% Canny detections


canny = canny_front(dss.thetao, tmin = 120, tmax = 220, sigma=0, apertureSize=5)
#np.where(canny == 0)


# plt.figure()
# cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
# plt.pcolormesh(dss.longitude, dss.latitude, canny, cmap='gray_r', alpha=0.2)
# plt.colorbar(cb,label = "SST")

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.title("ECS")
cb = plt.pcolormesh(dss.longitude, dss.latitude, ecs, cmap="Spectral_r", vmin = 00, vmax = 50)
plt.pcolormesh(dss.longitude, dss.latitude, canny, cmap='gray_r', alpha=0.2)
plt.colorbar(cb)

plt.subplot(122)
plt.title("MCP")
cb = plt.pcolormesh(dss.longitude, dss.latitude, mcp, cmap="cmo.dense", vmin = 100, vmax = 400)
plt.pcolormesh(dss.longitude, dss.latitude, canny, cmap='gray_r', alpha=0.2)
plt.colorbar(cb)




# %% BOA detections
boa = boa_wrapper(dss.thetao, threshold=0.2)

## travail sur la continuité
continuity = True
if continuity:

    from scipy.ndimage import label
    from skimage.morphology import skeletonize, closing, disk

    # fill 
    boa_filled = closing(boa, disk(1))
    boa_filled[np.isnan(boa_filled)] = 0
    
    # avoid small lines
    struct = np.ones((3, 3))  # connectivité 8
    labeled, num = label(boa_filled, structure=struct)
    
    # 2. Calcul de la taille de chaque composante
    sizes = np.bincount(labeled.ravel())
    
    # 3. Définir un seuil minimal
    min_size = 50  # <-- à ajuster selon ton cas
    
    # 4. Créer un masque filtré
    # garder seulement les composantes dont la taille >= min_size
    mask = np.isin(labeled, np.where(sizes >= min_size)[0])
    boa_filled*=mask
    boa_skeleton = skeletonize(boa_filled)


plt.figure()
cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
#plt.pcolormesh(dss.longitude, dss.latitude, boa, cmap='gray_r', alpha=0.2)
#plt.pcolormesh(dss.longitude, dss.latitude, boa_filled, cmap='gray_r', alpha=0.5)
plt.pcolormesh(dss.longitude, dss.latitude, boa_skeleton, cmap='gray_r', alpha=0.5)
plt.colorbar(cb)


contour_gen = contourpy.contour_generator(
    x=np.arange(boa.shape[1]),
    y=np.arange(boa.shape[0]),
    # z=boa,
    z=boa_skeleton,
    #z=boa_filled,
    line_type="Separate",  # renvoie des listes indépendantes
)

contours = contour_gen.lines(level=0.5)

nj = 7
plt.figure()
plt.title("BOA Algo (%s)"%dss.time.dt.strftime("%Y-%m-%d").values)
plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
#plt.quiver(dss.longitude[::nj], dss.latitude[::nj], dss.uo_detided[::nj, ::nj], dss.vo_detided[::nj, ::nj], color="k") 
lon = dss.longitude.values
lat = dss.latitude.values
for poly in contours:
    lon_poly = np.interp(poly[:, 0], np.arange(len(lon)), lon)
    lat_poly = np.interp(poly[:, 1], np.arange(len(lat)), lat)

    # plt.plot(poly[:, 0], poly[:, 1],linewidth=2)
    plt.plot(lon_poly, lat_poly, linewidth=0.1, c='k')


# %% CCA detections
# this algo is already based on line detection (converted to matrix front) 
# thus it does not require any filling 

## Juno param 
## On peut jouer sur la diff de température --> regarder par rapport au gradient moyen/std 
start = time.time()
cca, x_cca, y_cca = wrapper_cca(dss.thetao.copy(), 
                                minPopProp=0.2, 
                                minPopMeanDiff=2, 
                                minTheta=0.7, 
                                minSinglePopCohesion=0.9, 
                                minGlobalPopCohesion =0.7, 
                                algo ='sied',
                                njump=10) #used only for bf algo

end = time.time()
print("duaration for sied algo : %.2f min"%((end-start)/60))

# Victor Param
# cca, x_cca, y_cca = wrapper_cca(dss.thetao, 
#                                  minPopProp=0.2, 
#                                  minPopMeanDiff=0.4, 
#                                  minTheta=0.45, 
#                                  minSinglePopCohesion=0.8, 
#                                  minGlobalPopCohesion =0.9, 
#                                  algo ='bf')

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
plt.colorbar(cb)

plt.subplot(224)
plt.title("MCP")
cb = plt.pcolormesh(dss.longitude, dss.latitude, s_mcp, cmap="Spectral_r", vmin = 100, vmax = 400)
plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.5)
plt.colorbar(cb)


# contour_gen = contourpy.contour_generator(
#     x=np.arange(cca.shape[1]),
#     y=np.arange(cca.shape[0]),
#     z=cca,
#     line_type="Separate",  # renvoie des listes indépendantes
# )

# contours = contour_gen.lines(level=0.5)

# plt.figure()
# plt.title("CCA Algo (%s)"%dss.time.dt.strftime("%Y-%m-%d").values)
# plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
# lon = dss.longitude.values
# lat = dss.latitude.values
# for poly in contours:
#     lon_poly = np.interp(poly[:, 0], np.arange(len(lon)), lon)
#     lat_poly = np.interp(poly[:, 1], np.arange(len(lat)), lat)

#     # plt.plot(poly[:, 0], poly[:, 1],linewidth=2)
#     plt.plot(lon_poly, lat_poly, linewidth=1, c = "k")

# %% CCA perfs 

nbjumps = [5,10,15,20]
CCA = []
for i, nbjump in enumerate(nbjumps): 
    start = time.time()
    cca, x_cca, y_cca = wrapper_cca(dss.thetao.copy(), 
                                    minPopProp=0.2, 
                                    minPopMeanDiff=2, 
                                    minTheta=0.7, 
                                    minSinglePopCohesion=0.9, 
                                    minGlobalPopCohesion =0.7, 
                                    algo ='bf',
                                    njump=nbjump) #used only for bf algo
    end = time.time()
    print("duaration for %i nbjump : %.2f min"%(nbjump, (end-start)/60))
    CCA.append(cca)
    
plt.figure(figsize=(12,10))
for i, nbjump in enumerate(nbjumps):
    cca = CCA[i]
    plt.subplot(2,2,i+1)
    cb = plt.pcolormesh(dss.longitude, dss.latitude, dss.thetao, cmap="cmo.thermal")
    plt.pcolormesh(dss.longitude, dss.latitude, cca, cmap='gray_r', alpha=0.2)
"""