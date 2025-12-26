import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib.dates as mdates

path = '/var/data/tfreveletti/'

import sys
sys.path.append('/tank/users/tfreveletti')
from lanczos_filter import lanczos_filter

def create_LFP_EOF(lfps, indices, s, numLFPS):
    '''
    Input computed LFPs and reinstate nans into the correct locations.
    '''
    LFPs = []
    # k = number of modes wanted
    for k in range(numLFPS):
        LFP = np.full((s[1] * s[2]), np.nan) #original data shape
        LFP[indices[0,:]] = lfps[k, :].real  # fill the valid indices with the spatial data
        LFP = LFP.reshape(s[1], s[2])
        LFPs.append(LFP)
    
    return LFPs

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']


################### Atlantic #######################
atlantic_data = [np.load(path + f"FigureFiles/Atlantic_{model}_ensemble_lfca_outputs.npz", allow_pickle=True) for model in models]
Forced_Unforced_TS_data = [xr.open_dataset(path + f'processed_runs/filtered_data/{model}_Atlantic_forced_unforced_TS_25_modes.nc') for model in models]
stopVals = [(int(Forced_Unforced_TS_data[i].mode.values) + 1) for i in range(len(models))]
ref_lfps = [atlantic_data[i]["ref_lfps"] for i in range(len(models))]
indices = [atlantic_data[i]['indices'] for i in range(len(models))]
s = [atlantic_data[i]['s'] for i in range(len(models))]
ref_r = [atlantic_data[i]['ref_r'] for i in range(len(models))]
ref_pvar_LFP = [atlantic_data[i]['ref_pvar_LFP'] for i in range(len(models))]

ref_LFPs_filled = [create_LFP_EOF(ref_lfps[i], indices[i], s[i], stopVals[i]) for i in range(len(models))]

#set the LFCs and LFPs, fill models that don't have 4 reference modes with nan
ref_LFPs_plot = np.full((4,4,46,90), np.nan)
ref_LFPs_plot[0, :3] = ref_LFPs_filled[0][:3]
ref_LFPs_plot[1] = ref_LFPs_filled[1][:4]
ref_LFPs_plot[2] = ref_LFPs_filled[2][:4]
ref_LFPs_plot[3] = ref_LFPs_filled[3][:4]


lats = np.arange(-90,94,4)
lons = np.linspace(0, 360, 90, endpoint=False)
time = pd.date_range(start='1920-02', end='2025-01', freq='MS')

atlantic_proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(4, 5, height_ratios=[0.25, 0.25, 0.25, 0.25], width_ratios=[3, 1.66, 1.66, 1.66, 1.66])

#---------------------------------------------------------------------------------------------------------------------------------------
############ LFCs Plots
#---------------------------------------------------------------------------------------------------------------------------------------
ref_lfcs = [atlantic_data[i]["ref_lfcs"] for i in range(len(models))]

# have to set the time from 1: for the back 3 models because CESM1 starts from 1920-02
ref_lfcs_plot = np.full((4,1260,4), np.nan)
ref_lfcs_plot[0, :, :3] = ref_lfcs[0][:,:3]
ref_lfcs_plot[1] = ref_lfcs[1][1:,:4]
ref_lfcs_plot[2] = ref_lfcs[2][1:,:4]
ref_lfcs_plot[3] = ref_lfcs[3][1:,:4]


model_colors = {
    "CESM1": "tab:blue",
    "CESM2": "tab:orange",
    "GFDL-CM3": "tab:green",
    "E3SMv2": "tab:red",
}

for i in range(4): # loop over modes
    ax_line = fig.add_subplot(gs[i, 0])
    for j, model in enumerate(models):  # loop over models
        if (j == 0) & (i == 1):
            ax_line.plot(time, ref_lfcs_plot[j, :, i]*-1, color=model_colors[model]) #flips CESM1 LFC 2
        else:
            ax_line.plot(time, ref_lfcs_plot[j, :, i], color=model_colors[model])
    ax_line.set_title(f'Atlantic Ensemble Mean LFC {i+1}', fontsize='x-large')
    ax_line.xaxis.set_major_locator(mdates.YearLocator(10))
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_line.set_xlim(time[0], time[-1])
    ax_line.tick_params(axis="x", rotation=40)
    ax_line.set_ylim(-4, 4)
    ax_line.set_ylabel("Standard Deviations", fontsize='large')
    ax_line.grid()


#---------------------------------------------------------------------------------------------------------------------------------------
############ CESM1
#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[0, 1], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[0,0],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[0][0]:.2f}  {models[0]} LFP 1  {ref_pvar_LFP[0][0]:.2f}%', fontsize='large')



#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[1, 1], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[0,1]*-1,
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[0][1]:.2f}  {models[0]} LFP 2  {ref_pvar_LFP[0][1]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[2, 1], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[0,2],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[0][2]:.2f}  {models[0]} LFP 3  {ref_pvar_LFP[0][2]:.2f}%', fontsize='large')


#---------------------------------------------------------------------------------------------------------------------------------------
############ CESM2
#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[0, 2], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[1,0],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[1][0]:.2f}  {models[1]} LFP 1  {ref_pvar_LFP[1][0]:.2f}%', fontsize='large')



#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[1, 2], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[1,1],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[1][1]:.2f}  {models[1]} LFP 2  {ref_pvar_LFP[1][1]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[2, 2], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[1,2],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[1][2]:.2f}  {models[1]} LFP 3  {ref_pvar_LFP[1][2]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[3, 2], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[1,3],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[1][3]:.2f}  {models[1]} LFP 4  {ref_pvar_LFP[1][3]:.2f}%', fontsize='large')


#---------------------------------------------------------------------------------------------------------------------------------------
############ GFDL-CM3
#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[0, 3], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[2,0],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[2][0]:.2f}  {models[2]} LFP 1  {ref_pvar_LFP[2][0]:.2f}%', fontsize='large')



#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[1, 3], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[2,1],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[2][1]:.2f}  {models[2]} LFP 2  {ref_pvar_LFP[2][1]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[2, 3], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[2,2],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[2][2]:.2f}  {models[2]} LFP 3  {ref_pvar_LFP[2][2]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[3, 3], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[2,3],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[2][3]:.2f}  {models[2]} LFP 4  {ref_pvar_LFP[2][3]:.2f}%', fontsize='large')


#---------------------------------------------------------------------------------------------------------------------------------------
############ E3SMv2
#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[0, 4], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[3,0],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[3][0]:.2f}  {models[3]} LFP 1  {ref_pvar_LFP[3][0]:.2f}%', fontsize='large')



#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[1, 4], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[3,1],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[3][1]:.2f}  {models[3]} LFP 2  {ref_pvar_LFP[3][1]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[2, 4], projection=atlantic_proj)
ax_map.pcolormesh(lons, lats, ref_LFPs_plot[3,2],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[3][2]:.2f}  {models[3]} LFP 3  {ref_pvar_LFP[3][2]:.2f}%', fontsize='large')




#---------------------------------------------------------------------------------------------------------------------------------------
ax_map = fig.add_subplot(gs[3, 4], projection=atlantic_proj)
pcm = ax_map.pcolormesh(lons, lats, ref_LFPs_plot[3,3],
                  cmap='seismic', vmin=-0.5, vmax=0.5,
                  transform=atlantic_proj)
ax_map.coastlines()
ax_map.set_extent([-90, 40, -45, 62], crs=atlantic_proj)
ax_map.set_aspect('auto')
ax_map.set_title(f'{ref_r[3][3]:.2f}  {models[3]} LFP 4  {ref_pvar_LFP[3][3]:.2f}%', fontsize='large')


# -------------------------
# Add one global legend
# -------------------------
# Dummy handles for legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in model_colors.values()]
labels = list(model_colors.keys())

fig.legend(handles, labels, loc="upper center", ncol=5, fontsize="xx-large", frameon=False)

# Adjust layout and colorbar
plt.subplots_adjust(hspace=0.55, top=0.90)
cax = fig.add_axes([0.925, 0.225, 0.015, 0.55])
cb = fig.colorbar(pcm, cax=cax, orientation='vertical')
cb.set_label(r"$^\circ$C/STD", size=16)
