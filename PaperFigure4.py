import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy as sp
from scipy import io
import pandas as pd
import glob 
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import sem, t

path = '/var/data/tfreveletti/'

import sys
sys.path.append('/tank/users/tfreveletti')
from lanczos_filter import lanczos_filter

def lowpass_filter(ts):
    '''
    Apply a 10-year lowpass Lanczos filter.
    '''
    #reflected boundary conditions
    x  = np.concatenate((np.flipud(ts), ts, np.flipud(ts)))      
    xfilt  = lanczos_filter(x, 1, 1./120)[0] # 120 = 10 years * 12 months
    
    return xfilt[np.size(xfilt)//3:2*np.size(xfilt)//3]

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']
AMO_data = xr.open_dataset(path + 'processed_runs/AMO_results_ERSSTv5.nc')


lats = np.arange(-90,94,4)
lons = np.linspace(0, 360, 90, endpoint=False)
time = pd.date_range(start='1920-02', end='2025-01', freq='MS')

atlantic_proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, height_ratios=[0.33, 0.33, 0.33], width_ratios=[0.3, 0.2, 0.3, 0.2])

model_colors = {
    "CESM1": "tab:blue",
    "CESM2": "tab:orange",
    "GFDL-CM3": "tab:green",
    "E3SMv2": "tab:red",
}
mean_color = "k"
mean_model_color = "purple" 

# Set stippling by sign
#------------------------------------------------------------
stippling_arr = np.full((2, AMO_data.lat.size, AMO_data.lon.size), False, dtype = bool)

for n in range(2):
    for i, lat in enumerate(AMO_data.lat):
        for j, lon in enumerate(AMO_data.lon):
            if n == 0:
                cesm1_val = AMO_data.beta_forced.sel(model = models[0], lat = lat, lon = lon)
                cesm2_val = AMO_data.beta_forced.sel(model = models[1], lat = lat, lon = lon)
                GDFL_val = AMO_data.beta_forced.sel(model = models[2], lat = lat, lon = lon)
                E3SMv2_val = AMO_data.beta_forced.sel(model = models[3], lat = lat, lon = lon)
                vals = [cesm1_val, cesm2_val, GDFL_val, E3SMv2_val]
        
                if np.sum(vals) == 0:
                    stippling_arr[n, i, j] = False
                    
                if np.all(np.sign(vals) == -1):
                    stippling_arr[n, i, j] = True
                elif np.all(np.sign(vals) == 1):
                    stippling_arr[n, i, j] = True
                else:
                    stippling_arr[n, i, j] = False

            else:
                cesm1_val = AMO_data.beta_unforced.sel(model = models[0], lat = lat, lon = lon)
                cesm2_val = AMO_data.beta_unforced.sel(model = models[1], lat = lat, lon = lon)
                GDFL_val = AMO_data.beta_unforced.sel(model = models[2], lat = lat, lon = lon)
                E3SMv2_val = AMO_data.beta_unforced.sel(model = models[3], lat = lat, lon = lon)
                vals = [cesm1_val, cesm2_val, GDFL_val, E3SMv2_val]
        
                if np.sum(vals) == 0:
                    stippling_arr[n, i, j] = False
                    
                if np.all(np.sign(vals) == -1):
                    stippling_arr[n, i, j] = True
                elif np.all(np.sign(vals) == 1):
                    stippling_arr[n, i, j] = True
                else:
                    stippling_arr[n, i, j] = False


for i in range(3):
    ax_line = fig.add_subplot(gs[i, 0])
    ax_map = fig.add_subplot(gs[i, 1], projection=atlantic_proj)
    
    # Full AMO is unchanging for each member, so just pick the first one.
    if i == 0:
        ax_line.set_title('True AMV', fontsize='x-large')
        ax_line.plot(time, AMO_data['AMO_raw'].isel(model = 0), color=mean_color)    
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) > 0), interpolate=True, color='red')
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) < 0), interpolate=True, color='blue')
        ax_map.pcolormesh(lons, lats, AMO_data['beta_full'].isel(model = 0).values,
                      cmap='seismic', vmin=-1, vmax=1,
                      transform=atlantic_proj)
        ax_map.set_title('True Pattern', fontsize='x-large')

    elif i == 1:
        for model in models:
            ax_line.plot(time, AMO_data['forced_AMO_raw'].sel(model = model), color=model_colors[model], zorder = 3)
        ax_line.plot(time, AMO_data['AMO_raw'].isel(model = 0), color=mean_color, zorder = 1)    
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) > 0), interpolate=True, color='red', alpha = 0.2, zorder = 1)
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) < 0), interpolate=True, color='blue', alpha = 0.2, zorder = 1)    
        ax_line.plot(time, AMO_data['forced_AMO_raw'].mean(dim='model'), color=mean_model_color, linewidth=3, zorder=4)
        ax_map.pcolormesh(lons, lats, AMO_data['beta_forced'].mean(dim = 'model').values,
                  cmap='seismic', vmin=-1, vmax=1,
                  transform=atlantic_proj)
        ax_map.set_title('Multi-Model Mean', fontsize='x-large')
        ax_line.set_title('AMV Forced', fontsize='x-large')

        mask = stippling_arr[i-1]
        lat_idx, lon_idx = np.where(mask)
        lat_points = lats[lat_idx]
        lon_points = lons[lon_idx]
        ax_map.scatter(lon_points, lat_points, s=5, color="k", alpha=0.4)
        
    else:
        for model in models:
            ax_line.plot(time, AMO_data['unforced_AMO_raw'].sel(model = model), color=model_colors[model], zorder = 3)
        ax_line.plot(time, AMO_data['AMO_raw'].isel(model = 0), color=mean_color, zorder = 1)    
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) > 0), interpolate=True, color='red', alpha = 0.2, zorder = 1)
        ax_line.fill_between(time, 0, AMO_data['AMO_raw'].isel(model = 0), \
                         where=(AMO_data['AMO_raw'].isel(model = 0) < 0), interpolate=True, color='blue', alpha = 0.2, zorder = 1)
        ax_line.plot(time, AMO_data['unforced_AMO_raw'].mean(dim='model'), color=mean_model_color, linewidth=3, zorder=4)
        ax_map.pcolormesh(lons, lats, AMO_data['beta_unforced'].mean(dim = 'model').values,
                  cmap='seismic', vmin=-1, vmax=1,
                  transform=atlantic_proj)
        ax_map.set_title('Multi-Model Mean', fontsize='x-large')
        ax_line.set_title('AMV Unforced', fontsize='x-large')

        mask = stippling_arr[i-1]
        lat_idx, lon_idx = np.where(mask)
        lat_points = lats[lat_idx]
        lon_points = lons[lon_idx]
        ax_map.scatter(lon_points, lat_points, s=5, color="k", alpha=0.4)
        
    ax_line.xaxis.set_major_locator(mdates.YearLocator(10))
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_line.set_xlim(time[0], time[-1])
    ax_line.tick_params(axis="x", rotation=40)
    
    ax_line.set_ylim(-0.6, 0.7)
    ax_line.set_ylabel(r"Temperature Anomaly ($^\circ$C)", fontsize='x-large')
    ax_line.grid()

    
    ax_map.coastlines()
    ax_map.set_extent([-90, 40, -1, 61], crs=atlantic_proj)
    ax_map.set_aspect('auto')

    

# -------------------------
# Pacific
# -------------------------
PDO_data = xr.open_dataset(path + 'processed_runs/PDO_results_ERSSTv5.nc')

# Shift longitudes to make 180 the central longitude
lats = PDO_data.lat.values
central_longitude = 180
pacific_proj = ccrs.PlateCarree(central_longitude = central_longitude)
shifted_lon = (lons + 360 - central_longitude) % 360
sorted_indices = np.argsort(shifted_lon)
lons_shifted = lons[sorted_indices]

# Set stippling by sign
#------------------------------------------------------------
stippling_arr = np.full((2, PDO_data.lat.size, PDO_data.lon.size), False, dtype = bool)

for n in range(2):
    for i, lat in enumerate(PDO_data.lat):
        for j, lon in enumerate(PDO_data.lon):
            if n == 0:
                cesm1_val = PDO_data.beta_forced.sel(model = models[0], lat = lat, lon = lon)
                cesm2_val = PDO_data.beta_forced.sel(model = models[1], lat = lat, lon = lon)
                GDFL_val = PDO_data.beta_forced.sel(model = models[2], lat = lat, lon = lon)
                E3SMv2_val = PDO_data.beta_forced.sel(model = models[3], lat = lat, lon = lon)
                vals = [cesm1_val, cesm2_val, GDFL_val, E3SMv2_val]
        
                if np.sum(vals) == 0:
                    stippling_arr[n, i, j] = False
                    
                if np.all(np.sign(vals) == -1):
                    stippling_arr[n, i, j] = True
                elif np.all(np.sign(vals) == 1):
                    stippling_arr[n, i, j] = True
                else:
                    stippling_arr[n, i, j] = False

            else:
                cesm1_val = PDO_data.beta_unforced.sel(model = models[0], lat = lat, lon = lon)
                cesm2_val = PDO_data.beta_unforced.sel(model = models[1], lat = lat, lon = lon)
                GDFL_val = PDO_data.beta_unforced.sel(model = models[2], lat = lat, lon = lon)
                E3SMv2_val = PDO_data.beta_unforced.sel(model = models[3], lat = lat, lon = lon)
                vals = [cesm1_val, cesm2_val, GDFL_val, E3SMv2_val]
        
                if np.sum(vals) == 0:
                    stippling_arr[n, i, j] = False
                    
                if np.all(np.sign(vals) == -1):
                    stippling_arr[n, i, j] = True
                elif np.all(np.sign(vals) == 1):
                    stippling_arr[n, i, j] = True
                else:
                    stippling_arr[n, i, j] = False

for i in range(3):
    ax_line = fig.add_subplot(gs[i, -2])
    ax_map = fig.add_subplot(gs[i, -1], projection=pacific_proj)
    
    # Full AMO is unchanging for each member, so just pick the first one.
    if i == 0:
        ax_line.set_title('True PDO', fontsize='x-large')
        ax_line.plot(time, lowpass_filter(PDO_data['pc1'].isel(model = 0)), color=mean_color)
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) > 0), interpolate=True, color='red')
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) < 0), interpolate=True, color='blue')
        ax_map.pcolormesh(lons_shifted, lats, PDO_data['beta_full'].isel(model = 0).values[:, sorted_indices],
                      cmap='seismic', vmin=-1, vmax=1,
                      transform=ccrs.PlateCarree())
        ax_map.set_title('True Pattern', fontsize='x-large')

    elif i == 1:
        for model in models:
            ax_line.plot(time, lowpass_filter(PDO_data['pc1_forced'].sel(model = model)), color=model_colors[model], zorder = 3)

        ax_line.plot(time, lowpass_filter(PDO_data['pc1'].isel(model = 0)), color=mean_color, zorder = 1)
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) > 0), interpolate=True, color='red', alpha=0.2, zorder = 1)
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) < 0), interpolate=True, color='blue', alpha=0.2, zorder = 1)
        ax_line.plot(time, lowpass_filter(PDO_data['pc1_forced'].mean(dim='model')), color=mean_model_color, linewidth=3, zorder=4)
        ax_map.pcolormesh(lons_shifted, lats, PDO_data['beta_forced'].mean(dim = 'model').values[:, sorted_indices],
                    cmap='seismic', vmin=-1, vmax=1,
                    transform=ccrs.PlateCarree())
        ax_map.set_title('Multi-Model Mean', fontsize='x-large')
        ax_line.set_title('PDO Forced', fontsize='x-large')
        
        mask = stippling_arr[i-1]
        lat_idx, lon_idx = np.where(mask)
        lat_points = lats[lat_idx]
        lon_points = PDO_data.lon.values[lon_idx]
        
        # Shift longitudes to match plotted coordinates
        lon_points_shifted = (lon_points + 360 - central_longitude) % 360
        ax_map.scatter(lon_points_shifted, lat_points, s=5, color="k", alpha=0.4)

    
    else:
        for model in models:
            ax_line.plot(time, lowpass_filter(PDO_data['pc1_unforced'].sel(model = model)), color=model_colors[model], zorder = 3)

        ax_line.plot(time, lowpass_filter(PDO_data['pc1'].isel(model = 0)), color=mean_color, zorder = 1)
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) > 0), interpolate=True, color='red', alpha=0.2, zorder = 1)
        ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                         where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) < 0), interpolate=True, color='blue', alpha=0.2, zorder = 1)
        ax_line.plot(time, lowpass_filter(PDO_data['pc1_unforced'].mean(dim='model')), color=mean_model_color, linewidth=3, zorder=4)
        pcm = ax_map.pcolormesh(lons_shifted, lats, PDO_data['beta_unforced'].mean(dim = 'model').values[:, sorted_indices],
                    cmap='seismic', vmin=-1, vmax=1,
                    transform=ccrs.PlateCarree())
        ax_map.set_title('Multi-Model Mean', fontsize='x-large')
        ax_line.set_title('PDO Unforced', fontsize='x-large')

        mask = stippling_arr[i-1]
        lat_idx, lon_idx = np.where(mask)
        lat_points = lats[lat_idx]
        lon_points = PDO_data.lon.values[lon_idx]
        
        # Shift longitudes to match plotted coordinates
        lon_points_shifted = (lon_points + 360 - central_longitude) % 360
        ax_map.scatter(lon_points_shifted, lat_points, s=5, color="k", alpha=0.4)

    ax_line.set_ylim(-1.75, 1.1)
    event_years = [1925, 1946, 1977, 1999]
    event_dates = [pd.Timestamp(f"{year}-01") for year in event_years]
    for date in event_dates:
        ax_line.axvline(date, color='k', linestyle='--', linewidth=2)

    ax_line.xaxis.set_major_locator(mdates.YearLocator(10))
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_line.set_xlim(time[0], time[-1])
    ax_line.tick_params(axis="x", rotation=40)
    ax_line.grid()
    ax_map.coastlines()
    ax_map.set_extent([105,265,15,65])
    ax_map.set_aspect('auto')

# -------------------------
# Add one global legend
# -------------------------
# Dummy handles for legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in model_colors.values()]
labels = list(model_colors.keys())
# Add mean line handle
handles.append(plt.Line2D([0], [0], color=mean_color, lw=4))
labels.append(" True ERSSTv5 AMO/PDO Index")
# Add model-mean handle
handles.append(plt.Line2D([0], [0], color=mean_model_color, lw=4))
labels.append(" Multi-Model Mean")

fig.legend(handles, labels, loc="upper center", ncol=6, fontsize="xx-large", frameon=False)

# Adjust layout and colorbar
plt.subplots_adjust(hspace=0.4, wspace = 0.3, top=0.90)
cax = fig.add_axes([0.925, 0.225, 0.015, 0.55])
cb = fig.colorbar(pcm, cax=cax, orientation='vertical')
cb.set_label(r"$^\circ$C/STD", size=16)
