import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy as sp
import pandas as pd
import matplotlib.dates as mdates
from scipy.stats import sem, t
from matplotlib.lines import Line2D 

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


def weighted_time_series_lowpass(ts_da, lats, lons):
    '''
    Calculate a lowpass-weighted time series given an xarray dataArray.
    '''
    if "ensemble" not in ts_da.dims:
        ts_da = ts_da.expand_dims("ensemble", axis=0)

    ts_da = ts_da.transpose("ensemble", "time", "lat", "lon")
    ne = ts_da.sizes["ensemble"]
    ntime = ts_da.sizes["time"]

    lats = np.asarray(lats)
    lons = np.asarray(lons)
    y, x = np.meshgrid(lats, lons, indexing="ij")

    #  zero area-weights outside of the domain
    weights = np.cos(np.radians(y))
    weights = np.where(np.isnan(weights) | np.isnan(x) | np.isnan(y), 0.0, weights)

    weighted_ts = np.empty((ne, ntime))

    for i in range(ne):
        ts_data = ts_da.isel(ensemble=i).values
        valid_mask = weights > 0
        w_masked = weights * valid_mask
        
        # Area-weighted mean
        ts_mean = np.nansum(ts_data * w_masked[np.newaxis, :, :], axis=(1, 2)) / np.nansum(w_masked)
        
        # Low-pass filter
        weighted_ts[i] = lowpass_filter(ts_mean)

    # If the input had no ensemble, squeeze the result
    if "ensemble" not in ts_da.dims:
        return weighted_ts.squeeze()

    return weighted_ts

def ensemble_significance(ts_ens):
    """
    Compute effective sample size, and percentage of insignificant values.
    """
    ts_ens = np.asarray(ts_ens)
    ne = ts_ens.shape[0]
    
    mean = np.nanmean(ts_ens, axis=0)    
    std = np.nanstd(ts_ens, axis=0, ddof=1) # divide by ne - 1 instead of ne
    se = std / np.sqrt(ne)
    z = mean / se
    mask = np.abs(z) < 2  # True where indistinguishable from 0
    
    perc_insig = 100 * np.sum(mask) / len(mask)
    return mean, mask, perc_insig


# Supplementary Figure 1

if model == 'CESM1': 
    time = pd.date_range(start="1920-02", end="2025-01", freq="MS") 
else: 
    time = pd.date_range(start="1920-01", end="2025-01", freq="MS") 
    
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True) 

for i in range(4): 
    ts_ensemble = weighted_time_series_lowpass(TS_unforced.sel(mode=i), lats, lons) 
    mean, mask, perc_insig = ensemble_significance(ts_ensemble) 
    start = 0 
    while start < len(time): 
        end = start 
        while end < len(time) and mask[end] == mask[start]: 
            end += 1 
        linestyle = ":" if mask[start] else "-" # dotted if not significant 
        ax[i].plot(time[start:end], mean[start:end], c="k", linestyle=linestyle) 
        start = end 
    ax[i].set_title(f" 10-year Low-pass Filtered Ensemble-Mean Basin-Mean Unforced SST Anomalies {i+1} Reference Modes") 
    ax[i].grid() 
    ax[i].set_ylabel("SST Anomaly (K)") # Add percentage of insignificant values in top-right corner 
    ax[i].text( 0.98, 1.15, f"Insig: {perc_insig:.1f}%", transform=ax[i].transAxes, ha="right", va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3") ) 

# Custom legend 
legend_elements = [ Line2D([0], [0], color="k", linestyle="-", label="Significant (|z| ≥ 2)"), Line2D([0], [0], color="k", linestyle=":", label="Not significant (|z| < 2)"), ] 
ax[0].legend(handles=legend_elements, loc="upper left") 

# Format x-axis ticks every 10 years 
for axis in ax: 
    ticks = pd.date_range("1920-02", "2025-01", freq="10YS")
    axis.set_xticks(ticks) 
    axis.set_xticklabels([d.strftime("%Y") for d in ticks]) 
    
fig.suptitle(f"{model} {basin}") 
plt.tight_layout()



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Supplementary Figure 5

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[0.25, 0.25, 0.25, 0.25], width_ratios=[0.6, 0.4])

PDO_data = xr.open_dataset(path + 'processed_runs/PDO_results_ERSSTv5.nc')

lats = np.arange(-90,94,4)
lons = np.linspace(0, 360, 90, endpoint=False)
time = pd.date_range(start='1920-02', end='2025-01', freq='MS')

central_longitude = 180
pacific_proj = ccrs.PlateCarree(central_longitude = central_longitude)
shifted_lon = (lons + 360 - central_longitude) % 360
sorted_indices = np.argsort(shifted_lon)
lons_shifted = lons[sorted_indices]



for i in range(len(models)):
    ax_line = fig.add_subplot(gs[i, 0])
    ax_line.plot(time, lowpass_filter(PDO_data['pc1_forced'].sel(model = models[i])), color= 'r', zorder = 3, label = f'ERSSTv5/{models[i]}')
    ax_line.plot(time, lowpass_filter(PDO_data['pc1'].isel(model = 0)), color='k', zorder = 1, label = 'ERSSTv5')
    ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                     where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) > 0), interpolate=True, color='red', alpha=0.2, zorder = 1)
    ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                     where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) < 0), interpolate=True, color='blue', alpha=0.2, zorder = 1)
    
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
    ax_line.set_title(f'ERSSTv5/{models[i]} Forced PDO Index')
    ax_line.legend(loc = 'lower left')
    
    ax_map = fig.add_subplot(gs[i,1], projection=pacific_proj)
    pcm = ax_map.pcolormesh(lons_shifted, lats, PDO_data['beta_forced'].sel(model = models[i]).values[:, sorted_indices],
                    cmap='seismic', vmin=-1, vmax=1,
                    transform=ccrs.PlateCarree())
    ax_map.coastlines()
    ax_map.set_extent([105,265,18,20])
    ax_map.set_aspect('auto')
    ax_map.set_title(f'ERSSTv5/{models[i]} Forced PDO Pattern', fontsize='x-large')

# Adjust layout and colorbar
plt.subplots_adjust(hspace=0.4, wspace = 0.3, top=0.90)
cax = fig.add_axes([0.925, 0.225, 0.015, 0.55])
cb = fig.colorbar(pcm, cax=cax, orientation='vertical')
cb.set_label(r"$^\circ$C/STD", size=16)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Supplementary Figure 6

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[0.25, 0.25, 0.25, 0.25], width_ratios=[0.6, 0.4])

PDO_data = xr.open_dataset(path + 'processed_runs/PDO_results_ERSSTv5.nc')

lats = np.arange(-90,94,4)
lons = np.linspace(0, 360, 90, endpoint=False)
time = pd.date_range(start='1920-02', end='2025-01', freq='MS')

central_longitude = 180
pacific_proj = ccrs.PlateCarree(central_longitude = central_longitude)
shifted_lon = (lons + 360 - central_longitude) % 360
sorted_indices = np.argsort(shifted_lon)
lons_shifted = lons[sorted_indices]

for i in range(len(models)):
    ax_line = fig.add_subplot(gs[i, 0])
    ax_line.plot(time, lowpass_filter(PDO_data['pc1_unforced'].sel(model = models[i])), color= 'b', zorder = 3, label = f'ERSSTv5/{models[i]}')
    ax_line.plot(time, lowpass_filter(PDO_data['pc1'].isel(model = 0)), color='k', zorder = 1, label = 'ERSSTv5')
    ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                     where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) > 0), interpolate=True, color='red', alpha=0.2, zorder = 1)
    ax_line.fill_between(time, 0, lowpass_filter(PDO_data['pc1'].isel(model=0)), \
                     where=(lowpass_filter(PDO_data['pc1'].isel(model=0)) < 0), interpolate=True, color='blue', alpha=0.2, zorder = 1)
    
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
    ax_line.set_title(f'ERSSTv5/{models[i]} Unforced PDO Index')
    ax_line.legend(loc = 'lower left')
    
    ax_map = fig.add_subplot(gs[i, 1], projection=pacific_proj)
    pcm = ax_map.pcolormesh(lons_shifted, lats, PDO_data['beta_unforced'].sel(model = models[i]).values[:, sorted_indices],
                    cmap='seismic', vmin=-1, vmax=1,
                    transform=ccrs.PlateCarree())
    ax_map.coastlines()
    ax_map.set_extent([105,265,18,20])
    ax_map.set_aspect('auto')
    ax_map.set_title(f'ERSSTv5/{models[i]} Unforced PDO Pattern', fontsize='x-large')

# Adjust layout and colorbar
plt.subplots_adjust(hspace=0.4, wspace = 0.3, top=0.90)
cax = fig.add_axes([0.925, 0.225, 0.015, 0.55])
cb = fig.colorbar(pcm, cax=cax, orientation='vertical')
cb.set_label(r"$^\circ$C/STD", size=16)
