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

def mean_ci(ts, alpha=0.05):
    """
    Compute ensemble mean and 95% confidence intervals
    """
    mean = np.mean(ts, axis=0)
    n = ts.shape[0]
    ci = t.ppf(1 - alpha/2, n-1) * sem(ts, axis=0, nan_policy='omit')
    return mean, ci


model = 'CESM1'
Forced_Unforced_TS_atlantic = xr.open_dataset(path + f'processed_runs/filtered_data/{model}_Atlantic_forced_unforced_TS_25_modes.nc') 
Forced_Unforced_TS_pacific = xr.open_dataset(path + f'processed_runs/filtered_data/{model}_Pacific_forced_unforced_TS_25_modes.nc') 

time = pd.date_range(start='1920-02', end='2025-01', freq='MS')

fig = plt.figure(figsize = (20,15), dpi = 1500)
gs = fig.add_gridspec(4, 8, height_ratios=[0.25, 0.25, 0.25, 0.25], width_ratios=[0.125]*8)


####    Plot Forced and Unforced Stair Plots   ####
atlantic_stair_forced = fig.add_subplot(gs[0, :2])
atlantic_stair_unforced = fig.add_subplot(gs[0, 2:4])
pacific_stair_forced = fig.add_subplot(gs[0, 4:6])
pacific_stair_unforced = fig.add_subplot(gs[0, 6:])


nmembers = Forced_Unforced_TS_atlantic.ensemble.size
nt = Forced_Unforced_TS_atlantic.time.size
lats_a = Forced_Unforced_TS_atlantic.lat.where((Forced_Unforced_TS_atlantic.lat > -45) & (Forced_Unforced_TS_atlantic.lat < 60))
lons_a = Forced_Unforced_TS_atlantic.lon.where((((Forced_Unforced_TS_atlantic.lon > 260) & (Forced_Unforced_TS_atlantic.lon < 360)) | \
                                ((Forced_Unforced_TS_atlantic.lon >= 0) & (Forced_Unforced_TS_atlantic.lon < 60))))

r2_matrices = np.full((2, nmembers, nmembers), np.nan)

for component in range(2):
    if component == 0:
        lfc_filtered_basin_ts = weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['forced']).squeeze(), lats_a, lons_a)
        for i in range(nmembers):
            for j in range(nmembers):
                if i != j:  # Skip diagonal and put NaNs
                    x = lfc_filtered_basin_ts[i]
                    y = lfc_filtered_basin_ts[j]
                    corr = np.corrcoef(x, y)[0, 1]
                    r2_matrices[component, i, j] = corr
    elif component == 1:
        lfc_filtered_basin_ts = weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['unforced']).squeeze(), lats_a, lons_a)
        for i in range(nmembers):
            for j in range(nmembers):
                if i != j:  # Skip diagonal and put NaNs
                    x = lfc_filtered_basin_ts[i]
                    y = lfc_filtered_basin_ts[j]
                    corr = np.corrcoef(x, y)[0, 1]
                    r2_matrices[component, i, j] = corr

mask = np.triu(np.ones((nmembers, nmembers), dtype=bool), k=0)
ax = atlantic_stair_forced
sns.heatmap(
    r2_matrices[0],
    ax=ax,
    cmap="seismic",
    vmin=-1,
    vmax=1,
    square=True,
    cbar = True,
    cbar_kws={"shrink": 0.55, "label": 'Correlation'},
    xticklabels=np.arange(1,nmembers+1),
    yticklabels=np.arange(1,nmembers+1),
    mask = mask
)
ax.set_title(f'{model} Atlantic Forced Response', fontsize = 'x-large')
ax.set_xlabel('Member', fontsize = 'x-large')
ax.set_ylabel('Member', fontsize = 'x-large')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 13)

mean_r2 = np.nanmean(r2_matrices[0])

# Annotate the mean R² value in the top-right white space
ax.text(
    0.95, 0.65,
    f"mean r = {mean_r2:.2f}",
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=13
)

yticks = ax.yaxis.get_major_ticks()
if yticks:
    yticks[0].label1.set_visible(False)
    yticks[0].tick1line.set_visible(False)
    yticks[0].tick2line.set_visible(False)
xticks = ax.xaxis.get_major_ticks()
if xticks:
    xticks[-1].label1.set_visible(False)
    xticks[-1].tick1line.set_visible(False)
    xticks[-1].tick2line.set_visible(False)

#------------------------------------------------------------------------------------------------------------------------------------------

ax = atlantic_stair_unforced
sns.heatmap(
    r2_matrices[1],
    ax=ax,
    cmap="seismic",
    vmin=-1,
    vmax=1,
    square=True,
    cbar = True,
    cbar_kws={"shrink": 0.55, "label": 'Correlation'},
    xticklabels=np.arange(1,nmembers+1),
    yticklabels=np.arange(1,nmembers+1),
    mask = mask
)
ax.set_title(f'{model} Atlantic Unforced Response', fontsize = 'x-large')
ax.set_xlabel('Member', fontsize = 'x-large')
ax.set_ylabel('Member', fontsize = 'x-large')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 13)

mean_r2 = np.nanmean(r2_matrices[1])

# Annotate the mean R² value in the top-right white space
ax.text(
    0.95, 0.65,
    f"mean r = {mean_r2:.2f}",
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=13
)

yticks = ax.yaxis.get_major_ticks()
if yticks:
    yticks[0].label1.set_visible(False)
    yticks[0].tick1line.set_visible(False)
    yticks[0].tick2line.set_visible(False)
xticks = ax.xaxis.get_major_ticks()
if xticks:
    xticks[-1].label1.set_visible(False)
    xticks[-1].tick1line.set_visible(False)
    xticks[-1].tick2line.set_visible(False)
#------------------------------------------------------------------------------------------------------------------------------------------

lats_p = Forced_Unforced_TS_pacific.lat.where((Forced_Unforced_TS_pacific.lat > -45) & (Forced_Unforced_TS_pacific.lat < 70))
lons_p = Forced_Unforced_TS_pacific.lon.where((Forced_Unforced_TS_pacific.lon > 100) & (Forced_Unforced_TS_pacific.lon < 300))

r2_matrices = np.full((2, nmembers, nmembers), np.nan)

for component in range(2):
    if component == 0:
        lfc_filtered_basin_ts = weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['forced']).squeeze(), lats_p, lons_p)
        for i in range(nmembers):
            for j in range(nmembers):
                if i != j:  # Skip diagonal and put NaNs
                    x = lfc_filtered_basin_ts[i]
                    y = lfc_filtered_basin_ts[j]
                    corr = np.corrcoef(x, y)[0, 1]
                    r2_matrices[component, i, j] = corr
    elif component == 1:
        lfc_filtered_basin_ts = weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['unforced']).squeeze(), lats_p, lons_p)
        for i in range(nmembers):
            for j in range(nmembers):
                if i != j:  # Skip diagonal and put NaNs
                    x = lfc_filtered_basin_ts[i]
                    y = lfc_filtered_basin_ts[j]
                    corr = np.corrcoef(x, y)[0, 1]
                    r2_matrices[component, i, j] = corr

mask = np.triu(np.ones((nmembers, nmembers), dtype=bool), k=0)
ax = pacific_stair_forced
sns.heatmap(
    r2_matrices[0],
    ax=ax,
    cmap="seismic",
    vmin=-1,
    vmax=1,
    square=True,
    cbar = True,
    cbar_kws={"shrink": 0.55, "label": 'Correlation'},
    xticklabels=np.arange(1,nmembers+1),
    yticklabels=np.arange(1,nmembers+1),
    mask = mask
)
ax.set_title(f'{model} Pacific Forced Response', fontsize = 'x-large')
ax.set_xlabel('Member', fontsize = 'x-large')
ax.set_ylabel('Member', fontsize = 'x-large')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 13)

mean_r2 = np.nanmean(r2_matrices[0])

# Annotate the mean R² value in the top-right white space
ax.text(
    0.95, 0.65,
    f"mean r = {mean_r2:.2f}",
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=13
)

yticks = ax.yaxis.get_major_ticks()
if yticks:
    yticks[0].label1.set_visible(False)
    yticks[0].tick1line.set_visible(False)
    yticks[0].tick2line.set_visible(False)
xticks = ax.xaxis.get_major_ticks()
if xticks:
    xticks[-1].label1.set_visible(False)
    xticks[-1].tick1line.set_visible(False)
    xticks[-1].tick2line.set_visible(False)

#------------------------------------------------------------------------------------------------------------------------------------------

ax = pacific_stair_unforced
sns.heatmap(
    r2_matrices[1],
    ax=ax,
    cmap="seismic",
    vmin=-1,
    vmax=1,
    square=True,
    cbar = True,
    cbar_kws={"shrink": 0.55, "label": 'Correlation'},
    xticklabels=np.arange(1,nmembers+1),
    yticklabels=np.arange(1,nmembers+1),
    mask = mask
)
ax.set_title(f'{model} Pacific Unforced Response', fontsize = 'x-large')
ax.set_xlabel('Member', fontsize = 'x-large')
ax.set_ylabel('Member', fontsize = 'x-large')
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize = 13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = 13)

mean_r2 = np.nanmean(r2_matrices[1])

# Annotate the mean R² value in the top-right white space
ax.text(
    0.95, 0.65,
    f"mean r = {mean_r2:.2f}",
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=13
)

yticks = ax.yaxis.get_major_ticks()
if yticks:
    yticks[0].label1.set_visible(False)
    yticks[0].tick1line.set_visible(False)
    yticks[0].tick2line.set_visible(False)
xticks = ax.xaxis.get_major_ticks()
if xticks:
    xticks[-1].label1.set_visible(False)
    xticks[-1].tick1line.set_visible(False)
    xticks[-1].tick2line.set_visible(False)


#------------------------------------------------------------------------------------------------------------------------------------------
####    Plot forced time series and other estimates    ####
atlantic_forced_timeseries = fig.add_subplot(gs[1, :4])
pacific_forced_timeseries = fig.add_subplot(gs[1, 4:])

atlantic_forced_timeseries.axhline(
    y=0,
    color='k',
    linestyle='--',
    linewidth=2,
)

pacific_forced_timeseries.axhline(
    y=0,
    color='k',
    linestyle='--',
    linewidth=2,
)

for i in range(nmembers):
    atlantic_forced_timeseries.plot(time, weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['forced']).squeeze(), lats_a, lons_a)[i], linewidth = 0.8, zorder = 1, c = 'lightcoral')
    pacific_forced_timeseries.plot(time, weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['forced']).squeeze(), lats_p, lons_p)[i], linewidth = 0.8, zorder = 1, c = 'lightcoral')

atlantic_forced_timeseries.plot(time, np.mean(weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['forced']).squeeze(), lats_a, lons_a), axis = 0), linewidth = 2, zorder = 3, c = 'k')
pacific_forced_timeseries.plot(time, np.mean(weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['forced']).squeeze(), lats_p, lons_p), axis = 0), zorder = 3, linewidth = 2, c = 'k')


# Compute weighted, lowpass-filtered time series (ensemble x time)
atlantic_ts = weighted_time_series_lowpass(
    Forced_Unforced_TS_atlantic.TS.sel(component='forced').squeeze(),
    lats_a, lons_a
)
pacific_ts = weighted_time_series_lowpass(
    Forced_Unforced_TS_pacific.TS.sel(component='forced').squeeze(),
    lats_p, lons_p
)


# Compute for each basin
atl_mean, atl_ci = mean_ci(atlantic_ts)
pac_mean, pac_ci = mean_ci(pacific_ts)

# Plot Atlantic
atlantic_forced_timeseries.errorbar(
    time,
    atl_mean,
    yerr=atl_ci,
    fmt='none',
    ecolor='k',
    elinewidth=0.75,
    errorevery = 12,
    capsize=2,
    zorder = 3,
)

# Plot Pacific
pacific_forced_timeseries.errorbar(
    time,
    pac_mean,
    yerr=pac_ci,
    fmt='none',
    ecolor='k',
    errorevery =12,
    elinewidth=0.75,
    capsize=2,
    zorder = 3,
)


atlantic_forced_timeseries.set_xlim(time[0], time[-1])
pacific_forced_timeseries.set_xlim(time[0], time[-1])

atlantic_forced_timeseries.set_ylabel('Temperature Anomaly (K)', fontsize = 'x-large')
atlantic_forced_timeseries.set_title(f'{model} Atlantic Forced Time Series', fontsize = 'x-large')
pacific_forced_timeseries.set_title(f'{model} Pacific Forced Time Series', fontsize = 'x-large')

ylims = pacific_forced_timeseries.get_ylim()
atlantic_forced_timeseries.set_ylim(ylims)

pacific_forced_timeseries.grid()
atlantic_forced_timeseries.grid()


#------------------------------------------------------------------------------------------------------------------------------------------

####    Plot unforced time series    ####
atlantic_unforced_timeseries = fig.add_subplot(gs[2, :4])
pacific_unforced_timeseries = fig.add_subplot(gs[2, 4:])

atlantic_unforced_timeseries.axhline(
    y=0,
    color='k',
    linestyle='--',
    linewidth=2,
)

pacific_unforced_timeseries.axhline(
    y=0,
    color='k',
    linestyle='--',
    linewidth=2,
)

for i in range(nmembers):
    atlantic_unforced_timeseries.plot(time, weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['unforced']).squeeze(), lats_a, lons_a)[i], linewidth = 0.8, zorder = 1, c = 'cornflowerblue')
    pacific_unforced_timeseries.plot(time, weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['unforced']).squeeze(), lats_p, lons_p)[i], linewidth = 0.8, zorder = 1, c = 'cornflowerblue')

atlantic_unforced_timeseries.plot(time, np.mean(weighted_time_series_lowpass(Forced_Unforced_TS_atlantic.TS.sel(component = ['unforced']).squeeze(), lats_a, lons_a), axis = 0), linewidth = 2, zorder = 3, c = 'k')
pacific_unforced_timeseries.plot(time, np.mean(weighted_time_series_lowpass(Forced_Unforced_TS_pacific.TS.sel(component = ['unforced']).squeeze(), lats_p, lons_p), axis = 0), zorder = 3, linewidth = 2, c = 'k')


# Compute weighted, lowpass-filtered time series (ensemble x time)
atlantic_ts = weighted_time_series_lowpass(
    Forced_Unforced_TS_atlantic.TS.sel(component='unforced').squeeze(),
    lats_a, lons_a
)
pacific_ts = weighted_time_series_lowpass(
    Forced_Unforced_TS_pacific.TS.sel(component='unforced').squeeze(),
    lats_p, lons_p
)


# Compute for each basin
atl_mean, atl_ci = mean_ci(atlantic_ts)
pac_mean, pac_ci = mean_ci(pacific_ts)

# Plot Atlantic
atlantic_unforced_timeseries.errorbar(
    time,
    atl_mean,
    yerr=atl_ci,
    fmt='none',
    ecolor='k',
    elinewidth=0.75,
    errorevery = 12,
    capsize=2,
    zorder = 3,
)

# Plot Pacific
pacific_unforced_timeseries.errorbar(
    time,
    pac_mean,
    yerr=pac_ci,
    fmt='none',
    ecolor='k',
    errorevery =12,
    elinewidth=0.75,
    capsize=2,
    zorder = 3,
)


atlantic_unforced_timeseries.set_xlim(time[0], time[-1])
pacific_unforced_timeseries.set_xlim(time[0], time[-1])

atlantic_unforced_timeseries.set_ylabel('Temperature Anomaly (K)', fontsize = 'x-large')
atlantic_unforced_timeseries.set_title(f'{model} Atlantic Unforced Time Series', fontsize = 'x-large')
pacific_unforced_timeseries.set_title(f'{model} Pacific Unforced Time Series', fontsize = 'x-large')

#pacific_unforced_timeseries.set_yticks(np.arange(-0.02, 0.03, 0.01))
#atlantic_unforced_timeseries.set_yticks(np.arange(-0.02, 0.03, 0.01))

ylims = pacific_unforced_timeseries.get_ylim()
pacific_unforced_timeseries.set_ylim(ylims)
atlantic_unforced_timeseries.set_ylim(ylims)

pacific_unforced_timeseries.grid()
atlantic_unforced_timeseries.grid()


years = mdates.YearLocator(10)   # every 10 years
years_fmt = mdates.DateFormatter('%Y')

# Apply to each time series axis
for ax in [atlantic_forced_timeseries,
           pacific_forced_timeseries,
           atlantic_unforced_timeseries,
           pacific_unforced_timeseries]:
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.tick_params(axis='x', rotation=45)


#------------------------------------------------------------------------------------------------------------------------------------------

####    Plot AMO/PDO Dot plots    ####
models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']
amo_data = []
pdo_data = []
for model in models:
    amo_data.append(np.load(path + f"FigureFiles/Atlantic_{model}_amo_correlations_conservative.npz", allow_pickle=True))
    pdo_data.append(np.load(path + f"FigureFiles/Pacific_{model}_pdo_correlations_conservative.npz", allow_pickle=True))

amo_dot_plot = fig.add_subplot(gs[3, :4])
pdo_dot_plot = fig.add_subplot(gs[3, 4:])

# collect correlations
corr_full_forced = [amo['corr_full_forced'] for amo in amo_data]
corr_full_internal = [amo['corr_full_internal'] for amo in amo_data]

# mean R^2 per model
ave_corr = np.empty((len(models), 2))
for i in range(len(models)):
    ave_corr[i, 0] = np.mean(corr_full_forced[i]**2)
    ave_corr[i, 1] = np.mean(corr_full_internal[i]**2)

half_width = 1
x_positions = np.array([1,4,7,10])

# --- AMO ---
ax = amo_dot_plot
for i, (model, xpos) in enumerate(zip(models, x_positions)):
    n = len(corr_full_forced[i])
    
    x_min = xpos - half_width
    x_max = xpos + half_width
    
    x_forced = np.linspace(x_min, x_max, n, endpoint=False) + (x_max - x_min) / (2*n)
    x_internal = np.linspace(x_min, x_max, n, endpoint=False) + (x_max - x_min) / (2*n)
    
    ax.scatter(x_forced, corr_full_forced[i]**2, c='r', marker='o', alpha=0.7)
    ax.scatter(x_internal, corr_full_internal[i]**2, c='b', marker='o', alpha=0.7)
    
    # ensemble mean at model center
    ax.scatter(xpos, ave_corr[i, 0], c='r', marker='*', s=150, edgecolor='k', zorder=3)
    ax.scatter(xpos, ave_corr[i, 1], c='b', marker='*', s=150, edgecolor='k', zorder=3)

ax.set_xlim(-0.05, 11.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(x_positions)
ax.set_xticklabels([m + " Ensemble" for m in models])
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylabel(r'$R^{2}$')
ax.set_title('E2001 AMV Index')
ax.grid(alpha=0.5)
ax.legend(['Forced', 'Unforced'], loc='upper left', bbox_to_anchor=(0, 1.2))

#------------------------------------------------------------------------------------------------------------------------------------------
corr_full_forced = [pdo['corr_full_forced'] for pdo in pdo_data]
corr_full_internal = [pdo['corr_full_internal'] for pdo in pdo_data]

# mean R^2 per model
ave_corr = np.empty((len(models), 2))
for i in range(len(models)):
    ave_corr[i, 0] = np.mean(corr_full_forced[i]**2)
    ave_corr[i, 1] = np.mean(corr_full_internal[i]**2)

half_width = 1
x_positions = np.array([1,4,7,10])

# --- AMO ---
ax = pdo_dot_plot
for i, (model, xpos) in enumerate(zip(models, x_positions)):
    n = len(corr_full_forced[i])
    
    x_min = xpos - half_width
    x_max = xpos + half_width
    
    x_forced = np.linspace(x_min, x_max, n, endpoint=False) + (x_max - x_min) / (2*n)
    x_internal = np.linspace(x_min, x_max, n, endpoint=False) + (x_max - x_min) / (2*n)
    
    ax.scatter(x_forced, corr_full_forced[i]**2, c='r', marker='o', alpha=0.7)
    ax.scatter(x_internal, corr_full_internal[i]**2, c='b', marker='o', alpha=0.7)
    
    # ensemble mean at model center
    ax.scatter(xpos, ave_corr[i, 0], c='r', marker='*', s=150, edgecolor='k', zorder=3)
    ax.scatter(xpos, ave_corr[i, 1], c='b', marker='*', s=150, edgecolor='k', zorder=3)

ax.set_xlim(-0.05, 11.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(x_positions)
ax.set_xticklabels([m + " Ensemble" for m in models])
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylabel(r'$R^{2}$')
ax.set_title('PDO Index')
ax.grid(alpha=0.5)
ax.legend(['Forced', 'Unforced'], loc='upper left', bbox_to_anchor=(0, 1.2))


plt.tight_layout()
