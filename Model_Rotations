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
from signal_processing import lfca
from lanczos_filter import lanczos_filter

def create_LFP_EOF(lfps, indices, s, numLFPS):
    '''
    Input computed LFPs and reinstate nans into the correct locations.
    '''
    LFPs = []
    for k in range(numLFPS):
        LFP = np.full((s[1] * s[2]), np.nan) #original data shape
        LFP[indices[0,:]] = lfps[k, :].real  # fill the valid indices with the spatial data
        LFP = LFP.reshape(s[1], s[2])
        LFPs.append(LFP)
    
    return LFPs

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

def calc_r2(lfc_data, true_forcing_ts):
    '''
    Calulate the R-squared value.
    '''
    # calc corr
    corr = sp.stats.pearsonr(lfc_data, true_forcing_ts)[0]
    
    return corr**2

def pattern_rotation_with_unforced(LFCs, LFPs, nLFPs, ref_patterns, AREA_WEIGHTS):
    """
    Rotate LFPs and LFCs to best match reference patterns using area-weighted least squares.
    Also, returns the full unforced component by removing the forced variance in each mode and summing.

    Returns:
        LFCs_rotated
        LFPs_rotated
        unforced_field  --> full residual field from sum of 10 modes
    """

    # extract and weight patterns for regression
    L = LFPs[:nLFPs] * AREA_WEIGHTS.T
    ref = ref_patterns.T * AREA_WEIGHTS
    L = L.T

    # solve least squares for rotation matrix
    beta = np.linalg.inv(L.T @ L) @ (L.T @ ref)
    beta /= np.sqrt(np.sum(beta**2, axis=0, keepdims=True))

    # rotate patterns and time series
    LFPs_rotated = beta.T @ LFPs[:nLFPs, :]
    LFCs_rotated = LFCs[:, :nLFPs] @ beta

    # compute forced projection operator
    F = ref_patterns
    P = F.T @ np.linalg.inv(F @ F.T) @ F

    # compute residuals mode-by-mode
    ntime = LFCs.shape[0]
    nspace = LFPs.shape[1]
    unforced_field = np.zeros((ntime, nspace))

    for i in range(nLFPs):
        filtered_mode = np.outer(LFCs[:, i], LFPs[i, :])
        
        # forced projection for each timestep
        forced_part = filtered_mode @ P.T
        
        # unforced portion of each mode
        residual_part = filtered_mode - forced_part
        
        # summation of all unforced portions
        unforced_field += residual_part

    return LFCs_rotated, LFPs_rotated, unforced_field

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


def mean_ci(ts, alpha=0.05):
    """
    Compute ensemble mean and 95% confidence intervals
    """
    mean = np.mean(ts, axis=0)
    n = ts.shape[0]
    ci = t.ppf(1 - alpha/2, n-1) * sem(ts, axis=0, nan_policy='omit')
    return mean, ci


def remove_GMSSTA_regression(ts_da, gmssta_da):
    """
    Remove GMSSTA contribution via linear regression at each grid point.
    """
    ne, nt, nlat, nlon = ts_da.sizes['ensemble'], ts_da.sizes['time'], ts_da.sizes['lat'], ts_da.sizes['lon']

    # Flatten spatial dims
    ts_flat = ts_da.values.reshape(ne, nt, nlat*nlon)
    gmssta = gmssta_da.values

    residuals_flat = np.full_like(ts_flat, np.nan)
    for i in range(ne):
        X = np.vstack([np.ones(nt), gmssta[i]]).T
        Y = ts_flat[i]

        # Mask valid points
        mask = ~np.isnan(Y[0])
        # assuming nan locations are unchanging through time
        Y_valid = Y[:, mask]

        # Linear regression via least squares
        beta, _, _, _ = np.linalg.lstsq(X, Y_valid, rcond=None)
        # Remove linearly related portion 
        residual_flat = Y_valid - X @ beta
        # Fill non-nan locations
        residuals_flat[i, :, mask] = residual_flat.T

    residuals = xr.DataArray(
        residuals_flat.reshape(ne, nt, nlat, nlon),
        dims=('ensemble','time','lat','lon'),
        coords={'ensemble': ts_da.ensemble, 'time': ts_da.time, 'lat': ts_da.lat, 'lon': ts_da.lon},
        name='TS'
    )

    return residuals

def e2001(ssta):
    '''
    Calculates the standard AMO index defined by Enfield et al. 2001.

    ssta: 3d Xarray DataArray 
    '''
    
    lat = ssta.lat.where((ssta.lat > 0)&(ssta.lat < 60))
    lon = ssta.lon.where((ssta.lon > 280)&(ssta.lon <= 360))
    
    ssta_domain = ssta.where((ssta.lat == lat)&(ssta.lon == lon))
    
    #remove linear trend
    coeffs = ssta_domain.polyfit(dim='time', deg=1)
    fit = xr.polyval(ssta_domain['time'], coeffs.polyfit_coefficients)
    detrended_ssta = (ssta_domain - fit).mean(axis  = (1,2))
   
    return lowpass_filter(detrended_ssta)



def PCA(x, scale):
    '''
    Performs Principal Component Analysis.
    '''
    
    def peigs(a, rmax):
        (m, n) = a.shape
        rmax = min(rmax, min(m, n))
    
        # Use sparse eigendecomposition if rmax is significantly smaller
        if rmax < min(m, n) / 10.:
            d, v = scipy.sparse.linalg.eigs(a, rmax)
        else:
            d, v = np.linalg.eig(a)
    
        d = np.real(d) 
        v = np.real(v)
    
        # Sort eigenvalues in descending order
        idx = np.argsort(-d)
        d = d[idx]
        v = v[:, idx]
    
        # Estimate rank: discard near-zero eigenvalues
        d_min = np.max(d) * max(m, n) * np.finfo(float).eps
        r = np.sum(d > d_min)
    
        return v[:, :r], d[:r], r
    
    if x.ndim != 2:
        return
    else:
        covtot = np.cov(x, rowvar=False)
    (n, p) = x.shape
    if covtot.shape != (p, p):
        return

    # Center data
    x = x - np.nanmean(x, 0)[np.newaxis, ...]

    # Apply scaling
    xs = x * np.transpose(scale)

    # Weighted covariance
    covtot = np.transpose(scale) * covtot * scale
    pcvec, evl, rest = peigs(covtot, min(n - 1, p))
    trcovtot = np.trace(covtot)

    # Percent of total sample variation accounted for by each EOF
    pvar = evl / trcovtot * 100

    eof = np.transpose(pcvec) / np.transpose(scale)

    pcs = np.dot(xs, eof.T)

    return pcs, eof, pvar


# Options:
# CESM1
# CESM2
# GFDL-CM3
# E3SMv2
model = 'E3SMv2'

# Atlantic or Pacific
basin = 'Pacific'

if basin == 'Atlantic':
    ts_ens = xr.open_dataset(path + f'processed_runs/{model}_processed_Atlantic_ts.nc')
    ne = ts_ens.ensemble.size
    ts_ens = ts_ens.rename_vars({"__xarray_dataarray_variable__": "TS"})
    ensemble_mean_ts = ts_ens.mean(dim = 'ensemble').TS
    print(ensemble_mean_ts.shape)
    lats = ensemble_mean_ts.lat.where((ensemble_mean_ts.lat > -45) & (ensemble_mean_ts.lat < 60))
    lons = ensemble_mean_ts.lon.where((((ensemble_mean_ts.lon > 260) & (ensemble_mean_ts.lon < 360)) | \
                                    ((ensemble_mean_ts.lon >= 0) & (ensemble_mean_ts.lon < 60))))

if basin == 'Pacific':
    ts_ens = xr.open_dataset(path + f'processed_runs/{model}_processed_Pacific_ts.nc')
    ne = ts_ens.ensemble.size
    ts_ens = ts_ens.rename_vars({"__xarray_dataarray_variable__": "TS"})
    ensemble_mean_ts = ts_ens.mean(dim = 'ensemble').TS
    print(ensemble_mean_ts.shape)
    lats = ensemble_mean_ts.lat.where((ensemble_mean_ts.lat > -45) & (ensemble_mean_ts.lat < 70))
    lons = ensemble_mean_ts.lon.where((ensemble_mean_ts.lon > 100) & (ensemble_mean_ts.lon < 300))


# Perform ensemble mean LFCA
# LFCA parameters
truncation = 25  # % variance explained by EOFS
cutoff = 10 * 12  # Data is in months

s = ensemble_mean_ts.shape
x, y = np.meshgrid(lons.values, lats.values)
weights = np.cos(np.radians(y))
weights = np.reshape(weights, (s[1] * s[2],1))
domain_ts = np.reshape(ensemble_mean_ts.values, (s[0], s[1] * s[2]))

# Get valid indices
indices = ~np.isnan(domain_ts)
valid_ts = domain_ts[indices]
valid_weights = weights[indices[0, :]]

# Reshape valid weights
input_data = valid_ts.reshape(s[0], int(valid_ts.shape[0] / s[0]))

# Normalize and scale
normvec = valid_weights / np.sum(valid_weights)
scale = np.sqrt(normvec)

print('\nEnsemble LFCA beginning...\n')

ref_lfcs, ref_lfps, fps, ref_r, pvar, pcs, eof, N, ref_pvar_LFP = lfca(input_data,1,cutoff,truncation,scale)

print('\nEnsemble LFCA Performed...\n')

#np.savez(path + f"FigureFiles/{basin}_{model}_ensemble_lfca_outputs.npz",
#         ref_lfcs=ref_lfcs,
#         ref_lfps=ref_lfps,
#         ref_r=ref_r,
#         ref_pvar_LFP=ref_pvar_LFP,
#         indices=indices,
#         s = s)

## Perform LFCA on each member and perform the rotation toward the reference patterns

if basin == 'Pacific':
    lats = ensemble_mean_ts.lat.where((ensemble_mean_ts.lat > -45) & (ensemble_mean_ts.lat < 70))
    lons = ensemble_mean_ts.lon.where((ensemble_mean_ts.lon > 100) & (ensemble_mean_ts.lon < 300))

if basin == 'Atlantic':
    lats = ensemble_mean_ts.lat.where((ensemble_mean_ts.lat > -45) & (ensemble_mean_ts.lat < 60))
    lons = ensemble_mean_ts.lon.where((((ensemble_mean_ts.lon > 260) & (ensemble_mean_ts.lon < 360)) | \
                                    ((ensemble_mean_ts.lon >= 0) & (ensemble_mean_ts.lon < 60))))

LFCs = []
LFPs = []
fingerprints = []
r = []
pvar = []
PCs = []
EOF = []
N = []
pvar_LFP = []


# Reshape data
ne = ts_ens.ensemble.size

print('\nBeginning Single LFCA...\n')

for i in range(ne):
    TS = ts_ens.TS.sel(ensemble = i).values
    s = TS.shape
    x, y = np.meshgrid(lons.values, lats.values)
    weights = np.cos(np.radians(y))
    weights = np.reshape(weights, (s[1] * s[2],1))
    domain_ts = np.reshape(TS, (s[0], s[1] * s[2]))
    
    # Get valid indices
    indices = ~np.isnan(domain_ts)
    valid_ts = domain_ts[indices]
    valid_weights = weights[indices[0, :]]

    # Reshape valid weights
    input_data = valid_ts.reshape(s[0], int(valid_ts.shape[0] / s[0]))
    
    # Normalize and scale
    normvec = valid_weights / np.sum(valid_weights)
    scale = np.sqrt(normvec)

    
    print('\nEns Mem: ' + str(i))
    lfc,lfp,fp,r_vals,pvar_vals,pcs,eof,N_vals,pvar_LFP_vals = lfca(input_data,1,cutoff,truncation,scale)
    LFCs.append(lfc)
    LFPs.append(lfp)
    fingerprints.append(fp)
    r.append(r_vals)
    pvar.append(pvar_vals)
    PCs.append(pcs)
    EOF.append(eof)
    N.append(N_vals)
    pvar_LFP.append(pvar_LFP_vals)

print('\nSingle LFCA done...\n')

#############################################    NOW PERFORMING THE ROTATION    ###########################################
# ----------------------------------------------------------------------------------------------------------------------- #

# Rotation parameters
numModesLFP = 25  # Change this value depending on how many modes you'd like to keep for the rotation and unforced field reconstruction
numModesREF = 5   # Setting this here to make sure enough are included, but the number kept depends on later criteria

rotated_LFCs = np.empty([ne, LFCs[0].shape[0], numModesREF])
rotated_LFPs = np.empty([ne, numModesREF,  LFPs[0].shape[-1]])
unforced_field = np.empty([numModesREF, ne, LFCs[0].shape[0],LFPs[0].shape[-1]])

print('\nPerforming Rotation...\n')

for i in range(ne):
    rotated_LFCs[i], rotated_LFPs[i], _  = pattern_rotation_with_unforced(LFCs[i], LFPs[i], numModesLFP, ref_lfps[:numModesREF, :], scale)

# Calculate the unforced field based on how many reference patterns are included...
for i in range(numModesREF):
    for n in range(ne):
        _, _, unforced_field[i,n]  = pattern_rotation_with_unforced(LFCs[n], LFPs[n], numModesLFP, ref_lfps[:i, :], scale)

## Convert forced and unforced components back into gridded spatiotemporal data
print('\nReconstructing Spatiotemporal Data...\n')

s = ts_ens.TS.shape

Xf = np.empty((s[0], numModesREF, LFCs[0].shape[0], LFPs[0].shape[-1])) 

for n in range(s[0]):
    Xf[n] = np.einsum('tm,ms->mts', rotated_LFCs[n], rotated_LFPs[n])

# array with original spatial dimensions, filled with nan
Xf_filled = np.full((s[0], numModesREF, s[1], s[2] * s[3]), np.nan)
Xu_filled = np.full((numModesREF, s[0], s[1], s[2] * s[3]), np.nan)

# use valid indices from the last ensemble member; assume these indices don't change across members
inds = np.where(indices[0])[0]  # Get valid index locations for the first time step; assume they do not change across time

for n in range(s[0]):
    for mode in range(numModesREF):
        for p in range(len(inds)):
            Xf_filled[n, mode, :, inds[p]] = Xf[n, mode, :, p]
            Xu_filled[mode, n, :, inds[p]] = unforced_field[mode, n, :, p]


# Reshape to: (ensemble members, numModesREF, time, lat, lon)
Xf_filled = Xf_filled.reshape((s[0], numModesREF, s[1], s[2], s[3]))
Xu_filled = Xu_filled.reshape((numModesREF, s[0], s[1], s[2], s[3]))

TS_forced = xr.DataArray(
    Xf_filled,
    dims=["ensemble", "mode", "time", "lat", "lon"],
    coords={
        "ensemble": ts_ens.ensemble,
        "mode": np.arange(0, numModesREF),
        "time": ts_ens.time,
        "lat": ts_ens.lat,
        "lon": ts_ens.lon,
    },
    name="TS"
)

TS_unforced = xr.DataArray(
    Xu_filled,
    dims=["mode", "ensemble", "time", "lat", "lon"],
    coords={
        "mode": np.arange(0, numModesREF),
        "ensemble": ts_ens.ensemble,
        "time": ts_ens.time,
        "lat": ts_ens.lat,
        "lon": ts_ens.lon,
    },
    name="TS"
)

print('\nAnalysis Performed.')

# New Criteria:
# Number of reference modes included is determined by when 80% or greater of 
# the 10-year lowpass filtered ensemble mean unforced TS 
# is indistiguishable from 0... i.e. less than 2 STDs

perc_insig = np.empty((numModesREF))

for i in range(numModesREF):
    ts_ensemble = weighted_time_series_lowpass(TS_unforced.sel(mode=i), lats, lons)
    _, _, perc_insig[i] = ensemble_significance(ts_ensemble)
    print(f'Reference Modes Included: {i + 1}')
    print('% values insignificant = ', perc_insig[i], '\n')

stopVal = np.where(perc_insig > 80)[0][0] # mode number where the inclusion reaches the insignifigance level

print(f'Number of Reference Patterns Included: {stopVal + 1}')

print('\nCreating Final Dataset...')

forced = TS_forced.sel(mode=np.arange(stopVal+1)).sum(dim="mode")
unforced = TS_unforced.sel(mode=stopVal)

# Combine along a new dimension "component"
final_dataset = xr.concat(
    [forced, unforced],
    dim=xr.DataArray(["forced", "unforced"], dims="component", name="component")
).to_dataset(name="TS")

final_dataset.to_netcdf(path + f'processed_runs/filtered_data/{model}_{basin}_forced_unforced_TS_{truncation}_modes.nc')

print('\nFinished...')
