import numpy as np
import xarray as xr

path = '/var/data/tfreveletti/'

import sys
sys.path.append('/tank/users/tfreveletti')
from signal_processing import lfca
from lanczos_filter import lanczos_filter

def pattern_rotation_with_unforced(LFCs, LFPs, nLFPs, ref_patterns, AREA_WEIGHTS):
    """
    Rotate LFPs and LFCs to best match reference patterns using area-weighted least squares.
    Also, returns the full unforced component by removing the forced variance in each mode and summing.

    Returns:
        LFCs_rotated
        LFPs_rotated
        unforced_field  --> full residual field from sum of N modes
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
    w = AREA_WEIGHTS.flatten()          # sqrt weights
    W = w**2                      # full area weights
    P = F.T @ np.linalg.inv((F * W) @ F.T) @ (F * W)

    #P = F.T @ np.linalg.inv(F @ F.T) @ F

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

# Open for later use
Forced_Unforced_TS_data = xr.open_dataset(path + f'processed_runs/filtered_data/{model}_{basin}_forced_unforced_TS_25_modes.nc') 

# ERSSTv5 or HadISST1.1
obs_model = 'ERSSTv5'

observed_ssta = xr.open_dataset(path + f'processed_runs/{obs_model}_processed_{basin}_sst.nc')

if obs_model == 'ERSSTv5':
    observed_ssta = observed_ssta.rename_vars({"__xarray_dataarray_variable__": "SSTA"})
elif obs_model == 'HadISST1.1':
    observed_ssta = observed_ssta.rename_vars({"SST": "SSTA"})

print('Aligning Model and Observations along the time dimension...\n')
observed_ssta = observed_ssta.convert_calendar("noleap")

# Access year and month of cftime time coordinates
model_year = ensemble_mean_ts.time.dt.year
model_month = ensemble_mean_ts.time.dt.month

obs_year = observed_ssta.time.dt.year
obs_month = observed_ssta.time.dt.month

# Create a combined (year, month) tuple array for alignment
model_ym = list(zip(model_year.values, model_month.values))
obs_ym   = list(zip(obs_year.values, obs_month.values))

# Find common year-month combinations
common_ym = set(model_ym).intersection(set(obs_ym))

# Select only times in the common year-months
ensemble_mean_ts_aligned = ensemble_mean_ts.sel(
    time=[t for t in ensemble_mean_ts.time.values if (t.year, t.month) in common_ym]
)
observed_ssta_aligned = observed_ssta.sel(
    time=[t for t in observed_ssta.time.values if (t.year, t.month) in common_ym]
).SSTA


print('Times aligned. \n')

print(f'{obs_model} first and last times')
print(observed_ssta_aligned.time[0].values)
print(observed_ssta_aligned.time[-1].values)
print('\n')

print(f'{model} first and last times')
print(ensemble_mean_ts_aligned.time[0].values)
print(ensemble_mean_ts_aligned.time[-1].values)
print('\n')

print('Performing Spatial Alignment... \n')
# Since spatial Nans do not align perfectly, use observations as a mask for the model values
# Nans are static throughout time, so only need to use the first time step
obs_mask_first = observed_ssta_aligned.isel(time=0).isnull()

# Apply the same mask to all model times
ensemble_mean_ts_final = ensemble_mean_ts_aligned.where(~obs_mask_first)

# Verify NaN locations match
model_mask = ensemble_mean_ts_final.isel(time=0).isnull()
if not np.array_equal(model_mask.values, obs_mask_first.values):
    raise ValueError("NaN grid locations differ. Execution stopped.")

print('Performing LFCA... \n')

# LFCA parameters
truncation = 25  # % variance explained by EOFS
cutoff = 10 * 12  # Data is in months

s = observed_ssta_aligned.shape
x, y = np.meshgrid(lons.values, lats.values)
weights = np.cos(np.radians(y))
weights = np.reshape(weights, (s[1] * s[2],1))

model_domain_ts = np.reshape(ensemble_mean_ts_final.values, (s[0], s[1] * s[2]))
obs_domain_ts = np.reshape(observed_ssta_aligned.values, (s[0], s[1] * s[2]))

# Get valid indices
indices = ~np.isnan(obs_domain_ts)

valid_obs_ts = obs_domain_ts[indices]
valid_model_ts = model_domain_ts[indices]

valid_weights = weights[indices[0, :]]

# Reshape valid weights
obs_input_data = valid_obs_ts.reshape(s[0], int(valid_obs_ts.shape[0] / s[0]))
model_input_data = valid_model_ts.reshape(s[0], int(valid_model_ts.shape[0] / s[0]))

# Normalize and scale
normvec = valid_weights / np.sum(valid_weights)
scale = np.sqrt(normvec)

print('Model: ')
ref_lfcs, ref_lfps, _, _, _, _, _, _, _ = lfca(model_input_data, 1, cutoff, truncation, scale)
print('\n')
print('Observations:')
obs_lfcs, obs_lfps, _, _, _, _, _, _, _ = lfca(obs_input_data, 1, cutoff, truncation, scale)
print('\n')

print('LFCA on Ensemble Mean & Observations complete... \n')
print('Now Performing Rotation... \n')

# Rotation parameters
numModesLFP = 25
numModesREF = int(Forced_Unforced_TS_data.mode.values) + 1 # value determined via model experiments

rotated_LFCs = np.empty([obs_lfcs[0].shape[0], numModesREF])
rotated_LFPs = np.empty([numModesREF, obs_lfps[0].shape[-1]])
unforced_field = np.empty([obs_lfcs[0].shape[0], obs_lfps[0].shape[-1]])

rotated_LFCs, rotated_LFPs, unforced_field  = pattern_rotation_with_unforced(obs_lfcs.real, obs_lfps.real, numModesLFP, \
                                                                             ref_lfps[:numModesREF, :], scale)

print('\nReconstructing Spatiotemporal Data...\n')

Xf = np.empty((obs_lfcs[0].shape[0], obs_lfps[0].shape[-1])) 
Xf = np.einsum('tm,ms->ts', rotated_LFCs, rotated_LFPs)

# array with original spatial dimensions, filled with nan
Xf_filled = np.full((s[0], s[1] * s[2]), np.nan)
Xu_filled = np.full((s[0], s[1] * s[2]), np.nan)


inds = np.where(indices[0])[0]  # Get valid index locations
for p in range(len(inds)):
    Xf_filled[:, inds[p]] = Xf[:, p]
    Xu_filled[:, inds[p]] = unforced_field[:, p]


# Reshape to time, lat, lon
Xf_filled = Xf_filled.reshape((s[0], s[1], s[2]))
Xu_filled = Xu_filled.reshape((s[0], s[1], s[2]))

SSTA_forced = xr.DataArray(
    Xf_filled,
    dims=["time", "lat", "lon"],
    coords={
        "time": ensemble_mean_ts_final.time,
        "lat": ensemble_mean_ts_final.lat,
        "lon": ensemble_mean_ts_final.lon,
    },
    name="SSTA"
)

SSTA_unforced = xr.DataArray(
    Xu_filled,
    dims=["time", "lat", "lon"],
    coords={
        "time": ensemble_mean_ts_final.time,
        "lat": ensemble_mean_ts_final.lat,
        "lon": ensemble_mean_ts_final.lon,
    },
    name="SSTA"
)

print('\nAnalysis Performed.')

print('\nCreating Final Dataset...')

final_obs_dataset = xr.concat(
    [SSTA_forced, SSTA_unforced],
    dim=xr.DataArray(["forced", "unforced"], dims="component", name="component")
).to_dataset(name="SSTA")

final_obs_dataset.to_netcdf(path + f'processed_runs/filtered_data/{obs_model}_{model}_{basin}_forced_unforced_TS_25_modes.nc')

print('\nFinished.')
