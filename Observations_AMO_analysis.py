import numpy as np
import xarray as xr
import scipy as sp

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

def standardize(x):
    return (x - x.mean()) / x.std()

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']
obs_model = 'ERSSTv5'

# Storage initialization
first_model = True

for i_model, model in enumerate(models):
    print(model)
    Forced_Unforced_TS_data = xr.open_dataset(path + f'processed_runs/filtered_data/{obs_model}_{model}_Atlantic_forced_unforced_TS_25_modes.nc') 
    observed_ssta = xr.open_dataset(path + f'processed_runs/{obs_model}_processed_Atlantic_sst.nc')
    
    if obs_model == 'ERSSTv5':
        observed_ssta = observed_ssta.rename_vars({"__xarray_dataarray_variable__": "SSTA"})
    elif obs_model == 'HadISST1.1':
        observed_ssta = observed_ssta.rename_vars({"SST": "SSTA"})
    
    # Slice to common time
    Forced_Unforced_TS_data = Forced_Unforced_TS_data.sel(time = slice('1920-02','2025-01'))
    observed_ssta = observed_ssta.sel(time = slice('1920-02','2025-01'))
    
    observed_ssta = observed_ssta.convert_calendar("noleap")
    
    Xforced = Forced_Unforced_TS_data.SSTA.sel(component=['forced']).squeeze()
    Xunforced = Forced_Unforced_TS_data.SSTA.sel(component=['unforced']).squeeze()
    Xobs = observed_ssta.SSTA
    
    # Compute AMO indices (raw, not standardized)
    forced_AMO = e2001(Xforced)
    unforced_AMO = e2001(Xunforced)
    standard_AMO = e2001(Xobs)

    forced_AMO_std = standardize(forced_AMO)
    unforced_AMO_std = standardize(unforced_AMO)
    standard_AMO_std = standardize(standard_AMO)
    
    # Calculate the pattern based on standard deviations of the index
    beta_full = (Xobs * standard_AMO_std[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(standard_AMO_std**2)
    beta_forced = (Xobs * forced_AMO_std[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(forced_AMO_std**2)
    beta_unforced = (Xobs * unforced_AMO_std[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(unforced_AMO_std**2)
    
    # Initialize storage arrays once
    if first_model:
        n_models = len(models)
        ntime = Xobs.sizes['time']
        nlat = Xobs.sizes['lat']
        nlon = Xobs.sizes['lon']
        
        AMO_raw_arr = np.full((n_models, ntime), np.nan)
        forced_AMO_raw_arr = np.full((n_models, ntime), np.nan)
        unforced_AMO_raw_arr = np.full((n_models, ntime), np.nan)
        
        beta_full_arr = np.full((n_models, nlat, nlon), np.nan)
        beta_forced_arr = np.full((n_models, nlat, nlon), np.nan)
        beta_unforced_arr = np.full((n_models, nlat, nlon), np.nan)
        
        first_model = False
    
    AMO_raw_arr[i_model, :] = standard_AMO
    forced_AMO_raw_arr[i_model, :] = forced_AMO
    unforced_AMO_raw_arr[i_model, :] = unforced_AMO
    
    beta_full_arr[i_model, :, :] = beta_full
    beta_forced_arr[i_model, :, :] = beta_forced
    beta_unforced_arr[i_model, :, :] = beta_unforced

# Build dataset
AMO_dataset = xr.Dataset(
    data_vars={
        "AMO_raw": (("model", "time"), AMO_raw_arr),
        "forced_AMO_raw": (("model", "time"), forced_AMO_raw_arr),
        "unforced_AMO_raw": (("model", "time"), unforced_AMO_raw_arr),
        "beta_full": (("model", "lat", "lon"), beta_full_arr),
        "beta_forced": (("model", "lat", "lon"), beta_forced_arr),
        "beta_unforced": (("model", "lat", "lon"), beta_unforced_arr)
    },
    coords={
        "model": models,
        "time": Xobs.time.values,
        "lat": Xobs.lat.values,
        "lon": Xobs.lon.values
    }
)

# Save
AMO_dataset.to_netcdf(path + f"processed_runs/AMO_results_{obs_model}.nc")
# Save
AMO_dataset.to_netcdf(path + f"processed_runs/AMO_results_{obs_model}.nc")
