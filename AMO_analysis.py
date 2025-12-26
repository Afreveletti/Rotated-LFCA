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


Forced_Unforced_TS_data = xr.open_dataset(path + f'processed_runs/filtered_data/{model}_{basin}_forced_unforced_TS_25_modes_conservative.nc') 

ts_external = Forced_Unforced_TS_data.TS.sel(component = ['forced']).squeeze()
ts_internal = Forced_Unforced_TS_data.TS.sel(component = ['unforced']).squeeze()
s = ts_ens.TS.shape

full_e2001 = np.empty((s[0],s[1]))
forced_e2001 = np.empty((s[0],s[1]))
internal_e2001 = np.empty((s[0],s[1]))
residual_e2001 = np.empty((s[0],s[1]))

for i in range(s[0]):
    full_e2001[i] = e2001(ts_ens.TS[i])
    forced_e2001[i] = e2001(ts_external[i])
    internal_e2001[i] = e2001(ts_internal[i])

corr_full_forced = np.empty(s[0])
corr_full_internal = np.empty(s[0])

for i in range(s[0]):  
    corr_full_forced[i]   = sp.stats.pearsonr(full_e2001[i], forced_e2001[i])[0]
    corr_full_internal[i] = sp.stats.pearsonr(full_e2001[i], internal_e2001[i])[0]

np.savez(path + f"FigureFiles/{basin}_{model}_amo_correlations_conservative.npz",
         corr_full_forced=corr_full_forced,
         corr_full_internal=corr_full_internal)
