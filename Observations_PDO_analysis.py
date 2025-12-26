import numpy as np
import xarray as xr
import scipy as sp


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

def remove_GMSSTA_regression(ts_da, gmssta_da):
    """
    Remove GMSSTA contribution via linear regression at each grid point.
    """
    nt, nlat, nlon = ts_da.sizes['time'], ts_da.sizes['lat'], ts_da.sizes['lon']

    # Flatten spatial dims
    ts_flat = ts_da.values.reshape(nt, nlat*nlon)
    gmssta = gmssta_da.values

    residuals_flat = np.full_like(ts_flat, np.nan)
    X = np.vstack([np.ones(nt), gmssta]).T
    Y = ts_flat

    # Mask valid points
    mask = ~np.isnan(Y[0])
    # assuming nan locations are unchanging through time
    Y_valid = Y[:, mask]

    # Linear regression via least squares
    beta, _, _, _ = np.linalg.lstsq(X, Y_valid, rcond=None)
    # Remove linearly related portion 
    residual_flat = Y_valid - X @ beta
    # Fill non-nan locations
    residuals_flat[:, mask] = residual_flat

    residuals = xr.DataArray(
        residuals_flat.reshape(nt, nlat, nlon),
        dims=('time','lat','lon'),
        coords={'time': ts_da.time, 'lat': ts_da.lat, 'lon': ts_da.lon},
        name='TS'
    )

    return residuals

models = ['CESM1', 'CESM2', 'GFDL-CM3', 'E3SMv2']
obs_model = 'ERSSTv5'

# Storage arrays initialized after first model loop
first_model = True

for i_model, model in enumerate(models):
    print(model)
    Forced_Unforced_TS_data = xr.open_dataset(path + f'processed_runs/filtered_data/{obs_model}_{model}_Pacific_forced_unforced_TS_25_modes.nc') 
    GMSSTA_data = xr.open_dataset(path + f'{obs_model}_GMSSTA.nc')
    observed_ssta = xr.open_dataset(path + f'processed_runs/{obs_model}_processed_Pacific_sst.nc')

    Forced_Unforced_TS_data = Forced_Unforced_TS_data.sel(time = slice('1920-02','2025-01'))
    GMSSTA_data = GMSSTA_data.sel(time = slice('1920-02','2025-01'))
    observed_ssta = observed_ssta.sel(time = slice('1920-02','2025-01'))
    
    if obs_model == 'ERSSTv5':
        observed_ssta = observed_ssta.rename_vars({"__xarray_dataarray_variable__": "SSTA"})
    elif obs_model == 'HadISST1.1':
        observed_ssta = observed_ssta.rename_vars({"SST": "SSTA"})
    
    observed_ssta = observed_ssta.convert_calendar("noleap")
    GMSSTA_data = GMSSTA_data.convert_calendar("noleap")
    
    # Align times
    model_ym = list(zip(Forced_Unforced_TS_data.time.dt.year.values, Forced_Unforced_TS_data.time.dt.month.values))
    obs_ym = list(zip(observed_ssta.time.dt.year.values, observed_ssta.time.dt.month.values))
    common_ym = set(model_ym).intersection(set(obs_ym))
    
    Forced_Unforced_TS_aligned = Forced_Unforced_TS_data.sel(
        time=[t for t in Forced_Unforced_TS_data.time.values if (t.year, t.month) in common_ym]
    )
    observed_ssta_aligned = observed_ssta.sel(
        time=[t for t in observed_ssta.time.values if (t.year, t.month) in common_ym]
    )
    GMSSTA_aligned = GMSSTA_data.sel(
        time=[t for t in GMSSTA_data.time.values if (t.year, t.month) in common_ym]
    )
    
    # remove GMSSTA
    sst_external = remove_GMSSTA_regression(
        Forced_Unforced_TS_aligned.SSTA.sel(component = ['forced']).squeeze(),   
        GMSSTA_aligned.GMSSTA.squeeze()
    )
    sst_internal = remove_GMSSTA_regression(
        Forced_Unforced_TS_aligned.SSTA.sel(component = ['unforced']).squeeze(),
        GMSSTA_aligned.GMSSTA.squeeze()
    )
    sst_full = remove_GMSSTA_regression(observed_ssta_aligned.SSTA, GMSSTA_aligned.GMSSTA.squeeze())
    
    # PDO region subset
    lats = sst_full.lat.where((sst_full.lat > 20) & (sst_full.lat < 70))
    lons = sst_full.lon.where((sst_full.lon > 110) & (sst_full.lon < 260))
    
    ssta_pdo_region = sst_full.where((sst_full.lat == lats) & (sst_full.lon == lons))
    forced_ssta_pdo_region = sst_external.where((sst_external.lat == lats) & (sst_external.lon == lons))
    unforced_ssta_pdo_region = sst_internal.where((sst_internal.lat == lats) & (sst_internal.lon == lons))
    
    s = ssta_pdo_region.shape 
    x, y = np.meshgrid(lons.values, lats.values)
    weights = np.cos(np.radians(y)).flatten()
    
    ssta_pdo_region_r = ssta_pdo_region.values.reshape(s[0], s[1] * s[2])
    forced_ssta_pdo_region_r = forced_ssta_pdo_region.values.reshape(s[0], s[1] * s[2])
    unforced_ssta_pdo_region_r = unforced_ssta_pdo_region.values.reshape(s[0], s[1] * s[2])
    
    valid_mask = ~np.isnan(ssta_pdo_region_r)
    valid_ssta = ssta_pdo_region_r[:, valid_mask[0]]
    valid_forced_ssta = forced_ssta_pdo_region_r[:, valid_mask[0]]
    valid_unforced_ssta = unforced_ssta_pdo_region_r[:, valid_mask[0]]
    valid_weights = weights[valid_mask[0]]
    
    # Normalize and scale
    normvec = valid_weights / np.nansum(valid_weights)
    scale = np.sqrt(normvec)
    
    pcs, eof, pvar = PCA(valid_ssta, scale)
    
    pc1= pcs[:,0]
    eof1 = np.full((46, 90), np.nan)
    eof1.flat[valid_mask[0]] = eof[0, :]
    
    if eof1[33,58] < 0:
        eof1 *= -1
        pc1 *= -1
        valid_eof = eof[0,:] * -1
    else:
        valid_eof = eof[0,:]
    
    #normalize PC1 to unit standard deviation
    norm = np.std(pc1)
    pc1 = pc1 / norm
    
    #Calculate forced PC1s and normalize so that they add up to full PC1
    pc1_forced   = (np.dot(valid_forced_ssta * scale, valid_eof)) / norm
    pc1_unforced = (np.dot(valid_unforced_ssta * scale, valid_eof)) / norm

    #Standardize them for calculating the regression patterns
    pc1_std = (pc1 - np.mean(pc1)) / np.std(pc1)
    pc1_std_forced = (pc1_forced - np.mean(pc1_forced)) / np.std(pc1_forced)
    pc1_std_unforced = (pc1_unforced - np.mean(pc1_unforced)) / np.std(pc1_unforced)
    
    beta_full = (ssta_pdo_region * pc1_std[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(pc1_std**2)
    beta_forced = (ssta_pdo_region * pc1_std_forced[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(pc1_std_forced**2)
    beta_unforced = (ssta_pdo_region * pc1_std_unforced[:, np.newaxis, np.newaxis]).sum(dim='time') / np.sum(pc1_std_unforced**2)
    
    # Initialize storage arrays once
    if first_model:
        n_models = len(models)
        ntime = ssta_pdo_region.sizes['time']
        nlat = ssta_pdo_region.sizes['lat']
        nlon = ssta_pdo_region.sizes['lon']
        
        pc1_arr = np.full((n_models, ntime), np.nan)
        pc1_forced_arr = np.full((n_models, ntime), np.nan)
        pc1_unforced_arr = np.full((n_models, ntime), np.nan)
        
        beta_full_arr = np.full((n_models, nlat, nlon), np.nan)
        beta_forced_arr = np.full((n_models, nlat, nlon), np.nan)
        beta_unforced_arr = np.full((n_models, nlat, nlon), np.nan)
        
        first_model = False
    
    pc1_arr[i_model,:] = pc1
    pc1_forced_arr[i_model,:] = pc1_forced
    pc1_unforced_arr[i_model,:] = pc1_unforced
    
    beta_full_arr[i_model,:,:] = beta_full
    beta_forced_arr[i_model,:,:] = beta_forced
    beta_unforced_arr[i_model,:,:] = beta_unforced

# Build dataset
PDO_dataset = xr.Dataset(
    data_vars={
        "pc1": (("model", "time"), pc1_arr),
        "pc1_forced": (("model", "time"), pc1_forced_arr),
        "pc1_unforced": (("model", "time"), pc1_unforced_arr),
        "beta_full": (("model", "lat", "lon"), beta_full_arr),
        "beta_forced": (("model", "lat", "lon"), beta_forced_arr),
        "beta_unforced": (("model", "lat", "lon"), beta_unforced_arr)
    },
    coords={
        "model": models,
        "time": ssta_pdo_region.time.values,
        "lat": ssta_pdo_region.lat.values,
        "lon": ssta_pdo_region.lon.values
    }
)

# Save
PDO_dataset.to_netcdf(path + f"processed_runs/PDO_results_{obs_model}.nc")
