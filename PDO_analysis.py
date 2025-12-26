import numpy as np
import xarray as xr
import scipy as sp

path = '/var/data/tfreveletti/'

import sys
sys.path.append('/tank/users/tfreveletti')
from signal_processing import lfca
from lanczos_filter import lanczos_filter

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


Forced_Unforced_TS_data = xr.open_dataset(path + f'processed_runs/filtered_data/{model}_{basin}_forced_unforced_TS_25_modes.nc') 
GMSSTA_data = xr.open_dataset(path + f'processed_runs/{model}_GMSSTA_ensemble.nc')

# remove the GMSSTA from the data
ts_external = remove_GMSSTA_regression(Forced_Unforced_TS_data.TS.sel(component = ['forced']).squeeze(),   GMSSTA_data.GMSSTA)
ts_internal = remove_GMSSTA_regression(Forced_Unforced_TS_data.TS.sel(component = ['unforced']).squeeze(), GMSSTA_data.GMSSTA)
ts_ens_new = remove_GMSSTA_regression(ts_ens.TS, GMSSTA_data.GMSSTA)

s = ts_ens_new.shape

# Subset the PDO region
lats = ts_ens_new.lat.where((ts_ens_new.lat > 20) & (ts_ens_new.lat < 70))
lons = ts_ens_new.lon.where((ts_ens_new.lon > 110) & (ts_ens_new.lon < 260))

ts_pdo = []
forced_ts = []
unforced_ts = []

for i in range(s[0]):
    ts_pdo.append(ts_ens_new[i].where((ts_ens_new[i].lat == lats) & (ts_ens_new[i].lon == lons)))
    forced_ts.append(ts_external[i].where((ts_external[i].lat == lats) & (ts_external[i].lon == lons)))
    unforced_ts.append(ts_internal[i].where((ts_internal[i].lat == lats) & (ts_internal[i].lon == lons)))


PCs = []
EOF = []
PVAR = []

valid_ts_forced = []
valid_ts_unforced = []

x, y = np.meshgrid(lons.values, lats.values)
weights = np.cos(np.radians(y)).flatten()
for i in range(s[0]): 
    domain_ts = ts_pdo[i].values.reshape(s[1], s[2] * s[3])
    domain_ts_forced = forced_ts[i].values.reshape(s[1], s[2] * s[3])
    domain_ts_unforced = unforced_ts[i].values.reshape(s[1], s[2] * s[3])

    # Find valid spatial indices 
    valid_mask = ~np.isnan(domain_ts)
    
    valid_ts = domain_ts[:, valid_mask[0]]
    valid_ts_forced.append(domain_ts_forced[:, valid_mask[0]])
    valid_ts_unforced.append(domain_ts_unforced[:, valid_mask[0]])

    valid_weights = weights[valid_mask[0]]
    
    # Normalize and scale
    normvec = valid_weights / np.nansum(valid_weights)
    scale = np.sqrt(normvec)
    
    pcs, eof, pvar = PCA(valid_ts, scale)  
    PCs.append(pcs)
    EOF.append(eof)
    PVAR.append(pvar)

# Get the forced and unforced components of the PDO index by projecting the forced SSTAs onto the first EOF
pc1 = np.empty((s[0],s[1]))
eof1 = np.full((s[0], s[2], s[3]), np.nan)
valid_eof = []

# Get the first eofs and pcs from the ensemble members
for i in range(ne):
    pc1[i] = PCs[i][:,0]
    eof1[i].flat[valid_mask[0]] = EOF[i][0, :]
    valid_eof.append(EOF[i][0, :])

# Figure out if it is needed to flip the PCs/EOFs
# Indexing location is a point along the U.S. West Coast
for i in range(ne):
    if eof1[i][33,58] < 0:
        valid_eof[i] *= -1
        eof1[i] *= -1
        pc1[i] *= -1

pc1_forced = np.empty((s[0],s[1]))
pc1_unforced = np.empty((s[0],s[1]))

# Project the non-nan ssts onto the eofs and standardize the indices
for i in range(s[0]):
    
    #normalize PC1 to unit standard deviation
    norm = np.std(pc1[i])
   
    pc1[i] = pc1[i] / norm
    pc1_forced[i]   = (np.dot(valid_ts_forced[i] * scale, valid_eof[i])) / norm
    pc1_unforced[i] = (np.dot(valid_ts_unforced[i] * scale, valid_eof[i])) / norm


# Get the correlations between each member
corr_full_forced = np.empty(s[0])
corr_full_unforced = np.empty(s[0])

for i in range(s[0]):  
    corr_full_forced[i]   = sp.stats.pearsonr(pc1[i], pc1_forced[i])[0]
    corr_full_unforced[i] = sp.stats.pearsonr(pc1[i], pc1_unforced[i])[0]

np.savez(path + f"FigureFiles/Pacific_{model}_pdo_correlations.npz",
         corr_full_forced=corr_full_forced,
         corr_full_internal=corr_full_unforced)
