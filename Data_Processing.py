import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import glob 

path = '/var/data/tfreveletti/'

def order_files(files):
    import re
    def extract_member(file):
        match = re.search(r'/HIST_(\d{4})_', file)
        return int(match.group(1))

    def extract_sim(file):
        match = re.search(r'/HIST_1231_(\d{3})_', file)
        return int(match.group(1))
        
    file_keys = []

    for file in files:
        
        member = extract_member(file)

        if member == 1231:
            
            sim = extract_sim(file)
            key = (member, sim) 
            
        else:
            
            key = (member, 0)

        file_keys.append((file, key))
    
    # Sort by the key
    file_keys.sort(key=lambda x: x[1])

    sorted_files = [f for f, key in file_keys]
    
    return sorted_files

def regrid(ds,res):
    # Build the output dataset
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], np.arange(-90, 92, res)),
            "lon": (["lon"], np.arange(0, 360, res)),
        }
    )
    
    # Create the bilinear regridder
    regridder = xe.Regridder(ds, ds_out, "bilinear", periodic = True)
    
    # Regrid the dataset
    ds_out = regridder(ds)
    
    return ds_out

# Replace values of greater than 0.50 with np.nan and all other values with 1
def make_mask(ds):
    land_mask = xr.where(ds >=0.50, np.nan, 1)
    if 'type' in land_mask.coords:
        land_mask = land_mask.drop_vars('type')
    return land_mask

#multiply in np.nan values
def remove_land(ds,lm):
    return ds * lm[0]

#nan out other land blocks to get NA area
def rm_cells(da, lat_u, lat_d, lon_l, lon_r):
    # Extract the coordinates
    latitudes = da['lat'].values
    longitudes = da['lon'].values
    
    # Create boolean masks
    lat_mask = ((latitudes < lat_u) & (latitudes > lat_d))
    lon_mask = ((longitudes > lon_l) & (longitudes < lon_r))
    
    # Create a 2D boolean mask
    mask = np.outer(lat_mask, lon_mask)
    
    # Expand mask to 3D 
    # Determine the shape of the data array for broadcasting
    
    data_shape = da.TS.shape if hasattr(da, 'TS') else da.shape
    mask_3d = np.broadcast_to(mask, data_shape)

    # Apply the mask to all time steps
    modified_da = da.where(~mask_3d, np.nan)
    
    return modified_da

def adjust_temperature(ds):
    return ds.where(ds >= 273.15, 273.15)

def rm_seasonal_cycle(ds):
    return ds.groupby('time.month') - ds.groupby('time.month').mean(dim='time')


# CESM1
model = 'CESM1'
path = '/var/data/tfreveletti/CESM1/'

files = glob.glob(path + 'b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.TS.*.nc')
files = sorted(files, key=lambda f: int(f.split('.')[4])) # sort by ensemble number

data = [xr.open_dataset(i) for i in files]
data[0] = data[0].sel(time=slice("1920-02", "2006-01")) #remove first 70 years from first ensemble member
data = [ds.TS for ds in data]

land_data = xr.open_dataset(path + 'b.e11.B20TRC5CNBDRD.f09_g16.002.cam.h0.LANDFRAC.192001-200512.nc')

#Set the number of ensemble members
ne = len(files)

#CESM2
model = 'CESM2'
path = '/var/data/tfreveletti/CESM2/'

files = glob.glob(path + 'Combined_Data/HIST*.nc')

sorted_files = order_files(files)

ne = len(files)

data = [xr.open_dataset(i) for i in sorted_files]

land_data = xr.open_dataset(path + 'b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.LANDFRAC.185001-185912.nc')

data = [ds.TS.sel(time = slice('1920', '2005')) for ds in data]

#GFDL-CM3
model = 'GFDL-CM3'
path = '/var/data/tfreveletti/'

files = glob.glob(path + f'{model}/*.nc')
ne = len(files)

data = [xr.open_dataset(i) for i in files]

land_data = xr.open_dataset(path + 'CESM2/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.LANDFRAC.185001-185912.nc')

data = [ds.ts.sel(time = slice('1920-01', '2025-01')) for ds in data]

#E3SMv2
model = 'E3SMv2'
path = '/var/data/tfreveletti/'

files = glob.glob(path + f'{model}/*.nc')
ne = len(files)

data = [xr.open_dataset(i) for i in files]

land_data = xr.open_dataset(path + 'CESM2/b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h0.LANDFRAC.185001-185912.nc')

data = [ds.ts.sel(time = slice('1920-01', '2025-01')) for ds in data]

# 1 for Atlantic
# 2 for Pacific
basin_num = 2

####    regrid    ####
resolution = 4

datasets_regridded  = [regrid(ds,resolution) for ds in data]
land_regridded  = regrid(land_data,resolution)

####    Make land mask   ####
land_mask  = make_mask(land_regridded)

####    Remove land using masks and nan blocks    ####
datasets_regridded_noland  = [remove_land(ds, land_mask.LANDFRAC) for ds in datasets_regridded]

if basin_num == 1:
    basin = 'Atlantic'
    for i in range(ne):
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 10, 0, 250, 290)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 17, 0, 250, 278)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 70, 50, 260, 285)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 28, -90, 20, 100)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 50,25,40,60)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 5,-45,250,290)

elif basin_num == 2:
    basin = 'Pacific'
    for i in range(ne):
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], -7, -50, 100, 130)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], -25, -50, 100, 150)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 70, 11, 260, 300)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], 12, 8, 270,300)
        datasets_regridded_noland[i] = rm_cells(datasets_regridded_noland[i], -38, -45, 290,300)
    
    
####   correct for below-freezing temperatures    ####
data_corrected = [adjust_temperature(ds) for ds in datasets_regridded_noland]

####    remove seasonal cycle    ####
data_final = [rm_seasonal_cycle(ds) for ds in data_corrected]

####    CREATE CONCATENATED SURFACE TEMPERATURE ARRAYS    ####

#### initialize the lat and lon    ####
if basin_num == 1:
    #Atlantic
    lats = data_final[0].lat.where((data_final[0].lat > -45) & (data_final[0].lat < 60))
    lons = data_final[0].lon.where((((data_final[0].lon > 260) & (data_final[0].lon < 360)) | \
                                    ((data_final[0].lon >= 0) & (data_final[0].lon < 60))))

elif basin_num == 2:    
    #Pacific
    lats = data_final[0].lat.where((data_final[0].lat > -45) & (data_final[0].lat < 70))
    lons = data_final[0].lon.where((data_final[0].lon > 100) & (data_final[0].lon < 300))
    
ts = []
for i in range(ne):
    ts.append(data_final[i].where((data_final[i].lat == lats) & 
                                     (data_final[i].lon == lons)))

ts_final = xr.concat(ts, dim = 'ensemble')
ts_final.to_netcdf(path + f'processed_runs/{model}_processed_{basin}_ts.nc')
