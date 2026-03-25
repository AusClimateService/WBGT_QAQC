# Use this code to apply QAQC test to WBGT data produced for Treasury
# Author - Cassandra Rogers
# NOTE - these tests are specific to Treasury's aims. If WBGT are to be used for other purposes, other QAQC tests may be necessary


######################################################################################################################################


# Import modules - some of these can be removed if not used in the script
import numpy as np
import netCDF4 as nc
import xarray as xr
import math
import time as time_pack
import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import datetime
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import pandas as pd
import timeit
import glob
import argparse


######################################################################################################################################


# User inputs

crop_to_aust_region = 'yes'
mask_ocean = 'yes'
mask_nonaust_land = 'yes'


def main(inargs):

    input_dataset = inargs.dataset
    data_type = inargs.data_type
    start_year = inargs.StartYr
    end_year = inargs.EndYr
    set_model = inargs.model
    set_SSP = inargs.ssp
    dir_out = inargs.dir_out

    ######################################################################################################################################


    # BARRA      = 1979-2024
    # BARPA hist = 1960-2014
    # BARPA ssp  = 2015-2100
    # CCAM hist  = 1952-2014
    # CCAM ssp   = 2015-2100

    # If years specified by user are outside the possible range, adjust years to maximum range
    if (input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']) and (start_year > 2025):
        start_year = 1979
        end_year = 2024
    elif (set_SSP == 'historical') and (start_year > 2014):
        if input_dataset == 'CCAM':
            start_year = 1952
        else:
            start_year = 1960
        end_year = 2014
    elif (set_SSP in ['ssp126', 'ssp370']) and (start_year < 2015):
        start_year = 2015
        end_year = 2100

    if (input_dataset in ['CCAM']) and (end_year == 2100):
        end_year = 2099
    
    ######################################################################################################################################


    if input_dataset not in ['BARRA', 'BARRA-R', 'BARRA-R2']:
        if set_model in ['ACCESS-CM2']:
            set_run = 'r4i1p1f1'
        elif set_model in ['ACCESS-ESM1-5']:
            set_run = 'r6i1p1f1'
        elif set_model in ['CESM2']:
            set_run = 'r11i1p1f1'
        elif set_model in ['CMCC-ESM2', 'EC-Earth3', 'NorESM2-MM', 'MPI-ESM1-2-HR']:
            set_run = 'r1i1p1f1'
        elif set_model in ['CNRM-ESM2-1']:
            set_run = 'r1i1p1f2'


    ######################################################################################################################################


    if crop_to_aust_region == 'yes':
        geospatial_lat_min = -45.76
        geospatial_lat_max = -4.95
        geospatial_lon_min = 107.95
        geospatial_lon_max = 159.98


    ######################################################################################################################################


    # Specify function to allow reading in partial file

    from functools import partial
    def _preprocess(x, lon_bnds, lat_bnds):
        try:
            return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))
        except:
            return x.sel(longitude=slice(*lon_bnds), latitude=slice(*lat_bnds))

    lon_bnds, lat_bnds = (geospatial_lon_min, geospatial_lon_max), (geospatial_lat_min, geospatial_lat_max)
    partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)


    ######################################################################################################################################


    # Define functions for the relationship between WBGT and T

    def wbgt_t_min_limit(t,buffer=0):
        swbgt_t = -0.433 * t + 3.94 - buffer
        # -0.433 x + 3.94
        return swbgt_t

    def wbgt_t_max_limit(t,buffer=0):
        swbgt_t = 1.916 * 10**(-6) * (t**4) + 9.481 * 10**(-5) * (t**3) + 0.005353 * (t**2) - 0.2554 * t + 6.357 + buffer
        return swbgt_t


    ######################################################################################################################################


    # Get WBGT filenames

    # BARRA
    if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
        wbgt_var_name = 'wbgt'
        if data_type == 'raw':
            data_location_wbgt = '/scratch/tp28/wbgt/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/wbgt/v20250901/'
        elif data_type == 'bias_adjusted':
            sys.exit()
        fnames_wbgt = glob.glob('{}{}*.nc'.format(data_location_wbgt,wbgt_var_name))

    # BARPA
    elif input_dataset in ['BARPA', 'BARPA-R', 'BARPA-R2']:
        wbgt_var_name = 'wbgt'
        if data_type == 'raw':
            data_location_wbgt = '/scratch/tp28/wbgt/BARPA/output/CMIP6/DD/AUS-15/BOM/' + set_model + '/' + set_SSP + '/' + set_run + '/BARPA-R/v1-r1/1hr/wbgt/v20250901/'
        elif data_type == 'bias_adjusted':
            data_location_wbgt = '/g/data/ia39/australian-climate-service/test-data/CORDEX/output-CMIP6/bias-adjusted-output/AUST-11i/BOM/ACCESS-ESM1-5/historical/r6i1p1f1/CCAM-v2203-SN/v1-r1-ACS-QME-BARRAR2-1980-2022/1hr/wbgtAdjust/v20251120/'
        fnames_wbgt = glob.glob('{}{}*.nc'.format(data_location_wbgt,wbgt_var_name))

    # CCAM
    elif input_dataset in ['CCAM']:
        wbgt_var_name = 'wbgt'
        if data_type == 'raw':
            data_location_wbgt = '/scratch/e53/bxn599/aus10i_uncompressed_yearly/' + set_model + '_' + set_SSP + '/'
        elif data_type == 'bias_adjusted':
            data_location_wbgt = '/g/data/ia39/australian-climate-service/test-data/CORDEX/output-CMIP6/bias-adjusted-output/AUST-11i/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1/CCAM-v2203-SN/v1-r1-ACS-QME-BARRAR2-1980-2022/1hr/wbgtAdjust/v20251120/'
        fnames_wbgt = glob.glob('{}{}*.nc'.format(data_location_wbgt,wbgt_var_name))

    if data_type == 'bias_adjusted':
        wbgt_var_name = 'wbgtAdjust'

    fnames_wbgt.sort()

    year_array_wbgt = np.zeros(len(fnames_wbgt)) * np.nan
    month_array_wbgt = np.zeros(len(fnames_wbgt)) * np.nan


    for i in np.arange(len(year_array_wbgt)):
        if data_type == 'raw':
            if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2','BARPA', 'BARPA-R', 'BARPA-R2']:
                year_array_wbgt[i] = int(fnames_wbgt[i][-9:-5])
                month_array_wbgt[i] = int(fnames_wbgt[i][-5:-3])
            else:
                year_array_wbgt[i] = int(fnames_wbgt[i][-15:-11])
                month_array_wbgt[i] = int(fnames_wbgt[i][-11:-9])
        elif data_type == 'bias_adjusted':
            year_array_wbgt[i] = int(fnames_wbgt[i][-11:-7])
            month_array_wbgt[i] = int(fnames_wbgt[i][-7:-5])


    print(data_location_wbgt)
    print(fnames_wbgt[0])
    print(fnames_wbgt[-1])


    ######################################################################################################################################


    # T data
    # long_name = Near-Surface Air Temperature
    # standard_name = air_temperature
    # units = K

    # BARRA
    if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
        t_var_name = 'tas'
        data_location_t = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/' + t_var_name + '/latest/'
        fnames_t = glob.glob('{}{}*.nc'.format(data_location_t,t_var_name))

    # BARPA
    elif input_dataset in ['BARPA', 'BARPA-R', 'BARPA-R2']:
        t_var_name = 'tas'
        data_location_t = '/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/' + set_model + '/' + set_SSP + '/' + set_run + '/BARPA-R/v1-r1/1hr/' + t_var_name + '/latest/'
        fnames_t = glob.glob('{}{}*.nc'.format(data_location_t,t_var_name))

    # CCAM
    elif input_dataset in ['CCAM']:
        t_var_name = 'tas'
        if set_SSP == 'historical':
            data_location_t = '/g/data/hq89/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/1hr/' + t_var_name + '/v20231206/'
        else:
            data_location_t = '/g/data/xv83/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/1hr/' + t_var_name + '/v20250912/'
        fnames_t = glob.glob('{}{}*.nc'.format(data_location_t,t_var_name))

    fnames_t.sort()

    year_array_t = np.zeros(len(fnames_t)) * np.nan
    month_array_t = np.zeros(len(fnames_t)) * np.nan

    for i in np.arange(len(year_array_t)):
        if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2','BARPA', 'BARPA-R', 'BARPA-R2']:
            year_array_t[i] = int(fnames_t[i][-9:-5])
            month_array_t[i] = int(fnames_t[i][-5:-3])
        else:
            year_array_t[i] = int(fnames_t[i][-11:-7])
            month_array_t[i] = int(fnames_t[i][-7:-5])


    print(data_location_t)
    print(fnames_t[0])
    print(fnames_t[-1])


    ######################################################################################################################################


    # sfcWind
    # long_name = Near-Surface Wind Speed
    # standard_name = wind_speed
    # units = m s-1

    # BARRA
    if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
        wind_var_name = 'sfcWind'
        data_location_wind = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/1hr/' + wind_var_name + '/latest/'
        fnames_wind = glob.glob('{}{}*.nc'.format(data_location_wind,wind_var_name))

    # BARPA
    elif input_dataset in ['BARPA', 'BARPA-R', 'BARPA-R2']:
        wind_var_name = 'sfcWind'
        data_location_wind = '/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/' + set_model + '/' + set_SSP + '/' + set_run + '/BARPA-R/v1-r1/1hr/' + wind_var_name + '/latest/'
        fnames_wind = glob.glob('{}{}*.nc'.format(data_location_wind,wind_var_name))

    elif input_dataset in ['CCAM']:
        wind_var_name = 'sfcWind'
        if (set_model == 'ACCESS-ESM1-5') and (set_SSP == 'historical'):
            version_number = 'v20240327'
        elif set_SSP == 'historical':
            version_number = 'v20231206'
        if set_SSP == 'historical':
            data_location_wind = '/g/data/hq89/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/1hr/' + wind_var_name + '/' + version_number + '/'
        else:
            data_location_wind = '/g/data/xv83/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/1hr/' + wind_var_name + '/v20250912/'
        fnames_wind = glob.glob('{}{}*.nc'.format(data_location_wind,wind_var_name))

    fnames_wind.sort()

    year_array_wind = np.zeros(len(fnames_wind)) * np.nan
    month_array_wind = np.zeros(len(fnames_wind)) * np.nan

    for i in np.arange(len(year_array_wind)):
        if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2','BARPA', 'BARPA-R', 'BARPA-R2']:
            year_array_wind[i] = int(fnames_wind[i][-9:-5])
            month_array_wind[i] = int(fnames_wind[i][-5:-3])
        else:
            year_array_wind[i] = int(fnames_wind[i][-11:-7])
            month_array_wind[i] = int(fnames_wind[i][-7:-5])


    print(data_location_wind)
    print(fnames_wind[0])
    print(fnames_wind[-1])


    ######################################################################################################################################


    # Import land area fraction data
    if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
        laf_var_name = 'sftlf'
        laf_dir = '/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1/fx/' + laf_var_name + '/latest/'
    elif input_dataset in ['BARPA', 'BARPA-R', 'BARPA-R2']:
        laf_var_name = 'sftlf'
        laf_dir = '/g/data/py18/BARPA/output/CMIP6/DD/AUS-15/BOM/' + set_model + '/' + set_SSP + '/' + set_run + '/BARPA-R/v1-r1/fx/' + laf_var_name + '/latest/'
    elif input_dataset in ['CCAM']:
        laf_var_name = 'sftlf'
        if set_SSP == 'historical':
            laf_dir = '/g/data/hq89/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/fx/' + laf_var_name + '/v20231206/'
        else:
            laf_dir = '/g/data/xv83/CCAM/output/CMIP6/DD/AUS-10i/CSIRO/' + set_model + '/' + set_SSP + '/' + set_run + '/CCAM-v2203-SN/v1-r1/fx/' + laf_var_name + '/v20250912/'

    laf_fnames = glob.glob('{}*.nc'.format(laf_dir))
    laf_fnames.sort()

    if (len(laf_fnames) == 2) and (input_dataset in ['CCAM']) and (laf_fnames[0][:-3] + '_fx.nc' == laf_fnames[1]):
        laf_fnames = [laf_fnames[1]]

    assert len(laf_fnames) == 1

    laf_ds = xr.open_mfdataset(laf_fnames[0],preprocess=partial_func)


    ######################################################################################################################################


    # Create ocean mask (100% water = np.nan, <100% water = 1)
    land_regions_gr0pc = np.array(laf_ds[laf_var_name].data).copy()
    land_regions_gr0pc[np.where(land_regions_gr0pc == 0)] = np.nan
    land_regions_gr0pc[np.where(land_regions_gr0pc > 0)] = 1.0


    ######################################################################################################################################


    lats_array = np.ones(np.shape(land_regions_gr0pc)) * np.nan
    lons_array = np.ones(np.shape(land_regions_gr0pc)) * np.nan

    [num_lats, num_lons] = np.shape(land_regions_gr0pc)
    for i in np.arange(num_lons):
        lats_array[:,i] = laf_ds.lat.data
    for i in np.arange(num_lats):
        lons_array[i,:] = laf_ds.lon.data


    ######################################################################################################################################


    # Create mask to remove land north of Australia - Indonesia, PNG
    nonaust_mask = np.ones(np.shape(laf_ds[laf_var_name].data))
    nonaust_mask[np.where(lats_array > -9.5)] = np.nan
    nonaust_mask[np.where((lats_array > -15) * (lons_array > 147))] = np.nan
    nonaust_mask[np.where((lats_array > -12) * (lons_array < 130))] = np.nan


    ######################################################################################################################################


    # Read in data for each month/year, identify where WBGT data passes/fails test, output results to text files

    total_diff_gr_15_bin = 0
    total_diff_5_to_15_and_wind_gr1 = 0


    for i in np.arange(end_year - start_year + 1):
        year_year = start_year + i

        # Create empty arrays to store data to output in text files
        lats_out_diff_gr_15_bin = np.array([])
        lons_out_diff_gr_15_bin = np.array([])
        time_out_diff_gr_15_bin = []
        wbgt_out_diff_gr_15_bin = np.array([])
        t_out_diff_gr_15_bin = np.array([])
        wind_out_diff_gr_15_bin = np.array([])
        curve_out_diff_gr_15_bin = np.array([])

        lats_out_diff_wind_bin = np.array([])
        lons_out_diff_wind_bin = np.array([])
        time_out_diff_wind_bin = []
        wbgt_out_diff_wind_bin = np.array([])
        t_out_diff_wind_bin = np.array([])
        wind_out_diff_wind_bin = np.array([])
        curve_out_diff_wind_bin = np.array([])

        # Specify whether to read in one file per month (BARRA) or one file per year (BARPA, CCAM)
        if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']: # For monthly BARRA data
            set_range = np.arange(0,12)
        elif input_dataset in ['BARPA', 'BARPA-R', 'BARPA-R2', 'CCAM']: # For annual BARPA/CCAM data
            set_range = np.arange(0,1)

        for k in set_range:
            month_month = np.unique(month_array_wbgt)[k]

            #############################################################################################################################################
            # Import data and perform some basic tests

            # Determine indices for specifying file from filename list
            ind_wbgt = np.where((year_array_wbgt == year_year) * (month_array_wbgt == month_month))[0][0]
            ind_t = np.where((year_array_t == year_year) * (month_array_t == month_month))[0][0]
            ind_wind = np.where((year_array_wind == year_year) * (month_array_wind == month_month))[0][0]

            ds_wbgt = xr.open_mfdataset(fnames_wbgt[ind_wbgt],parallel=True,compat='no_conflicts',preprocess=partial_func)
            ds_t = xr.open_mfdataset(fnames_t[ind_t],parallel=True,compat='no_conflicts',preprocess=partial_func)
            ds_wind = xr.open_mfdataset(fnames_wind[ind_wind],parallel=True,compat='no_conflicts',preprocess=partial_func)

            # Make sure data are in the correct units - if not, variables will need to be converted
            assert ds_wbgt[wbgt_var_name].units in ['K','degK','degC']
            assert ds_t[t_var_name].units in ['K','degK']
            assert ds_wind[wind_var_name].units in ['m s-1']

            # Unchunk (compute) data and convert temperature variables from K to C
            if ds_wbgt[wbgt_var_name].units in ['K','degK']:
                ds_wbgt_computed = ds_wbgt[wbgt_var_name].compute() - 273.15
            elif ds_wbgt[wbgt_var_name].units in ['degC']:
                ds_wbgt_computed = ds_wbgt[wbgt_var_name].compute()
            ds_t_computed = ds_t[t_var_name].compute() - 273.15
            ds_wind_computed = ds_wind[wind_var_name].compute()

            # Make sure dimensions are in the correct order - if not, script will need to be altered
            try: ds_wbgt_computed = ds_wbgt_computed.transpose('time', 'lat', 'lon')
            except: ds_wbgt_computed = ds_wbgt_computed.transpose('time', 'latitude', 'longitude')
            assert ds_wbgt_computed.dims[0] == 'time'
            assert (ds_wbgt_computed.dims[1] == 'lat') or (ds_wbgt_computed.dims[1] == 'latitude')
            assert (ds_wbgt_computed.dims[2] == 'lon') or (ds_wbgt_computed.dims[2] == 'longitude')

            # Make sure dimensions are the same size
            assert np.shape(ds_wbgt_computed) == np.shape(ds_t_computed)
            assert np.shape(ds_wbgt_computed) == np.shape(ds_wind_computed)
            
            # APPLY OCEAN MASK
            if mask_ocean == 'yes':
                ds_wbgt_computed = ds_wbgt_computed * land_regions_gr0pc
            if mask_nonaust_land == 'yes':
                ds_wbgt_computed = ds_wbgt_computed * nonaust_mask

            #############################################################################################################################################
            # Perform analysis to determine where potential issues are

            wbgt_t_diff = ds_wbgt_computed.data - ds_t_computed.data
            curve = wbgt_t_max_limit(ds_t_computed.data,buffer=0) # Calculate the value along the curve for each t value in the input dataset
            dist_above_curve = wbgt_t_diff - curve

            # Create binary arrays to flag potential issues (1 = potential issue, 0 = passed test)
            diff_gr_15_bin = dist_above_curve > 15
            diff_5_to_15_bin = (dist_above_curve > 5) * (dist_above_curve <= 15)
            wind_gr1 = ds_wind_computed.data > 1
            diff_5_to_15_and_wind_gr1 = diff_5_to_15_bin * wind_gr1

            # Print simple output to screen to allow user to see number of potential problems per month or year (depending on input data file)
            print(str(int(year_year)) + ' ' + str(int(month_month)).zfill(2) + ' ' + str(np.sum(diff_gr_15_bin)) + ' ' + str(np.sum(diff_5_to_15_and_wind_gr1)))

            # Count the total number of potential issues for use within the script
            total_diff_gr_15_bin += np.sum(diff_gr_15_bin)
            total_diff_5_to_15_and_wind_gr1 += np.sum(diff_5_to_15_and_wind_gr1)

            # Add potential issues to arrays to output as text file later
            if np.sum(diff_gr_15_bin) > 0:
                inds_1 = np.where(diff_gr_15_bin == 1)
                for j in np.arange(len(inds_1[0])):
                    lats_out_diff_gr_15_bin = np.append(lats_out_diff_gr_15_bin,[ds_wbgt_computed.lat.data[inds_1[1][j]]])
                    lons_out_diff_gr_15_bin = np.append(lons_out_diff_gr_15_bin,[ds_wbgt_computed.lon.data[inds_1[2][j]]])
                    time_out_diff_gr_15_bin.append(str(ds_wbgt_computed.time.data[inds_1[0][j]]))
                    wbgt_out_diff_gr_15_bin = np.append(wbgt_out_diff_gr_15_bin,[ds_wbgt_computed.data[inds_1][j]])
                    t_out_diff_gr_15_bin = np.append(t_out_diff_gr_15_bin,[ds_t_computed.data[inds_1][j]])
                    wind_out_diff_gr_15_bin = np.append(wind_out_diff_gr_15_bin,[ds_wind_computed.data[inds_1][j]])
                    curve_out_diff_gr_15_bin = np.append(curve_out_diff_gr_15_bin,[curve[inds_1][j]])

            if np.sum(diff_5_to_15_and_wind_gr1) > 0:
                inds_2 = np.where(diff_5_to_15_and_wind_gr1 == 1)
                for j in np.arange(len(inds_2[0])):
                    lats_out_diff_wind_bin = np.append(lats_out_diff_wind_bin,[ds_wbgt_computed.lat.data[inds_2[1][j]]])
                    lons_out_diff_wind_bin = np.append(lons_out_diff_wind_bin,[ds_wbgt_computed.lon.data[inds_2[2][j]]])
                    time_out_diff_wind_bin.append(str(ds_wbgt_computed.time.data[inds_2[0][j]]))
                    wbgt_out_diff_wind_bin = np.append(wbgt_out_diff_wind_bin,[ds_wbgt_computed.data[inds_2][j]])
                    t_out_diff_wind_bin = np.append(t_out_diff_wind_bin,[ds_t_computed.data[inds_2][j]])
                    wind_out_diff_wind_bin = np.append(wind_out_diff_wind_bin,[ds_wind_computed.data[inds_2][j]])
                    curve_out_diff_wind_bin = np.append(curve_out_diff_wind_bin,[curve[inds_2][j]])


            del ind_wbgt, ind_t, ind_wind
            del ds_wbgt, ds_t, ds_wind
            del ds_wbgt_computed, ds_t_computed, ds_wind_computed

            del wbgt_t_diff, curve, dist_above_curve
            del diff_gr_15_bin, diff_5_to_15_bin, wind_gr1, diff_5_to_15_and_wind_gr1

        #############################################################################################################################################
        # Output diff_gr_15_bin results to text file

        if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
            text_fname_out_diff_gr_15 = 'wbgt_t_diff_gr15_' + input_dataset + '_' + str(int(year_year))
        else:
            text_fname_out_diff_gr_15 = 'wbgt_t_diff_gr15_' + input_dataset + '_' + set_model + '_' + set_SSP + '_' + set_run + '_' + str(int(year_year))
        if crop_to_aust_region == 'yes':
            text_fname_out_diff_gr_15 += '_AUST'
        if mask_nonaust_land == 'yes':
            text_fname_out_diff_gr_15 += '_ausonly'
        if mask_ocean == 'yes':
            text_fname_out_diff_gr_15 += '_NOocean'
        if data_type == 'bias_adjusted':
            text_fname_out_diff_gr_15 += '_BC'

        text_file_name_diff_gr_15 = text_fname_out_diff_gr_15 + '.txt'

        text_file_diff_gr_15 = open(dir_out + text_file_name_diff_gr_15, "w")

        if len(lats_out_diff_gr_15_bin) > 0:
            for l in np.arange(len(lats_out_diff_gr_15_bin)):
                if l == 0:
                    text_line = 'lat' + ',' + 'lon' + ',' + 'time' + ',' + wbgt_var_name + ',' + t_var_name + ',' + wind_var_name + ',' + 'curve' + '\n'
                    text_file_diff_gr_15.write(text_line)
                text_line = str(lats_out_diff_gr_15_bin[l]) + ',' + str(lons_out_diff_gr_15_bin[l]) + ',' + str(time_out_diff_gr_15_bin[l]) + ',' + str(wbgt_out_diff_gr_15_bin[l]) + \
                        ',' + str(t_out_diff_gr_15_bin[l]) + ',' + str(wind_out_diff_gr_15_bin[l]) + ',' + str(curve_out_diff_gr_15_bin[l]) + '\n'
                text_file_diff_gr_15.write(text_line)
        else:
            text_line = 'None'
            text_file_diff_gr_15.write(text_line)

        text_file_diff_gr_15.close()
        del text_line

        #############################################################################################################################################
        # Output diff_5_to_15_and_wind_gr1 results to text file

        if input_dataset in ['BARRA', 'BARRA-R', 'BARRA-R2']:
            text_fname_out_diff_wind = 'wbgt_t_diff_wind_' + input_dataset + '_' + str(int(year_year))
        else:
            text_fname_out_diff_wind = 'wbgt_t_diff_wind_' + input_dataset + '_' + set_model + '_' + set_SSP + '_' + set_run + '_' + str(int(year_year))
        if crop_to_aust_region == 'yes':
            text_fname_out_diff_wind += '_AUST'
        if mask_nonaust_land == 'yes':
            text_fname_out_diff_wind += '_ausonly'
        if mask_ocean == 'yes':
            text_fname_out_diff_wind += '_NOocean'
        if data_type == 'bias_adjusted':
            text_fname_out_diff_wind += '_BC'

        text_file_name_diff_wind = text_fname_out_diff_wind + '.txt'

        text_file_diff_wind = open(dir_out + text_file_name_diff_wind, "w")

        if len(lats_out_diff_wind_bin) > 0:
            for l in np.arange(len(lats_out_diff_wind_bin)):
                if l == 0:
                    text_line = 'lat' + ',' + 'lon' + ',' + 'time' + ',' + wbgt_var_name + ',' + t_var_name + ',' + wind_var_name + ',' + 'curve' + '\n'
                    text_file_diff_wind.write(text_line)
                text_line = str(lats_out_diff_wind_bin[l]) + ',' + str(lons_out_diff_wind_bin[l]) + ',' + str(time_out_diff_wind_bin[l]) + ',' + str(wbgt_out_diff_wind_bin[l]) + \
                        ',' + str(t_out_diff_wind_bin[l]) + ',' + str(wind_out_diff_wind_bin[l]) + ',' + str(curve_out_diff_wind_bin[l]) + '\n'
                text_file_diff_wind.write(text_line)
        else:
            text_line = 'None'
            text_file_diff_wind.write(text_line)

        text_file_diff_wind.close()
        del text_line


    ######################################################################################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", type=str, help="BARRA, BARPA, CCAM")
    parser.add_argument("--data_type", type=str, default='raw', help="raw, bias_adjusted")
    parser.add_argument("--dir_out", type=str, help="Enter output directory for text files")
    parser.add_argument("--ssp", type=str, default='historical', help="historical, ssp126, ssp370")
    parser.add_argument("--StartYr", type=int, default=1979, help="")
    parser.add_argument("--EndYr", type=int, default=2024, help="")
    parser.add_argument("--model", type=str, default='ACCESS-CM2', help="Name of model")

    args = parser.parse_args()
    main(args)


