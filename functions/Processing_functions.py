# Packages
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')
import datetime as dt
from datetime import timedelta
from cartopy.util import add_cyclic_point
from typing import List, Tuple
import pop_tools
from xgcm import Grid

lats = np.array([-90, -89.0575916230366, -88.1151832460733, -87.1727748691099, 
    -86.2303664921466, -85.2879581151832, -84.3455497382199, -83.4031413612565, 
    -82.4607329842932, -81.5183246073298, -80.5759162303665, -79.6335078534031, 
    -78.6910994764398, -77.7486910994764, -76.8062827225131, -75.8638743455497, 
    -74.9214659685864, -73.979057591623, -73.0366492146597, -72.0942408376963, 
    -71.151832460733, -70.2094240837696, -69.2670157068063, -68.3246073298429, 
    -67.3821989528796, -66.4397905759162, -65.4973821989529, -64.5549738219895, 
    -63.6125654450262, -62.6701570680628, -61.7277486910995, -60.7853403141361,
    -59.8429319371728, -58.9005235602094, -57.9581151832461, -57.0157068062827, 
    -56.0732984293194, -55.130890052356, -54.1884816753927, -53.2460732984293, 
    -52.303664921466, -51.3612565445026, -50.4188481675393, -49.4764397905759, 
    -48.5340314136126, -47.5916230366492, -46.6492146596859, -45.7068062827225, 
    -44.7643979057592, -43.8219895287958, -42.8795811518325, -41.9371727748691,
    -40.9947643979058, -40.0523560209424, -39.1099476439791, -38.1675392670157, 
    -37.2251308900524, -36.282722513089, -35.3403141361257, -34.3979057591623, 
    -33.455497382199, -32.5130890052356, -31.5706806282722, -30.6282722513089, 
    -29.6858638743456, -28.7434554973822, -27.8010471204189, -26.8586387434555, 
    -25.9162303664921, -24.9738219895288, -24.0314136125654, -23.0890052356021, 
    -22.1465968586387, -21.2041884816754, -20.261780104712, -19.3193717277487, 
    -18.3769633507853, -17.434554973822, -16.4921465968586, -15.5497382198953, 
    -14.6073298429319, -13.6649214659686, -12.7225130890052, -11.7801047120419, 
    -10.8376963350785, -9.89528795811519, -8.95287958115183, -8.01047120418848, 
    -7.06806282722513, -6.12565445026178, -5.18324607329843, -4.24083769633508, 
    -3.29842931937173, -2.35602094240838, -1.41361256544502, -0.471204188481678, 
    0.471204188481678, 1.41361256544502, 2.35602094240838, 3.29842931937172, 
    4.24083769633508, 5.18324607329843, 6.12565445026178, 7.06806282722513, 
    8.01047120418848, 8.95287958115183, 9.89528795811518, 10.8376963350785, 
    11.7801047120419, 12.7225130890052, 13.6649214659686, 14.6073298429319, 
    15.5497382198953, 16.4921465968586, 17.434554973822, 18.3769633507853, 
    19.3193717277487, 20.261780104712, 21.2041884816754, 22.1465968586387, 
    23.0890052356021, 24.0314136125654, 24.9738219895288, 25.9162303664921, 
    26.8586387434555, 27.8010471204188, 28.7434554973822, 29.6858638743455, 
    30.6282722513089, 31.5706806282723, 32.5130890052356, 33.455497382199, 
    34.3979057591623, 35.3403141361257, 36.282722513089, 37.2251308900524, 
    38.1675392670157, 39.1099476439791, 40.0523560209424, 40.9947643979058, 
    41.9371727748691, 42.8795811518325, 43.8219895287958, 44.7643979057592, 
    45.7068062827225, 46.6492146596859, 47.5916230366492, 48.5340314136126, 
    49.4764397905759,50.4188481675393, 51.3612565445026, 52.303664921466, 
    53.2460732984293, 54.1884816753927, 55.130890052356, 56.0732984293194, 
    57.0157068062827, 57.9581151832461, 58.9005235602094, 59.8429319371728, 
    60.7853403141361, 61.7277486910995, 62.6701570680628, 63.6125654450262, 
    64.5549738219895, 65.4973821989529, 66.4397905759162, 67.3821989528796, 
    68.3246073298429, 69.2670157068063, 70.2094240837696, 71.151832460733, 
    72.0942408376963, 73.0366492146597, 73.979057591623, 74.9214659685864, 
    75.8638743455497, 76.8062827225131, 77.7486910994764, 78.6910994764398, 
    79.6335078534031, 80.5759162303665, 81.5183246073298, 82.4607329842932, 
    83.4031413612565, 84.3455497382199, 85.2879581151832, 86.2303664921466, 
    87.17277486911, 88.1151832460733, 89.0575916230366, 90])
arclats = np.array([50.4188481675393, 51.3612565445026, 52.303664921466, 
        53.2460732984293, 54.1884816753927, 55.130890052356, 56.0732984293194, 
        57.0157068062827, 57.9581151832461, 58.9005235602094, 59.8429319371728, 
        60.7853403141361, 61.7277486910995, 62.6701570680628, 63.6125654450262, 
        64.5549738219895, 65.4973821989529, 66.4397905759162, 67.3821989528796, 
        68.3246073298429, 69.2670157068063, 70.2094240837696, 71.151832460733, 
        72.0942408376963, 73.0366492146597, 73.979057591623, 74.9214659685864, 
        75.8638743455497, 76.8062827225131, 77.7486910994764, 78.6910994764398, 
        79.6335078534031, 80.5759162303665, 81.5183246073298, 82.4607329842932, 
        83.4031413612565, 84.3455497382199, 85.2879581151832, 86.2303664921466, 
        87.17277486911, 88.1151832460733, 89.0575916230366, 90])
lons = np.array([-180, -178.75, -177.5, -176.25, -175, -173.75, -172.5, -171.25, -170, 
    -168.75, -167.5, -166.25, -165, -163.75, -162.5, -161.25, -160, -158.75, 
    -157.5, -156.25, -155, -153.75, -152.5, -151.25, -150, -148.75, -147.5, 
    -146.25, -145, -143.75, -142.5, -141.25, -140, -138.75, -137.5, -136.25, 
    -135, -133.75, -132.5, -131.25, -130, -128.75, -127.5, -126.25, -125, 
    -123.75, -122.5, -121.25, -120, -118.75, -117.5, -116.25, -115, -113.75, 
    -112.5, -111.25, -110, -108.75, -107.5, -106.25, -105, -103.75, -102.5, 
    -101.25, -100, -98.75, -97.5, -96.25, -95, -93.75, -92.5, -91.25, -90, 
    -88.75, -87.5, -86.25, -85, -83.75, -82.5, -81.25, -80, -78.75, -77.5, 
    -76.25, -75, -73.75, -72.5, -71.25, -70, -68.75, -67.5, -66.25, -65, 
    -63.75, -62.5, -61.25, -60, -58.75, -57.5, -56.25, -55, -53.75, -52.5, 
    -51.25, -50, -48.75, -47.5, -46.25, -45, -43.75, -42.5, -41.25, -40, 
    -38.75, -37.5, -36.25, -35, -33.75, -32.5, -31.25, -30, -28.75, -27.5, 
    -26.25, -25, -23.75, -22.5, -21.25, -20, -18.75, -17.5, -16.25, -15, 
    -13.75, -12.5, -11.25, -10, -8.75, -7.5, -6.25, -5, -3.75, -2.5, -1.25, 
    0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 12.5, 13.75, 15, 
    16.25, 17.5, 18.75, 20, 21.25, 22.5, 23.75, 25, 26.25, 27.5, 28.75, 30, 
    31.25, 32.5, 33.75, 35, 36.25, 37.5, 38.75, 40, 41.25, 42.5, 43.75, 45, 
    46.25, 47.5, 48.75, 50, 51.25, 52.5, 53.75, 55, 56.25, 57.5, 58.75, 60, 
    61.25, 62.5, 63.75, 65, 66.25, 67.5, 68.75, 70, 71.25, 72.5, 73.75, 75, 
    76.25, 77.5, 78.75, 80, 81.25, 82.5, 83.75, 85, 86.25, 87.5, 88.75, 90, 
    91.25, 92.5, 93.75, 95, 96.25, 97.5, 98.75, 100, 101.25, 102.5, 103.75, 
    105, 106.25, 107.5, 108.75, 110, 111.25, 112.5, 113.75, 115, 116.25, 
    117.5, 118.75, 120, 121.25, 122.5, 123.75, 125, 126.25, 127.5, 128.75, 
    130, 131.25, 132.5, 133.75, 135, 136.25, 137.5, 138.75, 140, 141.25, 
    142.5, 143.75, 145, 146.25, 147.5, 148.75, 150, 151.25, 152.5, 153.75, 
    155, 156.25, 157.5, 158.75, 160, 161.25, 162.5, 163.75, 165, 166.25, 
    167.5, 168.75, 170, 171.25, 172.5, 173.75, 175, 176.25, 177.5, 178.75])

def Wilks_pcrit(pvalues: np.ndarray, siglevel: float) -> float:
    '''
    This function calcules the p-critical level for the Wilks significance test

    INPUT:
    pvalues: array of p-values
    siglevel: significance level (i.e. 0.01 or 1%, 0.05 or 5%, ...)
    
    OUTPUT:
    pcrit: Wilks p-critical value (i.e. any p-value less than this is significant)
    '''

    # Calculate false detection rate
    alpha_fdr = 2*siglevel

    # Flatten p-values into 1D array & sort
    pvalues_fl = pvalues.flatten()
    pvalues_fl = np.sort(pvalues_fl)

    # Generate arrays to calculate differences
    x = np.arange(1,len(pvalues_fl)+1,1)
    x = x.astype(float)
    y = (x/len(x))*alpha_fdr

    # Calculate differences
    d = pvalues_fl-y

    # Grab index of first p-value where p-value > y
    w_out = np.where(d>0.0)
    k = -1 if w_out[0].size == 0 else w_out[0][0]

    # Find p-critical value & return it
    # None of the p-values are significant
    if k == 0:
        pcrit = 0.0
    # All p-values are significant
    elif k == -1:
        pcrit = pvalues_fl[-1]
    # Some p-values are significant
    else:
        pcrit = pvalues_fl[k-1]
        
    return pcrit

def AddCyclic(da: xr.DataArray) -> xr.DataArray:
    # Add cyclic point
    cyclic_data, cyclic_lon = add_cyclic_point(da.data, coord=da['lon'])
    cyclic_coords = {dim: da.coords[dim] for dim in da.dims}
    cyclic_coords['lon'] = cyclic_lon

    da = xr.DataArray(cyclic_data, dims=da.dims, coords=cyclic_coords, attrs=da.attrs, name=da.name)
    return da

def FixLongitude(da: xr.DataArray, add_cyclic: bool) -> xr.DataArray:
    '''
    Fixes CESM longitude 
    INPUT:
    da: xarray DataArray

    OUTPUT:
    da: modified xarray DataArray
    '''
    # Switch longitude from 0-360 to -180-180
    da = da.assign_coords(dict(lon=(((da.lon+180) % 360)-180)))

    # Sort longitude to fix plotting problems
    da = da.sortby('lon','ascending')

    if add_cyclic:
         da = AddCyclic(da)
    
    return da

def FixGrid(da: xr.DataArray, grid: str) -> xr.DataArray:
    '''
    Transforms CICE grid into lat/lon (0-360)
    INPUT:
    da: xarray DataArray

    OUTPUT:
    da: modified xarray DataArray
    '''

    # Get CICE grid from pop_tools
    grid = pop_tools.get_grid('POP_'+grid)

    # Change tarea to m2 instead of km2
    with xr.set_options(keep_attrs=True):
        grid['TAREA'] = grid['TAREA']/(1e4)
    grid['TAREA'].attrs['units'] = 'm^2'

    # Add lat, lon, and tarea coordinates
    da.coords['lat'] = (('nj','ni'),grid['TLAT'].values)
    da.coords['lon'] = (('nj','ni'),grid['TLONG'].values)
    da.coords['tarea'] = (('nj','ni'),grid['TAREA'].values)

    return da

    

def FixTime(da: xr.DataArray) -> xr.DataArray:
    '''
    Fixes CESM time coordinate for monthly data
    INPUT:
    da: xarray DataArray

    OUTPUT:
    da: modified xarray DataArray
    '''

    # Subtract 15 days to fix monthly timestamp
    da = da.assign_coords(dict(time=(da.time-timedelta(days=15))))

    return da

def CalcStatforDim(da: xr.DataArray, grpdim: str, dims: str|List[str]) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    '''
    Calculates mean, standard deviation, n for the DataArray over dimension(s)
    INPUT: 
    da: xarray DataArray
    dims: list of dimension(s)

    OUTPUT:
    da_avg: DataArray mean over dimension(s) 
    da_std: DataArray standard deviation over dimension(s)
    da_n: DataArray count over dimension(s)
    '''
    # Check if lat is one of average dimensions
    if 'lat' in dims:
        weights = np.cos(np.deg2rad(da.lat))
        da = da.weighted(weights)

    if grpdim == '':
        da_avg = da.mean(dims, skipna=True)
        da_std = da.std(dims, skipna=True, ddof=1)
        da_n = da.count(dims)
    else:
        # Calculate statistics over dimension(s)
        da_avg = da.groupby(grpdim).mean(dims, skipna=True)
        da_std = da.groupby(grpdim).std(dims, skipna=True, ddof=1)
        da_n = da.groupby(grpdim).count(dims)

    da_avg.compute()
    da_std.compute()
    da_n.compute()

    return da_avg, da_std, da_n



def CalcStatbyGrpDim(da: xr.DataArray, grpdim1: str, grpdim2: str, concatdim: str, avgdim: str, avgdim2: str|List[str]) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    '''
    Calculates mean, standard deviation, n for the DataArray over dimension(s) by a grouped by dimension
    INPUT: 
    da: xarray DataArray
    grpdim1: dimension to group data by
    grpdim2: second dimension to group data by
    concatdim: dimension to concatenate ungrouped data
    avgdims: dimension(s) to calculate statistics over group of data

    OUTPUT:
    da_avg: DataArray mean over dimension(s) 
    da_std: DataArray standard deviation over dimension(s)
    da_n: DataArray count over dimension(s)
    '''
    # Check if lat is in average dimension
    if 'lat' in avgdim:
        weights = np.cos(np.deg2rad(da.lat))
        da = da.weighted(weights)

    # Group dataArray
    da_grp = da.groupby(grpdim1)

    # Initialize lists
    da_avg_list = []
    lbl_list = []

    
    # Loop over all groups in grouped dataArray
    for sub, dss in da_grp:
        if grpdim2 != '':
            # Calculate mean for group
            dss_avg = dss.groupby(grpdim2).mean(avgdim, skipna=True)
        else:
            dss_avg = dss.mean(avgdim, skipna=True)

        # Add to lists
        da_avg_list.append(dss_avg)
        lbl_list.append(sub)

    # Calculate statistics over group
    da_avg_grp = xr.concat(da_avg_list,pd.Index(lbl_list,name=concatdim))   
    da_avg_grp.compute()
    
    da_avg = da_avg_grp.mean(avgdim2,skipna=True)
    da_std = da_avg_grp.std(avgdim2,skipna=True,ddof=1)
    da_n = da_avg_grp.count(avgdim2)

    da_avg.compute()
    da_std.compute()
    da_n.compute()

    return da_avg, da_std, da_n
    
def Ensemble(da_list, ens_index: pd.Index, return_mean=False, stat='avg') -> xr.DataArray:
    '''
    Takes list of DataArrarys, one for each ensemble member, and turns them into a single DataArray
    with a dimension of 'ensemble_member'. Optionally returns ensemble mean instead. If returning
    ensemble mean, stat is required
    INPUT:
    da_list: list of DataArrays, length is number of ensemble members
    ens_index: pandas Index with name 'ensemble_member'
    return_mean: (optional) boolean for returning ensemble mean. default is False
    stat: (optional) string describing statistic that ensemble mean will be calculated for. default is 'avg'
          must be one of 'avg', 'std', 'n' 

    OUTPUT:
    da_ens: DataArray containing ensemble
    da_ensmean: DataArray containing ensemble mean
    '''
    # Concatenate list with pandas index of ensemble members
    da_ens = xr.concat(da_list, ens_index)

    # Chunk data
    da_ens = da_ens.chunk({'ensemble_member': -1})
    da_ens.compute()

    # If returning ensemble mean
    if return_mean:
        # If statistic is average
        if stat == 'avg':
            da_ensmean = da_ens.mean('ensemble_member', skipna=True)
        
        # Else if statistic is standard deviation
        elif stat == 'std':
            num_em = da_ens.sizes['ensemble_member']
            da_ensmean = np.sqrt((da_ens**2).sum('ensemble_member', skipna=True)/num_em)

        # Else if statistic is count
        elif stat == 'n':
            da_ensmean = da_ens.sum('ensemble_member', skipna=True)
        
        # Else if stat is not the right value
        else:
            raise ValueError('\'stat\' value  must be one of \'avg\', \'std\', \'n\'')
        
        da_ensmean.compute()

        return da_ensmean
    
    # Else only return ensemble
    else:
        return da_ens