# N-point median pass filter of time series
#-----------------------------------------------------------------------
#  INPUT: time series, N (odd#)
# OUTPUT: filtered time series
#   PROCESS: calculate local N-point median filter of entire time series
#             %  N must be odd
#             %  boundaries = first and last problematic data points
#             %  @ boundaries, fix N-point median window at data bounds
#             %  ignores NaNs in N-point median
#             %  NaNs in time series replaced with local N-point median
# -----------------------------------------------------------------------

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from metpy.units import units

from timeseries.adjust import seconds_elapsed
from timeseries.filter import medianfilter, meanfilter

def flag_timechanges(times, values, thresh = 0.0002 * units('m/s')):
    
    """Flag changes in values over time above threshold.

    INPUT:
    - times: (M x 1) array of times
    - values: (M x 1) array of values (include units)
    - thresh: change threshold (include units, default = 0.0002 m/s)

    OUTPUT:
    - changes: (M x 1) array of change-over-time values
    - val_filter: (M x 1) array of u-component values with flagged values set to nan

    Latest recorded update:
    02-27-2025
    """

    val_filter = np.copy(values)
    
    # convert time differences to seconds
    dt = (np.diff(times).astype('timedelta64[s]') / np.timedelta64(1,'s')) * units('s')
    dv = np.diff(values)

    change_mag = abs(dv / dt)
    changes = np.append(np.array([0]), change_mag.magnitude) * change_mag.units
    
    val_filter[1:][change_mag > thresh] = np.nan

    return changes, val_filter
    

def flag_accelerations(times, u, v, thresh = 0.0002 * units('m/s2')):
    
    """Flag accelerations above threshold.

    INPUT:
    - times: (M x 1) array of times
    - u: (M x 1) array of u-component values (include units)
    - v: (M x 1) array of v-component values (include units)
    - thresh: acceleration threshold (include units, default = 0.0002 m/s2)

    OUTPUT:
    - accel: (M x 1) array of acceleration values
    - u_filter: (M x 1) array of u-component values with flagged values set to nan
    - v_filter: (M x 1) array of v-component values with flagged values set to nan

    Latest recorded update:
    02-27-2025
    """

    u_filter = np.copy(u)
    v_filter = np.copy(v)
    
    # convert time differences to seconds
    dt = (np.diff(times).astype('timedelta64[s]') / np.timedelta64(1,'s')) * units('s')
    du = np.diff(u_filter)
    dv = np.diff(v_filter)
    
    accel_mag = np.sqrt(du**2 + dv**2) / dt
    accel = np.append(np.array([0]), accel_mag.magnitude) * accel_mag.units
    
    u_filter[1:][accel_mag > thresh] = np.nan
    v_filter[1:][accel_mag > thresh] = np.nan

    return accel, u_filter, v_filter


def sigma3filter(times, og_series, L=5, N=1, min_frac = 0.5):
    
    """Filter data more than 3 sigma.

    INPUT:
    - times: (M x 1) array of times
    - og_series: (M x 1) array of original values
    - L: length of running median window (odd integer of # of points, or timedelta object)
    - N: number of iterations to apply filter
    - min_frac: minimum fraction of non-nan values required to calculate median, 
    otherwise return nan (default: 0.5)

    OUTPUT:
    - filter_series: (M x 1) array of filtered values (outliers replaced with nan)

    Latest recorded update:
    11-21-2025

    """
    
    filter_series = np.copy(og_series)
    
    for ii in range(N):

        # apply an L-point median filter
        medians = medianfilter(times, filter_series, L=L, min_frac = min_frac)

        # find residual of timeseries, and MAD (median absolute deviation)
        residual = abs(medians - filter_series)
       

        # find mean and standard deviation of residual timeseries
        mu = np.nanmean(residual)
        sigma = np.nanstd(residual)
        flags = np.abs(residual - mu) >= 3 * sigma

        # sigma = np.nanstd(residual)
        # flags = residual >= 3 * sigma

        filter_series[np.isnan(medians)] = np.nan
        filter_series[flags] = np.nan

    
    return filter_series