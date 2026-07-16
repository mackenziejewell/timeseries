# Functions for timeseries analysis

# DEPENDENCIES:
import xarray as xr

import numpy as np
import numpy.ma as ma
import pandas as pd

from datetime import datetime, timedelta

from metpy.units import units
from scipy.signal import butter, filtfilt


# FUNCTIONS:
#---------------------------------------------------------------------
def seconds_elapsed(time, t0 = None):
    
    """ Convert datetime array to seconds elapsed from specified start time
    INPUT:
    - time: (M x 1) array of datetimes
    - t0: datetime object of start time (if None, use first time in array)
    OUTPUT:
    - t: (M x 1) array of seconds elapsed from starttime
    Latest recorded update:
    02-27-2025
    """
    
    time = pd.to_datetime(time)
    
    if t0 == None:
        t0 = time[0]

    t = (time - t0) / np.timedelta64(1,'s')
    
    return t


def linear_interpolate(desired_x, og_x, og_values, max_dx = 1 * units('day')):
    
    """Function to linearly interpolate values to desired x-coords (which could be times). 
    For now, will not interpolate past nan bounds.

INPUT: 
- desired_x: (M x 1) array of x-coords (floats or datetimes) to interpolate to
- og_x: (M x 1) array of original x-coords (floats or datetimes)
- og_values: (M x 1) array of original values
- max_dx: maximum allowed gap for interpolation x-coords 
(floats or timedelta like 1 * units('day')), 
(otherwise return nan)


OUTPUT:
- interp_values: (M x 1) array of interpolated values
- dx_values: (M x 1) array of x-coord differences used in interpolation

Latest recorded update:
07-16-2026
    """

    x = np.asarray(og_x)
    desired = np.asarray(desired_x)

    # determine if x-coordinate is time
    if np.issubdtype(x.dtype, np.datetime64):
        x = seconds_elapsed(x)
        desired = seconds_elapsed(desired)

        if max_dx is not None:
            max_dx = max_dx.total_seconds()
    else:
        x = x.astype(float)
        desired = desired.astype(float)

        if isinstance(max_dx, timedelta):
            raise ValueError("timedelta max_dx only valid for datetime coordinates")

    # arrays to store interp values
    # and dt values (seconds) used in interp
    interp_values = np.array([])
    dx_values = np.array([])

    # find first and last non-nan times
    first_nn = x[np.where(np.isfinite(og_values))[0][0]]
    last_nn = x[np.where(np.isfinite(og_values))[0][-1]]

    for xxx in desired:
        
        # if time is outside of range of data, just add nans
        if (xxx < first_nn) or (xxx > last_nn):
            value = np.nan
            dx = np.nan

        else:
            # look first for exact match
            match_index = np.where(x == xxx)[0]

            # used this to debug multiple matches
            if (len(match_index) > 1):
                print(len(match_index))
                print(match_index)
                print(og_values[match_index])

            # if exact match, extract value
            elif (len(match_index) == 1):
                if np.isfinite(og_values[match_index]):
                    value = og_values[match_index]
                    dx = 0
                else:
                    value = np.nan
                    dx = np.nan
                    print('transferred nan value over... decide how want to handle this case?????')

            # otherwise, interp
            else:
                value = None

                # find non-nan times before and after values
                before_indices = np.where(x < xxx)[0]
                left_index = before_indices[np.isfinite(og_values)[before_indices]][-1]

                after_indices = np.where(x > xxx)[0]
                right_index = after_indices[np.isfinite(og_values)[after_indices]][0]
                # left_index = np.where(og_times < time)[0][-1]
                # right_index = np.where(og_times > time)[0][0]

                # linearly interpolate value across time gap
                DV = og_values[right_index] - og_values[left_index]
                DX = x[right_index] - x[left_index]

                # estimate new value
                dx = xxx - x[left_index]
                dv = DV * (dx/DX)
                value = og_values[left_index] + dv

                # save timestep used in interp
                # dx = dt.total_seconds()

        # save to array
        interp_values = np.append(interp_values, value)
        dx_values = np.append(dx_values, dx)
        
    # flag interp values with dx > max_dx
    if max_dx is not None:
        interp_values[dx_values > max_dx] = np.nan
    # max_dt_sec = max_dt.to('s').magnitude
    # # flag interp values with dt > max_dt_sec
    # interp_values[dx_values > max_dx] = np.nan
    
    return interp_values, dx_values


    # return u_fixed, v_fixed


def running_mean(series, length = 3, center = False, min_frac = 0.5, use_median = False):
    
    """Function to calculate running mean of a series (assumes evenly spaced data).

INPUT: 
- series: (M x 1) array of data to calculate running mean
- length: length of running mean window 
    (either one-sided (window=length+1), or one side of two-sided if center=True (window=2*length + 1))
- center: if True, calculate two-sided running mean
- min_frac: minimum fraction of non-nan values required to calculate median, 
    otherwise return nan (default: 0.5)
 - min_frac: minimum fraction of non-nan values required to calculate median, 
    otherwise return nan (default: 0.5)
 - use_median: if True, use median instead of mean (default: False)


OUTPUT:
- running_mean: (M x 1) array of running mean values

Latest recorded update:
11-21-2025
    """

    # copy data, retain xarray type if ds
    variable = series.copy()
    running_mean = variable.copy()

    # end values are nans
    if center:

        running_mean[:length] = np.nan
        running_mean[-length:] = np.nan
        for ii in range(length, len(running_mean)-length):

            current_vals = variable[ii-length:ii+length+1]

            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                if use_median:
                    running_mean[ii] = np.nanmedian(current_vals)
                else:
                    running_mean[ii] = np.nanmean(current_vals)
            else:
                running_mean[ii] = np.nan
        
    else:
        running_mean[:length] = np.nan

        for ii in range(length, len(running_mean)):

            current_vals = variable[ii-length:ii+1]

            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                if use_median:
                    running_mean[ii] = np.nanmedian(current_vals)
                else:
                    running_mean[ii] = np.nanmean(current_vals)
            else:
                running_mean[ii] = np.nan
        
        
    return running_mean


def dates_to_sequences(dates, dt_max = 1 * units('day')):
        
    """Convert list of dates to dictionary of consecutive date sequences.

INPUT: 
- dates: (M x 1) list of all dates to consider
- dt_max: maximum time gap to allow for sequence continuation

OUTPUT:
- seqs: dictionary of sequences, with keys as sequence number and values as list of dates in sequence

Latest recorded update:
01-30-2025
    """

    # start dict to store sequences
    seqs = {}
        
    if len(dates) > 0:
        
        seqs[0] = [dates[0]]

        for ii in range(1, len(dates)):

            # determine index and last date of last sequence
            last_seq = list(seqs.keys())[-1]
            last_date = seqs[last_seq][-1]

            # determine whether sequences continues
            # or new one must be defined
            DT = (dates[ii] - last_date).total_seconds() * units('s')

            if DT <= dt_max:
                seqs[last_seq].append(dates[ii])
            else:
                seqs[last_seq+1] = [dates[ii]]

    return seqs