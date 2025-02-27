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





def linear_interpolate(desired_times, og_times, og_values):
    
    """Function to linearly interpolate values to desired times.

INPUT: 
- desired_times: (M x 1) array of datetimes to interpolate to
- og_times: (M x 1) array of original datetimes
- og_values: (M x 1) array of original values

OUTPUT:
- interp_values: (M x 1) array of interpolated values
- dt_values: (M x 1) array of time differences used in interpolation

Latest recorded update:
01-29-2025
    """

    # arrays to store interp values
    # and dt values (seconds) used in interp
    interp_values = np.array([])
    dt_values = np.array([])

    for time in desired_times:
        
        # if time is outside of range of data, just add nans
        if (time < og_times[0]) or (time > og_times[-1]):
            value = np.nan
            dt_sec = np.nan

        else:
            # look first for exact match
            match_index = np.where(og_times == time)[0]

            # if exact match, extract value
            if (len(match_index) > 0) & np.isfinite(og_values[match_index]):
                value = og_values[match_index]
                dt_sec = 0

            # otherwise, interp
            else:
                value = None

                # find non-nan times before and after values
                before_indices = np.where(og_times < time)[0]
                left_index = before_indices[np.isfinite(og_values)[before_indices]][-1]

                after_indices = np.where(og_times > time)[0]
                right_index = after_indices[np.isfinite(og_values)[after_indices]][0]
                # left_index = np.where(og_times < time)[0][-1]
                # right_index = np.where(og_times > time)[0][0]

                # linearly interpolate value across time gap
                DV = og_values[right_index] - og_values[left_index]
                DT = og_times[right_index] - og_times[left_index]

                # estimate new value
                dt = time - og_times[left_index]
                dv = DV * (dt/DT)
                value = og_values[left_index] + dv

                # save timestep used in interp
                dt_sec = dt.total_seconds()

        # save to array
        interp_values = np.append(interp_values, value)
        dt_values = np.append(dt_values, dt_sec)
        
    return interp_values, dt_values


    return u_fixed, v_fixed


def running_mean(series, length = 3, center = False, address_nans = False):
    
    """Function to calculate running mean of a series (assumes evenly spaced data).

INPUT: 
- series: (M x 1) array of data to calculate running mean
- length: length of running mean window 
    (either one-sided (window=length+1), or one side of two-sided if center=True (window=2*length + 1))
- center: if True, calculate two-sided running mean
- address_nans: if True, address nans in data by taking mean of non-nan values in window
    (else carry any nans to final data)

OUTPUT:
- running_mean: (M x 1) array of running mean values

Latest recorded update:
01-30-2025
    """

    # copy data, retain xarray type if ds
    variable = series.copy()
    running_mean = variable.copy()

    # end values are nans
    if center:

        running_mean[:length] = np.nan
        running_mean[-length:] = np.nan
        for ii in range(length, len(running_mean)-length):
            if address_nans:
                running_mean[ii] = np.nanmean(variable[ii-length:ii+length+1])
            else:
                running_mean[ii] = np.sum(variable[ii-length:ii+length+1]) / len(variable[ii-length:ii+length+1])
        
    else:
        running_mean[:length] = np.nan
        for ii in range(length, len(running_mean)):
            if address_nans:
                running_mean[ii] = np.nanmean(variable[ii-length:ii+1])
            else:
                running_mean[ii] = np.sum(variable[ii-length:ii+1]) / len(variable[ii-length:ii+1])
        
        
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