# Functions for timeseries analysis

# DEPENDENCIES:
import xarray as xr

import numpy as np
import numpy.ma as ma

from datetime import datetime, timedelta

from metpy.units import units



# FUNCTIONS:
#---------------------------------------------------------------------
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
            if len(match_index) > 0:
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
01-29-2025
    """

    # copy data
    variable = np.copy(series)
    running_mean = np.copy(variable)

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

    