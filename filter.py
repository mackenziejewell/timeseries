

import numpy as np

from datetime import datetime, timedelta

from scipy.signal import butter, filtfilt

from timeseries.adjust import seconds_elapsed

# FUNCTIONS:
#---------------------------------------------------------------------

def butter_lowpass(t, y, cutoff_t, order = 5):
    
    """
    Digital butterworth filter
    
    INPUT:
    t: time (in seconds)
    y: initial values (cannot contain nans currently!)
    cutoff_t: cutoff period (in seconds)
    order: order of butterworth filter (default: 5)
    
    OUTPUT:
    yf : filtered timeseries
    """
    # resource 1:
    # https://stackoverflow.com/questions/63320705/what-are-order-and-critical-frequency-when-creating-a-low-pass-filter-using
    # resource 2:
    # https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7

    
    fs = 1/np.diff(t)[0]     # sample rate, Hz
    T = t[-1]                # Sample Period
    n = int(T * fs)          # total number of samples
    cutoff_f = 1/(cutoff_t)  # desired cutoff frequency of the filter (Hz)
    nyq = 0.5 * fs           # Nyquist Frequency

    # normal_cutoff = cutoff / nyq (only use this for didital filters if fs is not specified)
    # Get the filter coefficients 
    b, a = butter(order, cutoff_f, btype='low', fs = fs, analog=False)
    yf = filtfilt(b, a, y)
    
    # outputting with sos is recommended to reduce "numerical error"
    # but this seems to shift all the filtered data off form the true times?
    # from scipy.signal import sosfreqz, sosfilt
    # sos = butter(order, cutoff, btype='low', fs = fs, analog=False, output='sos')
    # y = sosfilt(sos, data)
    return yf


def loess(y,t,tls,tau):

    """Loess smoother. Based on Jim Lerczak's Matlab code from June 2012 (course OC 682).
    INPUT:
    - y: (M x 1) array of values
    - t: (M x 1) array of times (in units like seconds, not datetime)
    - tls: (M x 1) array of times (in units like seconds) where the loess smoothed values are computed
    - tau: half-span of the loess smoother (in units like seconds)
    OUTPUT:
    - yls: (M x 1) array of loess smoothed values
    - flag: (M x 1) array of flags (True for loess estimates that fall out of the range of the data, False otherwise)
    Latest recorded update:
    02-28-2025
    """
    
    # normalize t and tls by tau
    t = t/tau
    tls = tls/tau
    
    yls = np.full(len(tls), np.nan)
    flag = np.full(len(tls), False)

    # only apply loess smoother to times (tls) within the time range of the
    # data (t)
    # here grab indices to iterate over
    indices = np.arange(len(tls))[(tls>=t.min())&(tls<=t.max())]

    for ii in range(0, len(indices)):

        idx = indices[ii]         # time index for
        qn = (t-tls[idx])         # scaled times from current desired time
        mm = np.where(abs(qn)<=1)[0] # indices within tau of current time
        qn = qn[mm]               # times within tau
        ytmp = y[mm]              # values within tau
        ttmp = t[mm]-tls[idx]     # time offset

        # non-nan indices
        nn = np.where(~np.isnan(ttmp*ytmp))[0]

        # need at least three data points to do the regression
        if len(nn) >= 3:

            # select non-nan vals
            ytmp = ytmp[nn]
            ttmp = ttmp[nn]
            qn = qn[nn]

            # weighting function
            wn = ((1 - abs(qn)**3)**3)**2
            W = np.diag(wn,0)
            X = np.array([np.ones(len(ttmp)), ttmp, ttmp**2]).T

            XcT = X.conj().T # complex conjugate transpose

            M1 = np.matmul(XcT, np.matmul(W,X))    # eigenvalues
            M2 = np.matmul(XcT, np.matmul(W,ytmp)) # ??

            try:
                B = np.linalg.solve(M1, M2)            # solve linear system of equations
                yls[idx] = B[0]                        # grab first value, in matlab this was B(1)

                # if the solution is out of the range of the data used in the
                # regression, then flag that datapoint
                if B[0] < np.nanmin(ytmp) or B[0] > np.nanmax(ytmp):
                    flag[idx] = True
            except:
                flag[idx] = True
                
    return yls, flag




def medianfilter(times, og_series, L=5, min_frac = 0.5):

    """N-point median pass filter of time series.

    INPUT: 
    - times: (M x 1) array of times
    - og_series: (M x 1) array of original values
    - L: length of running median window (odd integer of # of points, or timedelta object)
    - min_frac: minimum fraction of non-nan values required to calculate median

    OUTPUT:
    - median_series: (M x 1) array of running-median values

    Latest recorded update:
    11-21-2025
    """

    # convert times to seconds
    t = seconds_elapsed(times)

    median_series = np.copy(og_series)

    # if L is a timedelta, convert to seconds
    # find length on either side of running time point to grab

    if isinstance(L, timedelta):

        L = L.total_seconds()
        w = L/2

        for ii in range(len(og_series)):

            # find median of values in current window
            window = abs(t-t[ii]) <= w

             # find median, only if at least min_frac of values are non-nan
            current_vals = og_series[window]
            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                median_series[ii] = np.nanmedian(current_vals)

            # otherwise return nan?
            else:
                median_series[ii] = np.nan



    # either fun over window specified as # of points
    elif type(L) == int:

        # find length on either side of running time point to grab
        w = int(L-(L+1)/2)

        #  apply local N-point median filter (with special conditions at bounds)    
        for ii in range(len(og_series)-w):

            # adjust window at beginning, end of series
            if ii < w: 
                current_vals = og_series[:ii+w+1]
            elif ii > len(og_series)-w:
                current_vals = og_series[ii-w:]
            else:
                current_vals = og_series[ii-w:ii+w+1]

            # find median, only if at least min_frac of values are non-nan
            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                median_series[ii] = np.nanmedian(current_vals)

            # otherwise return nan?
            else:
                median_series[ii] = np.nan


    return median_series


def meanfilter(times, og_series, L=5, min_frac = 0.5):

    """N-point mean pass filter of time series.

    INPUT: 
    - times: (M x 1) array of times
    - og_series: (M x 1) array of original values
    - L: length of running mean window (odd integer of # of points, or timedelta object)
    - min_frac: minimum fraction of non-nan values required to calculate mean

    OUTPUT:
    - mean_series: (M x 1) array of running-mean values

    Latest recorded update:
    11-21-2025
    """

    # convert times to seconds
    t = seconds_elapsed(times)

    mean_series = np.copy(og_series)

    # if L is a timedelta, convert to seconds
    # find length on either side of running time point to grab

    if isinstance(L, timedelta):

        L = L.total_seconds()
        w = L/2

        for ii in range(len(og_series)):

            # find mean of values in current window
            window = abs(t-t[ii]) <= w

            # find mean, only if at least min_frac of values are non-nan
            current_vals = og_series[window]
            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                mean_series[ii] = np.nanmean(current_vals)

            # otherwise return nan?
            else:
                mean_series[ii] = np.nan

    # either run over window specified as # of points
    elif type(L) == int:

        # find length on either side of running time point to grab
        w = int(L-(L+1)/2)

        # apply local N-point mean filter (with special conditions at bounds)
        for ii in range(len(og_series)-w):

            # adjust window at beginning, end of series
            if ii < w: 
                current_vals = og_series[:ii+w+1]
            elif ii > len(og_series)-w:
                current_vals = og_series[ii-w:]
            else:
                current_vals = og_series[ii-w:ii+w+1]

            # find mean, only if at least min_frac of values are non-nan
            if np.sum(np.isfinite(current_vals)) >= min_frac * len(current_vals):
                mean_series[ii] = np.nanmean(current_vals)

            # otherwise return nan?
            else:
                mean_series[ii] = np.nan

    return mean_series
