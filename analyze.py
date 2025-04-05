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
def variance_ellipses(xp, yp):
    
    """ Calculate principal axes and variances of 2D timeseries data

    Resources:
    https://www.researchgate.net/profile/Curtis-Mobley-2/publication/44529688_Principal_component_analysis_in_meteorology_and_oceanography_by_Rudolph_W_Preisendorfer_posthumously_compiled_and_edited_by_Curtis_D_Mobley/links/541875e90cf2218008bf3da1/Principal-component-analysis-in-meteorology-and-oceanography-by-Rudolph-W-Preisendorfer-posthumously-compiled-and-edited-by-Curtis-D-Mobley.pdf 
    https://jmlilly.net/course/pages/thevarianceellipse.html#12 
    https://www.mygeodesy.id.au/documents/Chapter%208.pdf

    INPUT:
    - xp: (M x 1) array of x-values
    - yp: (M x 1) array of y-values
    OUTPUT:
    - t: (M x 1) array of seconds elapsed from starttime
    Latest recorded update:
    03-26-2025
    """
    
    # mean of original vector components
    xb = xp.mean()
    yb = yp.mean()

    # translated coordinate system about the mean
    x = xp - xb
    y = yp - yb

    # points in rotated-by-theta frame
    # m = x * cos(theta) + y * sin(theta)
    # n = -x * sin(theta) + y * cos(theta)

    # variances along theta axis:
    # s2(theta) = (1/(n-1)) * np.sum(m*m)
    # s2(theta) = (1/(n-1)) * np.sum((x*x * cos2(theta) + 2*x*y * sin(theta)cos(theta) + y*y * sin2(theta)))
    # s2(theta) = sxx * cos2(theta) + 2 * sxy * sin(theta)cos(theta) + syy * sin2(theta)

    # def s2(sxx, syy, sxy, theta):
    #     s2 = (sxx * (np.cos(theta))**2) + (2 * sxy * np.sin(theta)*np.cos(theta)) + (syy * (np.sin(theta))**2)
    #     return s2

    # def ds2dtheta(sxx, syy, sxy, theta):
    #     ds2dtheta = (syy-sxx) * np.sin(2*theta)  + 2 * sxy * np.cos(2*theta)
    #     return ds2dtheta
        
    # principal angle found be setting:
    # ds2(theta) / d(theta) = 0 = (syy-sxx)*sin(2theta) + 2*sxy *cos(2theta)
    # which has solutions phi that satisfies:
    # tan(2phi) = 2*sxy/(sxx-syy)

    n = len(x)

    # variances of x, y, and covariance of x,y
    sxx = (1/(n-1)) * np.sum(x*x)
    syy = (1/(n-1)) * np.sum(y*y)
    sxy = (1/(n-1)) * np.sum(x*y)

    # principal angle theta (without convention adjustment)
    theta = (0.5 * np.arctan(2*sxy/(sxx-syy)))
    

    # convention to find 1st principal angle theta_m
    # from theta found from maximization
    if sxx - syy < 0:
        
        if sxy < 0:
            theta_m = theta-np.pi/2
            
        elif sxy == 0:
            theta_m = np.pi/2
            
        elif sxy > 0:
            theta_m = theta+np.pi/2
            
    elif sxx - syy == 0:
        
        if sxy < 0:
            theta_m = -np.pi/4
            
        elif sxy == 0:
            theta_m = 0
            
        elif sxy > 0:
            theta_m = np.pi/4
            
    elif sxx - syy > 0:
        
        if sxy < 0:
            theta_m = theta
            
        elif sxy == 0:
            theta_m = 0
            
        elif sxy > 0:
            theta_m = theta
        
            
    # variances along principle directions
    s11 = 0.5 * ((sxx+syy) + np.sqrt((sxx-syy)**2+(4*sxy**2)))
    s22 = 0.5 * ((sxx+syy) - np.sqrt((sxx-syy)**2+(4*sxy**2)))

    # standard deviations along principle directions
    a = np.sqrt(s11)
    b = np.sqrt(s22)

    # ellipse points
    thetas = np.linspace(0, 2*np.pi, 100)
    xe = a * np.cos(thetas)*np.cos(theta_m) - b * np.sin(thetas)*np.sin(theta_m)
    ye = a * np.cos(thetas)*np.sin(theta_m) + b * np.sin(thetas)*np.cos(theta_m)

    pca = {}
    pca['sxx'] = sxx
    pca['syy'] = syy
    pca['sxy'] = sxy
    pca['theta_m'] = theta_m
    pca['s11'] = s11
    pca['s22'] = s22
    pca['a'] = a
    pca['b'] = b
    pca['xe'] = xe
    pca['ye'] = ye

    return pca