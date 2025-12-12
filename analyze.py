# Functions for timeseries analysis

# DEPENDENCIES:
import xarray as xr

import numpy as np
import numpy.ma as ma
import pandas as pd

from datetime import datetime, timedelta

from metpy.units import units
from scipy.signal import butter, filtfilt

from scipy.stats import t

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


def linear_regress(xi, yi, alpha = 0.05, quiet = True):
    """
    Perform a simple linear regression with full statistical outputs, including
    slope, intercept, standard errors, RMSE, Pearson correlation, p-value, 
    and functions for prediction intervals and predicted values.

    This implementation is based on USGS calibration manuals:
    https://pubs.usgs.gov/tm/04/a03/tm4a3.pdf

    Parameters
    ----------
    xi : array-like
        Independent variable data
    yi : array-like
        Dependent variable data
    alpha : float, optional
        Significance level for confidence/prediction intervals (default 0.05 for 95% CI).
    quiet : bool, optional
        If False, prints regression info (default True).

    Returns
    -------
    reg : dict
        Dictionary containing regression results:
        - 'slope' : float, estimated slope (β1)
        - 'intercept' : float, estimated intercept (β0)
        - 'SE_slope' : float, standard error of slope
        - 'SE_intercept' : float, standard error of intercept
        - 'RMSE' : float, residual standard error (σ̂)
        - 'R' : float, Pearson correlation coefficient
        - 'p' : float, two-sided p-value for slope ≠ 0
        - 'x_line' : np.ndarray, x values for plotting regression line
        - 'y_line' : np.ndarray, predicted y values for x_line
        - 'y_line_CI' : np.ndarray, confidence interval for regression line
        - 'y_line_CI_upper' : np.ndarray, upper bound of CI
        - 'y_line_CI_lower' : np.ndarray, lower bound of CI
        - 'SE_predict' : function, SE of predicted y at x0
        - 'CI_predict' : function, confidence interval of predicted y at x0
        - 'predict' : function, predicted y at given x0
        - 'CI' : dict, containing 'dof', 'alpha', 't_crit' for reference

    Notes
    -----
    - SE_predict(x0) returns the standard error of prediction at a specific x0.
    - CI_predict(x0) returns the t-critical * SE_predict(x0), i.e., the width of
      the confidence/prediction interval around the regression line at x0.
    """

    # ---- means of calibration points ----
    xbar = np.nanmean(xi) # independent
    ybar = np.nanmean(yi) # dependent

    n = len(xi)  # number of calibration points
    dof = n - 2  # degrees of freedom for simple linear regression
    
    Syy = np.sum((yi - ybar)**2)          # sum of squares of y
    Sxx = np.sum((xi - xbar)**2)          # sum of squares of x
    Sxy = np.sum((xi - xbar)*(yi - ybar)) # sum of x-y cross-product
    
    beta1 = Sxy / Sxx            # slope estimate
    beta0 = ybar - beta1 * xbar  # intercept estimate

    # predicted y-values 
    yi_hat = beta0 + beta1 * xi
    residuals = (yi - yi_hat)
    MSE = np.sum(residuals**2) / dof  # Mean square error (MSE), "sigma" ** 2
    RMSE = np.sqrt(MSE)               # standard error of regression (RMSE), "sigma"
    
    # other ways to calc
    # s2 = (Syy - beta1*Sxy)/(n-2) # mean square error (MSE)
    # s = np.sqrt(s2)              # standard error or the regression

    # ----- standard errors of slope, intercept -----
    SE_beta1 = RMSE / np.sqrt(Sxx)              
    SE_beta0 = RMSE * np.sqrt(1/n + xbar**2/Sxx)

    # ----- confidence interval -----
    r = Sxy / np.sqrt(Sxx*Syy)

    # ----- student t test t-val -----
    t_crit = t.ppf(1 - alpha/2, dof)

    # two-sided t-test that slope is non-zero
    p_value = 2 * (1 - t.cdf(abs(beta1 / SE_beta1), df=dof))

    if not quiet:
        print(f'{(1-alpha)*100}% confidence, {dof} dof')
        print("Critical t-value:", t_crit)
        
    reg = {}

    # ------- functions to calculate ... -------
    # standard error of prediction
    reg['SE_predict'] = lambda x0 : RMSE * np.sqrt(1 + 1/n + ((x0 - xbar)**2) / Sxx)
    reg['CI_predict'] = lambda x0 : t_crit * reg['SE_predict'](x0)
    
    reg['slope'] = beta1
    reg['intercept'] = beta0
    reg['R'] = r                   # Pearson correlation coefficient between xi and yi
    reg['SE_slope'] = SE_beta1
    reg['SE_intercept'] = SE_beta0
    reg['RMSE'] = RMSE
    reg['p'] = p_value
    
    # predicted fit line values
    xi_dense = np.linspace(np.nanmin(xi), np.nanmax(xi), 100) 
    reg['x_line'] = xi_dense
    reg['y_line'] = beta0 + beta1 * xi_dense
    reg['y_line_CI'] = reg['CI_predict'](xi_dense)
    reg['y_line_CI_upper'] = reg['y_line'] + reg['y_line_CI']
    reg['y_line_CI_lower'] = reg['y_line'] - reg['y_line_CI']
    
    # reg['x_line'] = np.sort(xi)
    # reg['y_line'] = beta0 + beta1 * np.sort(xi)
    # reg['y_line_CI'] = reg['CI_predict'](np.sort(xi))

    reg['CI'] = {}
    reg['CI']['dof'] = dof
    reg['CI']['alpha'] = alpha
    reg['CI']['t_crit'] = t_crit

    # yhat_0 (regression-predicted y vals for each x0)
    reg['predict'] = lambda x0 : beta0 + beta1 * x0

    return reg
        