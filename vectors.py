# Functions for vectors

# DEPENDENCIES:
import xarray as xr

import numpy as np
import numpy.ma as ma

from datetime import datetime, timedelta

from metpy.units import units


def rotate_vector(u, v, angle):
    
    """Rotate vector [u,v] by angle (requires units)"""
    
    theta = angle.to('rad').magnitude
    
    ur = u * np.cos(theta) - v * np.sin(theta)
    vr = u * np.sin(theta) + v * np.cos(theta)
    
    return ur, vr


def dot_vectors(u1, v1, u2, v2):
    
    """Dot vectors [u1, v1] and [u2, v2]"""
    
    dot = (u1 * u2) + (v1 * v2)

    return dot


def comp_along_theta(u, v, angle):
    
    """Component of vector [u,v] along angle (requires units)"""
    
    theta = angle.to('rad').magnitude
    
    u1, v1 = u, v
    u2, v2 = np.cos(theta), np.sin(theta)
    
    dot = dot_vectors(u1, v1, u2, v2)

    return dot


def vector_angle(u, v):
    
    """Angle of vector [u,v]"""
    
    theta = np.arctan2(v, u)  * units('rad')

    return theta.to('deg')


def comp_along_vector(u1, v1, u2, v2):
    
    """Component of vector [u1, v1] along vector [u2, v2]"""

    dot = dot_vectors(u1, v1, u2, v2)

    mag2 = np.sqrt(u2**2 + v2**2)

    return dot / mag2
