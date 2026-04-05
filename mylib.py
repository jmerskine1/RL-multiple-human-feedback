# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:36:42 2018

@author: taku.yamagata
"""
import numpy as np

# ----------------------------------------------------------------------------
# Process data
# ----------------------------------------------------------------------------

def moving_average(d, len):
    prePadLen = len//2
    posPadLen = len - prePadLen
    d_ = np.append(d[0:prePadLen], d)
    d_ = np.append(d_, d[-posPadLen:])
    cs = np.cumsum(d_)
    ma = (cs[len:] - cs[:-len]) / len
    return ma

# ----------------------------------------------------------------------------
# Log calculation functions
# ----------------------------------------------------------------------------

# calculate log(a+b) from log(a) and log(b)
def logadd(a, b):
    if a > b:
        out = a + np.log( 1 + np.exp(b-a) )
    elif a < b:
        out = b + np.log( 1 + np.exp(a-b) )
    else:
        if np.abs(a) == np.inf:
            out = a
        else:
            out = a + np.log( 1 + np.exp(b-a) )
            
    return out
    
# calculate log( sum(a) ) from log(a)
def logsum(a):
    m = np.max(a)
    out = m + np.log( np.sum( np.exp(a-m) ) )
    return out
    
# normalise log-probability p
def lognorm(p):
    m = np.max(p)
    out = p - (m + np.log( np.sum( np.exp(p-m) ) ) )
    return out