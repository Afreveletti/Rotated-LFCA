#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:24:17 2018

Original Code at: https://github.com/rcjwills/lfca/blob/master/Python/signal_processing.py
Edited by: Anthony Freveletti
"""

import sys
sys.path.append('/tank/users/tfreveletti/')
from lanczos_filter import lanczos_filter
import numpy as np
import scipy as sp
from scipy.signal import butter, lfilter, filtfilt


def lfca(x, ne, cutoff, truncation, scale, **kwargs):
    if x.ndim != 2:
        return
    if 'covtot' in kwargs.keys():
        covtot = kwargs['covtot']
    else:
        covtot = np.cov(x, rowvar=False)
    (n, p) = x.shape
    if covtot.shape != (p, p):
        return


    # Number of time steps per ensemble member
    nt = n // ne

    # Center data
    x = x - np.nanmean(x, 0)[np.newaxis, ...]
    
    # Scale vector using np.tile
    xs = x * np.transpose(scale)
    
    # Eigendecomposition of covariance matrix
    covtot = np.transpose(scale) * covtot * scale
    pcvec, evl, rest = peigs(covtot, min(n - 1, p))
    trcovtot = np.trace(covtot)
    
    # Percent of total sample variation accounted for by each EOF
    pvar = evl / trcovtot * 100
    # Principal component time series
    pcs = np.dot(xs, pcvec)
    # Return EOFs in original scaling as patterns (row vectors)
    eof = np.transpose(pcvec) / np.transpose(scale)
                      
    # Truncation of EOFs
    if truncation < 1:
        truncation *= 100
        cum_pvar = np.cumsum(pvar)
        ntr = np.argmin(np.abs(cum_pvar - truncation)) + 1
    else:
        if truncation != round(truncation):
            raise ValueError('Truncation must be a fraction or an integer number of EOFs.')
        ntr = int(truncation)
    print('Number of EOFs = ', ntr)
    print(f'Percent Variance Explained = {np.sum(pvar[:ntr]):.2f}%')
    
    # Whitening transformation
    f = np.sqrt(np.squeeze(evl)[0:ntr])
    # Get transformation matrices
    s = np.dot(pcvec[:, 0:ntr], np.diag(1. / f))
    sadj = np.dot(np.diag(f), np.transpose(pcvec[:, 0:ntr]))

    
    # Filter data matrix
    t = np.arange(1, nt + 1)
    x_f = xs.copy()
    for j in range(ne):
        js = slice(j * nt, (j + 1) * nt)
        Y_cut = xs[js, :]
        for i in range(Y_cut.shape[1]):
            poly = np.polyfit(t, Y_cut[:, i], 1)
            tmp = Y_cut[:, i] - np.polyval(poly, t)
            tmp1 = np.concatenate((np.flipud(tmp), tmp, np.flipud(tmp)))
            tmp_filt = lanczos_filter(tmp1, 1, 1./cutoff)[0]
            x_f[js, i] = tmp_filt[len(tmp_filt) // 3: 2 * len(tmp_filt) // 3] + np.polyval(poly, t)

    print('Filtering Complete')

    # Whiten variables
    y = np.dot(x_f, s)    
    # Slow covariance matrix of whitened variables
    gamma = np.cov(y, rowvar=False)
    # SVD of slow variance matrix
    dummy, r, v = csvd(gamma)

    
    # Weight vectors and patterns
    weights = scale * np.dot(s, v)
    lfps = np.dot(np.transpose(v), sadj) / np.transpose(scale)
                 
    # Choose signs of patterns, weights, EOFs, and PCs
    for j in range(lfps.shape[0]):
        if np.dot(lfps[j,:][np.newaxis,...], scale)<0:
            lfps[j,:] = -lfps[j,:]
            weights[:,j] = -weights[:,j]
    for j in range(eof.shape[0]):
        if np.dot(eof[j,:][np.newaxis,...], scale)<0:
            eof[j,:] = -eof[j,:]
            pcs[:,j] = -pcs[:,j]
    
    # Low-frequency components
    xs = xs / np.transpose(scale)
    lfcs = np.dot(xs, weights)

    print('LFPs and LFCs Computed')
    
    # Fraction of variance in forced patterns
    w = weights / scale
    p = lfps * np.transpose(scale)

    pw_diag = np.diag(np.dot(p, w))
    tot_var = np.diag(np.dot(np.dot(p, covtot), w)) / pw_diag
    pvar_lfp = tot_var / trcovtot * 100


    
    return lfcs, lfps, weights, r, pvar, pcs, eof, ntr, pvar_lfp


def csvd(a):
    
    (m,n) = a.shape
    if m>=n:
        (u,s,v) = np.linalg.svd(a,0)
        v = np.transpose(v)
    else:
        (v,s,u) = np.linalg.svd(a.transpose(),0)
        u = np.transpose(u)   
    return u, s, v

def peigs(a, rmax):
    
    (m,n) = a.shape
    if rmax>min(m,n):
        rmax = min(m,n)
    
    if rmax<min(m,n)/10.:
        (d,v) = sp.sparse.linalg.eigs(a, rmax)
    else:
        (d,v) = np.linalg.eig(a)
    
    if d.size>max(d.shape):
        d = np.diag(d)
    
    # ensure that eigenvalues are monotonically decreasing    
    i = np.argsort(-d)
    d = -np.sort(-d)
    v = v[:,i]
    # estimate number of positive eigenvalues of a
    d_min = max(d)*max(m,n)*np.spacing(1)
    r = np.sum(d>d_min)
    # discard eigenpairs with eigenvalues that are close to or less than zero
    d = d[:r]
    v = v[:,:r]
    d = d[:]
    
    return v, d, r

    
