#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:46:51 2024

@author: hkular
"""
def maha_class( X_trn, y_trn, X_tst, y_tst ):
    '''
        js 08222024
        Computes two-class classification accuracy based on the Mahalanobis distance
        between each pattern in the test data and the mean patterns associated
        with each condition in independent training data
        input:
           X_trn = [num observation, num variable] matrix of data
           y_trn = [num observation,] vector of condition labels for each trial
           X_tst = samesies as X_trn but test data
           y_tst = samesies as y_trn but test data
    '''
    u_conds = np.unique( y_trn )            # unique conditions
    n_conds = len( u_conds )                # number of conditions
    # mean of each class and store counts for each class
    # and compute the pooled covariance matrix
    m_trn = np.full( ( n_conds,X_trn.shape[1] ), np.nan )
    c_cnts = np.full( n_conds,np.nan )
    S = np.zeros( ( X_trn.shape[1], X_trn.shape[1] ) )
    for c_idx, cond in enumerate( u_conds ):
        Xc = X_trn[ y_trn==cond,: ]
        m_trn[ c_idx,: ] = np.mean( Xc,axis=0 )
        c_cnts[ c_idx ] = np.count_nonzero( y_trn==cond )
        S += ( c_cnts[ c_idx ]-1 ) * np.cov( Xc,rowvar=False )#( np.cov( Xc,rowvar=False ) / ( c_cnts[ c_idx ]-1 ) )
    # norm cov by sum of n-1 for each cond
    S *= ( 1/np.sum(c_cnts-1) )
    # inv of cov matrix - using pinv
    # and not inv cause some cov matrices
    # are low-rank...not ideal, but...
    invS = np.linalg.pinv(S)
    # compute class as argmin weighted distance from each
    # condition mean (mahalanobis distance)
    dist = np.full( (n_conds,X_tst.shape[0]),np.nan )
    for cond in range( n_conds ):
        dist[ cond,: ] = np.diag( ( X_tst - m_trn[ cond,: ] ) @ invS @ ( X_tst - m_trn[ cond,: ] ).T )
    # get predicted class label for each trial
    actual = y_tst
    pred = np.argmin( dist,axis=0 )
    acc = np.sum( actual == pred ) / len( y_tst )
    # return acc and dicts
    return acc, actual, pred