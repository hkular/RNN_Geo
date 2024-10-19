#!/usr/bin/env python
# coding: utf-8

#  Name: Robert Kim\
#  Date: 03-29-2023\
#  Email: robert.f.kim@gmail.com\
#  fnc_eval_model.m\
#  Description: Implementation of rate model
# 
# Adapted from .m to .ipynb by Holly Kular\
# date: 02-15-2024
# update: 05-30-2024 added .item() to O[t] calculation to comply with numpy 1.25

# INPUT
#    - model_dir: full path to the trained model
#    - uu: input signal
#    - feedback: false or true (interlayer feedback)

# In[19]:


def fnc_eval_model(model_dir, uu, feedback, loadmat, np):
    
    
    # Load the data first
    data = loadmat(model_dir)
    
    # Total number of time-points
    T = uu.shape[1]
    DeltaT = 1
    
    # Time points-by-neurons
    N1 = data['N1'][0,0]
    N2 = data['N2'][0,0]
    N3 = data['N3'][0,0]
    
    X1 = np.zeros((T+1, N1))
    R1 = np.zeros((T+1, N1))
    
    X2 = np.zeros((T+1, N2))
    R2 = np.zeros((T+1, N2))
    
    X3 = np.zeros((T+1, N3))
    R3 = np.zeros((T+1, N3))
    
    X1[0, :] = np.random.randn(1, N1) / 100
    R1[0, :] = 1 / (1 + np.exp(-X1[0, :]))
    
    X2[0, :] = np.random.randn(1, N2) / 100
    R2[0, :] = 1 / (1 + np.exp(-X2[0, :]))
    
    X3[0, :] = np.random.randn(1, N3) / 100
    R3[0, :] = 1 / (1 + np.exp(-X3[0, :]))
    
    # Get all the weights
    ww1 = data['w1'] @ data['m1']
    ww2 = data['w2'] @ data['m2']
    ww3 = data['w3'] @ data['m3']
    
    ww21 = data['w21'] @ data['m1']
    ww32 = data['w32'] @ data['m2'] 
    
    
    if feedback:
        ww12 = data['w12'] @ data['m2']
        ww23 = data['w23'] @ data['m3']
        
    # Synaptic decay constants
    taus_sig1 = (1 / (1 + np.exp(-data['taus_gaus1']))) * (data['taus'][0,1] - data['taus'][0,0]) + data['taus'][0,0]
    taus_sig2 = (1 / (1 + np.exp(-data['taus_gaus2']))) * (data['taus'][0,1] - data['taus'][0,0]) + data['taus'][0,0]
    taus_sig3 = (1 / (1 + np.exp(-data['taus_gaus3']))) * (data['taus'][0,1] - data['taus'][0,0]) + data['taus'][0,0]
    
    O = np.zeros((T+1))
    for t in range(1, T+1):
        if feedback:          
            next_x1 = (1 - DeltaT / taus_sig1) * X1[t-1:t, :].T + \
                (DeltaT / taus_sig1) * (ww1 @ R1[t-1:t, :].T + ww12 @ R2[t-1:t, :].T + data['w_in'] @ uu[:, t-1:t]) + (np.random.randn(N1,1) / 10)
        else:
            next_x1 = ((1 - DeltaT / taus_sig1) * X1[t-1:t, :].T) + \
                (DeltaT / taus_sig1) * (ww1 @ R1[t-1:t, :].T + data['w_in'] @ uu[:, t-1:t]) + (np.random.randn(N1,1) / 10)
        next_r1 = 1 / (1 + np.exp(-next_x1))
        
        X1[t, :] = next_x1.T
        R1[t, :] = next_r1.T
        
        if feedback:
            next_x2 = (1 - DeltaT / taus_sig2) * X2[t-1:t, :].T + \
                (DeltaT / taus_sig2) * (ww2 @ (R2[t-1:t, :].T) + ww21 (R1[t-1:t, :].T) + ww23 @ (R2[t-1:t, :].T)) + np.random.randn(N2,1) / 10
        else:
            next_x2 = (1 - DeltaT / taus_sig2) * X2[t-1:t, :].T + \
                (DeltaT / taus_sig2) * (ww2 @ (R2[t-1:t, :].T) + ww21 @ (R1[t-1:t, :].T)) + np.random.randn(N2,1) / 10
        next_r2 = 1 / (1 + np.exp(-next_x2))
        
        X2[t, :] = next_x2.T
        R2[t, :] = next_r2.T
        
        next_x3 = (1 - DeltaT / taus_sig3) * X3[t-1:t, :].T + \
            (DeltaT / taus_sig3) * (ww3 @ (R3[t-1:t, :].T) + ww32 @ (R2[t-1:t, :].T)) + np.random.randn(N3,1) / 10
        next_r3 = 1 / (1 + np.exp(-next_x3))
        
        X3[t, :] = next_x3.T
        R3[t, :] = next_r3.T
        
        O[t] = (data['w_out'] @ R3[t, :].T + data['b_out']).item() # t:t+1
    
    out = {}

    out['X1'] = X1[1:, :]
    out['X2'] = X2[1:, :]
    out['X3'] = X3[1:, :]

    out['R1'] = R1[1:, :]
    out['R2'] = R2[1:, :]
    out['R3'] = R3[1:, :]

    out['O'] = O[1:]
                              
    return [out, O]


# In[ ]:




