#!/usr/bin/env python
# coding: utf-8

# Name: Holly Kular\
# Date: 08-14-2024\
# Email: hkular@ucsd.edu\
# Description: decoding RNN firing rate and trial type

# In[ ]:


# imports
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix


# In[ ]:


# RNN timing and info

RNN_params = {
    'prob_split': '70_30',
    'afc': [6, 2],
    'coh': ['hi', 'lo'],
    'feedback': False,
    'thresh': [.3, .7],
    'model': [0, 1 ,2],
    'fr': [1,3]
}

D_params = {
    'time_avg': False,
    't_win': [130, -1],
    'n_cvs': 5,
    'num_cgs': 30,
    'label': 'stim',  
    'units': 'all',  # 'all' or 'exc' or 'inh'
    'pred': 'unexpected'  # 'expected' or 'unexpected', 'neutral' MODIFY HERE
}
# Timing of task
task_info = {
    'trials': 1000,
    'trial_dur': 250,
    'stim_on': 80,
    'stim_dur': 50
}

window = 50 # size of time window to get sliding avg


# In[ ]:


# Define SVM 

n_cvs = 5
# penalties to eval
num_cgs = 30
Cs = np.logspace( -5,1,num_cgs )

# set up the grid
param_grid = { 'C': Cs, 'kernel': ['linear'] }

# define object - use a SVC that balances class weights (because they are biased, e.g. 70/30)
# note that can also specify cv folds here, but I'm doing it by hand below
grid = GridSearchCV( SVC(class_weight = 'balanced'),param_grid,refit=True,verbose=0 )


# In[ ]:


# Define custom funcs
# ------------------------------------------------------------------------------
# define sliding window of times to decode

def fnc_sliding_window(elements, window_size):
    """
    Description: create time windows to decode over
    Returns: [times]
    """
    if len(elements) <= window_size:
        return elements
    
    windows = []
    for i in range(len(elements) - window_size + 1):
      windows.append(elements[i:i + window_size])
    
    return windows


# -------------------------------------------------------------------------------
# define loop to decode over times and run boot strap samples over

def fnc_decode_times(n_boot):
    """
    Description: loop decoding over each time window
    Returns: [acc, class, t_step]
    """
    np.random.seed(n_boot)  # Ensure each bootstrap is different
    results = []
    for t in times:
        seed = np.random.randint(0, 1000000)  # Unique seed for each time point within the bootstrap
        result = fnc_fit_and_score(np.mean(data_d[:, t, :], axis=1), tri_ind, hold_out, n_cvs, afc, labs, D_params['label'], thresh, grid, seed)
        results.append(result)
    return results

# -------------------------------------------------------------------------------
# define decoding

def fnc_fit_and_score(data_slice, tri_ind, hold_out, n_cvs, afc, labs, label, thresh, grid, seeds):
    """
    Description: Script for decoding fitting linear SVM and scoring accuracy across CV folds
    Fits the model on each CV fold for a given time step
    Returns: [acc, class]
    """
    np.random.seed(seeds)  # Initialize the random number generator with the given seed
    acc = np.zeros(n_cvs)
    cm = np.zeros((n_cvs, afc))
      
    for i in range(n_cvs):
        # trials to hold out as test set on this cv fold
        tst_ind = tri_ind[ i*hold_out : (i+1)*hold_out ]
        trn_ind = np.random.choice(np.setdiff1d( tri_ind, tst_ind ), size = len(np.setdiff1d( tri_ind, tst_ind )), replace = True)

        # get the training data (X) and the training labels (y)
        X = data_slice[trn_ind,:]
        y = labs[trn_ind]
        
        # fit the model
        grid.fit( X,y )

        # get the test data (X) and the test labels (y)
        X_test = data_slice[tst_ind, :]
        y_test = labs[tst_ind]

        # predict and score
        y_pred = grid.predict(X_test)
        cm[i] = confusion_matrix(y_test, y_pred, normalize = "true").diagonal()
        
    acc = np.mean(cm, axis = 0) # acg across cvs
    
    return acc


# In[ ]:


# Now actually do decoding across conditions

modelnum = 0 # which model do we want 
nboots = 1000 # how many boot strap samples do we want?

combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh'], RNN_params['fr']))

for afc, coh, fr in combinations:

    # Load data
    data_dir = f"/data/{D_params['pred']}" 
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]# Get all the trained models (should be 40 .mat files)
    model_path = os.path.join(data_dir, mat_files[modelnum]) 
    model = loadmat(model_path)   
    data_file = f"{data_dir}/Trials{task_info['trials']}_model{model_path[-7:-4]}_{D_params['pred']}.npz"
    data = np.load(data_file)
    data_d = data[f'fr{fr}'] # this is a [trial x time step x unit] matrix
    labs = data['labs'].squeeze() # [trial x time step]

    # get some info about structure of the data
    tris = data_d.shape[0]             # number of trials
    tri_ind = np.arange(0,tris)      # list from 0...tris
    hold_out = int( tris / n_cvs )   # how many trials to hold out
    thresh = RNN_params.get('thresh', [.3, .7])
    
    times = fnc_sliding_window(range(task_info['stim_dur']+task_info['stim_on'],task_info['trial_dur']), window)  
    acc = np.zeros((nboots, len(times), afc))
    
    
    start_time = time.time() 
    if __name__ == "__main__":
        with Pool(processes=round(os.cpu_count() * .9)) as pool:
            results = pool.map(fnc_decode_times, range(nboots))    
        acc = np.array(results)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    
    full_file = f"{data_dir}/boot_{D_params['pred']}_all{modelnum}.npz" 
    np.savez(full_file, acc = acc)

