#!/usr/bin/env python
# coding: utf-8

# Name: Holly Kular\
# Date: 08-14-2024\
# Email: hkular@ucsd.edu\
# Description: decoding RNN firing rate and trial type

# In[1]:


# imports
import os
import itertools
import numpy as np
#import matplotlib.pyplot as plt
import sys
import time
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
#from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from skopt import BayesSearchCV

# In[2]:


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
    'n_cvs': 3,
    'num_cgs': 15,
    'label': 'choice',  
    'units': 'all',  # 'all' or 'exc' or 'inh'
    'pred': 'expected'  # 'expected' or 'unexpected', 'neutral'
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


# Number of cross-validation folds
n_cvs = 3

# Define search space for BayesSearchCV
param_space = {
    'C': (1e-5, 10.0, 'log-uniform'),  # Equivalent to np.logspace(-5, 1, num_cgs)
    'kernel': ['linear']
}

# Define the Bayesian search object
bayes_search = BayesSearchCV(
    estimator=SVC(class_weight='balanced'), 
    search_spaces=param_space, 
    n_iter=15,  # Number of Bayesian optimization steps
    refit=True,
    n_jobs=os.cpu_count() - 4,
    random_state=42
)

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

def fnc_decode_times(times, n_boot, task_info, data_d, tri_ind, hold_out, n_cvs, afc, labs, D_params, thresh):
    """
    Description: loop decoding over each time window
    Returns: [acc, class, t_step]
    """
    np.random.seed(n_boot)  # Ensure each bootstrap is different
    results = []
      
    for t in times:
        seed = np.random.randint(0, 1000000)  # Unique seed for each time point within the bootstrap
        result = fnc_fit_and_score(np.mean(data_d[:, t, :], axis=1), tri_ind, hold_out, n_cvs, afc, labs, D_params['label'], thresh)
        results.append(result)
    return results

# -------------------------------------------------------------------------------
# define decoding

def fnc_fit_and_score(data_slice, tri_ind, hold_out, n_cvs, afc, labs, label, thresh):
    """
    Description: Script for decoding fitting linear SVM and scoring accuracy across CV folds
    Fits the model on each CV fold for a given time step
    Returns: [acc, class]
    """
    #np.random.seed(seeds)  # Initialize the random number generator with the given seed
    acc = np.zeros(2)
    cm = np.zeros((n_cvs, 2)) # 2 and not afc because binary classification for choice
      
    for i in range(n_cvs):
        # trials to hold out as test set on this cv fold
        tst_ind = tri_ind[ i*hold_out : (i+1)*hold_out ]
        trn_ind = np.random.choice(np.setdiff1d( tri_ind, tst_ind ), size = len(np.setdiff1d( tri_ind, tst_ind )), replace = True)

        # get the training data (X) and the training labels (y)
        X = data_slice[trn_ind,:]
        y = np.select([labs[trn_ind] >= thresh[1], labs[trn_ind] <= thresh[0]], [0,1], default=0)
        
        # fit the model
        bayes_search.fit(X, y)

        # get the test data (X) and the test labels (y)
        X_test = data_slice[tst_ind, :]
        y_test = np.select([labs[tst_ind] >= thresh[1], labs[tst_ind] <= thresh[0]], [0,1], default=0)

        # predict and score
        y_pred = bayes_search.predict(X_test)
        cm[i] = confusion_matrix(y_test, y_pred, normalize = "true").diagonal()
        
    acc = np.mean(cm, axis = 0) # avg across cvs
    
    return acc



# In[ ]:


# Now actually do decoding across conditions

#modelnum = [1, 2] # which model do we want ranges 0 to 2?
nboots = 1 # how many boot strap samples do we want?

#combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh'], RNN_params['fr'], modelnum))



#for afc, coh, fr, modelnum in combinations:
afc = 6
coh = 'hi'
fr = 1
modelnum = 1
    

# Load data
data_dir = f"/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/rdk_{RNN_params['prob_split']}_{afc}afc/feedforward_only/{coh}_coh"
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]# Get all the trained models (should be 40 .mat files)
model_path = os.path.join(data_dir, mat_files[modelnum]) 
model = loadmat(model_path)   
data_file = f"{data_dir}/Trials{task_info['trials']}_model{model_path[-7:-4]}_0{D_params['pred']}.npz"
data = np.load(data_file)
data_d = data[f'fr{fr}'] # this is a [trial x time step x unit] matrix
labs = data['outs'][:,-1] # [trial x time step] choice is outs

# get some info about structure of the data
tris = data_d.shape[0]             # number of trials
tri_ind = np.arange(0,tris)      # list from 0...tris
hold_out = int( tris / n_cvs )   # how many trials to hold out
thresh = RNN_params.get('thresh', [.3, .7])

times = fnc_sliding_window(range(task_info['stim_dur']+task_info['stim_on'],task_info['trial_dur']), window)  
acc = np.zeros((nboots, len(times), 2))


start_time = time.time() 

for i in range(nboots):
    results = fnc_decode_times(times, i, task_info, data_d, tri_ind, hold_out, n_cvs, afc, labs, D_params, thresh)
    acc[i, : , :] = results


#if __name__ == "__main__":
#    with Pool(processes=round(os.cpu_count() * .9)) as pool:
#        results = pool.map(fnc_decode_times, range(nboots))    
#    acc = np.array(results)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
        
#full_file = f"/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/{D_params['pred']}/fr{fr}/{coh}_{afc}afc/boot_all{modelnum}_{D_params['label']}.npz"
#np.savez(full_file, acc = acc)
   
#print(f'done saving {full_file}')


# In[ ]:

