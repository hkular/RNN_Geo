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
#from sklearn.model_selection import RandomizedSearchCV
from scipy.io import loadmat
#from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut

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
n_cvs = 3;

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
    Decoding loop over each time window.
    """
    np.random.seed(n_boot)  # Ensure each bootstrap is different
    results = []

    for t in times:
        seed = np.random.randint(0, 1000000)  # Unique seed for each time point within the bootstrap
        result = fnc_fit_and_score(
            np.mean(data_d[:, t, :], axis=1),
            tri_ind,
            hold_out,
            n_cvs,
            afc,
            labs,
            D_params['label'],
            thresh,
            seed
        )
        results.append(result)

    return results

# -------------------------------------------------------------------------------
# define decoding objective - find best hyperparams, model training

def optuna_objective(trial, X_train, y_train):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # Suggest hyperparameters
    C = trial.suggest_float('C', 1e-3, 1e3, log = True) # maybe trye 1e-3 to 1e3
    #gamma = trial.suggest_loguniform('gamma', 1e-4, 1e-1) only do this if kernel is poly or rbf
    kernel = trial.suggest_categorical('kernel', ['linear'])

    model = SVC(C=C, kernel=kernel, class_weight='balanced', random_state = 40) # define gamma if we do non-linear kernel

       # Check if we have enough samples in each class
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    
    # If severe class imbalance, adjust cross-validation strategy
    if len(unique_classes) < 2 or any(count < 2 for count in class_counts):
        # Fallback to leave-one-out if not enough samples
        cv = LeaveOneOut()
        print("Class imbalance in tuning data.")
    else:
        # Prefer stratified k-fold when possible
        cv = StratifiedKFold(n_splits=min(5, np.min(class_counts)), 
                              shuffle=True, 
                              random_state=42)
    
    # Perform cross-validation on training data
    try:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')
        return np.mean(scores)
    except ValueError as e:
        # If cross-validation fails, return a low score
        print(f"Cross-validation failed: {e}")
        return 0.5  # Neutral score


# -------------------------------------------------------------------------------
# define train and test best hyper params - model prediction to evaluate

def fnc_fit_and_score(data_slice, tri_ind, hold_out, n_cvs, afc, labs, label, thresh, seeds):
    """
    Script for decoding, fitting SVM, and scoring accuracy across CV folds.
    """
    acc = np.zeros(2)
    cm = np.zeros((n_cvs, 2))  # Confusion matrix for binary classification

    data_slice = data_slice.reshape(-1, 1)

    for i in range(n_cvs):
        
        # Strict data split
        tst_ind = tri_ind[i * hold_out: (i + 1) * hold_out]
        remaining_ind = np.setdiff1d(tri_ind, tst_ind)

         # Use all remaining indices for both tuning and training
        # Perform an additional split within the remaining data
        tuning_size = int(len(remaining_ind) * 0.2)  # 20% for hyperparameter tuning
        np.random.shuffle(remaining_ind)
        tuning_ind = remaining_ind[:tuning_size]
        trn_ind = remaining_ind[tuning_size:]
        
        
        
        # Prepare data for this fold
        # Prepare training labels with thresholding
        X_tuning = data_slice[tuning_ind, :]
        y_tuning = np.select(
            [labs[tuning_ind] >= thresh[1], labs[tuning_ind] <= thresh[0]],
            [0, 1],
            default=np.nan
        )
        valid_mask = ~np.isnan(y_tuning)
        X_tuning = X_tuning[valid_mask, :]
        y_tuning = y_tuning[valid_mask]
        
        # Debugging thresholding
        unique_tuning_classes, tuning_class_counts = np.unique(y_tuning, return_counts=True)
        print(f"Tuning data classes: {unique_tuning_classes}")
        print(f"Tuning data class counts: {tuning_class_counts}")
        
        
        # Hyperparameter tuning on tuning set
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: optuna_objective(trial, X_tuning, y_tuning),
            n_trials=5
        )

       # Prepare training data for final model
        X_train = data_slice[trn_ind, :]
        y_train = np.select(
            [labs[trn_ind] >= thresh[1], labs[trn_ind] <= thresh[0]],
            [0, 1],
            default=0
        )
        
        # Create and train final model with best hyperparameters
        best_model = SVC(
            C=study.best_params['C'],
            kernel=study.best_params['kernel'],
            class_weight='balanced',
            random_state=40
        )

        best_model.fit(X_train, y_train)

        # Get test data and labels
        X_test = data_slice[tst_ind, :]
        y_test = np.select(
            [labs[tst_ind] >= thresh[1], labs[tst_ind] <= thresh[0]],
            [0, 1],
            default=0
        )

        # Predict and compute confusion matrix diagonal
        y_pred = best_model.predict(X_test)
        cm[i] = confusion_matrix(y_test, y_pred, normalize="true").diagonal()

    acc = np.mean(cm, axis=0)  # Average accuracy across CV folds
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

