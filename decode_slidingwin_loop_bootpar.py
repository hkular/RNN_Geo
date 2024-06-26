#!/usr/bin/env python
# coding: utf-8

# Name: Holly Kular\
# Date: 03-19-2024\
# Email: hkular@ucsd.edu\
# decode_L1.m\
# Description: Script for decoding analysis on layer 1 of probabilistic RNN

# In[54]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.io import loadmat
from fnc_fit_and_score_r import fnc_fit_and_score_r
from multiprocessing import Pool
import sliding_window

def sliding_window(elements, window_size):
  if len(elements) <= window_size:
    return elements

  windows = []
  for i in range(len(elements) - window_size + 1):
    windows.append(elements[i:i + window_size])

  return windows
# In[56]:


# MODIFY HERE
# what conditions were the RNNs trained on?
RNN_params = {
    'prob_split': '70_30',
    'afc': [6, 2],
    'coh': ['hi', 'lo'],
    'feedback': False,
    'thresh': [.3, .7],
    'model': [0, 1 ,2]
}

D_params = {
    'time_avg': False,
    't_win': [130, -1],
    'n_cvs': 5,
    'num_cgs': 30,
    'label': 'stim',  # 'stim' or 'choice'
    'units': 'all',  # 'all' or 'exc' or 'inh'
    'pred': 'all'  # 'expected' or 'unexpected', 'all'
}
# Timing of task
task_info = {
    'trials': 1000,
    'trial_dur': 250,
    'stim_on': 80,
    'stim_dur': 50
}
n_cvs = 5
window = 50
# penalties to eval
num_cgs = 30
Cs = np.logspace( -5,1,num_cgs )

# set up the grid
param_grid = { 'C': Cs, 'kernel': ['linear'] }

# define object - use a SVC that balances class weights (because they are biased, e.g. 70/30)
# note that can also specify cv folds here, but I'm doing it by hand below in a loop
grid = GridSearchCV( SVC(class_weight = 'balanced'),param_grid,refit=True,verbose=0 )

modelnum = RNN_params['model']


# In[57]:


#combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh'])) # add model back to combos later

# for now just focus on one model
model = 0
nboots = 10 # let's just do a few for now

#for afc, coh in combinations:
afc = 6
coh = 'lo'

# Load data
if sys.platform.startswith('linux'):
    data_dir = f"/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/rdk_{RNN_params['prob_split']}_{afc}afc/feedforward_only/{coh}_coh"
else:
    data_dir = f"/Volumes/serenceslab/holly/RNN_Geo/data/rdk_{RNN_params['prob_split']}_{afc}afc/feedforward_only/{coh}_coh"

# Chose the model
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]# Get all the trained models (should be 40 .mat files)
model_path = os.path.join(data_dir, mat_files[model]) 
model = loadmat(model_path) # model.keys()

# get the data from layer 1 decode stim
# this is a [trial x time step x unit] matrix
data_file = f"{data_dir}/Trials{task_info['trials']}_model{model_path[-7:-4]}_balanced.npz"
data = np.load(data_file)
data_d = data['fr3']

if D_params['label'] == 'stim':
    labs = data['labs'].squeeze()
elif D_params['label'] == 'choice':
    labs = data['outs'][:,-1]

# get some info about structure of the data
tris = data_d.shape[0]             # number of trials
tri_ind = np.arange(0,tris)      # list from 0...tris
hold_out = int( tris / n_cvs )   # how many trials to hold out
thresh = RNN_params.get('thresh', [.3, .7])
if D_params['label'] == 'stim':
    n_classes = len(np.unique(labs))
else:
    n_classes = 2

times = sliding_window(range(task_info['stim_dur']+task_info['stim_on'],task_info['trial_dur']), window)  
acc = np.zeros((nboots, len(times), n_classes))
    
# In[]:
# do bootstraps

def process_bootstrap(n_boot):
    np.random.seed(n_boot)  # Ensure reproducibility for each bootstrap
    results = []
    for t in times:
        seed = np.random.randint(0, 1000000)  # Unique seed for each time point within the bootstrap
        result = fnc_fit_and_score_r(t, np.mean(data_d[:, t, :], axis=1), tri_ind, hold_out, n_cvs, n_classes, labs, D_params['label'], thresh, grid, seed)
        results.append(result)
    if n_boot % 5 == 0:
        print(f'done decoding boot: {n_boot}')
    return results
# In[]:
start_time = time.time() 
if __name__ == "__main__":
    with Pool(processes=round(os.cpu_count() * .9)) as pool:
        results = pool.map(process_bootstrap, range(nboots))    
    acc = np.array(results)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# In[]:

# get CI over bootstraps

# In[58]:


full_file = f'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr3/results_balanced_all{modelnum}.npz'
np.savez(full_file, lo_2_stim_all2 = lo_2_stim_all2, lo_6_stim_all2 = lo_6_stim_all2, hi_2_stim_all2 = hi_2_stim_all2, hi_6_stim_all2 = hi_6_stim_all2, lo_2_choice_all2 = lo_2_choice_all2, lo_6_choice_all2 = lo_6_choice_all2, hi_2_choice_all2 = hi_2_choice_all2, hi_6_choice_all2 = hi_6_choice_all2)
print(f'{"done saving"}')


# In[38]:


# load and graph
full_file = '/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/results_stim_balanced_all1.npz'
accs1 = np.load(full_file)


# In[53]:


# plots
fig, axs = plt.subplots(1,2, figsize=(12, 8))

# Plot 1
axs[0].plot(accs1['hi_6_stim_all1'][:,0], color='blue', label='6afc exp')
axs[0].plot(np.mean(accs1['hi_6_stim_all1'][:,1:5], axis = 1), color='red',  label='6fac unexp')
axs[0].plot(accs1['hi_2_stim_all1'][:,0], color='blue',alpha=0.4, label='2afc exp')
axs[0].plot(np.mean(accs1['hi_2_stim_all1'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[0].set_xlabel('Time after stimulus offset')
axs[0].set_ylabel('Decoding Accuracy')
axs[0].set_title('Decoding Stimulus, hi coh')
axs[0].set_ylim(0, 1)
axs[0].legend()

# Plot 2
axs[1].plot(accs1['lo_6_stim_all1'][:,0], color='blue', label='6afc exp')
axs[1].plot(np.mean(accs1['lo_6_stim_all1'][:,1:5], axis = 1), color='red',  label='6fac unexp')
axs[1].plot(accs1['lo_2_stim_all1'][:,0], color='blue',alpha = 0.4, label='2afc exp')
axs[1].plot(np.mean(accs1['lo_2_stim_all1'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[1].set_xlabel('Time after stimulus offset')
axs[1].set_ylabel('Decoding Accuracy')
axs[1].set_title('Decoding Stimulus, lo coh')
axs[1].set_ylim(0, 1)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.rcParams.update({'font.size': 12})

plt.savefig(f"{'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/stim_all1_plots_06032024.png'}")


# In[52]:


full_file = '/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/results_stim_balanced_all0.npz'
accs0 = np.load(full_file)
# plots
fig, axs = plt.subplots(1,2, figsize=(12, 8))

# Plot 1
axs[0].plot(accs0['hi_6_stim_all0'][:,0], color='blue', label='6afc exp')
axs[0].plot(np.mean(accs0['hi_6_stim_all0'][:,1:5], axis = 1), color='red', label='6fac unexp')
axs[0].plot(accs0['hi_2_stim_all0'][:,0], color='blue',alpha = 0.4, label='2afc exp')
axs[0].plot(np.mean(accs0['hi_2_stim_all0'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[0].set_xlabel('Time after stimulus offset')
axs[0].set_ylabel('Decoding Accuracy')
axs[0].set_title('Decoding Stimulus, hi coh')
axs[0].set_ylim(0, 1)
axs[0].legend()

# Plot 2
axs[1].plot(accs0['lo_6_stim_all0'][:,0], color='blue', label='6afc exp')
axs[1].plot(np.mean(accs0['lo_6_stim_all0'][:,1:5], axis = 1), color='red', label='6fac unexp')
axs[1].plot(accs0['lo_2_stim_all0'][:,0], color='blue', alpha = 0.4,label='2afc exp')
axs[1].plot(np.mean(accs0['lo_2_stim_all0'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[1].set_xlabel('Time after stimulus offset')
axs[1].set_ylabel('Decoding Accuracy')
axs[1].set_title('Decoding Stimulus, lo coh')
axs[1].set_ylim(0, 1)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.rcParams.update({'font.size': 12})

plt.savefig(f"{'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/stim_all0_plots_06032024.png'}")


# In[51]:


full_file = f'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/results_balanced_all{modelnum}.npz'
accs2 = np.load(full_file)
# plots
fig, axs = plt.subplots(1,2, figsize=(12, 8))

# Plot 1
axs[0].plot(accs2['hi_6_stim_all2'][:,0], color='blue', label='6afc exp')
axs[0].plot(np.mean(accs2['hi_6_stim_all2'][:,1:5], axis = 1), color='red', label='6fac unexp')
axs[0].plot(accs2['hi_2_stim_all2'][:,0], color='blue', label='2afc exp', alpha = 0.4)
axs[0].plot(np.mean(accs2['hi_2_stim_all2'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[0].set_xlabel('Time after stimulus offset')
axs[0].set_ylabel('Decoding Accuracy')
axs[0].set_title('Decoding Stimulus, hi coh')
axs[0].set_ylim(0, 1)
axs[0].legend()

# Plot 2
axs[1].plot(accs2['lo_6_stim_all2'][:,0], color='blue', label='6afc exp')
axs[1].plot(np.mean(accs2['lo_6_stim_all2'][:,1:5], axis = 1), color='red', label='6fac unexp')
axs[1].plot(accs2['lo_2_stim_all2'][:,0], color='blue', label='2afc exp', alpha = 0.4)
axs[1].plot(np.mean(accs2['lo_2_stim_all2'][:,1:2], axis = 1), color='red', alpha=0.4, label='2afc unexp')
axs[1].set_xlabel('Time after stimulus offset')
axs[1].set_ylabel('Decoding Accuracy')
axs[1].set_title('Decoding Stimulus, lo coh')
axs[1].set_ylim(0, 1)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.rcParams.update({'font.size': 12})

plt.savefig(f"{'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/stim_all2_plots_06032024.png'}")


# In[42]:


# plot all three models


