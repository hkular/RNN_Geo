#!/usr/bin/env python
# coding: utf-8

# Name: Holly Kular\
# Date: 03-19-2024\
# Email: hkular@ucsd.edu\
# decode_L1.m\
# Description: Script for decoding analysis on layer 1 of probabilistic RNN
import os
# In[54]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.io import loadmat
from fnc_fit_and_score_r import fnc_fit_and_score_r
from multiprocessing import Pool
import sliding_window
from scipy.stats import sem, t

def sliding_window(elements, window_size):
  if len(elements) <= window_size:
    return elements

  windows = []
  for i in range(len(elements) - window_size + 1):
    windows.append(elements[i:i + window_size])

  return windows


def process_bootstrap(n_boot):
    np.random.seed(n_boot)  # Ensure reproducibility for each bootstrap
    results = []
    for t in times:
        seed = np.random.randint(0, 1000000)  # Unique seed for each time point within the bootstrap
        result = fnc_fit_and_score_r(np.mean(data_d[:, t, :], axis=1), tri_ind, hold_out, n_cvs, n_classes, labs, D_params['label'], thresh, grid, seed)
        results.append(result)
#    if n_boot % 100 == 0:
#        print(f'done decoding boot: {n_boot}')
    return results
# In[56]:


# MODIFY HERE
# what conditions were the RNNs trained on?
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

window = 50

n_cvs = 5
# penalties to eval
num_cgs = 30
Cs = np.logspace( -5,1,num_cgs )

# set up the grid
param_grid = { 'C': Cs, 'kernel': ['linear'] }

# define object - use a SVC that balances class weights (because they are biased, e.g. 70/30)
# note that can also specify cv folds here, but I'm doing it by hand

grid = GridSearchCV( SVC(class_weight = 'balanced'),param_grid,refit=True,verbose=0 )

#modelnum = RNN_params['model']


# In[57]:


#combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh'])) # add model back to combos later

# for now just focus on one model
modelnum = 1
nboots = 10 # let's just do a few for now

#for afc, coh in combinations:
afc = 6
coh = 'lo'
fr = 1

combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh'], RNN_params['fr']))

for afc, coh, fr in combinations:

    # Load data
    if sys.platform.startswith('linux'):
        data_dir = f"/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/rdk_{RNN_params['prob_split']}_{afc}afc/feedforward_only/{coh}_coh"
    else:
        data_dir = f"/Volumes/serenceslab/holly/RNN_Geo/data/rdk_{RNN_params['prob_split']}_{afc}afc/feedforward_only/{coh}_coh"
    
    # Chose the model
    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]# Get all the trained models (should be 40 .mat files)
    model_path = os.path.join(data_dir, mat_files[modelnum]) 
    model = loadmat(model_path) # model.keys()
    
    # get the data from layer 1 decode stim
    # this is a [trial x time step x unit] matrix
    data_file = f"{data_dir}/Trials{task_info['trials']}_model{model_path[-7:-4]}_balanced.npz"
    data = np.load(data_file)
    data_d = data[f'fr{fr}']
    
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
    
    
    start_time = time.time() 
    if __name__ == "__main__":
        with Pool(processes=round(os.cpu_count() * .9)) as pool:
            results = pool.map(process_bootstrap, range(nboots))    
        acc = np.array(results)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    
    full_file = f'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr{fr}/{coh}_{afc}afc/boot_balanced_all{modelnum}.npz'
    np.savez(full_file, acc = acc)
    #np.savez(full_file, lo_2_stim_all2 = lo_2_stim_all2, lo_6_stim_all2 = lo_6_stim_all2, hi_2_stim_all2 = hi_2_stim_all2, hi_6_stim_all2 = hi_6_stim_all2, lo_2_choice_all2 = lo_2_choice_all2, lo_6_choice_all2 = lo_6_choice_all2, hi_2_choice_all2 = hi_2_choice_all2, hi_6_choice_all2 = hi_6_choice_all2)
    print(f'{"done saving"}')
        

# In[65]:
start_time = time.time() 
if __name__ == "__main__":
    with Pool(processes=round(os.cpu_count() * .9)) as pool:
        results = pool.map(process_bootstrap, range(nboots))    
    acc = np.array(results)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
pool.close()
# In[]:

# get CI over bootstraps
y_data = np.mean(acc[:, :, 1:], axis = 2)

# Calculate the mean over axis 0
mean_y = np.mean(y_data, axis=0)

# Calculate the standard error of the mean (SEM)
sem_y = sem(y_data, axis=0)

# Define the confidence interval (95%)
confidence_interval = 1.96 * sem_y

# Define the x-axis data
x_data = np.arange(y_data.shape[1])

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(x_data, mean_y, label='Mean', color='blue')

# Plot the confidence interval as a ribbon
plt.fill_between(x_data, mean_y - confidence_interval, mean_y + confidence_interval, color='blue', alpha=0.3, label='95% CI')

# Labels and title
plt.xlabel('time steps')
plt.ylabel('Decoding accuracy')
plt.title(f'{afc} afc {coh} coh')
plt.legend()

# Show plot
plt.show()

# In[58]:


full_file = f'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr{fr}/{coh}_{afc}afc/boot_balanced_all{modelnum}.npz'
np.savez(full_file, lo_6afc = acc)
#np.savez(full_file, lo_2_stim_all2 = lo_2_stim_all2, lo_6_stim_all2 = lo_6_stim_all2, hi_2_stim_all2 = hi_2_stim_all2, hi_6_stim_all2 = hi_6_stim_all2, lo_2_choice_all2 = lo_2_choice_all2, lo_6_choice_all2 = lo_6_choice_all2, hi_2_choice_all2 = hi_2_choice_all2, hi_6_choice_all2 = hi_6_choice_all2)
print(f'{"done saving"}')


# In[38]:

fr = 3
modelnum = 0


combinations = list(itertools.product(RNN_params['afc'], RNN_params['coh']))

for afc, coh in combinations:

    # load and graph
    #full_file = f'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/neutral/fr{fr}/{coh}_{afc}afc/boot_balanced_all{modelnum}.npz'
    full_file = f'/Volumes/serenceslab/holly/RNN_Geo/data/decoding/neutral/fr{fr}/{coh}_{afc}afc/boot_balanced_all{modelnum}.npz'
    acc = np.load(full_file)
    
    for var in acc:
        globals()[var] = acc[var]
        
afc6 = np.stack((hi_6afc, lo_6afc), axis = 3) 
afc2 = np.stack((hi_2afc, lo_2afc), axis = 3)



# In[43]:
    
# plot for each cond separately
mean_y = np.zeros((71,2))
ci = np.zeros((71,2))
y0 = np.zeros((71,2))
ci0 = np.zeros((71,2))
afc = 6
cohs = ['hi', 'lo']


for i in range(5):
    plt.plot( afc6[i, :, 0, ])

# Labels and title
plt.rcParams.update({'font.size': 15})
plt.xlabel('time steps after stim offset')
plt.ylabel('Decoding accuracy')
plt.title(f'{afc} afc')
plt.legend()

# Show plot
plt.show()

# In[49]:
mean_y = np.zeros((71,2))
ci = np.zeros((71,2))
y0 = np.zeros((71,2))
ci0 = np.zeros((71,2))
afc = 2
#coh = 'hi'
cohs = ['hi', 'lo']
confidence = .95
dof = afc2.shape[0] - 1
t_crit = np.abs(t.ppf((1-confidence)/2,dof))

for m in range(2):

    # get CI over bootstraps
    
    y_data = np.mean(afc2[:, :, 1:, m], axis = 2)
    
    # Calculate the mean over axis 0
    mean_y[:,m] = np.mean(y_data, axis=0)
    y0[:,m] = np.mean(afc2[:,:,0,m], axis = 0)
    
    # Calculate the standard error of the mean (SEM)
    s_y = y_data.std(axis=0)
    s_y0 = afc2[:,:,0,m].std(axis = 0)
    
    # Define the confidence interval (95%)
    ci[:,m] = s_y*t_crit/np.sqrt(y_data.shape[0])
    ci0[:,m] = s_y0*t_crit/np.sqrt(y_data.shape[0])
    
    # Define the x-axis data
    x_data = np.arange(y_data.shape[1])

# Plotting
plt.figure(figsize=(10, 6))

for m in range(2):
    coh = cohs[m]
    # Plot the mean line
    plt.plot(x_data, mean_y[:,m], label=f'unexp {coh}')
    plt.plot(x_data, y0[:,m], label=f'exp {coh}')
    
    # Plot the confidence interval as a ribbon
    plt.fill_between(x_data, mean_y[:,m] - ci[:,m], mean_y[:,m] + ci[:,m], alpha=0.3)
    plt.fill_between(x_data, y0[:,m] - ci0[:,m], y0[:,m] + ci0[:,m], alpha=0.3)

# Labels and title
plt.rcParams.update({'font.size': 15})
plt.xlabel('time steps after stim offset')
plt.ylabel('Decoding accuracy')
plt.title(f'{afc} afc')
plt.legend()

# Show plot
plt.show()

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


