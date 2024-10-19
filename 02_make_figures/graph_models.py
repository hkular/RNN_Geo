#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:05:51 2024

@author: hkular
"""
# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import itertools
from scipy.stats import sem
import math
# In[2]:
# load and graph
mod0 = np.load('/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr1/results_stim_balanced_all0.npz')
mod1 = np.load('/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr1/results_balanced_all1.npz')
mod2 = np.load('/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr1/results_balanced_all2.npz')


for var in mod0:
    globals()[var] = mod0[var]
for var in mod1:
    globals()[var] = mod1[var]
for var in mod2:
    globals()[var] = mod2[var]
# In[3]:
    
# which one
coh = 'lo'
afc  = 2
label = 'stim'    

array_names = [f"{coh}_{afc}_{label}_all0", f"{coh}_{afc}_{label}_all1", f"{coh}_{afc}_{label}_all2"]
arrays = [globals()[name] for name in array_names]


# mean
stacked = np.stack(arrays, axis=2)
mean_mod = np.mean(stacked, axis = 2)
exec(f'{coh}_{afc}_{label}_mean = mean_mod')

# sem
sem_mod = sem(stacked, axis = 2)
exec(f'{coh}_{afc}_{label}_sem = sem_mod')

print(f'{"done stats"}')
# In[4]:

#plots

fig, axs = plt.subplots(1,2, figsize=(12, 8))

# Plot 1
    # 6afc
axs[0].plot(hi_6_stim_mean[:,0], color='blue', label='6afc exp')
axs[0].plot(np.mean(hi_6_stim_mean[:,1:5], axis = 1), color='red',  label='6fac unexp')
axs[0].fill_between(range(0,71),hi_6_stim_mean[:,0] - hi_6_stim_sem[:,0], hi_6_stim_mean[:,0] + hi_6_stim_sem[:,0], color='blue', alpha=0.2)
axs[0].fill_between(range(0,71),np.mean(hi_6_stim_mean[:,1:5], axis = 1) - np.mean(hi_6_stim_sem[:,1:5], axis = 1), np.mean(hi_6_stim_mean[:,1:5], axis = 1) + np.mean(hi_6_stim_sem[:,1:5], axis = 1), color='red', alpha=0.2)
    #2afc
axs[0].plot(hi_2_stim_mean[:,0], color='green',alpha=1, label='2afc exp')
axs[0].plot(np.mean(hi_2_stim_mean[:,1:2], axis = 1), color='orange', alpha=1, label='2afc unexp')
axs[0].fill_between(range(0,71),hi_2_stim_mean[:,0] - hi_2_stim_sem[:,0], hi_2_stim_mean[:,0] + hi_2_stim_sem[:,0], color='green', alpha=0.1)
axs[0].fill_between(range(0,71),np.mean(hi_2_stim_mean[:,1:2], axis = 1) - np.mean(hi_2_stim_sem[:,1:2], axis = 1), np.mean(hi_2_stim_mean[:,1:2], axis = 1) + np.mean(hi_2_stim_sem[:,1:2], axis = 1), color='orange', alpha=0.1)
    
axs[0].set_xlabel('Time after stimulus offset')
axs[0].set_ylabel('Decoding Accuracy')
axs[0].set_title('Decoding Stimulus, hi coh')
axs[0].set_ylim(0, 1)
axs[0].legend()


# Plot 2
    #6afc
axs[1].plot(lo_6_stim_mean[:,0], color='blue', label='6afc exp')
axs[1].plot(np.mean(lo_6_stim_mean[:,1:5], axis = 1), color='red',  label='6fac unexp')
axs[1].fill_between(range(0,71),lo_6_stim_mean[:,0] - lo_6_stim_sem[:,0], lo_6_stim_mean[:,0] + lo_6_stim_sem[:,0], color='blue', alpha=0.2)
axs[1].fill_between(range(0,71),np.mean(lo_6_stim_mean[:,1:5], axis = 1) - np.mean(lo_6_stim_sem[:,1:5], axis = 1), np.mean(lo_6_stim_mean[:,1:5], axis = 1) + np.mean(lo_6_stim_sem[:,1:5], axis = 1), color='red', alpha=0.2)
    
    #2afc
axs[1].plot(lo_2_stim_mean[:,0], color='green',alpha = 1, label='2afc exp')
axs[1].plot(np.mean(lo_2_stim_mean[:,1:2], axis = 1), color='orange', alpha=1, label='2afc unexp')
axs[1].fill_between(range(0,71),lo_2_stim_mean[:,0] - lo_2_stim_sem[:,0], lo_2_stim_mean[:,0] + lo_2_stim_sem[:,0], color='green', alpha=0.1)
axs[1].fill_between(range(0,71),np.mean(lo_2_stim_mean[:,1:2], axis = 1) - np.mean(lo_2_stim_sem[:,1:2], axis = 1), np.mean(lo_2_stim_mean[:,1:2], axis = 1) + np.mean(lo_2_stim_sem[:,1:2], axis = 1), color='orange', alpha=0.1)
   

axs[1].set_xlabel('Time after stimulus offset')
axs[1].set_ylabel('Decoding Accuracy')
axs[1].set_title('Decoding Stimulus, lo coh')
axs[1].set_ylim(0, 1)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.rcParams.update({'font.size': 12})

#plt.savefig(f"{'/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/stim_all1_plots_06032024.png'}")

# In[5]:
# plot garbage as baseline decoding accuracy model 0 6 afc lo coh

#boot = np.load('/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr3/boot_balanced_all0.npz')
#for var in boot:
#    globals()[var] = boot[var]

# lo_6afc and garbage to plot   
   

# get CI over bootstraps
y_data = lo_6afc[:, :, 0]
y_g = garbage[:,:,0]
y_datau = np.mean(lo_6afc[:, :, 1:], axis = 2)
y_gu = np.mean(garbage[:,:,1:], axis = 2)
# Calculate the mean over axis 0
mean_y = np.mean(y_data, axis=0)
mean_y_g = np.mean(y_g, axis = 0)
mean_yu = np.mean(y_datau, axis=0)
mean_y_gu = np.mean(y_gu, axis = 0)
# Calculate the standard error of the mean (SEM)
sem_y = sem(y_data, axis=0)
sem_y_g = sem(y_g, axis =0)
sem_yu = sem(y_datau, axis=0)
sem_y_gu = sem(y_gu, axis =0)
# Define the confidence interval (95%)
confidence_interval = 1.96 * sem_y
ci_g = 1.96 *sem_y_g
confidence_intervalu = 1.96 * sem_yu
ci_gu = 1.96 *sem_y_gu
# Define the x-axis data
x_data = np.arange(y_data.shape[1])

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(x_data, mean_y, label='expected', color='blue')
plt.plot(x_data, mean_y_g, label = 'Garbage exp', color = 'red')

plt.plot(x_data, mean_yu, label='unexpected', color='green')
plt.plot(x_data, mean_y_gu, label = 'Garbage unexp', color = 'orange')
# Plot the confidence interval as a ribbon
plt.fill_between(x_data, mean_y - confidence_interval, mean_y + confidence_interval, color='blue', alpha=0.3)
plt.fill_between(x_data, mean_y_g - ci_g, mean_y_g + ci_g, color = 'red', alpha = 0.3)
plt.fill_between(x_data, mean_yu - confidence_intervalu, mean_yu + confidence_intervalu, color='green', alpha=0.3)
plt.fill_between(x_data, mean_y_gu - ci_gu, mean_y_gu + ci_gu, color = 'orange', alpha = 0.3)
# Labels and title
plt.xlabel('time steps')
plt.ylabel('Decoding accuracy')
plt.title(f'{afc} afc {coh} coh')
plt.legend()

# Show plot
plt.show()
    
# In[6]:
# plot results unexpected , where stim 1 instead of stim 0 is predominant in trials fed into decoder

#unexpected = np.load('/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/decoding/fr1/results_unexpected_all_allmodels.npz')
#for var in unexpected:
#    globals()[var] = unexpected[var]

fig, axs = plt.subplots(1,2, figsize=(12, 8))

# Plot 1
    # 6afc
axs[0].plot(hi_6_stim_all0[:,0], color='blue', label='6afc exp')
axs[0].plot(np.mean(hi_6_stim_all0[:,1:5], axis = 1), color='red',  label='6fac unexp')
#axs[0].fill_between(range(0,71),hi_6_stim_all0[:,0] - hi_6_stim_sem[:,0], hi_6_stim_mean[:,0] + hi_6_stim_sem[:,0], color='blue', alpha=0.2)
#axs[0].fill_between(range(0,71),np.mean(hi_6_stim_all0[:,1:5], axis = 1) - np.mean(hi_6_stim_sem[:,1:5], axis = 1), np.mean(hi_6_stim_mean[:,1:5], axis = 1) + np.mean(hi_6_stim_sem[:,1:5], axis = 1), color='red', alpha=0.2)
    #2afc
axs[0].plot(hi_2_stim_all0[:,0], color='green',alpha=1, label='2afc exp')
axs[0].plot(np.mean(hi_2_stim_all0[:,1:2], axis = 1), color='orange', alpha=1, label='2afc unexp')
#axs[0].fill_between(range(0,71),hi_2_stim_all0[:,0] - hi_2_stim_sem[:,0], hi_2_stim_mean[:,0] + hi_2_stim_sem[:,0], color='green', alpha=0.1)
#axs[0].fill_between(range(0,71),np.mean(hi_2_stim_all0[:,1:2], axis = 1) - np.mean(hi_2_stim_sem[:,1:2], axis = 1), np.mean(hi_2_stim_mean[:,1:2], axis = 1) + np.mean(hi_2_stim_sem[:,1:2], axis = 1), color='orange', alpha=0.1)
    
axs[0].set_xlabel('Time after stimulus offset')
axs[0].set_ylabel('Decoding Accuracy')
axs[0].set_title('Decoding Stimulus, hi coh')
axs[0].set_ylim(0, 1)
axs[0].legend()


# Plot 2
    #6afc
axs[1].plot(lo_6_stim_all0[:,0], color='blue', label='6afc exp')
axs[1].plot(np.mean(lo_6_stim_all0[:,1:5], axis = 1), color='red',  label='6fac unexp')
#axs[1].fill_between(range(0,71),lo_6_stim_mean[:,0] - lo_6_stim_sem[:,0], lo_6_stim_mean[:,0] + lo_6_stim_sem[:,0], color='blue', alpha=0.2)
#axs[1].fill_between(range(0,71),np.mean(lo_6_stim_mean[:,1:5], axis = 1) - np.mean(lo_6_stim_sem[:,1:5], axis = 1), np.mean(lo_6_stim_mean[:,1:5], axis = 1) + np.mean(lo_6_stim_sem[:,1:5], axis = 1), color='red', alpha=0.2)
    
    #2afc
axs[1].plot(lo_2_stim_all0[:,0], color='green',alpha = 1, label='2afc exp')
axs[1].plot(np.mean(lo_2_stim_all0[:,1:2], axis = 1), color='orange', alpha=1, label='2afc unexp')
#axs[1].fill_between(range(0,71),lo_2_stim_mean[:,0] - lo_2_stim_sem[:,0], lo_2_stim_mean[:,0] + lo_2_stim_sem[:,0], color='green', alpha=0.1)
#axs[1].fill_between(range(0,71),np.mean(lo_2_stim_mean[:,1:2], axis = 1) - np.mean(lo_2_stim_sem[:,1:2], axis = 1), np.mean(lo_2_stim_mean[:,1:2], axis = 1) + np.mean(lo_2_stim_sem[:,1:2], axis = 1), color='orange', alpha=0.1)
   

axs[1].set_xlabel('Time after stimulus offset')
axs[1].set_ylabel('Decoding Accuracy')
axs[1].set_title('Decoding Stimulus, lo coh')
axs[1].set_ylim(0, 1)
axs[1].legend()

# Adjust layout
plt.tight_layout()
plt.rcParams.update({'font.size': 12})





