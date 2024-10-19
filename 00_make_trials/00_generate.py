#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# Name: Robert Kim\
# Date: 06-09-2023\
# Email: rkim@salk.edu\
# generate_trials.m\
# Description: Script to generate trials\
# 
# coverted from .m to .pynb by Holly Kular\
# date: 02-15-2024


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime
import fnc_generate_trials
import fnc_eval_model



# MODIFY HERE
# what conditions were the RNNs trained on?
prob_split = '70_30' # the probability of stimulus 1 vs all
afc = '2' # number of alternatives
coh = 'hi' # coherence
feedback = False # interlayer feedback (true or false)



data_dir = f"/mnt/neurocube/local/serenceslab/holly/RNN_Geo/data/rdk_{prob_split}_{afc}afc/feedforward_only/{coh}_coh"


# Get all the trained models (should be 40 .mat files)
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

# Choose one model as an example
model_path = os.path.join(data_dir, mat_files[1])


# **Testing task condition 
task_info = {}
task_info['trials'] = 100
task_info['trial_dur'] = 250  # trial duration (timesteps)
task_info['stim_on'] = 80
task_info['stim_dur'] = 50
task_info['num_stims'] = int(afc) # nAFC
if coh == 'hi': # hi_coh = 0.7 vs. lo_coh = 0.6
    task_info['coh'] = 0.7 
else:
    task_info['coh'] = 0.6
    
task_info['units'] = 200 # number of units
# NOTE: adjust pred and primary_prob to change the testing environement
# Ex. Evenly distributed 6-AFC => task_info.primary_prob = 1/6;
# Ex. pred=5 and primary_prob = 0.7 => stim "5" will be predominant (70% of the time)
task_info['pred'] = 0 # predominant stimulus is "1" aka 0
task_info['primary_prob'] = 0.70; # 70-30 split



# Store firing rates, outputs, and labels for each trials
fr1 = np.zeros((task_info['trials'], task_info['trial_dur'], task_info['units']))
fr2 = np.zeros((task_info['trials'], task_info['trial_dur'], task_info['units']))
fr3 = np.zeros((task_info['trials'], task_info['trial_dur'], task_info['units']))
outs = np.zeros((task_info['trials'], task_info['trial_dur']))
labs = np.zeros((task_info['trials'], 1))

for tri in range(0, task_info['trials']):
    # Generate trials
    u, lab = fnc_generate_trials('rdk', task_info)

    # Now test the trained model
    out, O = fnc_eval_model(model_path, u, feedback)

    outs[tri, :] = out['O']  # Store the output signal
    labs[tri] = lab
    
    fr1[tri, :, :] = out['R1']
    fr2[tri, :, :] = out['R2']
    fr3[tri, :, :] = out['R3']
print(f'{"done generating trials"}')




# save trial data to load into other notebooks
today = datetime.now().strftime("%y%m%d")
full_file = os.path.join(data_dir, f"/Trials_{today}.npz")
np.savez(full_file,fr1 = fr1, fr3=fr3, outs=outs, labs=labs)

