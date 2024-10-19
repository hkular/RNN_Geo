#!/usr/bin/env python
# coding: utf-8

# Name: Robert Kim\
# Date: 03-29-2023\
# Email: robert.f.kim@gmail.com\
# fnc_generate_trials.m\
# Description: Function to generate trials for a task
# 
# Adapted from .m to .ipynb by Holly Kular\
# date: 02-15-2024

# INPUT
#  - task: task name ('instr', 'xor', etc...)
#  - task_info: struct containing task information (stim onset time, 
#  - stim duration, delay duration, etc...)\
#  
# OUTPUT
#  - u: input matrix for fnc_eval_model.ipynb
#  - lab: struct containing trial info (instruction timing/amp and stim info)

# In[7]:
    
def fnc_generate_trials(task, task_info, np, trial_info):
    if task.lower() == 'rdk':
        
        u = np.zeros((task_info['num_stims'], task_info['trial_dur']))
        u += np.random.randn(*u.shape) # add noise

        # Generate a random number
        rand_num = np.random.rand()

        if rand_num < task_info['primary_prob']:
            # Increase the stimulus intensity for the predominant stimulus
            u[task_info['pred'], task_info['stim_on']:task_info['stim_on']+task_info['stim_dur']] += task_info['coh']
            lab = task_info['pred']
        else:
            # Randomly select another stimulus
            rand_choice = np.random.permutation(task_info['num_stims']-1)
            choices = np.setdiff1d(np.arange(task_info['num_stims']),task_info['pred'])
            rand_choice = choices[rand_choice[0]]
            u[rand_choice, task_info['stim_on']:task_info['stim_on']+task_info['stim_dur']] += task_info['coh']
            lab = rand_choice

    elif task.lower() == 'instr':  
        
        if task_info['force']: 
            # task_info.force for 'instr' is [stim1, stim2, instr_amp, instr_t]
            u = np.zeros((3, trial_info['trial_dur']))
            if task_info['force'][-1] == -1:
                instr_on = trial_info['stim_on'] - 50
            else:
                instr_on = trial_info['stim_on'] + trial_info['stim_dur']
            instr_amp = task_info['force'][2] # (3) in matlab, but here 2 because zero indexing?

            u[-1, instr_on:instr_on+50] = instr_amp
            u[0, task_info['stim_on']:task_info['stim_on']+task_info['stim_dur']] = task_info['force'][0]
            u[1, task_info['stim_on']+task_info['stim_dur']+task_info['delay']:task_info['stim_on']+2*task_info['stim_dur']+task_info['delay']] = task_info['force'][1]

            lab = {}
            lab['stim_lab'] = np.prod(task_info['force'][:2])
            lab['instr_t'] = task_info['force'][-1]
            lab['instr_amp'] = instr_amp
            lab['stims_ids'] = task_info['force'][:2]
                 
        else: # Generate specific trial instead of random trial
            
            u = np.zeros((3, trial_info['trial_dur']))
            
            # First set the instruction signal onset and amplitude
            # Instruction timing (either before or after first stim)
            # instr_t indicates the timing (-1 = before the first stim, +1 = after the first stim)
            if np.random.rand() < 0.50:
                instr_on = task_info['stim_on'] - 50
                instr_t = -1
            else:
                instr_on = task_info['stim_on'] + task_info['stim_dur']
                instr_t = 1
            
            # Instruction amplitude (+1 for DMS and -1 for anti-DMS)
            if np.random.rand() < 0.50:
                instr_amp = 1 # DMS task
            else:
                instr_amp = -1 # anti-DMS task
            u[-1, instr_on:instr_on+50] = instr_amp
            
            # Now set the first and second stimuli
            stim_labs = np.zeros((1,2))
            if np.random.rand() < 0.5:
                u[0, task_info['stim_on']:task_info['stim_on']+task_info['stim_dur']+1] = 1
                stim_labs[0, 0] = 1
            else:
                u[0, task_info['stim_on']:task_info['stim_on']+task_info['stim_dur']+1] = -1
                stim_labs[0, 0] = -1
            
            if np.random.rand() < 0.5:
                u[1, task_info['stim_on']+task_info['stim_dur']+task_info['delay']:task_info['stim_on']+2*task_info['stim_dur']+task_info['delay']] = 1
                stim_labs[0,1] = 1
            else:
                u[1, task_info['stim_on']+task_info['stim_dur']+task_info['delay']:task_info['stim_on']+2*task_info['stim_dur']+task_info['delay']] = -1
                stim_labs[0,1] = -1
            lab['stim_lab'] = np.prod(stim_labs)
            lab['instr_amp'] = instr_t
            lab['stim_ids'] = stim_labs
        
        
    elif task.lower() == 'instr2':
        u = np.zeros((6, task_info['trial_dur']))
        
        # First set the instruction signal onset and amplitude
        # Instruction timing (either before or after first stim)
        # instr_t indicates the timing (-1 = before the first stim, +1 = after the first stim)
        if np.random.rand() < 0.50:
            instr_on = task_info['stim_on'] - 50
            instr_t = -1
        else:
            instr_on = task_info['stim_on'] + task_info['stim_dur']
            instr_t = 1
            
        # Instruction amplitude (+1 for DMS and -1 for ant-DMS)
        if np.random.rand() < 0.50:
            instr_amp = 1 # DMS task
        else:
            instr_amp = -1 # anti-DMS task
        
        u[4, instr_on:instr_on+51] = instr_amp
        
        # Cue singal (-1 for the first modality and +1 for the second modality)
        if np.random.rand() < 0.50:
            cue_amp = -1 # focus on the first modelity
        else:
            cue_amp = 1 # focus on second modality
        
        u[5, instr_on:instr_on+51] = cue_amp
        
        # Set the first and second sitmuli for the first modality
        stim_labs1 = np.zeros((1,2))
        if np.random.rand() < 0.5:
            u[0, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = 1
            stim_labs1[0, 0] = 1
        else:
            u[0, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = -1
            stim_labs1[0, 0] = -1
            
        if np.random.rand() < 0.5:
            u[1, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = 1
            stim_labs1[0, 1] = 1
        else:
            u[1, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = -1
            stim_labs1[0, 1] = -1
            
        # Set the first and second sitmuli for the second modality
        stim_labs2 = np.zeros((1,2))
        if np.random.rand() < 0.5:
            u[2, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = 1
            stim_labs2[0, 0] = 1
        else:
            u[2, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = -1
            stim_labs2[0, 0] = -1
            
        if np.random.rand() < 0.5:
            u[3, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = 1
            stim_labs2[0, 1] = 1
        else:
            u[3, task_info['stim_on']:task_info['stim_on']+ task_info['stim_dur']+1] = -1
            stim_labs2[0, 1] = -1
            
        lab['stim_mod1'] = stim_labs1
        lab['stim_mod2'] = stim_labs2
        lab['stim_lab1'] = np.prod(stim_labs1)
        lab['stim_lab2'] = np.prod(stim_labs2)
        
        lab['instr_t'] = instr_t
        lab['instr_amp'] = instr_amp
        lab['cue_amp'] = cue_amp

            
            
            
    return [u, lab]

