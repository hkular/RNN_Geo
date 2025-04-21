#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: Oct. 11, 2019
# Email: rkim@salk.edu
# Description: main script for training continuous-variable rate RNN models 
# For more info, refer to 
# Kim R., Li Y., & Sejnowski TJ. Simple Framework for Constructing Functional Spiking 
# Recurrent Neural Networks. Preprint at BioRxiv 
# https://urldefense.com/v3/__https://www.biorxiv.org/content/10.1101/579706v2__;!!Mih3wA!AJnCrORPcWvfJh_kGWX2FP0Jb3UM5gN5qI1oUp4fX738thKjt2umx4OvS0zE7QnieVGZUT6GCLF3my8vD3tf$  (2019).
# Note: This version uses 2 stimuli for the RDK tasks (one expected and one unexpected)
# NOTE: Edited on 12/31/2021 to add 1) a new version of RDK with 70/30 exp/unexp probability
#       (RDK_opposite for this hasn't been set up yet (the current one is of the 80/20 version); and
#       2) 3-feature Mante task
#     : Edited on 01/19/2022 to add variations of RDK tasks e.g., the 6AFC versions
# Edited on 4/14/25 using tf_upgrade_v2 to allow for tf 2.0 install to run
# EDited on 4/18/25 completed overhaul for TF2.0 compatibility

import os, sys
import time
import scipy.io
import numpy as np
import tensorflow as tf
import argparse
import datetime

# Import utility functions
from utils_v2 import set_gpu
from utils_v2 import restricted_float
from utils_v2 import str2bool

# Import the continuous rate model
from model_feedback_v2 import FR_RNN_dale

# Import the tasks

from model_feedback_v2 import generate_input_stim_rdk_70_30 # 2AFC; stim1 = 70%, stim2 = 30%
from model_feedback_v2 import generate_input_stim_rdk_80_20 # 2AFC; stim1 = 80%, stim2 = 20%
from model_feedback_v2 import generate_input_stim_rdk_equal # 2AFC; both stims are equally likely

from model_feedback_v2 import generate_input_stim_rdk_70_30_6AFC # 6AFC; stim1 = 70%, sum(unexpected stim) = 30%
from model_feedback_v2 import generate_input_stim_rdk_80_20_6AFC # 6AFC; stim1 = 80%, sum(unexpected stim) = 20%
from model_feedback_v2 import generate_input_stim_rdk_equal_6AFC # 6AFC; all stims are equally likely

from model_feedback_v2 import generate_target_continuous_rdk


# Parse input arguments
parser = argparse.ArgumentParser(description='Training rate RNNs')
parser.add_argument('--gpu', required=False,
        default='0', help="Which gpu to use")
parser.add_argument("--gpu_frac", required=False,
        type=restricted_float, default=0.4,
        help="Fraction of GPU mem to use")
parser.add_argument("--n_trials", required=True,
        type=int, default=200, help="Number of epochs")
parser.add_argument("--mode", required=True,
        type=str, default='Train', help="Train or Eval")
parser.add_argument("--output_dir", required=True,
        type=str, help="Model output path")
parser.add_argument("--N1", required=True,
        type=int, help="Number of neurons for the FIRST LAYER")
parser.add_argument("--N2", required=True,
        type=int, help="Number of neurons for the SECOND LAYER")
parser.add_argument("--N3", required=True,
        type=int, help="Number of neurons for the THIRD LAYER")
parser.add_argument("--gain", required=False,
        type=float, default = 1.5, help="Gain for the connectivity weight initialization")
parser.add_argument("--P_inh", required=False,
        type=restricted_float, default = 0.20,
        help="Proportion of inhibitory neurons")
parser.add_argument("--P_rec", required=False,
        type=restricted_float, default = 0.20,
        help="Connectivity probability")
parser.add_argument("--som_N", required=True,
        type=int, default = 0, help="Number of SST neurons")
parser.add_argument("--task", required=True,
        type=str, help="Task (XOR, sine, etc...)")
parser.add_argument("--act", required=True,
        type=str, default='sigmoid', help="Activation function (sigmoid, clipped_relu)")
parser.add_argument("--loss_fn", required=True,
        type=str, default='l2', help="Loss function (either L1 or L2)")
parser.add_argument("--apply_dale", required=True,
        type=str2bool, default='True', help="Apply Dale's principle?")
parser.add_argument("--decay_taus", required=True,
        nargs='+', type=float,
        help="Synaptic decay time-constants (in time-steps). If only one number is given, then all\
        time-constants set to that value (i.e. not trainable). Otherwise specify two numbers (min, max).")
args = parser.parse_args()

# Select GPU and limit Mem
set_gpu(args.gpu, args.gpu_frac)

# Set up the output dir where the output model will be saved
out_dir = os.path.join(args.output_dir, 'models', args.task.lower())
if args.apply_dale == False:
    out_dir = os.path.join(out_dir, 'NoDale')
if len(args.decay_taus) > 1:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Taus_' + str(args.decay_taus[0]) + '_' + str(args.decay_taus[1]))
else:
    out_dir = os.path.join(out_dir, 'P_rec_' + str(args.P_rec) + '_Tau_' + str(args.decay_taus[0]))

if os.path.exists(out_dir) == False:
    os.makedirs(out_dir)

# Number of units/neurons
N1 = args.N1
N2 = args.N2
N3 = args.N3
som_N = args.som_N; # number of SST neurons 

# Define task-specific parameters
# NOTE: Each time step is 5 ms

if args.task.lower() == 'rdk_70_30':
    # Simple discrimination task (stim1 = 70%, stim2 = 30%)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'rdk_80_20':
    # Simple discrimination task (stim1 = 80%, stim2 = 20%)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'rdk_equal':
        # Simple discrimination task (both stims are equally likely)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name                                  
            }
elif args.task.lower() == 'rdk_70_30_6afc':
    # Simple discrimination task (stim1 = 70%, sum(unexpected stim) = 30%)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'rdk_80_20_6afc':
    # Simple discrimination task (stim1 = 80%, sum(unexpected stim) = 20%)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name
            }
elif args.task.lower() == 'rdk_equal_6afc':
        # Simple discrimination task (both stims are equally likely)
    settings = {
            'T': 200, # trial duration (in steps)
            'stim_on': 50, # input stim onset (in steps)
            'stim_dur': 25, # input stim duration (in steps)
            'DeltaT': 1, # sampling rate
            'taus': args.decay_taus, # decay time-constants (in steps)
            'task': args.task.lower(), # task name                                  
            }

'''
Initialize the input and output weight matrices
'''

# Simple discrimination task (stim1 = 80%, stim2 = 20%)
if args.task.lower() == 'rdk_80_20':
    w_in = np.float32(np.random.randn(N1, 2))
    w_out = np.float32(np.random.randn(1, N3)/100)

# Simple discrimination task (both stims are equally likely)
elif args.task.lower() == 'rdk_equal':
    w_in = np.float32(np.random.randn(N1, 2))
    w_out = np.float32(np.random.randn(1, N3)/100)

# Simple discrimination task (stim1 = 70%, stim2 = 30%)
elif args.task.lower() == 'rdk_70_30':
    w_in = np.float32(np.random.randn(N1, 2))
    w_out = np.float32(np.random.randn(1, N3)/100)


# Simple discrimination task (stim1 = 70%)
elif args.task.lower() == 'rdk_70_30_6afc':
    w_in = np.float32(np.random.randn(N1, 6))
    w_out = np.float32(np.random.randn(1, N3)/100)

# Simple discrimination task (stim1 = 80%)
elif args.task.lower() == 'rdk_80_20_6afc':
    w_in = np.float32(np.random.randn(N1, 6))
    w_out = np.float32(np.random.randn(1, N3)/100)

# Simple discrimination task (both stims are equally likely)
elif args.task.lower() == 'rdk_equal_6afc':
    w_in = np.float32(np.random.randn(N1, 6))
    w_out = np.float32(np.random.randn(1, N3)/100)

'''
Define the training parameters (learning rate, training termination criteria, etc...)
'''
training_params = {
        'learning_rate': 0.01, # learning rate
        'loss_threshold': 5, # loss threshold (when to stop training)
        'eval_freq': 100, # how often to evaluate task perf
        'eval_tr': 100, # number of trials for eval
        'eval_amp_threh': 0.7, # amplitude threshold during response window
        'activation': args.act.lower(), # activation function
        'loss_fn': args.loss_fn.lower(), # loss function ('L1' or 'L2')
        'P_rec': 0.20
        }


'''
Initialize the continuous rate model
'''
P_inh = args.P_inh # inhibitory neuron proportion
P_rec = args.P_rec # initial connectivity probability (i.e. sparsity degree)
print('P_rec set to ' + str(P_rec))

w_dist = 'gaus' # recurrent weight distribution (Gaussian or Gamma)
net = FR_RNN_dale(N1, N2, N3, P_inh, P_rec, w_in, som_N, w_dist, args.gain, args.apply_dale, w_out, settings, training_params)
print('Intialized the network...')


'''
DEFINE LOSS AND OPTIMIZER
'''
# define optimizer TF2 - happens in training script
optimizer = tf.keras.optimizers.Adam(learning_rate=training_params['learning_rate'])

# define loss TF2 
def loss_op(o, z, training_params):
    """
    Compute loss between output `o` and target `z`.

    Args:
        o (tf.Tensor or list of tf.Tensor): model outputs (e.g., [T,] or list of [batch, 1])
        z (tf.Tensor): target values (same shape as `o`)
        training_params (dict): must include 'loss_fn'

    Returns:
        loss: scalar loss value
    """
    loss_fn = training_params['loss_fn'].lower()

    if isinstance(o, list):
        o = tf.stack(o, axis=0)

    if loss_fn == 'l1':
        loss = tf.reduce_sum(tf.abs(o - z))
    elif loss_fn == 'l2':
        loss = tf.reduce_mean(tf.square(o - z)) # Maybe change to make like TF1.0 version
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    return loss


'''
Train the network
'''


if args.mode.lower() == 'train':
    print('Training started...')
    training_success = False

    for tr in range(args.n_trials):
        start_time = time.time()

        # Generate a task-specific input signal
        if args.task.lower() == 'rdk_70_30':
            u, label = generate_input_stim_rdk_70_30(settings)
            target = generate_target_continuous_rdk(settings, label)
        elif args.task.lower() == 'rdk_80_20':
            u, label = generate_input_stim_rdk_80_20(settings)
            target = generate_target_continuous_rdk(settings, label)
        elif args.task.lower() == 'rdk_equal':
            u, label = generate_input_stim_rdk_equal(settings)
            target = generate_target_continuous_rdk(settings, label)
        elif args.task.lower() == 'rdk_70_30_6afc':
            u, label = generate_input_stim_rdk_70_30_6AFC(settings)
            target = generate_target_continuous_rdk(settings, label)
        elif args.task.lower() == 'rdk_80_20_6afc':
            u, label = generate_input_stim_rdk_80_20_6AFC(settings)
            target = generate_target_continuous_rdk(settings, label)
        elif args.task.lower() == 'rdk_equal_6afc':
            u, label = generate_input_stim_rdk_equal_6AFC(settings)
            target = generate_target_continuous_rdk(settings, label)

            print("Trial " + str(tr) + ': ' + str(label))
            
        # For storing all the loss vals
        losses = np.zeros((args.n_trials,))

        # convert stim and target to tensor
        stim_tensor = tf.convert_to_tensor(u, dtype=tf.float32)
        target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)

        # Training step
        with tf.GradientTape() as tape:   
            output = net(stim_tensor)
            t_loss = loss_op(output, target_tensor, training_params)
        grads = tape.gradient(t_loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))

        print('Loss: ', t_loss.numpy())
        losses[tr] = t_loss.numpy()

        '''
        Evaluate the model and determine if the training termination criteria are met
        '''
        # Simple discrimination task (stim1 = 70%, stim2 = 30%)
        if args.task.lower() == 'rdk_70_30':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_70_30(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minimum number of trials required is set to 2500 to 
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # Simple discrimination task (stim1 = 80%, stim2 = 20%)
        if args.task.lower() == 'rdk_80_20':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_80_20(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minimum number of trials required is set to 2500 to 
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # Simple discrimination task (both stims are equally likely)
        if args.task.lower() == 'rdk_equal':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_equal(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minumum number of trials required is set to 2500 to
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # 6AFC-Simple discrimination task (stim1 = 70%)
        if args.task.lower() == 'rdk_70_30_6afc':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_70_30_6AFC(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minimum number of trials required is set to 2500 to
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # 6AFC-Simple discrimination task (stim1 = 80%, stim2 = 20%)
        if args.task.lower() == 'rdk_80_20_6afc':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_80_20_6AFC(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minimum number of trials required is set to 2500 to
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        # 6AFC-Simple discrimination task (both stims are equally likely)
        if args.task.lower() == 'rdk_equal_6afc':
            resp_onset = settings['stim_on'] + settings['stim_dur']
            if (tr-1)%training_params['eval_freq'] == 0:
                eval_perf = np.zeros((1, training_params['eval_tr']))
                eval_losses = np.zeros((1, training_params['eval_tr']))
                eval_os = np.zeros((training_params['eval_tr'], settings['T']-1))
                eval_labels = np.zeros((training_params['eval_tr'], ))
                for ii in range(eval_perf.shape[-1]):
                    eval_u, eval_label = generate_input_stim_rdk_equal_6AFC(settings)
                    eval_target = generate_target_continuous_rdk(settings, eval_label)
                    eval_stim_tensor = tf.convert_to_tensor(eval_u, dtype=tf.float32)
                    eval_target_tensor = tf.convert_to_tensor(eval_target[1:], dtype=tf.float32)
                    eval_target_tensor = tf.squeeze(eval_target_tensor)
                    eval_o = net(eval_stim_tensor)
                    eval_o = tf.squeeze(eval_o).numpy()
                    eval_l = loss_op(eval_o, eval_target_tensor, training_params)
                    eval_losses[0, ii] = eval_l
                    eval_os[ii, :] = eval_o
                    eval_labels[ii, ] = eval_label
                    if eval_label == 1:
                        if np.max(eval_o[resp_onset:]) > training_params['eval_amp_threh']:
                            eval_perf[0, ii] = 1
                    else:
                        # if np.min(eval_o[resp_onset:]) < -training_params['eval_amp_threh']:
                        if np.max(np.abs(eval_o[resp_onset:])) < 0.3:
                            eval_perf[0, ii] = 1

                eval_perf_mean = np.nanmean(eval_perf, 1)
                eval_loss_mean = np.nanmean(eval_losses, 1)
                print("Perf: %.2f, Loss: %.2f"%(eval_perf_mean.item(), eval_loss_mean.item()))

                # if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                if eval_loss_mean < training_params['loss_threshold'] and eval_perf_mean > 0.85 and tr > 5000:
                    # For this task, the minumum number of trials required is set to 2500 to
                    # ensure that the trained rate model is stable.
                    training_success = True
                    break

        elapsed_time = time.time() - start_time
        
    
    # Save the trained params in a .mat file
    var = {}
    var['x1_0'] = net.x1_0
    var['r1_0'] = net.r1_0
    var['w1_0'] = net.w1_0

    var['x2_0'] = net.x2_0
    var['r2_0'] = net.r2_0
    var['w2_0'] = net.w2_0

    var['x3_0'] = net.x3_0
    var['r3_0'] = net.r3_0
    var['w3_0'] = net.w3_0

    var['w21_0'] = net.w21_0
    var['w32_0'] = net.w32_0
    var['w12_0'] = net.w12_0
    var['w23_0'] = net.w23_0

    var['taus_gaus1_0'] = net.taus_gaus1_0.numpy()
    var['taus_gaus2_0'] = net.taus_gaus2_0.numpy()
    var['taus_gaus3_0'] = net.taus_gaus3_0.numpy()

    var['u'] = u
    var['o'] = output.numpy()

    var['w1'] = net.W1
    var['w2'] = net.W2
    var['w3'] = net.W3
    var['w21'] = net.W21
    var['w32'] = net.W32
    var['w12'] = net.W12
    var['w23'] = net.W23

    var['target'] = target
    var['w_out'] = net.w_out

    var['N1'] = net.N1
    var['N2'] = net.N2
    var['N3'] = net.N3
    var['exc1'] = net.exc1
    var['inh1'] = net.inh1
    var['exc2'] = net.exc2
    var['inh2'] = net.inh2
    var['exc3'] = net.exc3
    var['inh3'] = net.inh3

    var['w_in'] = net.w_in
    var['b_out'] = net.b_out
    var['som_N'] = net.som_N
    var['losses'] = losses
    var['taus'] = settings['taus']
    var['eval_perf_mean'] = eval_perf_mean.item()
    var['eval_loss_mean'] = eval_loss_mean.item()
    var['eval_os'] = eval_os
    var['eval_labels'] = eval_labels
    var['taus_gaus1'] = net.taus_gaus1.numpy()
    var['taus_gaus2'] = net.taus_gaus2.numpy()
    var['taus_gaus3'] = net.taus_gaus3.numpy()
    var['tr'] = tr
    var['activation'] = training_params['activation']
    fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    if len(settings['taus']) > 1:
        fname = 'Task_{}_N1_{}_N2_{}_N3_{}_Taus_{}_{}_Act_{}_{}.mat'.format(args.task.lower(),\
                N1, N2, N3, settings['taus'][0], 
                settings['taus'][1], training_params['activation'], fname_time)
    elif len(settings['taus']) == 1:
        fname = 'Task_{}_N1_{}_N2_{}_N3_{}_Tau_{}_Act_{}_{}.mat'.format(args.task.lower(),\
                N1, N2, N3, settings['taus'][0], 
                training_params['activation'], fname_time)
    full_filepath = os.path.join(out_dir, fname) #Create the full filepath
    scipy.io.savemat(full_filepath, var)
    print(f"Saving file: {full_filepath}") # Print the full filepath that is being saved.
    print(f"elapsed time: {elapsed_time}")
