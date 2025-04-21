#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Implementation of the continuous rate RNN model
# Note: This version uses 2 stimuli for the RDK tasks (one expected and one unexpected) 
# NOTE: Edited on 12/26/2021 for 3-layer RNN model
#     : Edited on 01/19/2022 to add 1) a new version of RDK with 70/30 exp/unexp probability
#       (RDK_opposite for this hasn't been set up yet (the current one is of the 80/20 version); and
#       2) 3-feature Mante task
# Edited on 4/14/25 using tf_upgrade_v2 to allow for tf 2.0 install to run
# Edited on 4/17/25 complete overhaul to switch to TF2.0 compatibility

import os, sys
import numpy as np
import tensorflow as tf
import scipy.io

'''
CONTINUOUS FIRING-RATE RNN CLASS
'''

class FR_RNN_dale(tf.keras.Model): # updated TF1 to TF2, now its a TF2 model
    """
    Firing-rate RNN model for excitatory and inhibitory neurons
    Initialization of the firing-rate model with recurrent connections
    """
    def __init__(self, N1, N2, N3, P_inh, P_rec, w_in, som_N, w_dist, gain, apply_dale, w_out, settings, training_params):
        """
        Network initialization method
        N: number of units (neurons)
        P_inh: probability of a neuron being inhibitory
        P_rec: recurrent connection probability
        w_in: NxN weight matrix for the input stimuli
        som_N: number of SOM neurons (set to 0 for no SOM neurons)
        w_dist: recurrent weight distribution ('gaus' or 'gamma')
        apply_dale: apply Dale's principle ('True' or 'False')
        w_out: Nx1 readout weights

        Based on the probability (P_inh) provided above,
        the units in the network are classified into
        either excitatory or inhibitory. Next, the
        weight matrix is initialized based on the connectivity
        probability (P_rec) provided above.
        """
        super().__init__()
        self.settings = settings
        self.training_params = training_params

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3

        self.P_inh = P_inh
        self.P_rec = P_rec
        self.som_N = som_N
        self.w_dist = w_dist
        self.gain = gain
        self.apply_dale = apply_dale

        # Assign each unit as excitatory or inhibitory
        inh1, exc1, NI1, NE1, inh2, exc2, NI2, NE2, inh3, exc3, NI3, NE3, som_inh = self.assign_exc_inh()

        self.inh1 = inh1
        self.exc1 = exc1
        self.NI1 = NI1
        self.NE1 = NE1

        self.inh2 = inh2
        self.exc2 = exc2
        self.NI2 = NI2
        self.NE2 = NE2

        self.inh3 = inh3
        self.exc3 = exc3
        self.NI3 = NI3
        self.NE3 = NE3

        self.som_inh = som_inh

        # not sure if this is right
        self.b_out = self.add_weight(
            name='b_out',
            shape=(),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        self.w_out = self.add_weight(
            name='w_out',
            shape=w_out.shape,
            initializer=tf.constant_initializer(w_out),
            trainable=True,
            dtype=tf.float32
        )

        self.w_in = self.add_weight(
            name='w_in',
            shape=w_in.shape,
            initializer=tf.constant_initializer(w_in),
            trainable=False,  # or True, depending on your setup
            dtype=tf.float32
        )

        self.taus_gaus1 = self.add_weight(
            name="taus_gaus1",
            shape=(self.N1, 1),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )
        self.taus_gaus2 = self.add_weight(
            name="taus_gaus2",
            shape=(self.N2, 1),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )
        self.taus_gaus3 = self.add_weight(
            name="taus_gaus3",
            shape=(self.N3, 1),
            initializer=tf.random_normal_initializer(),
            trainable=True
        )
        # Initialize the weight matrix
        self.W1, self.mask1, self.W2, self.mask2, self.W3, self.mask3, \
                self.W21, self.w21_mask, self.W32, self.w32_mask, \
                self.W12, self.w12_mask, self.W23, self.w23_mask, self.som_mask = self.initialize_W()
        
        # # Recast masks as non-trainable weights
        # self.mask1 = self.add_weight("mask1", shape=self.mask1.shape,
        #                             initializer=tf.constant_initializer(self.mask1),
        #                             trainable=False, dtype=tf.float32)

        # self.mask2 = self.add_weight("mask2", shape=self.mask2.shape,
        #                             initializer=tf.constant_initializer(self.mask2),
        #                             trainable=False, dtype=tf.float32)

        # self.mask3 = self.add_weight("mask3", shape=self.mask3.shape,
        #                             initializer=tf.constant_initializer(self.mask3),
        #                             trainable=False, dtype=tf.float32)

        # self.w21_mask = self.add_weight("w21_mask", shape=self.w21_mask.shape,
        #                                 initializer=tf.constant_initializer(self.w21_mask),
        #                                 trainable=False, dtype=tf.float32)

        # self.w12_mask = self.add_weight("w12_mask", shape=self.w12_mask.shape,
        #                                 initializer=tf.constant_initializer(self.w12_mask),
        #                                 trainable=False, dtype=tf.float32)

        # self.w32_mask = self.add_weight("w32_mask", shape=self.w32_mask.shape,
        #                                 initializer=tf.constant_initializer(self.w32_mask),
        #                                 trainable=False, dtype=tf.float32)

        # self.w23_mask = self.add_weight("w23_mask", shape=self.w23_mask.shape,
        #                                 initializer=tf.constant_initializer(self.w23_mask),
        #                                 trainable=False, dtype=tf.float32)

        # self.som_mask = self.add_weight("som_mask", shape=self.som_mask.shape,
        #                                 initializer=tf.constant_initializer(self.som_mask),
        #                                 trainable=False, dtype=tf.float32)



    def assign_exc_inh(self):
        """
        Method to randomly assign units as excitatory or inhibitory (Dale's principle)

        Returns
            inh: bool array marking which units are inhibitory
            exc: bool array marking which units are excitatory
            NI: number of inhibitory units
            NE: number of excitatory units
            som_inh: indices of "inh" for SOM neurons
        """
        # Apply Dale's principle
        if self.apply_dale == True:
            # First layer
            inh1 = np.random.rand(self.N1, 1) < self.P_inh
            exc1 = ~inh1
            NI1 = len(np.where(inh1 == True)[0])
            NE1 = self.N1 - NI1

            # Second layer
            inh2 = np.random.rand(self.N2, 1) < self.P_inh
            exc2 = ~inh2
            NI2 = len(np.where(inh2 == True)[0])
            NE2 = self.N2 - NI2

            # Third layer
            inh3 = np.random.rand(self.N3, 1) < self.P_inh
            exc3 = ~inh3
            NI3 = len(np.where(inh3 == True)[0])
            NE3 = self.N3 - NI3

        # Do NOT apply Dale's principle
        else:
            inh1 = np.random.rand(self.N1, 1) < 0 # no separate inhibitory units
            exc1 = ~inh1
            NI1 = len(np.where(inh1 == True)[0])
            NE1 = self.N1 - NI1

            inh2 = np.random.rand(self.N2, 1) < 0 # no separate inhibitory units
            exc2 = ~inh2
            NI2 = len(np.where(inh2 == True)[0])
            NE2 = self.N2 - NI2

            inh3 = np.random.rand(self.N3, 1) < 0 # no separate inhibitory units
            exc3 = ~inh3
            NI3 = len(np.where(inh3 == True)[0])
            NE3 = self.N3 - NI2

        if self.som_N > 0:
            som_inh = np.where(inh==True)[0][:self.som_N]
        else:
            som_inh = 0

        return inh1, exc1, NI1, NE1, inh2, exc2, NI2, NE2, inh3, exc3, NI3, NE3, som_inh

    def initialize_W(self):
        """
        Method to generate and initialize the connectivity weight matrix, W
        The weights are drawn from either gaussian or gamma distribution.

        Returns
            w: NxN weights (all positive)
            mask: NxN matrix of 1's (excitatory units)
                  and -1's (for inhibitory units)
        NOTE: To compute the "full" weight matrix, simply
        multiply w and mask (i.e. w*mask)
        """
        # FIRST LAYER
        # Weight matrix
        w1 = np.zeros((self.N1, self.N1), dtype = np.float32)
        idx1 = np.where(np.random.rand(self.N1, self.N1) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w1[idx1[0], idx1[1]] = np.random.gamma(2, 0.003, len(idx1[0]))
        elif self.w_dist.lower() == 'gaus':
            w1[idx1[0], idx1[1]] = np.random.normal(0, 1.0, len(idx1[0]))
            w1 = w1/np.sqrt(self.N1*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # SECOND LAYER
        # Weight matrix
        w2 = np.zeros((self.N2, self.N2), dtype = np.float32)
        idx2 = np.where(np.random.rand(self.N2, self.N2) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w2[idx2[0], idx2[1]] = np.random.gamma(2, 0.003, len(idx2[0]))
        elif self.w_dist.lower() == 'gaus':
            w2[idx2[0], idx2[1]] = np.random.normal(0, 1.0, len(idx2[0]))
            w2 = w2/np.sqrt(self.N2*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # THIRD LAYER
        # Weight matrix
        w3 = np.zeros((self.N3, self.N3), dtype = np.float32)
        idx3 = np.where(np.random.rand(self.N3, self.N3) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w3[idx3[0], idx3[1]] = np.random.gamma(2, 0.003, len(idx3[0]))
        elif self.w_dist.lower() == 'gaus':
            w3[idx3[0], idx3[1]] = np.random.normal(0, 1.0, len(idx3[0]))
            w3 = w3/np.sqrt(self.N3*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # First -> Second layer weights
        w21 = np.zeros((self.N2, self.N1), dtype = np.float32)
        idx21 = np.where(np.random.rand(self.N2, self.N1) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w21[idx21[0], idx21[1]] = np.random.gamma(2, 0.003, len(idx21[0]))
        elif self.w_dist.lower() == 'gaus':
            w21[idx21[0], idx21[1]] = np.random.normal(0, 1.0, len(idx21[0]))
            # w21 = w21/np.sqrt(self.N1*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # Second -> First layer weights
        w12 = np.zeros((self.N1, self.N2), dtype = np.float32)
        idx12 = np.where(np.random.rand(self.N2, self.N1) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w12[idx12[0], idx12[1]] = np.random.gamma(2, 0.003, len(idx12[0]))
        elif self.w_dist.lower() == 'gaus':
            w12[idx12[0], idx12[1]] = np.random.normal(0, 1.0, len(idx12[0]))
            # w21 = w21/np.sqrt(self.N1*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # Second -> Third layer weights
        w32 = np.zeros((self.N3, self.N2), dtype = np.float32)
        idx32 = np.where(np.random.rand(self.N3, self.N2) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w32[idx32[0], idx32[1]] = np.random.gamma(2, 0.003, len(idx32[0]))
        elif self.w_dist.lower() == 'gaus':
            w32[idx32[0], idx32[1]] = np.random.normal(0, 1.0, len(idx32[0]))
            # w21 = w21/np.sqrt(self.N1*self.P_rec)*self.gain # scale by a gain to make it chaotic

        # Third -> Second layer weights
        w23 = np.zeros((self.N2, self.N3), dtype = np.float32)
        idx23 = np.where(np.random.rand(self.N3, self.N2) < self.P_rec)
        if self.w_dist.lower() == 'gamma':
            w23[idx23[0], idx23[1]] = np.random.gamma(2, 0.003, len(idx23[0]))
        elif self.w_dist.lower() == 'gaus':
            w23[idx23[0], idx23[1]] = np.random.normal(0, 1.0, len(idx23[0]))
            # w21 = w21/np.sqrt(self.N1*self.P_rec)*self.gain # scale by a gain to make it chaotic

        if self.apply_dale == True:
            w1 = np.abs(w1)
            w2 = np.abs(w2)

            w21 = np.abs(w21)
            w12 = np.abs(w12)

            w32 = np.abs(w32)
            w23 = np.abs(w23)
        
        # Mask matrix
        mask1 = np.eye(self.N1, dtype=np.float32)
        mask1[np.where(self.inh1==True)[0], np.where(self.inh1==True)[0]] = -1

        mask2 = np.eye(self.N2, dtype=np.float32)
        mask2[np.where(self.inh2==True)[0], np.where(self.inh2==True)[0]] = -1

        mask3 = np.eye(self.N3, dtype=np.float32)
        mask3[np.where(self.inh3==True)[0], np.where(self.inh3==True)[0]] = -1

        # Layer 1 -> Layer 2 mask (for removing inhibitory projections)
        w21_mask = np.ones((self.N2, self.N1), dtype=np.float32)
        w12_mask = np.ones((self.N1, self.N2), dtype=np.float32) # feedback

        w32_mask = np.ones((self.N3, self.N2), dtype=np.float32)
        w23_mask = np.ones((self.N2, self.N3), dtype=np.float32) # feedback

        # SOM mask matrix
        som_mask = np.ones((self.N1, self.N1), dtype=np.float32)
        if self.som_N > 0:
            Nsom = self.N1 - self.som_N
            som_mask[0:Nsom, Nsom:] = 0
            # for i in self.som_inh:
                # som_mask[i, np.where(self.inh==True)[0]] = 0

        return w1, mask1, w2, mask2, w3, mask3, w21, w21_mask, w32, w32_mask, w12, w12_mask, w23, w23_mask, som_mask

    def load_net(self, model_dir):
        """
        Method to load pre-configured network settings
        """
        settings = scipy.io.loadmat(model_dir)
        self.N1 = settings['N1'][0][0]
        self.som_N = settings['som_N'][0][0]
        self.inh1 = settings['inh1']
        self.exc1 = settings['exc1']
        self.inh1 = self.inh1 == 1
        self.exc1 = self.exc1 == 1
        self.NI1 = len(np.where(settings['inh1'] == True)[0])
        self.NE1 = len(np.where(settings['exc1'] == True)[0])
        self.mask = settings['m1']
        self.som_mask = settings['som_m']
        self.W = settings['w1']
        self.w_in = settings['w_in']
        self.b_out = settings['b_out']
        self.w_out = settings['w_out']

        return self
    
    def display(self):
        """
        Method to print the network setup
        """
        print('Network Settings for the FIRST LAYER')
        print('====================================')
        print('Number of Units: ', self.N1)
        print('\t Number of Excitatory Units: ', self.NE1)
        print('\t Number of Inhibitory Units: ', self.NI1)
        print('Weight Matrix, W')
        full_w = self.W1*self.mask1
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N1*self.N1)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N1*self.N1)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N1*self.N1)*100))

        print('====================================')
        print('Network Settings for the SECOND LAYER')
        print('====================================')
        print('Number of Units: ', self.N2)
        print('\t Number of Excitatory Units: ', self.NE2)
        print('\t Number of Inhibitory Units: ', self.NI2)
        print('Weight Matrix, W')
        full_w = self.W2*self.mask2
        zero_w = len(np.where(full_w == 0)[0])
        pos_w = len(np.where(full_w > 0)[0])
        neg_w = len(np.where(full_w < 0)[0])
        print('\t Zero Weights: %2.2f %%' % (zero_w/(self.N2*self.N2)*100))
        print('\t Positive Weights: %2.2f %%' % (pos_w/(self.N2*self.N2)*100))
        print('\t Negative Weights: %2.2f %%' % (neg_w/(self.N2*self.N2)*100))

    def call(self, stim):
        """
        Method to construct a TF graph and return nodes with
        Dale's principle
        INPUT
            self: firing-rate RNN class
            settings: dict containing the following keys
                T: duration of a single trial (in steps)
                stim_on: stimulus starting time (in steps)
                stim_dur: stimulus duration (in steps)
                delay: delay b/w two stimuli (in steps)
                taus: time-constants (in steps)
                DeltaT: sampling rate
            training_params: dictionary containing training parameters
                learning_rate: learning rate
        OUTPUT
            TF graph
        """

        # Task params
        T = self.settings['T']
        taus = self.settings['taus']
        DeltaT = self.settings['DeltaT']
        task = self.settings['task']

        # Training params
        learning_rate = self.training_params['learning_rate']

        # Excitatory units
        exc_idx_tf1 = tf.constant(np.where(self.exc1 == True)[0], name='exc_idx1')
        exc_idx_tf2 = tf.constant(np.where(self.exc2 == True)[0], name='exc_idx2')
        exc_idx_tf3 = tf.constant(np.where(self.exc3 == True)[0], name='exc_idx3')

        # Inhibitory units
        inh_idx_tf1 = tf.constant(np.where(self.inh1 == True)[0], name='inh_idx1')
        inh_idx_tf2 = tf.constant(np.where(self.inh2 == True)[0], name='inh_idx2')
        inh_idx_tf3 = tf.constant(np.where(self.inh3 == True)[0], name='inh_idx3')

        som_inh_idx_tf = tf.constant(self.som_inh, name='som_inh_idx')


        # Synaptic currents and firing-rates for FIRST LAYER
        x1 = [tf.random.normal([self.N1, 1]) / 100]
        r1 = [] # firing-rates

        # Synaptic currents and firing-rates for SECOND LAYER
        x2 = [tf.random.normal([self.N2, 1]) / 100]
        r2 = [] # firing-rates

        # Synaptic currents and firing-rates for THIRD LAYER
        x3 = [tf.random.normal([self.N3, 1]) / 100]
        r3 = [] # firing-rates

        # Transfer function options
        if self.training_params['activation'] == 'sigmoid':
            r1.append(tf.sigmoid(x1[0]))
            r2.append(tf.sigmoid(x2[0]))
            r3.append(tf.sigmoid(x3[0]))
            
        elif self.training_params['activation'] == 'clipped_relu': 
            r1.append(tf.clip_by_value(tf.nn.relu(x1[0]), 0, 20))
            r2.append(tf.clip_by_value(tf.nn.relu(x2[0]), 0, 20))
            r3.append(tf.clip_by_value(tf.nn.relu(x3[0]), 0, 20))

        elif self.training_params['activation'] == 'softplus':
            r1.append(tf.clip_by_value(tf.nn.softplus(x1[0]), 0, 20))
            r2.append(tf.clip_by_value(tf.nn.softplus(x2[0]), 0, 20))
            r3.append(tf.clip_by_value(tf.nn.softplus(x3[0]), 0, 20))
        else:
            raise ValueError(f"Unsupported activation: {self.training_params['activation']}")


        # Prepare weights and masks as tensors
        w1 = tf.nn.relu(self.W1) if self.apply_dale else self.W1
        w2 = tf.nn.relu(self.W2) if self.apply_dale else self.W2
        w3 = tf.nn.relu(self.W3) if self.apply_dale else self.W3

        w21 = tf.nn.relu(tf.convert_to_tensor(self.W21, dtype=tf.float32))
        w12 = tf.nn.relu(tf.convert_to_tensor(self.W12, dtype=tf.float32))
        w32 = tf.nn.relu(tf.convert_to_tensor(self.W32, dtype=tf.float32))
        w23 = tf.nn.relu(tf.convert_to_tensor(self.W23, dtype=tf.float32))

        m1 = tf.convert_to_tensor(self.mask1, dtype=tf.float32)
        m2 = tf.convert_to_tensor(self.mask2, dtype=tf.float32)
        m3 = tf.convert_to_tensor(self.mask3, dtype=tf.float32)
        som_m = tf.convert_to_tensor(self.som_mask, dtype=tf.float32)
        w21_m = tf.convert_to_tensor(self.w21_mask, dtype=tf.float32)
        w32_m = tf.convert_to_tensor(self.w32_mask, dtype=tf.float32)
        w12_m = tf.convert_to_tensor(self.w12_mask, dtype=tf.float32)
        w23_m = tf.convert_to_tensor(self.w23_mask, dtype=tf.float32)

        # Pass the synaptic time constants thru the sigmoid function
        if len(taus) > 1:
            taus_sig1 = tf.sigmoid(self.taus_gaus1)*(taus[1] - taus[0]) + taus[0]
            taus_sig2 = tf.sigmoid(self.taus_gaus2)*(taus[1] - taus[0]) + taus[0]
            taus_sig3 = tf.sigmoid(self.taus_gaus3)*(taus[1] - taus[0]) + taus[0]
        elif len(taus) == 1: # one scalar synaptic decay time-constant
            taus_sig1 = taus[0]
            taus_sig2 = taus[0]
            taus_sig3 = taus[0]

        # Forward pass # !!! might need to change t and t-1 to -1
        o = [] # output (i.e. weighted linear sum of rates, r)
        for t in range(1, T):
            u_t = tf.expand_dims(stim[:, t-1], 1)  # [n_inputs, 1]

            # next_x is [N x 1]
            ww1 = tf.matmul(w1, m1)
            ww1 = tf.multiply(ww1, som_m)

            ww2 = tf.matmul(w2, m2)

            ww3 = tf.matmul(w3, m3)

            ww21 = tf.matmul(w21, m1)
            ww21 = tf.multiply(ww21, w21_m)

            ww12 = tf.matmul(w12, m2) # feedback
            ww12 = tf.multiply(ww12, w12_m) # feedback

            ww32 = tf.matmul(w32, m2)
            ww32 = tf.multiply(ww32, w32_m)

            ww23 = tf.matmul(w23, m3) # feedback
            ww23 = tf.multiply(ww23, w23_m) # feedback

            next_x1 = tf.multiply((1 - DeltaT/taus_sig1), x1[t-1]) + \
                    tf.multiply((DeltaT/taus_sig1), ((tf.matmul(ww1, r1[t-1]))\
                    + tf.matmul(ww12, r2[t-1]) + tf.matmul(self.w_in, tf.expand_dims(stim[:, t-1], 1)))) +\
                    tf.random.normal([self.N1, 1], dtype=tf.float32)/10
            x1.append(next_x1)

            next_x2 = tf.multiply((1 - DeltaT/taus_sig2), x2[t-1]) + \
                    tf.multiply((DeltaT/taus_sig2), ((tf.matmul(ww2, r2[t-1]))\
                    + tf.matmul(ww21, r1[t-1]) + tf.matmul(ww23, r3[t-1]))) +\
                    tf.random.normal([self.N2, 1], dtype=tf.float32)/10
            x2.append(next_x2)

            next_x3 = tf.multiply((1 - DeltaT/taus_sig3), x3[t-1]) + \
                    tf.multiply((DeltaT/taus_sig3), ((tf.matmul(ww3, r3[t-1]))\
                    + tf.matmul(ww32, r2[t-1]))) +\
                    tf.random.normal([self.N3, 1], dtype=tf.float32)/10
            x3.append(next_x3)

            if self.training_params['activation'] == 'sigmoid':
                r1.append(tf.sigmoid(next_x1))
                r2.append(tf.sigmoid(next_x2))
                r3.append(tf.sigmoid(next_x3))
            elif self.training_params['activation'] == 'clipped_relu': 
                r1.append(tf.clip_by_value(tf.nn.relu(next_x1), 0, 20))
                r2.append(tf.clip_by_value(tf.nn.relu(next_x2), 0, 20))
                r3.append(tf.clip_by_value(tf.nn.relu(next_x3), 0, 20))
            elif self.training_params['activation'] == 'softplus':
                r1.append(tf.clip_by_value(tf.nn.softplus(next_x1), 0, 20))
                r2.append(tf.clip_by_value(tf.nn.softplus(next_x2), 0, 20))
                r3.append(tf.clip_by_value(tf.nn.softplus(next_x3), 0, 20))

            next_o = tf.matmul(self.w_out, r3[t]) + self.b_out

            o.append(next_o)
        # save out initial values explicitly
        if not hasattr(self, 'x1_0'):
            self.x1_0 = x1[0]
            self.r1_0 = r1[0]
            self.w1_0 = w1[0]

            self.x2_0 = x2[0]
            self.r2_0 = r2[0]
            self.w2_0 = w2[0]

            self.x3_0 = x3[0]
            self.r3_0 = r3[0]
            self.w3_0 = w3[0]

            self.w21_0 = w21[0]
            self.w32_0 = w32[0]
            self.w12_0 = w12[0]
            self.w23_0 = w23[0]

            self.taus_gaus1_0 = self.taus_gaus1[0, :]
            self.taus_gaus2_0 = self.taus_gaus2[0, :]
            self.taus_gaus3_0 = self.taus_gaus3[0, :]

            self.w_in0 = self.w_in[0]


        return tf.stack(o, axis = 0) # [T-1,1]


'''
Task-specific input signals
'''
def generate_input_stim_rdk_70_30(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (stim1 is expected (70%); 
    and stim2 is unexpected (30%))

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-2 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 2
    stimAmp = 0.6 # coherence level: 0.6 = low coh; 0.7 = high coh

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    randnum = np.random.rand()

    if randnum > 0.30:
        #u[0, stim_on:stim_on+stim_dur] = 1 # expected direction
        u[0, stim_on:stim_on+stim_dur] = u[0, stim_on:stim_on+stim_dur] + stimAmp # expected direction
        label = 1
    else:
        #u[rand_choice + 1, stim_on:stim_on+stim_dur] = 1 # expected dirction
        u[1, stim_on:stim_on+stim_dur] = u[1, stim_on:stim_on+stim_dur] + stimAmp # unexpected direction
        label = 2

    return u, label

def generate_input_stim_rdk_70_30_6AFC(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (stim1 is expected (70%); 
    and stim2-6 are unexpected (6% each))

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-6 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 6
    stimAmp = 0.6 # ~coherence level: 0.6 = low coh; 0.7 = high coh

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    randnum = np.random.rand()

    if randnum > 0.30:
        #u[0, stim_on:stim_on+stim_dur] = 1 # expected direction
        u[0, stim_on:stim_on+stim_dur] = u[0, stim_on:stim_on+stim_dur] + stimAmp # expected direction
        label = 1
    else:
        rand_choice = np.random.choice(5)
        #u[rand_choice + 1, stim_on:stim_on+stim_dur] = 1 # expected dirction
        u[rand_choice + 1, stim_on:stim_on+stim_dur] = u[rand_choice + 1, stim_on:stim_on+stim_dur] + stimAmp # unexpected direction
        label = rand_choice + 2

    return u, label

def generate_input_stim_rdk_80_20(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (stim1 is expected (80%); 
    and stim2 is unexpected (20%))

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-2 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 2
    stimAmp = 0.6 # coherence level: 0.6 = low coh; 0.7 = high coh

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    randnum = np.random.rand()

    if randnum > 0.20:
        #u[0, stim_on:stim_on+stim_dur] = 1 # expected direction
        u[0, stim_on:stim_on+stim_dur] = u[0, stim_on:stim_on+stim_dur] + stimAmp # expected direction
        label = 1
    else:
        #u[rand_choice + 1, stim_on:stim_on+stim_dur] = 1 # expected dirction
        u[1, stim_on:stim_on+stim_dur] = u[1, stim_on:stim_on+stim_dur] + stimAmp # unexpected direction
        label = 2

    return u, label

def generate_input_stim_rdk_80_20_6AFC(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (stim1 is expected (80%); 
    and stim2-6 are unexpected (4% each))

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-6 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 6
    stimAmp = 0.6 # ~coherence level: 0.6 = low coh; 0.7 = high coh

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    randnum = np.random.rand()

    if randnum > 0.20:
        #u[0, stim_on:stim_on+stim_dur] = 1 # expected direction
        u[0, stim_on:stim_on+stim_dur] = u[0, stim_on:stim_on+stim_dur] + stimAmp # expected direction
        label = 1
    else:
        rand_choice = np.random.choice(5)
        #u[rand_choice + 1, stim_on:stim_on+stim_dur] = 1 # expected dirction
        u[rand_choice + 1, stim_on:stim_on+stim_dur] = u[rand_choice + 1, stim_on:stim_on+stim_dur] + stimAmp # unexpected direction
        label = rand_choice + 2

    return u, label

def generate_input_stim_rdk_equal(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (both stims are equally likely)

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-2 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 2
    stimAmp = 0.6

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    rand_choice = np.random.choice(num_stims)
    # u[rand_choice, stim_on:stim_on+stim_dur] = 1
    u[rand_choice, stim_on:stim_on+stim_dur] = u[rand_choice, stim_on:stim_on+stim_dur] + stimAmp

    label = rand_choice + 1

    return u, label

def generate_input_stim_rdk_equal_6AFC(settings):
    """
    Method to generate the input stimulus matrix for the
    simple discrimination task (all stims are equally likely)

    INPUT
    settings: dict containing the following keys
    T: duration of a single trial (in steps)
    stim_on: stimulus starting time (in steps)
    stim_dur: stimulus duration (in steps)
    taus: time-constants (in steps)
    DeltaT: sampling rate
    OUTPUT
    u: 1xT stimulus matrix
    label: 1-6 (indicating what direction is the presented stim)
    """

    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    num_stims = 6
    stimAmp = 0.6

    u = np.zeros((num_stims, T)) #+ np.random.randn(1, T)
    u = u + np.random.randn(np.shape(u)[0], np.shape(u)[1])

    rand_choice = np.random.choice(num_stims)
    # u[rand_choice, stim_on:stim_on+stim_dur] = 1
    u[rand_choice, stim_on:stim_on+stim_dur] = u[rand_choice, stim_on:stim_on+stim_dur] + stimAmp

    label = rand_choice + 1

    return u, label


    """
    Method to generate the input stimulus matrix for the
    mante task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
    OUTPUT
        u: 6xT stimulus matrix (first 3 rows for color/motion/contrast and the second
        3 rows for context
        label: either +1 or -1
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    # Color/motion/contrast sensory inputs
    u = np.zeros((3, T))
    u_lab = np.zeros((3, 1))
    if np.random.rand() <= 0.50:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[0, 0] = 1
    else:
        u[0, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[0, 0] = -1

    if np.random.rand() <= 0.50:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[1, 0] = 1
    else:
        u[1, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[1, 0] = -1

    if np.random.rand() <= 0.50:
        u[2, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) + 0.5
        u_lab[2, 0] = 1
    else:
        u[2, stim_on:stim_on+stim_dur] = np.random.randn(1, stim_dur) - 0.5
        u_lab[2, 0] = -1

    # Context input
    c = np.zeros((3, T))
    label = 0
    rr = np.random.rand()
    if rr <= 0.33:
        c[0, :] = 1

        if u_lab[0, 0] == 1:
            label = 1
        elif u_lab[0, 0] == -1:
            label = -1
    elif rr > 0.33 and rr <= 0.67:
        c[1, :] = 1

        if u_lab[1, 0] == 1:
            label = 1
        elif u_lab[1, 0] == -1:
            label = -1
    else:
        c[2, :] = 1

        if u_lab[2, 0] == 1:
            label = 1
        elif u_lab[2, 0] == -1:
            label = -1

    return np.vstack((u, c)), label

'''
Task-specific target signals
'''

def generate_target_continuous_rdk(settings, label):
    """
    Method to generate a continuous target signal (z) 
    for the RDK task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    else:
        # z[0, stim_on+stim_dur:] = -1
        z[0, stim_on+stim_dur:] = 0 

    return np.squeeze(z)


    """
    Method to generate a continuous target signal (z) 
    for the MANTE task

    INPUT
        settings: dict containing the following keys
            T: duration of a single trial (in steps)
            stim_on: stimulus starting time (in steps)
            stim_dur: stimulus duration (in steps)
            taus: time-constants (in steps)
            DeltaT: sampling rate
        label: either +1 or -1
    OUTPUT
        z: 1xT target signal
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    z = np.zeros((1, T))
    if label == 1:
        z[0, stim_on+stim_dur:] = 1
    else:
        z[0, stim_on+stim_dur:] = -1

    return np.squeeze(z)


'''
EVALUATE THE TRAINED MODEL
'''
# maybe change this for tf2 
def eval_tf(model_dir, settings, u):
    """
    Method to evaluate a trained TF graph
    INPUT
        model_dir: full path to the saved model .mat file
        stim_params: dictionary containig the following keys
        u: 12xT stimulus matrix
            NOTE: There are 12 rows (one per dot pattern): 6 cues and 6 probes.
    OUTPUT
        o: 1xT output vector
    """
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']
    DeltaT = settings['DeltaT']

    # Load the trained mat file
    var = scipy.io.loadmat(model_dir)

    # Get some additional params
    N1 = var['N1'][0][0]
    N2 = var['N2'][0][0]
    N3 = var['N3'][0][0]
    exc_ind1 = [np.bool(i) for i in var['exc1']]
    exc_ind2 = [np.bool(i) for i in var['exc2']]
    exc_ind3 = [np.bool(i) for i in var['exc3']]

    # Get the delays
    taus_gaus1 = var['taus_gaus1']
    taus_gaus2 = var['taus_gaus2']
    taus_gaus3 = var['taus_gaus3']
    taus = var['taus'][0] # tau [min, max]
    taus_sig1 = (1/(1+np.exp(-taus_gaus1))*(taus[1] - taus[0])) + taus[0]
    taus_sig2 = (1/(1+np.exp(-taus_gaus2))*(taus[1] - taus[0])) + taus[0]
    taus_sig3 = (1/(1+np.exp(-taus_gaus3))*(taus[1] - taus[0])) + taus[0]

    # Synaptic currents and firing-rates
    x1 = np.zeros((N1, T)) # synaptic currents
    x2 = np.zeros((N2, T))
    x3 = np.zeros((N3, T))
    r1 = np.zeros((N1, T)) # firing-rates
    r2 = np.zeros((N2, T))
    r3 = np.zeros((N3, T))
    x1[:, 0] = np.random.randn(N1, )/100
    x2[:, 0] = np.random.randn(N2, )/100
    x3[:, 0] = np.random.randn(N3, )/100
    r1[:, 0] = 1/(1 + np.exp(-x1[:, 0]))
    r1[:, 0] = 1/(1 + np.exp(-x2[:, 0]))
    r1[:, 0] = 1/(1 + np.exp(-x3[:, 0]))

    # Output
    o = np.zeros((T, ))
    o_counter = 0

    # Recurrent weights and masks
    # w = var['w0'] #!!!!!!!!!!!!
    w1 = var['w1']
    w2 = var['w2']
    w3 = var['w3']
    w21 = var['w21']
    w32 = var['w32']
    w12 = var['w12']
    w23 = var['w23']
    w21_m = var['w21_m']
    w32_m = var['w32_m']
    #w12_m = var['w12_m']
    #w23_m = var['w23_m']

    m1 = var['m1']
    m2 = var['m2']
    m3 = var['m3']
    som_m = var['som_m']
    som_N = var['som_N'][0][0]

    # Identify excitatory/inhibitory neurons
    exc1 = var['exc1']
    exc_ind1 = np.where(exc1 == 1)[0]
    exc2 = var['exc2']
    exc_ind2 = np.where(exc2 == 1)[0]
    exc3 = var['exc3']
    exc_ind3 = np.where(exc3 == 1)[0]
    inh1 = var['inh1']
    inh_ind1 = np.where(inh1 == 1)[0]
    inh2 = var['inh2']
    inh_ind2 = np.where(inh2 == 1)[0]
    inh3 = var['inh3']
    inh_ind3 = np.where(inh3 == 1)[0]

    for t in range(1, T):
        # next_x is [N x 1]
        ww1 = np.matmul(w1, m1)
        ww1 = np.multiply(ww1, som_m)

        ww2 = np.matmul(w2, m2)

        ww3 = np.matmul(w3, m3)

        ww21 = np.matmul(w21, m1)
        ww21 = np.multiply(ww21, w21_m)

        ww32 = np.matmul(w32, m2)
        ww32 = np.multiply(ww32, w32_m)

        ww12 = np.matmul(w12, m2)

        ww23 = np.matmul(w23, m3)


        next_x1 = np.multiply((1 - DeltaT/taus_sig1), np.expand_dims(x1[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig1), ((np.matmul(ww1, np.expand_dims(r1[:, t-1], 1)))\
                + np.matmul(ww12, np.expand_dims(r2[:, t-1], 1)) + \
                np.matmul(var['w_in'], np.expand_dims(u[:, t-1], 1)))) +\
                np.random.randn(N1, 1)/10

        x1[:, t] = np.squeeze(next_x1)
        r1[:, t] = 1/(1 + np.exp(-x1[:, t]))

        next_x2 = np.multiply((1 - DeltaT/taus_sig2), np.expand_dims(x2[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig2), ((np.matmul(ww2, np.expand_dims(r2[:, t-1], 1)))\
                + np.matmul(ww21, np.expand_dims(r1[:, t-1], 1)) +\
                np.matmul(ww23, np.expand_dims(r3[:, t-1], 1)))) +\
                np.random.randn(N2, 1)/10

        x2[:, t] = np.squeeze(next_x2)
        r2[:, t] = 1/(1 + np.exp(-x2[:, t]))

        next_x3 = np.multiply((1 - DeltaT/taus_sig3), np.expand_dims(x3[:, t-1], 1)) + \
                np.multiply((DeltaT/taus_sig3), ((np.matmul(ww3, np.expand_dims(r3[:, t-1], 1)))\
                + np.matmul(ww32, np.expand_dims(r2[:, t-1], 1)))) +\
                np.random.randn(N3, 1)/10

        x3[:, t] = np.squeeze(next_x3)
        r3[:, t] = 1/(1 + np.exp(-x3[:, t]))

        wout = var['w_out']

        r_exc1 = r1[exc_ind1, :]
        r_inh1 = r1[inh_ind1, :]
        r_exc2 = r2[exc_ind2, :]
        r_inh2 = r2[inh_ind2, :]
        r_exc3 = r3[exc_ind3, :]
        r_inh3 = r3[inh_ind3, :]

        o[o_counter] = np.matmul(wout, r3[:, t]) + var['b_out']
        # o[o_counter] = np.matmul(wout_exc, r[exc_ind, t]) + var['b_out'] # excitatory output
        # o[o_counter] = np.matmul(wout_inh, r[inh_ind, t]) + var['b_out'] # inhibitory output
        o_counter += 1
    return x1, x2, x3, r1, r2, r3, o

