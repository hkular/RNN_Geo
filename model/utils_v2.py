#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Contains several general-purpose utility functions
# Edited  4/18/25 set_gpu for TF2.0 compatiblity

import os
import tensorflow as tf
import argparse

def set_gpu(gpu: str = '0', memory_fraction: float = 0.3):
    """
    Sets the visible GPU and limits memory usage.

    Args:
        gpu (str): GPU ID to use (e.g., '0')
        memory_fraction (float): Fraction of GPU memory to allocate
    """
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            selected_gpu = gpus[int(gpu)]
            
            # Use memory growth if allocating all memory
            if memory_fraction >= 1.0:
                tf.config.experimental.set_memory_growth(selected_gpu, True)
                print(f"Using GPU {gpu} with memory growth enabled.")
            else:
                # Try to get total memory info (may not work on all systems)
                details = tf.config.experimental.get_device_details(selected_gpu)
                total_mem = details.get('memory_limit')  # May return None

                if total_mem is not None:
                    limit_mem = total_mem * memory_fraction
                else:
                    # Default to 16GB if memory can't be queried (adjust as needed)
                    limit_mem = 16 * 1024**3 * memory_fraction

                tf.config.experimental.set_virtual_device_configuration(
                    selected_gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit_mem / (1024**2))]  # in MB
                )
                print(f"Using GPU {gpu} with memory limited to {int(memory_fraction * 100)}%.")
        except RuntimeError as e:
            print(f"Failed to set GPU memory configuration: {e}")
    else:
        print("No GPU found or accessible.")



def restricted_float(x):
    """
    Helper function for restricting input arg to range from 0 to 1

    INPUT
        x: string representing a float number

    OUTPUT
        x or raises an argument type error
    """
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r no in range [0.0, 1.0]"%(x,))
    return x

def str2bool(v):
    """
    Helper function to parse boolean input args

    INPUT
        v: string representing true or false
    OUTPUT
        True or False or raises an argument type error
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




