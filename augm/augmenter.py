#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:59:09 2022

@author: Nicola Braile

"""

import numpy as np
import torch
from augm.augment import *
from functools import partial

'''
    Dictionary of soft augmentation partial functions. This way we can set all the parameters
    except the input data.

    L2R  ->  Left to Right flip
    CR   ->  Crop Resize
    RN   ->  Random Noise

'''
soft_augments = {'L2R': partial(left2rightFlip, show_plot=False, feature=0),
                 'CR': partial(crop_resize, show_plot=False, feature=0),
                 'RN': partial(random_noise, scale=1, show_plot=False, feature=0)
                 }

'''
    Dictionary of hard augmentation partial functions. Parameters setting.

    BLK  ->  Blockout
    MW   ->  Magnitude Warping
    PC   ->  Permute Channels

'''
hard_augments = {'BLK': partial(blockout, duration=10, show_plot=False, feature=0),
                 'MW': partial(magnitude_warping, scale=3, n=2, show_plot=False, feature=0),
                 'PC': partial(permute_channels)
                 }

'''
    This function applies soft augmentations and optionally hard ones too.
    For soft augmentations it is possible to apply only one 
    or a series of augmentations as specified in soft_augm_list (with a certain probability ) .
'''

def augmenter(datas, is_hard_augm=False, hard_augm='', is_multiple_augm=False, soft_augm_list=['L2R', 'CR', 'RN'], soft_augm=''):
   
    def _random_apply_augment(transformation, data, p=0.5):
        prob = np.random.uniform(low=0.0, high=1.0, size=None)
        return transformation(data) if prob < p else data

    augm_data = np.array(datas)

    # Apply hard augmentation
    if is_hard_augm:
        augm_data = hard_augments[hard_augm](datas)

    # Apply soft augmentations
    if is_multiple_augm:
        # for all augmentations
        for aug_pos in range(len(soft_augm_list)):
            # apply each one in a random way
            augm_data = _random_apply_augment(
                soft_augments[soft_augm_list[aug_pos]], augm_data)
    else:
        # apply a single augmentation in particular the one stated in soft_augm
        augm_data = soft_augments[soft_augm](augm_data)

    return augm_data

def augment_batch(batchdata,config):
    '''    
    Parameters
    ----------
    batchdata : Batch of TS .
    config : Contains the current setting for the augmentations to be applied. These 
    info can be updated in config/config_augmenter.json
    
    Returns
    -------
    For each window of the batch applies the proper augmentation.
    Returns the augmented batch.
    
    Note
    -------
    This function must be called after the batch has been doubled. 
    '''
    
    aug_datas = []
    # augment all batch
    for window in batchdata:
        z = augmenter(datas=window, 
                      is_hard_augm=config['is_hard_augm'], 
                      hard_augm=config['hard_augm'],        # The chosen hard augm
                      is_multiple_augm=config['is_multiple_augm'], 
                      soft_augm_list=config['soft_augm_list'],  # The chosen permutation of soft augm
                      soft_augm=config['soft_augm'])        # The chosen soft augm
        
        aug_datas.append(z.transpose())

    return torch.tensor(np.array(aug_datas), dtype=torch.float32)
