#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 12:59:09 2022

@author: Nicola Braile 

"""

import numpy as np
import random
import itertools

from augm.augment import * 
from functools import partial

'''
    Dictionary of soft augmentation partial functions. This way we can set all the parameters 
    except the input data.
    
    L2R  ->  Left to Right flip
    CR   ->  Crop Resize
    RN   ->  Random Noise
    
'''
soft_augments = { 'L2R' : partial(left2rightFlip,show_plot=True,feature=0),
                  'CR'  : partial(crop_resize,show_plot=True,feature=0),
                  'RN'  : partial(random_noise,scale=1,show_plot=True,feature=0)
                }

'''
    Dictionary of hard augmentation partial functions. Parameters setting.
    
    BLK  ->  Blockout
    MW   ->  Magnitude Warping
    PC   ->  Permute Channels
    
'''
hard_augments = { 'BLK' : partial(blockout,duration=10,show_plot=False,feature=0),
                  'MW'  : partial(magnitude_warping,scale=3,n=2,show_plot=False,feature=0),
                  'PC'  : partial(permute_channels)
                }

'''
    This function applies soft augmentations and optionally hard ones too.
    For soft augmentations it is possible to apply only one soft augmentation 
    or a series of augmentations as specified in soft_order. 
    This way we can reprocude both papers.
 
'''
def augmenter(datas, is_hard_augm=False, hard_augm='', is_multiple_augm=False, soft_order=['L2R','CR','RN'],single_augm=''):
    #orders = list(itertools.permutations(['L2R', 'CR', 'RN'])) //this liine is for final traing and must be moved from here
    
    def _random_apply_augment(transformation, data ,augm,p=0.5):
        prob = np.random.uniform(low=0.0, high=1.0, size=None)
        print('Applying ',augm,' augmentation : ', prob < p)
        return transformation(data) if prob < p else data
    
    augm_data = np.array(datas)
    
    # Apply hard augmentation
    if is_hard_augm:
        augm_data = hard_augments[hard_augm](datas)
    
    # Apply soft augmentations
    if is_multiple_augm:
        #for all augmentations
        for aug_pos in range( len(soft_order)):
            #apply each one in a random way  
            augm_data = _random_apply_augment(soft_augments[soft_order[aug_pos]],augm_data,soft_order[aug_pos])
    else:
        #apply a single augmentation in particular the one stated in single_augm
        augm_data = soft_augments[single_augm](augm_data)
    
    return augm_data




