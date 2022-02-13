#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:00:38 2022

@author: Nicola Braile - Marianna Del Corso
"""

import torch.nn as nn

'''
    Feature Extractor ( Encoder )
        3x Encoder layer
        {
            Conv1D
            LeakyRELU
            BatchNorm    
        }
    
'''
# This is 
class Encoder(nn.Module):
    
    def __init__(self,config):
        super(Encoder, self).__init__()
        
        bsz = config['batch_size']
        wsz = config['window_size']
        kernel_sz = config['encoder_parameters']['kernel_size_layer1']
        strd = config['encoder_parameters']['stride_layer1']
        in_ch = config['encoder_parameters']['in_channels_layer1']
        out_ch = int(1 + ( wsz - kernel_sz) / strd)
        print(out_ch)
                
        eps = config['encoder_parameters']['eps']
        momentum = config['encoder_parameters']['momentum']
        
        self.encoder_layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_sz, strd),
            nn.ReLU(),
            nn.BatchNorm1d( out_ch , eps, momentum)
        )

    def forward(self, inputs):
        print("\n input fwd ", inputs)
    

class SimCLR_TS(nn.Module):
    '''
     def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder_layer = nn.Sequential(
            nn.Conv1d( inch , outch, kernel_size, stride= strd),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )

    
    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer(features)

        if _return_aux:
            return output, _aux

        return output
    '''

