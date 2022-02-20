#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:00:38 2022

@author: Nicola Braile - Marianna Del Corso
"""
import tensorflow as tf
import torch.nn as nn
import torch

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
class EncoderLayer(nn.Module):
    
    def __init__(self,in_ch, out_ch, kernel_sz, strd,eps,momentum,name):
        super(EncoderLayer, self).__init__()
        self.name = name
        self.encoder_layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_sz, strd),
            nn.ReLU(),
            nn.BatchNorm1d( out_ch , eps, momentum)
        )
    
    def forward(self, inputs):
        print( self.name)
        return self.encoder_layer(inputs)
    

class SimCLR_TS(nn.Module):
    
    def __init__(self,config):
        super(SimCLR_TS, self).__init__()
        
        #    LAYER 1
        wsz = config['window_size']
        kernel_sz = config['encoder_parameters']['kernel_size_layer1']
        strd = config['encoder_parameters']['stride_layer1']
        in_ch = config['encoder_parameters']['in_channels_layer1']
        out_ch = int(1 + ( wsz - kernel_sz) / strd)                
        eps = config['encoder_parameters']['eps']
        momentum = config['encoder_parameters']['momentum']
        wsz_out = int(( (wsz - 3) / strd) + 1)
        
        #   LAYER 2
        kernel_sz2 = config['encoder_parameters']['kernel_size_layer2']
        strd2 = config['encoder_parameters']['stride_layer2']
        out_ch2 = int(1 + ( wsz_out - kernel_sz2) / strd2)  
        wsz_out2 = int(((wsz_out - 3)/strd2) + 1)
        
        #   LAYER 3
        kernel_sz3 = config['encoder_parameters']['kernel_size_layer3']
        strd3 = config['encoder_parameters']['stride_layer3']
        out_ch3 = int(1 + ( wsz_out2 - kernel_sz3) / strd3)  
        wsz_out3 = int(((wsz_out2 - 3)/strd3) + 1)
        
        self.encoder = nn.Sequential(
           EncoderLayer(in_ch, out_ch, kernel_sz, strd,eps,momentum,'enc1'),
           EncoderLayer(out_ch, out_ch2, kernel_sz2, strd2,eps,momentum,'enc2'),
           EncoderLayer(out_ch2, out_ch3, kernel_sz3, strd3,eps,momentum,'enc3')
        )
        
        #   Classification layer
        ''' 
        This classifier is used to measure augmentation accuracy on TEP Dataset.
        We use it to understand if multiple augmentation improve or not.
        '''        
        self.augm_cls_linear = nn.Linear(out_ch3 * wsz_out3, 22) #for TEP there are 22 classes 
    
    def forward(self, inputs, penultimate=False, simclr=False, shift=False, joint=False):
        #print(inputs.type())
        #print("forward SimCLR_TS")

        features = self.encoder(inputs)
        
        # this must be called only when Pretrain is complete 
        if tep_linear:
            features = torch.flatten(features, start_dim=1)
            return self.augm_cls_linear(features)
        else:
            return features

