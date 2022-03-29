#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:00:38 2022

@author: Nicola Braile - Marianna Del Corso
"""
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

def initialize_weights(m):
  if isinstance(m, nn.Conv1d):
      #print('conv1d')
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
      #print('batchnorm')
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      #print('linear')
      nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
      nn.init.constant_(m.bias.data, 0)
  #else:
      #print(f"else {m}")

class EncoderLayer(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_sz, strd, eps, momentum, name):
        super(EncoderLayer, self).__init__()
        self.name = name
        self.encoder_layer = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_sz, strd),
            nn.ReLU(),
            nn.BatchNorm1d(out_ch, eps, momentum)
        )
        self.encoder_layer.apply(initialize_weights)

    def forward(self, inputs):
        #print(self.name)
        x = self.encoder_layer(inputs)
        # print(x.size() )
        return x


class SimCLR_TS(nn.Module):

    def __init__(self, config):
        super(SimCLR_TS, self).__init__()

        #    LAYER 1
        wsz = config['window_size']
        kernel_sz = config['encoder_parameters']['kernel_size_layer1']
        strd = config['encoder_parameters']['stride_layer1']
        in_ch = config['encoder_parameters']['in_channels_layer1']

        eps = config['encoder_parameters']['eps']
        momentum = config['encoder_parameters']['momentum']

        #   LAYER 2
        kernel_sz2 = config['encoder_parameters']['kernel_size_layer2']
        strd2 = config['encoder_parameters']['stride_layer2']

        #   LAYER 3
        kernel_sz3 = config['encoder_parameters']['kernel_size_layer3']
        strd3 = config['encoder_parameters']['stride_layer3']

        self.encoder = nn.Sequential(
            EncoderLayer(in_ch, 64, kernel_sz, strd, eps, momentum, 'enc1'),
            EncoderLayer(64, 128, kernel_sz2, strd2, eps, momentum, 'enc2'),
            EncoderLayer(128, 256, kernel_sz3, strd3, eps, momentum, 'enc3'),
        )
        #self.avg_pool = nn.AvgPool1d(kernel_size=10, stride=1)
        #   Classification layer
        '''
        This classifier is used to measure augmentation accuracy on TEP Dataset.
        We use it to understand if multiple augmentation improve or not.
        '''
        '''
        self.cls_linear = nn.Linear(256*7, 22)  # for TEP there are 22 classes
        '''
        
        self.cls_linear = nn.Sequential(
            nn.Linear(256*22,22)
        )
        self.cls_linear.apply(initialize_weights)


    def forward(self, inputs, pretrain=True):
        # print(inputs.type())
        # print("forward SimCLR_TS")

        features = self.encoder(inputs)

        # this must be called only when Pretrain is complete
        if not pretrain:
            #print(features.size())
            #features = self.avg_pool(features)
            #print(features.size())
            features = torch.flatten(features, start_dim=1)
            return self.cls_linear(features)
        else:
            return features
    
    