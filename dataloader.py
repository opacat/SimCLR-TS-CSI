# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:42:13 2022

@author: Nicola Braile - Marianna Del Corso
"""
import numpy as np
import torch
from utils import build_TEP_dataset

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        if(self.y):
            return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
        else:
            return (self.X[index:index+self.seq_len]).transpose()


def dataloader(dataset_file, config):
    '''
    # NAB
    data = np.load(dataset_file)
    train = data['training'].astype('float32')
    train = np.expand_dims(train, axis=1)
    '''
    train, y_train, test, y_test = build_TEP_dataset()
    config['encoder_parameters']['in_channels_layer1'] = train.shape[1]

    train_dataset = TimeseriesDataset(train, seq_len=100)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = False)
    return loader