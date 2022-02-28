# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:42:13 2022

@author: Nicola Braile - Marianna Del Corso
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils import build_TEP_dataset

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, seq_len=100):

        self.X = torch.tensor(X.values)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = y

        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        start = index
        end = start + self.seq_len
        if(self.y is not None):
            return self.X[start:end].transpose(1,0), self.y[end-1].squeeze()
        else:
            return self.X[start:end], []

class Dataloader:
    def __init__(self, config):
        dataset_name = config['dataset']
        # NAB
        if dataset_name == 'NAB':
            data = np.load('dataset/NAB/nyc_taxi.npz')
            train = data['training'].astype('float32')
            train = np.expand_dims(train, axis=1)

        # TEP
        elif dataset_name == 'TEP':
            train, y_train, test, y_test = build_TEP_dataset()

        config['encoder_parameters']['in_channels_layer1'] = train.shape[1]

        self.train_dataset = TimeseriesDataset(X=train, y=y_train)
        self.test_dataset = TimeseriesDataset(X=test, y=y_test)

    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size = 64, shuffle = False)

    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size = 64, shuffle = False)
