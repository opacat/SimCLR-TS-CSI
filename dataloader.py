# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:42:13 2022

@author: Nicola Braile - Marianna Del Corso
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from utils.utils import build_TEP_dataset

class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, seq_len=1):

        #print(X.dtypes)
        #self.X = np.squeeze(np.array([X], dtype=(np.float32)))
        self.X = torch.tensor(X.values)
        #print(self.X.dtype)
        if y:
            self.y = torch.tensor(y.values)
        else:
            self.y = y

        self.seq_len = seq_len
        self.transform = Compose([ToTensor()])


    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        start = index
        end = start + self.seq_len
        if(self.y):
            return (self.X[start:end], self.y[end-1])
        else:
            return self.X[start:end].transpose(1,0)


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
    loader = DataLoader(train_dataset, batch_size = 64, shuffle = False)
    return loader