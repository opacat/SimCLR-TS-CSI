#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:53:06 2022

@author: Nicola Braile - Marianna Del Corso
"""
import json
import pandas as pd
import numpy as np

# Load all config parameters
def get_config_json(json_file):
    with open(json_file,'r') as config:
        _config = json.load(config)
    return _config

def build_TEP_dataset():

    # Load .dat files
    def _get_dat_file(filename):
        res = pd.read_csv(filename, sep=' ',header=None, skipinitialspace=True, dtype=(np.float32))
        return res

    # Concatenate TEP training/test .dat files
    def _concat_files(is_training=True, postfix=''):

        dataset_list = []
        dims = []
        path = 'dataset/TEP/'
        for i in range(22):
            file_name = path + 'd' + "{:02d}".format(i) + postfix + '.dat'
            x = _get_dat_file(file_name)
            # first training files has wrong format
            if i==0 and is_training:
                x = x.transpose()

            # skip first 160 rows of test files 1 to 22
            if i>0 and not is_training:
                x = x.iloc[160:,:]

            # saving dataset rows - This info will be used to build labels correctly per dataset
            dims.append(x.shape[0])
            dataset_list.append(x)

        return  pd.concat(dataset_list), dims

    # Build labels
    def _get_labels(sizes, rows):
        start = 0
        labels = np.ones( (rows,1))
        for i in range(22):
            end = sizes[i] + start
            labels[start:end] = i * np.ones((sizes[i],1))
            start = end
        return labels


    # read training files and get training labels
    train_dataset, train_sizes = _concat_files()
    train_labels = _get_labels(train_sizes, train_dataset.shape[0])

    # read test files and get test labels
    test_dataset, test_sizes = _concat_files(is_training=False, postfix='_te')
    test_labels = _get_labels(test_sizes, test_dataset.shape[0])

    return train_dataset, train_labels, test_dataset, test_labels

