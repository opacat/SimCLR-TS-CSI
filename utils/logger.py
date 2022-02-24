#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:35:18 2022

@author: Nicola Braile - Marianna Del Corso
"""

import logging
from os import path
import os
import sys
import numpy as np
import torch
from collections import deque, defaultdict

def logger_warmup(name,save_dir='logger_data'):

    if not path.exists(save_dir):
        os.mkdir(save_dir)

    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

        if save_dir:
            file_handler = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.value = np.nan
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        self.value = value

    @property
    def avg(self):
        values = np.array(self.deque)
        return np.mean(values)

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            #print(type(v))
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.3f} ({:.3f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


'''
x = MetricLogger()
data = 4

dictn =  { 'data1': 1,'data2':100}
dictn2 = { 'data1': 10,'data2':103}
dictn3 = { 'data1': 20,'data2':102}
dictn4 = { 'data1': 15,'data2':120}

x.update(**dictn)
x.update(**dictn2)

print(x)
'''