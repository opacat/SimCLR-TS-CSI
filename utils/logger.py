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
