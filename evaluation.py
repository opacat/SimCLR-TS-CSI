#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:01:21 2022

@author: Nicola Braile - Marianna Del Corso
"""
from models.simCLR_TS import SimCLR_TS
from utils.checkpoint import Checkpoint
from dataloader import Dataloader
from utils.logger import logger_warmup

import logging
import torch

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
logger_warmup('Logger_')
log = logging.getLogger('Logger_')


def evaluate(config):
    dataloader = Dataloader(config['NET'])
    test_loader = dataloader.test_loader()
    
    with open('config/comb_list.txt') as f:
        model = SimCLR_TS(config['NET']).to(device)
        comb = f.readline()
        ckp = Checkpoint(name=comb, model=model, save_dir='checkpoints')
        ckp.load_eval()
        
        model.eval()
        running_corrects = 0
    
        for batchdata, labels in test_loader:
            batchdata = batchdata.to(device)
            labels = labels.to(device)
            
            cls_output = model(batchdata, False)
            
            # Get predictions
            _, preds = torch.max(cls_output.data, dim=1)
           
            print(f"preds : {preds}")
            print(f'labels : {labels.data}')
            #print(len(preds))
            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()
    
        # Calculate Accuracy
        accuracy = running_corrects / float(len(test_loader)*config['NET']['batch_size'])
        print(running_corrects)
        print(f"Accuracy with {comb} augmentation: {accuracy}")
        log.info(f"Accuracy with {comb} augmentation: {accuracy}")
