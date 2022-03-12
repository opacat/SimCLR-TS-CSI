#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:01:21 2022

@author: Nicola Braile - Marianna Del Corso
"""
from utils.utils import get_config_json
from models.simCLR_TS import SimCLR_TS
from utils.checkpoint import Checkpoint
from dataloader import Dataloader
from utils.logger import logger_warmup

import logging
import torch
import torch.nn as nn

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
logger_warmup('Logger_')
log = logging.getLogger('Logger_')

config = {}
config.update(NET=get_config_json('config/config_net.json'))

dataloader = Dataloader(config['NET'])
test_loader = dataloader.test_loader()
criterion = nn.CrossEntropyLoss()

with open('config/comb_list.txt') as f:
    model = SimCLR_TS(config['NET']).to(device)
    comb = f.readline()
    ckp = Checkpoint(comb, model, save_dir='checkpoints')
    ckp.load_eval()
    
    model.eval()
    running_corrects = 0

    for batchdata, labels in test_loader:
        batchdata = batchdata.to(device)
        labels = labels.to(device)
        
        cls_output = model(batchdata, False)
        
        # Get predictions
        _, preds = torch.max(cls_output.data, dim=1)
       
        #print(f"preds : {cls_output.data}")
        #print(len(preds))
        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_loader))
    print(f"Accuracy with {comb} augmentation: {accuracy}")
    log.info(f"Accuracy with {comb} augmentation: {accuracy}")
