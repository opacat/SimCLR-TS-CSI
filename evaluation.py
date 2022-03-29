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
from utils.utils import check_labels

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

        #model.eval()
        running_corrects = 0
        tot_samples = 0
        cur_label = 0
        j = 0
        for i,(batchdata, labels) in enumerate(test_loader):
            #print(f"label {labels.data[0]}, cur_label {cur_label}, j {j}")
            '''if labels.data[0] == 0 and j<2: # training e test sui primi 2 batch di ogni classe (2816 samples)
              j += 1
              print(labels.data[0])
            else:
              if j>1:
                j = 0
                cur_label += 1
              continue'''
            '''if i >0: # training e test su 1 batch
              break'''
            if not check_labels(labels):
              continue
            batchdata = batchdata.to(device)
            labels = labels.to(device)

            cls_output = model(batchdata, False)

            # Get predictions
            _, preds = torch.max(cls_output.data, dim=1)

            #print(f"preds : {preds}")
            #print(f'labels : {labels.data}')
            #print(len(preds))
            # Update Corrects
            #print(f"cur_label {cur_label}")
            running_corrects += torch.sum(preds == labels.data).data.item()
            tot_samples += len(batchdata)
            #print(f"len( labels ): {len(labels.data)}")
        # Calculate Accuracy
        #print(float(len(test_loader)*config['NET']['batch_size']))
        #accuracy = running_corrects / float(len(test_loader)*config['NET']['batch_size'])
        #i=22*2
        print(f"tot samples: {tot_samples}")
        accuracy = running_corrects / float(tot_samples)
        print(f"running_corrects {running_corrects}")
        print(f"Accuracy with {comb} augmentation: {accuracy}")
        log.info(f"Accuracy with {comb} augmentation: {accuracy}")
        return accuracy
