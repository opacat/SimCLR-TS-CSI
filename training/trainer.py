#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:08:39 2022

@author: Nicola Braile - Marianna Del Corso
"""

from utils.utils import all_combinations
from utils.logger import logger_warmup, MetricLogger
from metrics import get_sim_matrix, NT_xent
from augm.augmenter import augment_batch
from utils.checkpoint import Checkpoint
from dataloader import dataloader
from models.simCLR_TS import SimCLR_TS

import numpy as np
import itertools
import logging
import torch
from torch.optim import Adam, lr_scheduler
    
logger_warmup('Logger_')
log = logging.getLogger('Logger_')

def training_warmup(config):
    
    train_loader, test_loader = dataloader('TEP', config['NET']) # NAB, TEP
    model = SimCLR_TS(config['NET'])
    
    # Optimizer
    optimizer = Adam(model.parameters())
    
    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100)
    
    args = {'start_epoch' : 0}
    
    ckp = Checkpoint(model, optimizer, scheduler, 'checkpoints/')
    # Load checkpoint, if exists. extra_ckp contains other important information like epoch number
    extra_ckp = ckp.load()
    args.update(extra_ckp)
    
    train_dict={}
    train_dict.update(model=model)
    train_dict.update(optimizer=optimizer)
    train_dict.update(scheduler=scheduler)
    train_dict.update(train_loader=train_loader)
    train_dict.update(test_loader=test_loader)
    train_dict.update(args=args)
    train_dict.update(ckp=ckp)
    
    config.update(TRAINING = train_dict)
     
def train_single_soft_augm(config):
    
    for s_a in config['AUGMENTER']['soft_augm_list']:
        
        config['AUGMENTER']['soft_augm'] = s_a
        log.info("Start pre-training single soft augmentation : {}".format(config['AUGMENTER']['soft_augm']))
        if config['AUGMENTER']['is_hard_augm']:
            log.info("with hard augmentation : {}".format(config['AUGMENTER']['hard_augm']))
        
        #crea dentro config la chiave Training con tutti i parametri di cui ha bisogno 
        training_warmup(config)
        
        pre_train(config)
        
        log.info("Start training... ")
        cls_train(config)


def train_multiple_soft_augm(config):

    for soft_comb in all_combinations(config['AUGMENTER']['soft_augm_list']):

        # Generate all soft permutations
        soft_perm = np.array(list(itertools.permutations(soft_comb)))

        for perm in soft_perm:

            # Setting a certain permutation
            config['AUGMENTER']['soft_augm_list'] = perm
            
            log.info("Start pre-training multiple soft augmentations : {} ".format(perm))
            if config['AUGMENTER']['is_hard_augm']:
                log.info("with hard augmentation : {}".format(config['AUGMENTER']['hard_augm']))
            
            training_warmup(config)
            pre_train(config)
            
            log.info("Start training... ")
            cls_train(config)


def train_single_soft_augm_with_hard_augm(config):
    for h_a in config['AUGMENTER']['hard_augm_list']:
        config['AUGMENTER']['hard_augm'] = h_a
        train_single_soft_augm(config)


def train_multiple_soft_augm_with_hard_augm(config):
    for h_a in config['AUGMENTER']['hard_augm_list']:
        config['AUGMENTER']['hard_augm'] = h_a
        train_multiple_soft_augm(config)


def pre_train(config):
    epochs = config['NET']['epochs']
    model = config['TRAINING']['model']
    optimizer = config['TRAINING']['optimizer']
    scheduler = config['TRAINING']['scheduler']
    args = config['TRAINING']['args']
    train_loader = config['TRAINING']['train_loader']
    ckp = config['TRAINING']['ckp']
    meters = MetricLogger()
    
    start_epoch = args['start_epoch']
    for epoch in range(start_epoch, epochs):

        loss_list = []
        # for each batch
        for batchdata, _ in train_loader:

            # Double the batch
            batchdata = batchdata.repeat(2, 1, 1)

            # Applies data augmentation
            augmented_batch = augment_batch(batchdata, config['AUGMENTER'])

            output = model(augmented_batch)

            # flatten output
            output = torch.flatten(output, start_dim=1)

            # TODO
            ''' WE MUST ADD HERE AVG_POOL FOR OUTPUT '''

            sim_mat = get_sim_matrix(output)
            loss = NT_xent(sim_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            meters.update(loss=loss.item())

        scheduler.step()
        # At the end of epoch, log all information
        log.info(meters.delimiter.join([
            "epoch: {epoch:03d}",
            "lr: {lr:.5f}",
            '{meters}',
        ]).format(
            epoch=epoch,
            lr=optimizer.param_groups[0]['lr'],
            meters=str(meters),
        ))

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            args['start_epoch'] = epoch
            ckp.save('model_pretrain_{:03d}'.format(epoch), **args)


def cls_train(config):
    log.info("Start training classifier...")
    epochs_cls = config['NET']['epochs_cls']
    model = config['TRAINING']['model']
    train_loader = config['TRAINING']['train_loader']
    
    # Fine tuning on fault classification
    for epoch in range(epochs_cls):
        for batchdata, _ in train_loader:

            cls_output = model(batchdata, True)
