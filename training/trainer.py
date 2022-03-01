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
from dataloader import Dataloader
from models.simCLR_TS import SimCLR_TS

import numpy as np
import itertools
import logging
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn

logger_warmup('Logger_')
log = logging.getLogger('Logger_')
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def training_warmup(config):

    dataloader = Dataloader(config['NET'])
    train_loader = dataloader.train_loader()
    model = SimCLR_TS(config['NET'])

    # Optimizers
    optimizer = Adam(model.encoder.parameters())
    linear_optimizer = Adam(model.cls_linear.parameters())

    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100)

    # Linear loss criterion
    criterion = nn.CrossEntropyLoss()

    args = {'start_epoch': 0}

    ckp = Checkpoint(model, optimizer, scheduler, 'checkpoints/')
    # Load checkpoint, if exists. extra_ckp contains other important information like epoch number
    extra_ckp = ckp.load()
    args.update(extra_ckp)

    train_dict = {}
    train_dict.update(model=model)
    train_dict.update(optimizer=optimizer)
    train_dict.update(linear_optimizer=linear_optimizer)
    train_dict.update(criterion=criterion)
    train_dict.update(scheduler=scheduler)
    train_dict.update(train_loader=train_loader)
    train_dict.update(args=args)
    train_dict.update(ckp=ckp)

    config.update(TRAINING=train_dict)


def train_single_soft_augm(config):

    for s_a in config['AUGMENTER']['soft_augm_list']:

        config['AUGMENTER']['soft_augm'] = s_a
        log.info(
            "Start pre-training single soft augmentation : {}".format(config['AUGMENTER']['soft_augm']))
        if config['AUGMENTER']['is_hard_augm']:
            log.info("with hard augmentation : {}".format(
                config['AUGMENTER']['hard_augm']))

        # crea dentro config la chiave TRAINING con tutti i parametri di cui ha bisogno
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

            log.info(
                "Start pre-training multiple soft augmentations : {} ".format(perm))
            if config['AUGMENTER']['is_hard_augm']:
                log.info("with hard augmentation : {}".format(
                    config['AUGMENTER']['hard_augm']))

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
    model = config['TRAINING']['model'].to(device)
    optimizer = config['TRAINING']['optimizer']
    scheduler = config['TRAINING']['scheduler']
    args = config['TRAINING']['args']
    train_loader = config['TRAINING']['train_loader']
    ckp = config['TRAINING']['ckp']
    meters = MetricLogger()

   # model.to(device)
    model.train()
    loss_list = []
    start_epoch = args['start_epoch']
    for epoch in range(start_epoch, epochs):

        # for each batch
        for batchdata, _ in train_loader:

            # Double the batch
            batchdata = batchdata.repeat(2, 1, 1).to(device)
            #batchdata.to(device)
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
            args['meters'] = meters
            ckp.save('model_pretrain_{:03d}'.format(epoch), **args)
            config['TRAINING'].update(model=model)

    meters.plot_loss()


def cls_train(config):
    log.info("Start training classifier...")
    epochs_cls = config['NET']['epochs_cls']
    model = config['TRAINING']['model'].to(device)
    train_loader = config['TRAINING']['train_loader']
    criterion = config['TRAINING']['criterion'].to(device)
    linear_optimizer = config['TRAINING']['linear_optimizer']
    scheduler = config['TRAINING']['scheduler']
    ckp = config['TRAINING']['ckp']

    #model.to(device)
    #criterion.to(device)
    model.train()
    # Freeze first part of the model
    for param in model.encoder.parameters():
        param.requires_grad = False

    loss_list = []
    # Fine tuning on fault classification
    for epoch in range(epochs_cls):
        for batchdata, labels in train_loader:
            batchdata = batchdata.to(device)
            labels = labels.to(device)
            cls_output = model(batchdata, False)  # pretrain=False
            loss_linear = criterion(cls_output, labels)

            linear_optimizer.zero_grad()
            loss_linear.backward()
            linear_optimizer.step()
            loss_list.append(loss_linear.item())

        scheduler.step()

    ckp.save('model_train_aug_{}'.format( config['AUGMENTER']['soft_augm']))
