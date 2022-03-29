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
import time

logger_warmup('Logger_')
log = logging.getLogger('Logger_')
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def training_warmup(config):

    dataloader = Dataloader(config['NET'])
    train_loader = dataloader.train_loader()
    model = SimCLR_TS(config['NET']).to(device)

    # Optimizers
    optimizer = Adam(model.encoder.parameters())
    linear_optimizer = Adam(model.cls_linear.parameters())

    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100)

    # Linear loss criterion
    criterion = nn.CrossEntropyLoss()

    args = {'start_epoch': 0, 'augm_done':[], 'meters':None}

    ckp = Checkpoint(config['name'], model, optimizer, scheduler, 'checkpoints')
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
        name_conf = s_a
        config['AUGMENTER']['soft_augm'] = s_a
        log.info(
            "Start pre-training single soft augmentation : {}".format(config['AUGMENTER']['soft_augm']))
        if config['AUGMENTER']['is_hard_augm']:
            name_conf = name_conf+'_'+config['AUGMENTER']['hard_augm']
            log.info("with hard augmentation : {}".format(
                config['AUGMENTER']['hard_augm']))



        config.update(name=name_conf)

        # crea dentro config la chiave TRAINING con tutti i parametri di cui ha bisogno
        training_warmup(config)
        #print('augm_done ',config['TRAINING']['args']['augm_done'])
        #print('name_conf ', name_conf)
        if name_conf in config['TRAINING']['args']['augm_done']:
            log.info('[{}] already trained.'.format(name_conf))
            continue

        pre_train(config)

        log.info("Start training... ")
        cls_train(config)


def train_multiple_soft_augm(config):

    for soft_comb in all_combinations(config['AUGMENTER']['soft_augm_list']):

        # Generate all soft permutations
        soft_perm = np.array(list(itertools.permutations(soft_comb)))

        for perm in soft_perm:

            name_conf = '_'.join(perm)
            # Setting a certain permutation
            config['AUGMENTER']['soft_augm_list'] = perm

            log.info(
                "Start pre-training multiple soft augmentations : {} ".format(perm))
            if config['AUGMENTER']['is_hard_augm']:
                name_conf = name_conf+'_'+config['AUGMENTER']['hard_augm']
                log.info("with hard augmentation : {}".format(
                    config['AUGMENTER']['hard_augm']))



            config.update(name=name_conf)
            training_warmup(config)
            if name_conf in config['TRAINING']['args']['augm_done']:
                log.info('[{}] already trained.'.format(name_conf))
                continue
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
    if args['meters']:
        print('Meters recuperato!')
        meters = args['meters']
    else:
        print('no meters')
        meters = MetricLogger()

   # model.to(device)
    model.train()
    loss_list = []
    start_epoch = args['start_epoch']
    for epoch in range(start_epoch, epochs):
        #start_time = time.time()

        # for each batch
        for batchdata,_ in train_loader:
            # Double the batch
            batchdata = batchdata.repeat(2, 1, 1)
            start_epoch = args['start_epoch']

            #start_time = time.time()
            # Applies data augmentation
            augmented_batch = augment_batch(batchdata, config['AUGMENTER']).to(device)
            #print("--- Augment Batch execution time : %s seconds ---" % (time.time() - start_time))


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
        
        #print("--- Epoch execution time : %s seconds ---" % (time.time() - start_time))

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
        if (epoch+1) % config['NET']['save_epoch'] == 0 and epoch > 0:
            args['start_epoch'] = epoch+1
            args.update(meters=meters)

            if epoch == epochs-1:
                args['augm_done'].append(config['name'])
                #print('last epoch ',args['augm_done'])
            ckp.save('model_pretrain_{:03d}'.format(epoch), **args)
            
            config['TRAINING'].update(model=model)

    meters.plot_loss(config['name'])


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

def local_eval(model,loader):
  running_corrects = 0
  tot_samples = 0
  cur_label = 0
  j = 0
  for i,(batchdata, labels) in enumerate(loader):
    '''if labels.data[0] == cur_label and j<2:
      j += 1
    else:
      if j>1:
        j = 0
        cur_label += 1
      continue'''
    '''if i >0:
      break'''
    if (not all(l == 0 for l in labels.data)) and (not all(l == 1 for l in labels.data)) and (not all(l == 2 for l in labels.data)) and (not all(l == 3 for l in labels.data)) and (not all(l == 4 for l in labels.data)) and (not all(l == 5 for l in labels.data)) and (not all(l == 6 for l in labels.data)) and (not all(l == 7 for l in labels.data)) and (not all(l == 8 for l in labels.data)) and (not all(l == 9 for l in labels.data)) and (not all(l == 10 for l in labels.data)) and (not all(l == 11 for l in labels.data)) and (not all(l == 12 for l in labels.data)) and (not all(l == 13 for l in labels.data)) and (not all(l == 14 for l in labels.data)) and (not all(l == 15 for l in labels.data)) and (not all(l == 16 for l in labels.data)) and (not all(l == 17 for l in labels.data)) and (not all(l == 18 for l in labels.data)) and (not all(l == 19 for l in labels.data)) and (not all(l == 20 for l in labels.data)) and (not all(l == 21 for l in labels.data)):
      continue
    batchdata = batchdata.to(device)
    labels = labels.to(device)
    
    cls_output = model(batchdata, False)
    
    # Get predictions
    _, preds = torch.max(cls_output.data, dim=1)
    running_corrects += torch.sum(preds == labels.data).data.item()
    tot_samples += len(batchdata)
  '''print(f"tot samples: {tot_samples}")
  print(f"running_corrects {running_corrects}")'''
  return running_corrects / float(tot_samples)

def baseline_train(config):
    dataloader = Dataloader(config['NET'])
    train_loader = dataloader.train_loader()
    model = SimCLR_TS(config['NET']).to(device)
    ckp = Checkpoint(name='BASE', model=model, save_dir='checkpoints')
    #print(f"model parameters\n{list(model.parameters())}")
    #print(model)
    for name, param in model.named_parameters():
      print(f"{name}, {param.data}")
    return
    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['NET']['lr'], weight_decay=config['NET']['weight_decay'])

    # Scheduler
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=200)

    # Linear loss criterion
    criterion = nn.CrossEntropyLoss()
    meters = MetricLogger()

    model.train()
    acc_list = []
    for epoch in range(config['NET']['epochs']):
        cur_label = 0
        j=0        
        for i,(batchdata,labels) in enumerate(train_loader):
            '''if labels.data[0] == cur_label and j<2: # training e test sui primi 2 batch di ogni classe (2816 samples)
              j += 1
            else:
              if j>1:
                j = 0
                cur_label += 1
              continue'''
            '''if i >0: # training e test su 1 batch
              break'''
            if (not all(l == 0 for l in labels.data)) and (not all(l == 1 for l in labels.data)) and (not all(l == 2 for l in labels.data)) and (not all(l == 3 for l in labels.data)) and (not all(l == 4 for l in labels.data)) and (not all(l == 5 for l in labels.data)) and (not all(l == 6 for l in labels.data)) and (not all(l == 7 for l in labels.data)) and (not all(l == 8 for l in labels.data)) and (not all(l == 9 for l in labels.data)) and (not all(l == 10 for l in labels.data)) and (not all(l == 11 for l in labels.data)) and (not all(l == 12 for l in labels.data)) and (not all(l == 13 for l in labels.data)) and (not all(l == 14 for l in labels.data)) and (not all(l == 15 for l in labels.data)) and (not all(l == 16 for l in labels.data)) and (not all(l == 17 for l in labels.data)) and (not all(l == 18 for l in labels.data)) and (not all(l == 19 for l in labels.data)) and (not all(l == 20 for l in labels.data)) and (not all(l == 21 for l in labels.data)):
              continue
            batchdata = batchdata.to(device)
            labels = labels.to(device)
            out = model(batchdata, False)
            loss = criterion(out, labels)
            meters.update(loss=loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        ''' local_acc = local_eval(model, train_loader)
        print(f"local_acc : {local_acc}")
        acc_list.append(local_acc)'''
         
        log.info(meters.delimiter.join([
            "epoch: {epoch:03d}",
            "lr: {lr:.6f}",
            "wd: {wd:.6f}",
            '{meters}',
        ]).format(
            epoch=epoch,
            lr=optimizer.param_groups[0]['lr'],
            wd=optimizer.param_groups[0]['weight_decay'],
            meters=str(meters),
        ))
        scheduler.step()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    timestamps = len(acc_list)
    x = np.arange(timestamps)
    ax = fig.add_subplot(111)
    plt.plot(x,acc_list)
    ax.set_xlim([0, timestamps])
    
    plt.savefig('plots/train_acc.png') 
    plt.show()
    
    meters.plot_loss('BASE')
    ckp.save('model_train_aug_BASE')