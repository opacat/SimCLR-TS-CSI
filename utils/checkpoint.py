# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:50:18 2022

@author: Nicola Braile - Marianna Del Corso
"""
import os
import torch
import logging

log = logging.getLogger('Logger_')
class Checkpoint:

    #_last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self, name, model, optimizer=None, scheduler=None, save_dir="", save_to_disk=True):

        self.name = name
        self.last_checkpoint_name = name+'_last_checkpoint.txt'
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name, **args):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}

        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(args)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if not os.path.exists(self.save_dir+'/'+self.name):
            os.mkdir(self.save_dir+'/'+self.name)

        save_file = os.path.join(self.save_dir+'/'+self.name, "{}.pth".format(name))
        log.info('Saving checkpoint {}'.format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
            log.info('Loading from checkpoint {}...'.format(f))
        if not f:
            # no checkpoint could be found
            log.info('No checkpoint found.')
            return {}

        checkpoint = self._load_file(f)
        model = self.model

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint
    
    def load_eval(self):
        save_file = os.path.join(self.save_dir+'/'+self.name, 'model_train_aug_'+self.name+'.pth')
        #save_file = os.path.join(self.save_dir+'/'+self.name, 'model_baseline.pth')
        
        checkpoint = self._load_file(save_file)
        self.model.load_state_dict(checkpoint.pop("model"))

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir+'/'+self.name, self.last_checkpoint_name)
        return save_file

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir+'/'+self.name, self.last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        if last_filename.find('aug') >= 0:
            return
        save_file = os.path.join(self.save_dir+'/'+self.name, self.last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))
