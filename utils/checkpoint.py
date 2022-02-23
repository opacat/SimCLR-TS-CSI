# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:50:18 2022

@author: Nicola Braile - Marianna Del Corso
"""
import os
import torch

class Checkpoint:

    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self, model,optimizer=None,scheduler=None,save_dir="",save_to_disk=True):

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

       save_file = os.path.join(self.save_dir, "{}.pth".format(name))
       torch.save(data, save_file)

       self.tag_last_checkpoint(save_file)


    def load(self, f=None, use_latest=True):
       if self.has_checkpoint() and use_latest:
           # override argument with existing checkpoint
           f = self.get_checkpoint_file()
       if not f:
           # no checkpoint could be found
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


    def get_checkpoint_file(self):
       save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
       try:
           with open(save_file, "r") as f:
               last_saved = f.read()
               last_saved = last_saved.strip()
       except IOError:
           # if file doesn't exist, maybe because it has just been
           # deleted by a separate process
           last_saved = ""
       return last_saved

    def has_checkpoint(self):
       save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
       return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
       save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
       with open(save_file, "w") as f:
           f.write(last_filename)

    def _load_file(self, f):
       return torch.load(f, map_location=torch.device("cpu"))
