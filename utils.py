#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:53:06 2022

@author: Nicola Braile - Marianna Del Corso
"""
import json

# Load all config parameters
def get_config_json(json_file):
    with open(json_file,'r') as config:
        _config = json.load(config)
    return _config