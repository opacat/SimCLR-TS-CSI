#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:37:44 2022

@author: Nicola Braile - Marianna Del Corso
"""

import torch

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def get_sim_matrix(outputs):
    
    outputs = normalize(outputs)  
    sim_matrix = torch.mm(outputs, outputs.t())  
    return sim_matrix

def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss

