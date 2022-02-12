#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:28:51 2022

@author: Nicola Braile , Marianna Del Corso

"""

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from functools import partial

'''
 The input matrix has T rows and M columns where:
     T  = window size 
     M  = number of features ( or channels )
'''

def apply_augm_xChannel(datas,augmentation):
    augmented_datas = np.array(datas)
    channels = datas.shape[1]
    for chi in range(channels):
       augmented_datas[:,chi] = augmentation(datas[:,chi])
    return augmented_datas

# Left to right flipping through anti diagonal matrix multiplication
def left2rightFlip(datas,show_plot=False):
    
    datas_ltor = np.array(datas) 
    def _xChannel_left2rightFlip(channel):
         dim = channel.shape[0]    
         antiDiag_Eye_Matrix = np.fliplr(np.eye(dim))
         return np.dot(channel,antiDiag_Eye_Matrix)
     
    datas_ltor = apply_augm_xChannel(datas, _xChannel_left2rightFlip )
    #plots first channel only for now
    if show_plot:
        plot_data(datas[:,0], datas_ltor[:,0]) 
    

# Interpolation and selection of a sub signal of size dim
def crop_resize(datas):
    
    datas_cr = np.array(datas)
    def _xChannel_crop_resize(channel):
        outdata=[]
        dim = channel.shape[0]    
        #interpolation between odd values --> this generate a signal of 2*dim - 1 sample
        for i in range(dim):
            if i==0:
                outdata.append(channel[i])
                continue    
            outdata.append( (channel[i]+ channel[i-1]) /2 )
            outdata.append(channel[i])
        
        #selection of dim samples starting in a random point of the first window
        start = random.randint(0, dim-1)
        end = start+dim   
        return outdata[start:end]
    
    datas_cr = apply_augm_xChannel(datas, _xChannel_crop_resize)
   
    plot_data(datas[:,0], datas_cr[:,0]) 
    return datas_cr

# Applies random noise to the signal by adding or removing std * p values 
# where p is drawn from a uniform distribution
def random_noise(datas,scale=1):
    dim1 = datas.shape[0]        
    dim2 = datas.shape[1]
    datas_rn = np.array(datas)
    std = np.std(datas,axis=1)
    ru = np.random.uniform(-1.0, 1.0 , (dim1,dim2))
    
    noise_matrix = np.array(ru)
    for chi in range(dim2):
        noise_matrix[chi] = std[chi] * ru[chi]
    datas_rn = datas + (scale* noise_matrix )
    plot_data(datas[:,0], datas_rn[:,0]) 
    
#              HARD AUGMENTATIONS

# Select a random point and makes the signal zeroed
def blockout(datas,duration):
    
    dim = datas.shape[0]    
    datas_bo = np.array(datas)
    #starting blockout point equal for all channels
    start = random.randint(0,dim-duration)
    
    def _blockout(data,start,duration):
        end = start + duration
        data[start:end] = 0
        return data
    
    bo_augmentation = partial(_blockout , start=start, duration=duration)
    datas_bo = apply_augm_xChannel(datas_bo, bo_augmentation)
    plot_data(datas[:,0], datas_bo[:,0]) 
    return datas_bo
    
# Sum n full sine periods to the signal
def magnitude_warping(datas,scale,n):
    
    datas_mw = np.array(datas)
    def _magnitude_warping(data,scale,n=2):
        # T is the window size
        
        T = data.shape[0]   
        phase = np.random.uniform(0,2*math.pi) 
        #sampling T values from sin function
        sin_values = []
        for t in range(T):
            x = ( 2*math.pi*n*t / T ) + phase
            sin_values.append(scale*(1+math.sin(x)))
        
        return data*sin_values
    
    mw_augmentation = partial(_magnitude_warping,scale=scale,n=n)
    datas_mw = apply_augm_xChannel(datas_mw, mw_augmentation)
    plot_data(datas[:,0], datas_mw[:,0]) 
    return datas_mw
    
# Random shuffle of channels
def permute_channels(data):
    t = data.transpose()
    rs =  random.sample(list(t), len(t))
    return np.array(rs).transpose()
    
def plot_data(data,data_augm):
    fig = plt.figure()
    x= np.arange(50)
    ax = fig.add_subplot(111)
    plt.plot(x,data)
    plt.plot(x,data_augm)
    ax.set_xlim([-10, 100])
    ax.set_ylim([-10, 100])
    plt.show()  

#TESTING CODE

#b = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.])
#left2rightFlip(b)
#crop_resize(b)
#blockout(b, 5)
#random_noise(b)

#res = magnitude_warping(b,1,5)
#plot_data(b,res)

#b = b.reshape((3, 3))

#r = permute_channels(b)
#print(r)

#a = b
#data = np.ones((100,2))

#random_noise(data,3)           #OK
#blockout(data, 20)             #OK
#permute_channels(a)    

#testing C_R OK it works        
#val = magnitude_warping(data, 1, 4) #OK
#crop_resize(val)


#testing L2R - OK it works
#val = np.arange(100)
#crop_resize(val.reshape(50,2))

#left2rightFlip(val.reshape(50,2),True)

'''
Nel caso di dati 1D occorre chiamare prima expand_dims in modo 
da ottenere una matrice
'''
c = np.arange(50)
print(c.shape)
c=np.expand_dims(c, axis=1)
print(c.shape)
blockout(c, 20) 