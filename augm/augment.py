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

'''
 The input matrix has T rows and M columns where:
     T  = window size 
     M  = number of features ( or channels )
'''

def apply_augm_xChannel(datas,augmentation):
    augmented_datas = np.array(datas)
    channels = datas.shape[1]
    for chi in range(channels):
       augmented_datas[chi] = augmentation(datas[:,chi])
    return augmented_datas.transpose()

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
def random_noise(datas):
    dim1 = datas.shape[0]        
    dim2 = datas.shape[1]
    datas_rn = np.array(datas)
    std = np.std(datas,axis=1)
    print("std")
    print(std)
    ru = np.random.uniform(-1.0, 1.0 , (dim1,dim2))
    print("random uniform matrix ")
    print(ru)
    prob_matrix = std* ru
    print("prob matrix ")
    print(prob_matrix)
    datas_rn = np.multiply(datas,prob_matrix)
    print(datas_rn)
    plot_data(datas[:,0], datas_rn[:,0]) 
    
    '''
    def _xChannel_random_noise(channel):
        dim = channel.shape[0]        
        std = np.std(channel)
        prob = std*np.random.uniform(-1.0, 1.0 , dim)
        return channel*prob
    '''

#              HARD AUGMENTATIONS

# Select a random point and makes the signal zeroed
def blockout(data,duration):
    dim = data.shape[0]    
    start = random.randint(0,dim-duration)
    end = start + duration
    data[start:end] = 0
    return data

# Sum n full sine periods to the signal
def magnitude_warping(data,lamb,n=2):
    T = data.shape[0]   
    sin_values = []
    for t in range(T):
        val = 2*n*math.pi* t / T
        sin_values.append(lamb*(1+math.sin(val)))
    return data*sin_values 

# Random shuffle of channels
def permute_channels(data):
    rs =  random.sample(list(data), len(data))
    x = np.array(rs)
    return x
    
def plot_data(data,data_augm):
    fig = plt.figure()
    x= [1,2,3]
    ax = fig.add_subplot(111)
    plt.plot(x,data)
    plt.plot(x,data_augm)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    plt.show()  

#TESTING CODE

b = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.])
#left2rightFlip(b)
#crop_resize(b)
#blockout(b, 5)
#random_noise(b)

#res = magnitude_warping(b,1,5)
#plot_data(b,res)

b = b.reshape((3, 3))

#r = permute_channels(b)
#print(r)

a = b
#left2rightFlip(a,True)
#crop_resize(a)

random_noise(a)