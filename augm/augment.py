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


# Left to right flipping through antidiagonal matrix multiplication

def left2rightFlip(data):
     dim = data.shape[0]    
     antiD_Eye_Matrix = np.fliplr(np.eye(dim))
     return data.dot(antiD_Eye_Matrix)

# Interpolation and selection of a sub signal of size dim
def crop_resize(data):
    outdata=[]
    dim = data.shape[0]    
    #interpolation between odd values --> this generate a signal of 2*dim - 1 sample
    for i in range(dim):
        if i==0:
            outdata.append(data[i])
            continue
    
        outdata.append( (data[i]+ data[i-1]) /2 )
        outdata.append(data[i])
    
    #selection of dim samples starting in a random point of the first window
    start = random.randint(0, dim-1)
    end = start+dim    
    return outdata[start:end]


# Applies random noise to the signal by adding or removing std * p values 
# where p is drawn from a uniform distribution
def random_noise(data):
    dim = data.shape[0]        
    std = np.std(data)
    prob = std*np.random.uniform(-1.0, 1.0 , dim)
    return data*prob


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
    plt.plot(data)
    plt.plot(data_augm)
    plt.show()  

#TESTING CODE

b = np.array([1,2,3,4,5,6,7,8,9])
#left2rightFlip(b)
#crop_resize(b)
#blockout(b, 5)
#random_noise(b)

#res = magnitude_warping(b,1,5)
#plot_data(b,res)

b = b.reshape((3, 3))
print(b)
r = permute_channels(b)
print(r)
'''
arr = np.arange(9).reshape((3, 3))
print(arr)
np.random.shuffle(arr)
print(arr)
'''