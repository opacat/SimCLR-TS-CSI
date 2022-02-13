import numpy as np
from augm.augment import *
from augm.augmenter import *
from tensorflow.keras.utils import timeseries_dataset_from_array

data = np.load('dataset/nyc_taxi.npz')

training = data['training'][:150]
training = np.expand_dims(training, axis=1)
print(training.shape)

#TEST WINDOWS
train = data['training']
print('training : ',train.shape)

loader = timeseries_dataset_from_array(data=train,targets=None,sequence_length=10)
for a,i in zip(loader,range(2)):
  print(a)

#TESTING CODE for Augment.py

'''
left2rightFlip(training, True)
random_noise(training)
crop_resize(training)
blockout(training,50)
magnitude_warping(training,2,3)
'''

#TESTING CODE for Augmenter.py

#Notes:
#CR must be checked because of zig zag visualitazion ( it may not be a fault )

#augmenter(datas=training,is_hard_augm=False,hard_augm='',is_multiple_augm=True,soft_order=['CR','RN','L2R'],single_augm='')
#PASS

#augmenter(datas=training,is_hard_augm=False,hard_augm='',is_multiple_augm=False,soft_order=[],single_augm='RN')
#PASS

augmenter(datas=training,is_hard_augm=True,hard_augm='MW',is_multiple_augm=False,soft_order=[],single_augm='L2R')
#PASS 

#augmenter(datas=training,is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
#PASS
