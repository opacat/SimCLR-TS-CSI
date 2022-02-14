import numpy as np
from utils import *
from models.simCLR_TS import *
from augm.augment import *
from augm.augmenter import *
import torch
from dataloader import *

'''
data = np.load('dataset/nyc_taxi.npz')
train = data['training']
train = np.expand_dims(train, axis=1)
print(train.shape)
'''
'''
#TESTING CODE for Augmenter
#Notes:
#CR must be checked because of zig zag visualitazion ( it may not be a fault )
augmenter(datas=training,is_hard_augm=False,hard_augm='',is_multiple_augm=True,soft_order=['CR','RN','L2R'],single_augm='')
#PASS
augmenter(datas=training,is_hard_augm=False,hard_augm='',is_multiple_augm=False,soft_order=[],single_augm='RN')
#PASS
augmenter(datas=training,is_hard_augm=True,hard_augm='MW',is_multiple_augm=False,soft_order=[],single_augm='L2R')
#PASS
augmenter(datas=training,is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
#PASS
'''

#augmenter(datas=training,is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')


config = get_config_json('config.json')

loader = dataloader('dataset/nyc_taxi.npz', config)

'''
for batch, i in zip(loader, range(1)):
    print('batch : ',batch.shape)
    for t,j in zip(batch, range(1)):
        augmenter(datas=t.transpose(1,0),is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
'''

model = SimCLR_TS(config)

x=[]
for batchdata in loader:
    # augment all batch
    for t in batchdata:
        z = augmenter(datas=t.transpose(1,0),is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
        x.append(z.transpose())
    
    #print(np.array(x).shape)
    
    tmp = torch.tensor(np.array(x), dtype=torch.float32)
    model(tmp)
