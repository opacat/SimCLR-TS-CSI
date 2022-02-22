import numpy as np
from utils import *
from models.simCLR_TS import *
from augm.augment import *
from augm.augmenter import *
import torch
from torch.optim import Adam, lr_scheduler
from dataloader import *
from metrics import *

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

config = get_config_json('config.json')

loader = dataloader('dataset/NAB/nyc_taxi.npz', config)

model = SimCLR_TS(config)

# Optimizer
optimizer = Adam(model.parameters())
# Scheduler
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100)

# Apply One Epoch 

for batchdata in loader:
    x=[]
    #print('batch type ', type(batchdata))
    # augment all batch
    #TODO il batch va sdoppiato prima di applicare le augmentations
    for window in batchdata:
        z = augmenter(datas=window.transpose(1,0),is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
        x.append(z.transpose())

    tmp = torch.tensor(np.array(x), dtype=torch.float32)
    print(tmp.size())
    
    output = model(tmp)
    print(output.size())
    
    sim_mat = get_sim_matrix(output)
    loss = NT_xent(sim_mat)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('loss = ',loss.item())
    