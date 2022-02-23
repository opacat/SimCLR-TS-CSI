import numpy as np
from utils.utils import *
from utils.checkpoint import *
from utils.logger import *
from models.simCLR_TS import *
from augm.augment import *
from augm.augmenter import *
import torch
from torch.optim import Adam, lr_scheduler
from dataloader import *
from metrics import *
import logging

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

logger_warmup('Logger1')
log = logging.getLogger('Logger1')
print(log.handlers)

config = get_config_json('config.json')

loader = dataloader('dataset/NAB/nyc_taxi.npz', config)

model = SimCLR_TS(config)

# Optimizer
optimizer = Adam(model.parameters())
# Scheduler
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100)
args = {'start_epoch' : 0}
ckp = Checkpoint(model, optimizer, scheduler, 'checkpoints/')
# Load checkpoint, if exists. extra_ckp contains other important information like epoch number
extra_ckp = ckp.load()
args.update(extra_ckp)

#log.log(logging.DEBUG,'I topini non aveno nipotini',args)
#log.log(logging.INFO,'Le lame non avevano la lama',args)

log.info('Message Info')
log.debug('Message Debug Info')

epochs = config['epochs']
epochs_cls = config['epochs_cls']
start_epoch = args['start_epoch']
# Apply One Epoch
loss_list =[]

for epoch in range(start_epoch,epochs):
    #for each batch 
    for batchdata in loader:
        x=[]
        batchdata = batchdata.repeat(2,1,1)
        #print(batchdata.size)
        # augment all batch
        #TODO il batch va sdoppiato prima di applicare le augmentations
        for window in batchdata:
            z = augmenter(datas=window.transpose(1,0),is_hard_augm=True,hard_augm='BLK',is_multiple_augm=True,soft_order=['RN','CR','L2R'],single_augm='')
            x.append(z.transpose())
    
        tmp = torch.tensor(np.array(x), dtype=torch.float32)
        #print(tmp.size())
    
        output = model(tmp)
    
        output = torch.flatten(output, start_dim=1)
        #print(output.size())
    
        sim_mat = get_sim_matrix(output)
        loss = NT_xent(sim_mat)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_list.append(loss.item())
   
    # Save checkpoint every 10 epochs
    if epoch%10==0:
        args['start_epoch'] = epoch
        ckp.save('model_pretrain_{:03d}'.format(epoch), **args)
    

#for epoch in range(epochs_cls):
    #training for classifier ( 22 classes )
