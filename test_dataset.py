from training.trainer import train_multiple_soft_augm, train_multiple_soft_augm_with_hard_augm, train_single_soft_augm, train_single_soft_augm_with_hard_augm
from training.trainer import baseline_train
from utils.utils import get_config_json
from evaluation import evaluate
from utils.logger import logger_warmup

import random
import logging

logger_warmup('Logger_')
log = logging.getLogger('Logger_')

config = {}
config.update(NET=get_config_json('config/config_net.json'))
config.update(AUGMENTER=get_config_json('config/config_augmenter.json'))


baseline_train(config) # training senza contrastive learning per replicare baseline
#evaluate(config)      # evaluation su testset usando il modello di baseline


'''
#  RANDOM SEARCH

random.seed(10)

lr_values = [0.001]
weight_dec_values = [0.002]
best_acc = 0.0
best_lr = -1
best_wd = -1

for lr in range(10):
  lr = float(format(random.uniform(1e-6, 1e-3), ".6f"))
  for wd in range(10):
    wd = float(format(random.uniform(0, 1e-1), ".6f"))
    log.info(f"START training with lr:{lr}, wd:{wd}")
    config['NET'].update(lr=lr)
    config['NET'].update(weight_decay=wd)
    baseline_train(config)
    acc = evaluate(config)

    if acc > best_acc:
      log.info(f"New best accuracy: {best_acc} -> {acc}")
      best_acc = acc
      best_lr = lr
      best_wd = wd

log.info(f"END : best accuracy {best_acc}, best lr {best_lr}, best wd {best_wd}")
'''

'''
# PER TROVARE LA COMBINAZIONE MIGLIORE DI TRASFORMAZIONI

# config aumenter in the proper way and then call the right training procedure
if config['AUGMENTER']['is_multiple_augm']:

    # ANALISING MULTIPLE SOFT AUGM.

    if config['AUGMENTER']['is_hard_augm']:
        train_multiple_soft_augm_with_hard_augm(config)
    else:
        train_multiple_soft_augm(config)

else:
    # ANALISING SINGLE SOFT AUGM

    if config['AUGMENTER']['is_hard_augm']:
        train_single_soft_augm_with_hard_augm(config)
    else:
        train_single_soft_augm(config)
'''