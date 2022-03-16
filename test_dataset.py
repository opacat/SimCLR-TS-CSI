from training.trainer import train_multiple_soft_augm, train_multiple_soft_augm_with_hard_augm, train_single_soft_augm, train_single_soft_augm_with_hard_augm
from training.trainer import baseline_train
from utils.utils import get_config_json
from evaluation import evaluate

'''
data = np.load('dataset/nyc_taxi.npz')
train = data['training']
train = np.expand_dims(train, axis=1)
print(train.shape)
'''

config = {}
config.update(NET=get_config_json('config/config_net.json'))
config.update(AUGMENTER=get_config_json('config/config_augmenter.json'))

'''
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
baseline_train(config)
'''
TODO PRIMA DI ESEGUIRE EVALUATE !!
cambiare il nome del file di checkpoint in load_eval in model_baseline.pth
'''
evaluate(config)