import numpy as np
from augm.augment import *

data = np.load('dataset/nyc_taxi.npz')

training = data['training'][:200]
training = np.expand_dims(training, axis=1)
print(training.shape)

left2rightFlip(training, True)
random_noise(training)
crop_resize(training)
blockout(training,50)
magnitude_warping(training,2,3)
