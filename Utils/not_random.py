# this file has a very nice name
import random
import os
import numpy as np

# random seed
seed = 18

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

print(f'Setting seed random library, os and numpy seed to {seed}')
