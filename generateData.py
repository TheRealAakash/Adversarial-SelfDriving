import os
import sklearn.utils

import tensorflow as tf
from settings.main_settings import get_settings
import trainBehavior
from carlaEnv import CarEnv
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass
if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("misclass"):
    os.mkdir("misclass")

env_name = "CarEnv"
env = CarEnv(False)

kerasModel = None
settings, config = get_settings()
agent = trainBehavior.PPOAgent(env, env_name, True, 50_000)
agent.load()
if os.path.exists("frame_data.p"):
    with open('frame_data.p') as f:
        x_train, y_train = pickle.load(f)
else:
    x_train, y_train = [], []
print(len(x_train), len(y_train))
for i in range(10):
    x_train_new, y_train_new = agent.generate(5_000, True)
    x_train.extend(x_train_new)
    y_train.extend(y_train_new)
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
    with open("frame_data.p", 'wb') as f:
        pickle.dump([x_train, y_train], f)
        f.close()
    print(i + 1, end="/10\n")
with open('frame_data.p') as f:
    x_train, y_train = pickle.load(f)
print(len(x_train), len(y_train))
