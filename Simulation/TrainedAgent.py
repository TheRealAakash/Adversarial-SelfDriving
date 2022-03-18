import os
import time

import cv2
from tensorflow import keras
import numpy as np

from carlaEnv import CarEnv
# from trainingBehavior import processIMG
import natsort
from Simulation.AutoEncoder import AutoEncoder
import tensorflow as tf
from settings.SimulationSettings import Config

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs successfully enabled")
    except RuntimeError as e:
        print(e)


def main():
    #    model = training.build_model((224, 224, 3), 10)
    env = CarEnv(False)
    state = env.reset()
    file = natsort.natsorted(os.listdir("checkpoints/"))[-1]
    model = keras.models.load_model(f"checkpoints/model-024-0.0064.h5")
    autoEncoder = AutoEncoder((128, 128, 3), 10, Config.numOutputs, "models/Sim-", True)
    print("loaded learner")
    while True:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.Canny(image, threshold1=119, threshold2=250)
        cv2.waitKey(1)
        # state = processIMG(state)
        cv2.imshow("AI Peek", state)
        last_time = time.time()
        state = np.array(state) / 255.0
        state = autoEncoder.encode(state)
        print(state)
        print(state.shape)
        result = model.predict(state)
        print(result)
        state, reward, done, actions = env.step(result)
        print(actions[0] - result[0], actions[1] - result[1], actions[2] - result[2])
        # print(actions, result)
        if done:
            state = env.reset()

        print(f'loop took {time.time() - last_time} seconds or {round(1 / max((time.time() - last_time), 0.00000000001), 2)} fps')


main()
