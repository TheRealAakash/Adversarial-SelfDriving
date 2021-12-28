import time

import cv2
from tensorflow import keras
import numpy as np

from carlaEnv import CarEnv
import trainingBehavior

model = keras.models.load_model("model.h5")
print("loaded learner")


def label_func(x): return x.parent.name


def main():
    #    model = training.build_model((224, 224, 3), 10)
    env = CarEnv(False)
    state = env.reset()
    while True:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.Canny(image, threshold1=119, threshold2=250)
        cv2.imshow("AI Peek", state)
        cv2.waitKey(1)
        last_time = time.time()
        state = np.array([state])
        result = model.predict(state)[0]
        state, reward, done, actions = env.step(result)
        if done:
            state = env.reset()

        print(f'loop took {time.time() - last_time} seconds or {round(1 / max((time.time() - last_time), 0.00000000001), 2)} fps')


main()
