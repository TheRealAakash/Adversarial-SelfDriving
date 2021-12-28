import numpy as np
import cv2
import time
import os
import getKeys
from carlaEnv import CarEnv

file_name = "data/training_data.npy"
file_name2 = "data/target_data.npy"


def get_data():
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        image_data = list(np.load(file_name, allow_pickle=True))
        targets = list(np.load(file_name2, allow_pickle=True))
    else:
        print('File does not exist, starting fresh!')
        image_data = []
        targets = []
    return image_data, targets


def save_data(image_data, targets):
    numFiles = len(os.listdir("data")) + 1
    data = np.array([image_data, targets], dtype=object)
    np.save(f"data/train{numFiles}.npy", data, allow_pickle=True)


def main():
    env = CarEnv(True)

    count = 0
    image_data, targets = [], []
    while True:
        count += 1
        last_time = time.time()
        state, reward, done, actions = env.step([1, 2, 3])
        image_data.append(state)
        targets.append(actions)
        # print(actions)
        # Debug line to show image
        cv2.imshow("AI Peak", state)
        cv2.waitKey(1)

        # Convert to numpy array
        # print(actions, len(image_data))
        keys = getKeys.key_check()
        if len(image_data) % 1000 == 0:
            print(len(os.listdir("data")))
            print("Saving...")
            save_data(image_data, targets)
            image_data = []
            targets = []
            print("Saved")
        # print(f'loop took {time.time() - last_time} seconds or {round(1 / max((time.time() - last_time), 0.00000000001), 2)} fps')


if __name__ == '__main__':
    main()
