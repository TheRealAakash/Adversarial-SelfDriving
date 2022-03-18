import os

import cv2
import numpy as np
import trainingBehavior
from sklearn.utils import shuffle


def saveProcessedData():
    batch_x, batch_y = [], []
    for file in os.listdir("data/"):
        images, actions = np.load("data/" + file, allow_pickle=True)
        for image, action in zip(images, actions):
            cv2.imshow("image", image)
            cv2.waitKey(0)
            quit()
            image = trainingBehavior.processIMG(image)
            batch_x.append(image)
            batch_y.append(action)
    # batch_x, batch_y = shuffle(batch_x, batch_y)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    # data = np.array([batch_x, batch_y], dtype=object)
    np.save("processData/datay.npy", batch_y, allow_pickle=True)
    np.save("processData/datax.npy", batch_x, allow_pickle=True)


if __name__ == '__main__':
    saveProcessedData()
