import os
import pickle
import time

import tensorflow as tf
import tqdm
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, ELU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import random
import cv2
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.callbacks import TensorBoard
from Simulation import AutoEncoder
from settings.SimulationSettings import Config

LOG_DIR = "logs"
tensorboard = TensorBoard(log_dir=LOG_DIR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs successfully enabled")
    except RuntimeError as e:
        print(e)

file_name = "data/training_data.npy"
file_name2 = "data/target_data.npy"
autoEncoder = AutoEncoder.AutoEncoder((128, 128, 3), 10, Config.numOutputs, "models/Sim-", True)


def buildModel(input_shape, action_space):
    X_input = Input(shape=input_shape)
    model = Dense(256, activation="relu")(X_input)
    model = Dense(128, activation="relu")(model)
    model = Dense(64, activation="relu")(model)
    model = Dense(32, activation="relu")(model)
    # base_model = Xception(weights=None, include_top=False, input_shape=input_shape)
    # model = base_model.output
    predictions = Dense(action_space, activation="tanh")(model)

    actor = Model(inputs=X_input, outputs=predictions)
    actor.compile(loss="mse", optimizer=Adam(lr=0.001, decay=0.0001))
    return actor


def generateData():
    # batch_x, batch_y = [], []
    while True:
        files = os.listdir("data/")
        random.shuffle(files)
        for file in files:
            images, actions = np.load("data/" + file, allow_pickle=True)
            for image, action in zip(images, actions):
                image = processIMG(image)
                yield np.array(image), np.array(action)
                # batch_x, batch_y = [], []


def processIMG(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (img.shape[0], img.shape[1], 1)) / 255
    return img


def getData():
    batch_x, batch_y = [], []
    files = os.listdir("data/")
    print(len(files))
    fileNum = 1
    for file in os.listdir("data/"):
        images, actions = np.load("data/" + file, allow_pickle=True)
        for image, action in tqdm.tqdm(zip(images, actions), total=len(images)):
            image = image / 255.0
            cv2.imshow("image", autoEncoder.decode(image))
            cv2.waitKey(1)
            # image = processIMG(image)

            batch_x.append(image)
            batch_y.append(action)
        print(f"File {fileNum} of {len(files)}")
        fileNum += 1
    # print(batch_x[0].shape)
    return np.array(batch_x), np.array(batch_y)


def getProcessedData():
    batch_x = np.load("processData/datax.npy", allow_pickle=True)
    batch_y = np.load("processData/datay.npy", allow_pickle=True)
    print(batch_x.shape, batch_y.shape)
    return np.array(batch_x), np.array(batch_y)


def run():
    print("Loaded")

    # start_time = time.time()
    # test = learn.predict('g1-j5.png')
    # print("--- %imageSocket seconds ---" % (time.time() - start_time))
    # print(test)
    # X_train, y_train = np.load(file_name), np.load(file_name2)
    # X_train = X_train
    # X_train, y_train = shuffle(X_train, y_train)
    early_stop = EarlyStopping(monitor='loss', patience=3,
                               verbose=0, mode='min')
    checkpoint = ModelCheckpoint('checkpoints/' + "model" + '-{epoch:03d}-{loss:.4f}.h5',
                                 monitor='loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    trainX, trainY = getData()
    autoEncoder = AutoEncoder.AutoEncoder((128, 128, 3), 10, Config.numOutputs, "models/Sim-", True)
    trainX = autoEncoder.encodeList(trainX)
    model = buildModel(input_shape=(Config.numOutputs,), action_space=3)
    model.fit(trainX, trainY, batch_size=64, epochs=30, callbacks=[early_stop, checkpoint, tensorboard])
    model.save("model.h5")
    # dataset = tf.data.Dataset.from_generator(generateData, output_signature=(tf.type_spec_from_value(np.zeros(shape=(128, 128, 1))), tf.type_spec_from_value(np.zeros(shape=(3,)))))
    # dataset = dataset.shuffle(buffer_size=4096,
    #                           reshuffle_each_iteration=True)
    # dataset = dataset.repeat()
    # dataset = dataset.batch(64)#
    # model.fit(dataset, batch_size=64, epochs=100, steps_per_epoch=2000, callbacks=[early_stop, checkpoint])


if __name__ == '__main__':
    run()
