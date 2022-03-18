import os
import pickle
import time

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, ELU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import random
import cv2
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.regularizers import l2
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.callbacks import TensorBoard

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


def label_func(x): return x.parent.name


def build_model(hp):  # random search passes this hyperparameter() object
    activations = ["relu", "tanh", "sigmoid", "softmax"]
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int('input_units',
                            min_value=32,
                            max_value=256,
                            step=32), (hp.Int('input_strides1', min_value=1, max_value=5, step=1), hp.Int('input_strides1', min_value=1, max_value=5)),
                     input_shape=(128, 128, 1)))

    model.add(Activation(hp.Choice(f'conv_input_activation', activations)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_conv_layers', 1, 10)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (hp.Int(f'conv_{i}_strides1', min_value=1, max_value=5), hp.Int(f'conv_{i}_strides2', min_value=1, max_value=5))))
        model.add(Activation(hp.Choice(f'conv_{i}_activation', activations)))
    model.add(Flatten())
    for i in range(hp.Int('n_dense_layers', 1, 10)):  # adding variation of layers.
        model.add(Dense(hp.Int(f'dense_{i}_units', 8, 512), activation=hp.Choice(f'dense{i}_activation', activations)))
    model.add(Dense(3))
    model.add(Activation(hp.Choice('output_activation', activations)))

    model.compile(optimizer="adam",
                  loss="mse",
                  metrics=["accuracy"])

    return model


def generateData(batch_size=128):
    while True:
        batch_x, batch_y = [], []
        for file in os.listdir("data/"):
            images, actions = np.load("data/" + file, allow_pickle=True)
            for image, action in zip(images, actions):
                batch_x.append(image)
                batch_y.append(action)
                if len(batch_x) == batch_size:
                    batch_x, batch_y = shuffle(batch_x, batch_y)
                    yield np.vstack(batch_x), np.vstack(batch_y)
                    batch_x, batch_y = [], []


def processIMG(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    return img


def getData():
    batch_x, batch_y = [], []
    for file in os.listdir("data/"):
        images, actions = np.load("data/" + file, allow_pickle=True)
        for image, action in zip(images, actions):
            image = processIMG(image)
            batch_x.append(image)
            batch_y.append(action)
    batch_x, batch_y = shuffle(batch_x, batch_y)
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
    checkpoint = ModelCheckpoint('checkpoints/' + "model" + '-{epoch:02d}-{loss:.4f}.h5',
                                 monitor='loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    trainX, trainY = getData()
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=30,  # how many model variations to test?
        executions_per_trial=3,  # how many trials per variation? (same model could perform differently)
        directory="logs")
    num_validation_samples = int(0.1 * len(trainX))
    testX = trainX[:num_validation_samples]
    testY = trainY[:num_validation_samples]
    trainX = trainX[num_validation_samples:]
    trainY = trainY[num_validation_samples:]
    tuner.search_space_summary()

    tuner.search(x=trainX,
                 y=trainY,
                 epochs=10,
                 batch_size=64,
                 callbacks=[tensorboard, early_stop],
                 validation_data=(testX, testY))

    tuner.results_summary()

    with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
        pickle.dump(tuner, f)

    tuner.search()
    print(tuner.get_best_hyperparameters()[0].values)
    tuner.get_best_models()[0].summary()
    tuner.results_summary()


if __name__ == '__main__':
    run()
