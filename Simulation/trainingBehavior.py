import os

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, ELU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import random
import cv2
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.keras.regularizers import l2

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


def build_model(img_shape, classes):
    # input_image = keras.layers.Input(shape=(224, 224, 3))
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=img_shape,
                     output_shape=img_shape))
    INIT = 'glorot_uniform'  # 'he_normal', glorot_uniform
    keep_prob = 0.2
    reg_val = 0.01
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid", kernel_initializer=INIT, kernel_regularizer=l2(reg_val)))
    # W_regularizer=l2(reg_val)
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid", kernel_initializer=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid", kernel_initializer=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="valid", kernel_initializer=INIT))
    model.add(ELU())
    model.add(Dropout(keep_prob))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(classes))
    model.compile(optimizer=Adam(), loss="mse", metrics=['accuracy'])
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


def getData():
    batch_x, batch_y = [], []
    for file in os.listdir("data/"):
        images, actions = np.load("data/" + file, allow_pickle=True)
        for image, action in zip(images, actions):
            batch_x.append(image)
            batch_y.append(action)
    batch_x, batch_y = shuffle(batch_x, batch_y)
    return np.array(batch_x), np.array(batch_y)


def run():
    print("Loaded")

    # start_time = time.time()
    # test = learn.predict('g1-j5.png')
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(test)
    # X_train, y_train = np.load(file_name), np.load(file_name2)
    # X_train = X_train
    # X_train, y_train = shuffle(X_train, y_train)
    # early_stop = EarlyStopping(monitor='loss', patience=15,
    #                            verbose=0, mode='min')
    checkpoint = ModelCheckpoint('checkpoints/' + "model" + '-{epoch:02d}-{loss:.4f}.h5',
                                 monitor='loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    model = build_model((128, 128, 3), 3)
    trainX, trainY = getData()
    model.fit(trainX, trainY, epochs=50, callbacks=[checkpoint], verbose=1)
    model.save("model.h5")
    import TrainedAgent


if __name__ == '__main__':
    run()
