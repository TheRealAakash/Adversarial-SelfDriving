from __future__ import print_function

from keras.layers import Conv2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, LeakyReLU


def build_discriminator(inputs):
    D = Conv2D(32, 4, strides=(2, 2))(inputs)
    D = LeakyReLU()(D)
    D = Dropout(0.4)(D)
    D = Conv2D(64, 4, strides=(2, 2))(D)
    D = BatchNormalization()(D)
    D = LeakyReLU()(D)
    D = Dropout(0.4)(D)
    D = Flatten()(D)
    D = Dense(64)(D)
    D = BatchNormalization()(D)
    D = LeakyReLU()(D)
    D = Dense(1, activation='sigmoid')(D)
    return D
