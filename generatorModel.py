from __future__ import print_function
from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from settings import main_settings

settings, configs = main_settings.get_settings()


def generator_loss(y_true, y_pred):
    return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - settings.c, 0), axis=-1)
    # ||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3


def build_generator(inputs):
    # c3s1-8
    G = Conv2D(8, 3, padding='same')(inputs)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    # d16
    G = Conv2D(16, 3, strides=(2, 2), padding='same')(G)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    # d32
    G = Conv2D(32, 3, strides=(2, 2), padding='same')(G)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    residual = G
    # four r32 blocks
    for _ in range(4):
        G = Conv2D(32, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)
        G = Conv2D(32, 3, padding='same')(G)
        G = BatchNormalization()(G)
        G = layers.add([G, residual])
        residual = G

    # u16
    G = Conv2DTranspose(16, 3, strides=(2, 2), padding='same')(G)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    # u8
    G = Conv2DTranspose(8, 3, strides=(2, 2), padding='same')(G)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    # c3s1-3
    G = Conv2D(1, 3, padding='same')(G)
    G = InstanceNormalization()(G)
    G = Activation('relu')(G)
    G = layers.add([G, inputs])
    return G
