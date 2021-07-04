from __future__ import print_function

from keras import layers, Model
from keras.layers import Input
from keras.models import Sequential
from settings import GermanTrafficSigns as settings


def build_target(n_out):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    # Fully connected layer

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_out))

    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model.add(layers.Activation('softmax'))

    img = Input(shape=settings.IMG_SHAPE)
    validity = model(img)

    return Model(img, validity)
