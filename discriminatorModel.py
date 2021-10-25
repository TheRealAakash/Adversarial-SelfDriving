from __future__ import print_function

from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from collections import defaultdict

import visualkeras

from PIL import ImageFont
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model


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

if __name__ == '__main__':
    font = ImageFont.truetype("times.ttf", 18)  # using comic sans is strictly prohibited!
    color_map = defaultdict(dict)
    color_map[Dense]['fill'] = 'grey'
    inputs = Input(shape=(32, 32, 3))
    model = build_discriminator(inputs)
    model = Model(inputs, model)
    img = visualkeras.layered_view(model, color_map=color_map, legend=True, font=font)
    img.show()
    img.save("discriminatorArch.png")