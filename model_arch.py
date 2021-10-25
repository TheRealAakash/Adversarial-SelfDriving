from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


def model_conv_64x3(X_input):
    model = Conv2D(64, (3, 3), padding='same', activation="relu")(X_input)
    model = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(model)
    model = Conv2D(64, (3, 3), padding='same', activation="relu")(model)
    model = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(model)
    model = Conv2D(64, (3, 3), padding='same', activation="relu")(model)
    model = MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(model)
    model = Flatten()(model)
    return model


def model_conv_256_512(X_input):
    model = Conv2D(128, (3, 3), padding='same', activation="relu")(X_input)
    model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
    model = Conv2D(256, (3, 3), padding='same', activation="relu")(model)
    model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
    model = Conv2D(512, (3, 3), padding='same', activation="relu")(model)
    model = MaxPooling2D(pool_size=(2, 2), padding='same')(model)
    model = Dropout(0.2)(model)
    model = Flatten()(model)
    model = Dense(512, activation="relu", kernel_initializer='he_uniform')(model)
    model = Dense(256, activation="relu", kernel_initializer='he_uniform')(model)
    model = Dense(64, activation="relu", kernel_initializer='he_uniform')(model)
    return model
