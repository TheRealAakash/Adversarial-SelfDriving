import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from settings.SimulationSettings import Config


class AutoEncoder:
    def __init__(self, shape, epochs, outputs, path, load_model=False):
        self.shape = shape
        # self.shape = x_train[0].shape
        # self.shape = (28, 28, 1)
        self.opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
        self.encoder = None
        self.autoencoder = None
        self.path = path
        self.epochs = epochs
        self.outputs = outputs

        self.buildModels(self.shape)
        if load_model:
            self.load()

    def setTrainable(self, trainable):
        self.encoder.trainable = trainable
        self.autoencoder.trainable = trainable

    def buildModels(self, shape):
        encoder_input = keras.Input(shape=shape, name="img")
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoder_input)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = keras.layers.Flatten()(x)
        encoder_output = keras.layers.Dense(self.outputs, activation="relu")(x)

        self.encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        decoder_input = keras.layers.Dense(2048, activation="relu")(encoder_output)  # 784
        x = keras.layers.Reshape((8, 8, 32))(decoder_input)
        x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoder_output = keras.layers.Conv2D(3, (3, 3), activation="relu", padding="same")(x)

        self.autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
        self.autoencoder.compile(self.opt, loss="mse")
        self.autoencoder.summary()

    def train(self, x_train):
        self.autoencoder.fit(x_train, x_train, epochs=self.epochs, batch_size=32, validation_split=0.1, shuffle=True)

    def encode(self, img):
        return self.encoder.predict([img.reshape((-1, self.shape[0], self.shape[1], self.shape[2]))])[0]

    def encodeList(self, imgs):
        imgs = np.array(imgs)
        return self.encoder.predict(imgs.reshape((-1, self.shape[0], self.shape[1], self.shape[2])))

    def decode(self, encoded):
        return self.autoencoder.predict([encoded.reshape(-1, self.shape[0], self.shape[1], self.shape[2])])[0]

    def save(self):
        self.encoder.save(self.path + "encoder.h5")
        self.autoencoder.save(self.path + "autoencoder.h5")

    def load(self):
        self.encoder = keras.models.load_model(self.path + "encoder.h5")
        self.autoencoder = keras.models.load_model(self.path + "autoencoder.h5")


def main():
    import trainingBehavior
    (x_train, y_train) = trainingBehavior.getData()
    autoEncoder = AutoEncoder(x_train[0].shape, 10, Config.numOutputs, "models/Sim-")
    autoEncoder.train(x_train)
    autoEncoder.save()
    cv2.imshow("img", x_train[0])
    cv2.imshow("ae_out", autoEncoder.decode(x_train[0]))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
