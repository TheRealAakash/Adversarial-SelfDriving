from __future__ import print_function

import os
import random
import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.set_random_seed(1)
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from numpy.random import seed

import Classification
from discriminatorModel import build_discriminator
from generatorModel import build_generator, generator_loss
from settings import settings
import sys

base_path = os.path.dirname(__file__)

if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("misclass"):
    os.mkdir("misclass")


def custom_acc(y_true, y_pred):
    return binary_accuracy(K.round(y_true), K.round(y_pred))


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

kerasModel = None
(x_train, y_train), (x_test, y_test), (_, _), _ = settings.DATA_LOADER()
x_train = (x_train * 2. / 255 - 1).reshape((len(x_train), settings.IMG_W, settings.IMG_H, settings.CHANNELS))  # pixel values in range [-1., 1.] for D


def save_generated_images(filename, batch, directory):
    batch = batch.reshape(batch.shape[0], settings.IMG_W, settings.IMG_H, settings.CHANNELS)
    batch = (batch + 1) / 2.
    batch.clip(0, 1)
    rows, columns = 5, 5

    fig, axs = plt.subplots(rows, columns)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            if cnt < len(batch):
                axs[i, j].imshow(batch[cnt], interpolation='nearest', cmap='gray')
                cnt += 1
            axs[i, j].axis('off')
    fig.savefig("%s/%s.png" % (directory, filename))
    plt.close()


def showComps(batch, labels):
    batch = (batch + 1) / 2.
    batch = batch * 255
    batch.clip(0, 255)
    x = random.randint(0, len(batch) - 5)
    add = random.randint(1, 6)
    Classification.showTestImagesWithLabels(np.array(batch[x:x + add], np.uint8), labels[x:x + add], kerasModel)


# noinspection DuplicatedCode
class DCGAN:
    def __init__(self):
        global kerasModel
        # input image dimensions

        optimizer_g = Adam(settings.AdvGAN_Discriminator_LR)
        optimizer_d = SGD(settings.AdvGAN_Discriminator_LR)
        print(settings.IMG_SHAPE)
        inputs = Input(shape=x_train[0].shape)
        outputs = build_generator(inputs)
        self.generatorModel = Model(inputs, outputs)
        self.generatorModel.summary()

        outputs = build_discriminator(self.generatorModel(inputs))
        self.discriminatorModel = Model(inputs, outputs)
        self.discriminatorModel.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer_d, metrics=[custom_acc])
        # self.discriminatorModel.summary()

        self.kerasModel = Classification.KerasModel(n_out=settings.N_CLASSES)
        kerasModel = self.kerasModel
        self.targetModel = self.kerasModel.model
        self.targetModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                 metrics=['accuracy'])
        if settings.LOAD_TARGET_WEIGHTS:
            self.targetModel.load_weights(f"models/{settings.model_name}")
        # self.target.summary()

        self.stacked = Model(inputs=inputs,
                             outputs=[self.generatorModel(inputs), self.discriminatorModel(self.generatorModel(inputs)), self.targetModel(self.generatorModel(inputs))])
        self.stacked.compile(
            loss=[generator_loss, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy],
            optimizer=optimizer_g)
        # self.stacked.summary()
        if settings.load_generator or settings.mode != "train":
            self.generatorModel.load_weights(f"{settings.models_dir}/{settings.generator_model_name}")
        if settings.load_discriminator or settings.mode != "train":
            self.discriminatorModel.load_weights(f"{settings.models_dir}/{settings.discriminator_model_name}")

    # build the cnn
    def get_batches(self, start, end, x_data, y_data):
        x_batch = x_data[start:end]
        Gx_batch = self.generatorModel.predict_on_batch(x_batch)
        y_batch = y_data[start:end]
        return x_batch, Gx_batch, y_batch

    def train_D_on_batch(self, batches):
        x_batch, Gx_batch, _ = batches

        # for each batch:
        # predict noise on generator: G(z) = batch of fake images
        # train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        # train real images on discriminator: D(x) = update D params per classification for real images

        # Update D params
        self.discriminatorModel.trainable = True
        d_loss_real = self.discriminatorModel.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)))  # real=1, positive label smoothing
        d_loss_fake = self.discriminatorModel.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)))  # fake=0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        return d_loss  # (loss, accuracy) tuple

    def train_stacked_on_batch(self, batches):
        x_batch, _, y_batch = batches
        self.discriminatorModel.trainable = False
        self.targetModel.trainable = False
        if not settings.targeted:
            y_batch_cat = 1. - to_categorical(y_batch, settings.N_CLASSES)
        else:
            y_batch_cat = []
            for label in np.array(y_batch).reshape(len(y_batch), ):
                if label in settings.targets:
                    y_batch_cat.append(settings.targets[label])
                else:
                    y_batch_cat.append(settings.target)
            y_batch_cat = to_categorical(y_batch_cat, settings.N_CLASSES)
        # for each batch:
        # train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        # Update only G params
        # print(flipped_y_batch)
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)),
                                                             y_batch_cat])
        # stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(y_batch, settings.N_CLASSES)])
        # input to full GAN is original image
        # output 1 label for generated image is original image
        # output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        # output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
        return stacked_loss  # (total loss, hinge loss, gan loss, adv loss) tuple

    def trainGAN(self):
        # x_train = np.array(x_train)[:6000]
        # y_train = np.array(y_train)[:6000]

        # self.targetModel.fit(x_train, to_categorical(y_train, settings.N_CLASSES), epochs=5)  # pretrain target

        epochs = settings.AdvGAN_epochs
        batch_size = settings.AdvGAN_batch_size
        num_batches = len(x_train) // batch_size
        if len(x_train) % batch_size != 0:
            num_batches += 1

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch_index = 0

            for batch in range(num_batches - 1):
                start = batch_size * batch_index
                end = batch_size * (batch_index + 1)
                batches = self.get_batches(start, end, x_train, y_train)
                self.train_D_on_batch(batches)
                self.train_stacked_on_batch(batches)
                batch_index += 1

            start = batch_size * batch_index
            end = len(x_train)
            x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)

            (d_loss, d_acc) = self.train_D_on_batch((x_batch, Gx_batch, y_batch))
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch((x_batch, Gx_batch, y_batch))

            target_acc = self.targetModel.test_on_batch(Gx_batch, to_categorical(y_batch, settings.N_CLASSES))[1]
            target_predictions = self.targetModel.predict_on_batch(Gx_batch)  # (96,2)

            misclassified = np.where(y_batch.reshape((len(x_train) % batch_size,)) != np.argmax(target_predictions, axis=1))[0]

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: "
                  "%f\tAccuracy:%.2f%%" % (d_loss, d_acc * 100., gan_loss, hinge_loss, adv_loss, target_acc * 100.))

            if epoch == 0:
                save_generated_images("orig", x_batch, 'images')
            if epoch % 1 == 0:
                save_generated_images(str(epoch), Gx_batch, 'images')
                save_generated_images(str(epoch), Gx_batch[misclassified], 'misclass')
                showComps(Gx_batch[misclassified], y_batch.reshape((len(x_train) % batch_size,))[misclassified])
            self.generatorModel.save(f"{settings.models_dir}/{settings.generator_model_name}", save_format="h5")
            self.discriminatorModel.trainable = True
            self.discriminatorModel.save(f"{settings.models_dir}/{settings.discriminator_model_name}", save_format="h5")
            self.discriminatorModel.trainable = False

    def testGAN(self):
        # x_train = np.array(x_train)[:6000]
        # y_train = np.array(y_train)[:6000]

        # self.targetModel.fit(x_train, to_categorical(y_train, settings.N_CLASSES), epochs=5)  # pretrain target

        batch_size = settings.AdvGAN_batch_size
        num_batches = len(x_train) // batch_size
        if len(x_train) % batch_size != 0:
            num_batches += 1

        print("Testing...")
        batch_index = 0
        start = batch_size * batch_index
        end = len(x_train)
        x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)

        (d_loss, d_acc) = self.train_D_on_batch((x_batch, Gx_batch, y_batch))
        (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch((x_batch, Gx_batch, y_batch))

        target_acc = self.targetModel.test_on_batch(Gx_batch, to_categorical(y_batch, settings.N_CLASSES))[1]
        target_predictions = self.targetModel.predict_on_batch(Gx_batch)  # (96,2)

        misclassified = np.where(y_batch.reshape((len(x_train) % batch_size,)) != np.argmax(target_predictions, axis=1))[0]

        print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: "
              "%f\tAccuracy:%.2f%%" % (d_loss, d_acc * 100., gan_loss, hinge_loss, adv_loss, target_acc * 100.))

        save_generated_images(str("Test-images"), Gx_batch, 'images')
        save_generated_images(str("Test-images"), Gx_batch[misclassified], 'misclass')
        showComps(Gx_batch[misclassified], y_batch.reshape((len(x_train) % batch_size,))[misclassified])
        np.save("data.npy", [Gx_batch, y_batch])

    def genImages(self):
        print("Generating...")
        start = 0
        end = len(x_train)
        x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)
        np.save(f"{settings.images_name}.npy", (Gx_batch, y_batch))


if __name__ == '__main__':
    tf.compat.v1.set_random_seed(1)
    seed(1)
    dcgan = DCGAN()
    if settings.mode == "train":
        dcgan.trainGAN()
    elif settings.mode == "test":
        dcgan.testGAN()
    elif settings.mode == "gen":
        dcgan.genImages()
