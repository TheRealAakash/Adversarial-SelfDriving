from __future__ import print_function

import os
import warnings

import cv2
import sklearn.utils
import tqdm

warnings.filterwarnings("ignore", module="matplotlib\..*")
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from numpy.random import seed
from discriminatorModel import build_discriminator
from generatorModel import build_generator, generator_loss
from settings.main_settings import get_settings
import trainBehavior
from carlaEnv import CarEnv
from tqdm.keras import TqdmCallback
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass
if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("misclass"):
    os.mkdir("misclass")

env_name = "CarEnv"
env = CarEnv(False)


def custom_acc(y_true, y_pred):
    return binary_accuracy(K.round(y_true), K.round(y_pred))


kerasModel = None
settings, config = get_settings()
agent = trainBehavior.PPOAgent(env, env_name, True, 50_000)
agent.load()

with open('frame_data.p') as f:
    x_train, y_train = pickle.load(f)

x_train, y_train = sklearn.utils.shuffle(x_train, y_train)


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
    # classifier.showTestImagesWithLabels(np.array(batch[:5], np.uint8), labels[:5], kerasModel)


# noinspection DuplicatedCode
class AdvGAN:
    def __init__(self):
        global kerasModel
        # input image dimensions

        optimizer_g = Adam(settings.AdvGAN_Discriminator_LR)
        optimizer_d = SGD(settings.AdvGAN_Discriminator_LR)

        inputs = Input(shape=x_train[0].shape)
        outputs = build_generator(inputs)
        self.generatorModel = Model(inputs, outputs)
        # self.generatorModel.summary()

        outputs = build_discriminator(self.generatorModel(inputs))
        self.discriminatorModel = Model(inputs, outputs)
        self.discriminatorModel.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer_d, metrics=[custom_acc])
        # self.discriminatorModel.summary()
        self.agent = agent
        self.kerasModel = self.agent.actor
        kerasModel = self.kerasModel
        self.targetModel = self.kerasModel
        self.targetModel.load_weights("CarEnv_PPO_Actor_orig.h5")
        # self.target.summary()

        self.stacked = Model(inputs=inputs,
                             outputs=[self.generatorModel(inputs), self.discriminatorModel(self.generatorModel(inputs)), self.targetModel(self.generatorModel(inputs))])
        self.stacked.compile(
            loss=[generator_loss, keras.losses.binary_crossentropy, keras.losses.mean_squared_error],
            optimizer=optimizer_g)
        print("")
        # self.stacked.summary()
        if settings.load:
            self.load()

    # build the cnn
    def get_batches(self, start, end, x_data, y_data):
        x_batch = x_data[start:end]
        x_batch = np.array(x_batch)
        Gx_batch = self.generatorModel.predict_on_batch(x_batch)
        y_batch = y_data[start:end]
        y_batch = np.array(y_batch)
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
            y_batch_cat = []
            for y in y_batch:
                y_batch_cat.append([.9, 0, 0])
        else:
            y_batch_cat = [1, 0, 0]
            # y_batch_cat = []
            # for label in np.array(y_batch).reshape(len(y_batch), ):
            #     if label in settings.targets:
            #         y_batch_cat.append(settings.targets[label])
            #     else:
            #         y_batch_cat.append(settings.target)
            # y_batch_cat = to_categorical(y_batch_cat, settings.N_CLASSES)
        # for each batch:
        # train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        # Update only G params
        # print(flipped_y_batch)
        x_batch = np.array(x_batch)
        y_batch_cat = np.array(y_batch_cat)
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)),
                                                             y_batch_cat])
        # stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(y_batch, settings.N_CLASSES)])
        # input to full GAN is original image
        # output 1 label for generated image is original image
        # output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        # output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
        return stacked_loss  # (total loss, hinge loss, gan loss, adv loss) tuple

    def load(self):
        #  self.stacked.load_weights(f"{settings.models_dir}/{settings.stacked_model_name}")
        self.generatorModel.load_weights(f"{settings.models_dir}/{settings.generator_model_name}")
        self.discriminatorModel.load_weights(f"{settings.models_dir}/{settings.discriminator_model_name}")

    def trainGAN(self):
        global x_train, y_train
        # x_train = np.array(x_train)[:6000]
        # y_train = np.array(y_train)[:6000]

        # self.targetModel.fit(x_train, to_categorical(y_train, settings.N_CLASSES), epochs=5)  # pretrain target

        epochs = settings.AdvGAN_epochs
        batch_size = settings.AdvGAN_batch_size
        num_batches = len(x_train) // batch_size

        if len(x_train) % batch_size != 0:
            num_batches += 1
        self.load()
        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch_index = 0

            for batch in tqdm.tqdm(range(num_batches - 1)):
                start = batch_size * batch_index
                end = batch_size * (batch_index + 1)
                batches = self.get_batches(start, end, x_train, y_train)
                self.train_D_on_batch(batches)
                self.train_stacked_on_batch(batches)
                batch_index += 1

            start = batch_size * batch_index
            end = len(x_train)
            x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)
            print(x_batch.shape)
            print(Gx_batch.shape)
            print(len(Gx_batch))
            print(len(x_batch))
            (d_loss, d_acc) = self.train_D_on_batch((x_batch, Gx_batch, y_batch))
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch((x_batch, Gx_batch, y_batch))

            target_acc = self.targetModel.test_on_batch(Gx_batch, y_batch)[1]
            target_predictions = self.targetModel.predict_on_batch(Gx_batch)  # (96,2)
            print("Predictions...")
            print(target_predictions)
            print("Gx_Batch")
            print(Gx_batch[0])
            print("X_Batch")
            print(x_batch[0])
            print("X_Batch diff")
            print(x_batch[0] - Gx_batch[0])
            cv2.imshow("Gx_Batch", cv2.resize(np.array(Gx_batch[0] * 255, np.uint8), dsize=(256, 256)))
            cv2.imshow("X_Batch", cv2.resize(np.array(x_batch[0] * 255, np.uint8), dsize=(256, 256)))
            print("Y action")
            print(y_batch[0])
            print("Target action")
            print(target_predictions[0])
            cv2.waitKey(1)

            # misclassified = np.where(y_batch.reshape((len(x_train) % batch_size,3)) != np.argmax(target_predictions, axis=1))[0]

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: "
                  "%f\tAccuracy:%.2f%% \n %d states" % (d_loss, d_acc * 100., gan_loss, hinge_loss, adv_loss, target_acc * 100., len(x_train)))
            # x_new, y_new = self.agent.testPerturbed(self.generatorModel, 5)
            # x_train.extend(x_new)
            # y_train.extend(y_new)
            # x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
            # x_train = x_train[:10_000]
            # y_train = y_train[:10_000]
            # if epoch == 0:
            #     save_generated_images("orig", x_batch, 'images')
            # if epoch % 1 == 0:
            #     save_generated_images(str(epoch), Gx_batch, 'images')
            #     save_generated_images(str(epoch), Gx_batch[misclassified], 'misclass')
            #     showComps(Gx_batch[misclassified], y_batch.reshape((len(x_train) % batch_size,))[misclassified])
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

        target_acc = self.targetModel.test_on_batch(Gx_batch, y_batch)[1]
        target_predictions = self.targetModel.predict_on_batch(Gx_batch)  # (96,2)
        # misclassified = np.where(y_batch.reshape((len(x_train) % batch_size,)) != np.argmax(target_predictions, axis=1))[0]

        print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: "
              "%f\tAccuracy:%.2f%% \n %d states" % (d_loss, d_acc * 100., gan_loss, hinge_loss, adv_loss, target_acc * 100., len(x_train)))
        self.agent.testPerturbed(self.generatorModel)

        # save_generated_images(str("Test-images"), Gx_batch, 'images')
        # save_generated_images(str("Test-images"), Gx_batch[misclassified], 'misclass')
        # showComps(Gx_batch[misclassified], y_batch.reshape((len(x_train) % batch_size,))[misclassified])

    def genImages(self):
        print("Generating...")
        start = 0
        end = len(x_train)
        x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)
        np.save(f"{settings.images_name}.npy", (Gx_batch, y_batch))


if __name__ == '__main__':
    seed(1)
    tf.random.set_seed(1)
    dcgan = AdvGAN()
    if settings.mode == "train":
        dcgan.trainGAN()
    elif settings.mode == "test":
        dcgan.testGAN()
    elif settings.mode == "gen":
        dcgan.genImages()
