import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from tensorflow.keras.optimizers import Adam
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Input, Model
from settings import main_settings
import wandb
from wandb.keras import WandbCallback
from load_datasets import load_data_traffic_signs
from tensorflow import keras

settings, configs = main_settings.get_settings()
(X_train, y_train), (X_valid, y_valid), (X_test, y_test), signs = settings.DATA_LOADER()
n_train = X_train.shape[0]  # Number of training examples
n_test = X_test.shape[0]  # Number of testing examples
n_validation = X_valid.shape[0]  # Number of validation examples
n_classes = len(np.unique(y_train))  # Number of highres in dataset
EPOCHS = settings.target_model_epochs
BATCH_SIZE = settings.target_batch_size
SHOW_DATASET = False


def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            dataset: An np.array compatible with plt.imshow.
            dataset_y (Default = No label): A string to be used as a label for each image.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        indx = random.randint(0, len(dataset) - 1)
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap=cmap)
        font = {
            'color': 'black',
            'weight': 'normal',
            'size': 17,
        }
        plt.xlabel(signs[dataset_y[indx]], fontdict=font)
        # plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def histogram_plot(dataset, label):
    """
    Plots a histogram of the input data.
        Parameters:
            dataset: Input data to be plotted as a histogram.
            lanel: A string to be used as a label for the histogram.
    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()


def visualizeDataset():
    # Plotting sample examples, before pre-processing
    list_images(X_train, y_train, "Training example")
    # list_images(X_test, y_test, "Testing example")
    # list_images(X_valid, y_valid, "Validation example")
    # Show frequency of each label
    histogram_plot(y_train, "Training examples")
    histogram_plot(y_test, "Testing examples")
    histogram_plot(y_valid, "Validation examples")


def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


def image_normalize(image):
    """
    Normalize images to [0, 1] scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    image = np.divide(image, 255)
    return image


def preprocess(data):  # step 3
    data = (data * 2. / 255 - 1).reshape((len(data), settings.IMG_W, settings.IMG_H, settings.CHANNELS))  # pixel values in range [-1., 1.] for D
    return data
    # Sample images after greyscaling
    gray_images = list(map(gray_scale, data))
    # list_images(gray_images, y_train, "Gray Scale image", "gray")
    # Equalize images using skimage to improve contrast
    # Sample images after Local Histogram Equalization
    equalized_images = list(map(local_histo_equalize, gray_images))
    # list_images(equalized_images, y_train, "Equalized Image", "gray")

    # Normalize images
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    # list_images(normalized_images, y_train, "Normalized Image", "gray")
    normalized_images = normalized_images[..., None]
    return normalized_images


class KerasModel:
    def __init__(self, n_out=n_classes):
        self.n_out = n_out
        self.model = self.setup_model_keras()
        # self.model.summary()

    def inceptionModel(self):
        model = keras.applications.InceptionV3(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        model.compile()
        return model

    def setup_model_keras(self):
        model = self.build_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return model

    def build_model(self):
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
        model.add(layers.Dense(self.n_out))

        # model.add(Convolution2D(10,3,3, border_mode='same'))
        # model.add(GlobalAveragePooling2D())
        model.add(layers.Activation('softmax'))

        img = Input(shape=settings.IMG_SHAPE)
        validity = model(img)

        return Model(img, validity)

    def y_predict(self, X_data):
        predictions = []
        ps = self.model.predict(X_data)
        for p in ps:
            predictions.append(np.argmax(p))
        return np.array(predictions)

    def y_predict_prob(self, X_data):
        prob = self.model.predict(X_data)
        return prob

    def y_predict_topk_prob_and_pred(self, data, top_k=5):
        probs = self.model.predict(data)
        y_probs = []
        y_preds = []
        for prob in probs:
            y_pred = []
            y_prob = []
            for i in range(top_k):
                y_pred.append(np.argmax(prob))
                y_prob.append(prob[y_pred[-1]])
                prob[y_pred[-1]] = 0
            y_probs.append(y_prob)
            y_preds.append(y_pred)
        return np.array(y_probs), np.array(y_preds)

    def y_predict_topk(self, data, top_k=5):
        probs = self.model.predict(data)
        y_preds = []
        for prob in probs:
            y_pred = []
            for i in range(top_k):
                y_pred.append(np.argmax(prob))
                prob[y_pred[-1]] = 0
            y_preds.append(y_pred)
        return np.array(y_preds)

    def evaluate(self, X_data, y_data):
        score = self.model.evaluate(X_data, y_data)
        return score

    def load_model(self, path):
        self.model.load_weights(path)


def trainModelKeras(normalized_images):
    global X_train, y_train
    kerasModel = KerasModel(n_out=n_classes)

    # Validation set preprocessing
    X_valid_preprocessed = preprocess(X_valid)
    y_train_onehot = to_categorical(y_train, n_classes)
    y_valid_onehot = to_categorical(y_valid, n_classes)
    kerasModel.model.fit(normalized_images, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(X_valid_preprocessed, y_valid_onehot), callbacks=[WandbCallback()])
    kerasModel.model.save(f"{settings.models_dir}/{settings.model_name}", save_format='h5')
    return kerasModel


def showTestImagesWithLabels(test_data, test_labels, model, print_new_acc=False):
    new_test_images_preprocessed = preprocess(np.asarray(test_data))
    font = {
        'color': 'black',
        'weight': 'normal',
        'size': 20,
    }
    # get predictions
    y_prob, y_pred = model.y_predict_topk_prob_and_pred(new_test_images_preprocessed)
    # generate summary of results
    test_accuracy = 0
    for i in enumerate(new_test_images_preprocessed):
        accu = test_labels[i[0]] == np.asarray(y_pred[i[0]])[0]
        if accu == True:
            test_accuracy += 0.2
    if print_new_acc:
        print("New Images Test Accuracy = {:.1f}%".format(test_accuracy * 100))

    plt.figure(figsize=(15, 16))
    new_test_images_len = len(new_test_images_preprocessed)
    for i in range(new_test_images_len):
        plt.subplot(new_test_images_len * 2, 2, 2 * i * 2 + 1)
        plt.imshow(test_data[i])
        plt.title(signs[test_labels[i]], fontdict=font)
        #  plt.title(signs[y_pred[i][0]])
        plt.axis('off')
        plt.subplot(new_test_images_len, 2, 2 * i + 2)
        plt.barh(np.arange(1, 6, 1), y_prob[i, :] * 100)
        labels = [signs[j] for j in y_pred[i]]
        plt.tick_params(axis='y', labelsize=18)
        plt.yticks(np.arange(1, 6, 1), labels)

    plt.show()


def showTestImages(test_data, model):
    new_test_images_preprocessed = preprocess(np.asarray(test_data))
    # get predictions
    y_prob, y_pred = model.y_predict_topk_prob_and_pred(new_test_images_preprocessed)
    # generate summary of results
    plt.figure(figsize=(15, 16))
    new_test_images_len = len(new_test_images_preprocessed)
    for i in range(new_test_images_len):
        plt.subplot(new_test_images_len, 2, 2 * i + 1)
        plt.imshow(test_data[i])
        plt.title(signs[y_pred[i][0]])
        plt.axis('off')
        plt.subplot(new_test_images_len, 2, 2 * i + 2)
        plt.barh(np.arange(1, 6, 1), y_prob[i, :])
        labels = [signs[j] for j in y_pred[i]]
        plt.yticks(np.arange(1, 6, 1), labels)
    plt.show()


def main():
    global X_train, y_train

    # Step 3, preprocessing
    if SHOW_DATASET:
        visualizeDataset()
    # Randomize dataset to improve training, using sklearn
    X_train, y_train = shuffle(X_train, y_train)
    # X_train_normalized = preprocess(X_train)
    #  Step 4, training
    kerasModel = KerasModel(n_out=n_classes)
    kerasModel.load_model("models/TrafficSignRecognition-KerasModel.model")
    # kerasModel = trainModelKeras(X_train_normalized)

    # Step 5, testing
    # X_test_preprocessed = preprocess(X_test)
    # y_test_onehot = to_categorical(y_test, n_classes)

    # y_pred = kerasModel.y_predict(X_test_preprocessed)
    # print(sum(y_test == y_pred))
    # test_accuracy = sum(sum(y_test == y_pred)) / len(y_test)
    # print("Test Accuracy = {:.1f}%".format(test_accuracy * 100))

    # Show model results, and failures
    # cm = confusion_matrix(y_test, y_pred)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm = np.log(.0001 + cm)
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Log of normalized Confusion Matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()
    showTestImagesWithLabels(test_data=X_test[0:5], test_labels=y_train[0:5], model=kerasModel)
    # Step 6, testing new images(outside dataset)
    # new_test_images = []
    # path = 'datasets/traffic-signs-data/new_test_images/'
    # for image in os.listdir(path):
    #     img = cv2.imread(path + image)
    #     img = cv2.resize(img, (32, 32))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     new_test_images.append(img)
    # new_IDs = [13, 3, 14, 27, 17]
    # print("Number of new testing examples: ", len(new_test_images))

    # plt.figure(figsize=(15, 16))
    # for i in range(len(new_test_images)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(new_test_images[i])
    #     plt.xlabel(signs[new_IDs[i]])
    #     plt.ylabel("New testing image")
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.show()


#
# # New test data preprocessing
# showTestImagesWithLabels(new_test_images, new_IDs, kerasModel)


if __name__ == '__main__':
    # wandb.init(project='Adversarial-SelfDriving', entity='therealaakash', config=configs)
    main()
