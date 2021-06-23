import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from keras.optimizers import Adam
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.utils import np_utils

# Step 1, Load data


training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

signs = []
with open('signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames, None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

# Step 2, dataset info

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
n_train = X_train.shape[0]  # Number of training examples
n_test = X_test.shape[0]  # Number of testing examples
n_validation = X_valid.shape[0]  # Number of validation examples
n_classes = len(np.unique(y_train))  # Number of classes in dataset

EPOCHS = 15
BATCH_SIZE = 64
SHOW_DATASET = False
model_name = "KerasModel"
save_dir = "models"


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
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
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
    # list_images(X_train, y_train, "Training example")
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


class Model:
    def __init__(self, n_out=n_classes):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
        self.model.add(Activation('relu'))
        BatchNormalization(axis=-1)
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        BatchNormalization(axis=-1)
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        BatchNormalization(axis=-1)
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        # Fully connected layer

        BatchNormalization()
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        BatchNormalization()
        self.model.add(Dropout(0.2))
        self.model.add(Dense(n_out))

        # model.add(Convolution2D(10,3,3, border_mode='same'))
        # model.add(GlobalAveragePooling2D())
        self.model.add(Activation('softmax'))

        # self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    def y_predict(self, X_data):
        predictions = self.model.predict_classes(X_data)
        return predictions

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
    kerasModel = Model(n_out=n_classes)

    # Validation set preprocessing
    X_valid_preprocessed = preprocess(X_valid)
    y_train_onehot = np_utils.to_categorical(y_train, n_classes)
    y_valid_onehot = np_utils.to_categorical(y_valid, n_classes)
    kerasModel.model.fit(normalized_images, y_train_onehot, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(X_valid_preprocessed, y_valid_onehot), verbose=0)
    kerasModel.model.save(f"{save_dir}/TrafficSignRecognition-{model_name}.model")
    return kerasModel


def showTestImagesWithLabels(test_data, test_labels, model):
    new_test_images_preprocessed = preprocess(np.asarray(test_data))

    # get predictions
    y_prob, y_pred = model.y_predict_topk_prob_and_pred(new_test_images_preprocessed)
    # generate summary of results
    test_accuracy = 0
    for i in enumerate(new_test_images_preprocessed):
        accu = test_labels[i[0]] == np.asarray(y_pred[i[0]])[0]
        if accu == True:
            test_accuracy += 0.2
    print("New Images Test Accuracy = {:.1f}%".format(test_accuracy * 100))

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
    X_train_normalized = preprocess(X_train)
    #  Step 4, training
    kerasModel = trainModelKeras(X_train_normalized)

    # Step 5, testing
    X_test_preprocessed = preprocess(X_test)
    y_test_onehot = np_utils.to_categorical(y_test, n_classes)

    y_pred = kerasModel.y_predict(X_test_preprocessed)
    test_accuracy = sum(y_test == y_pred) / len(y_test)
    print("Test Accuracy = {:.1f}%".format(test_accuracy * 100))

    # Show model results, and failures
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.log(.0001 + cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Log of normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Step 6, testing new images(outside dataset)
    new_test_images = []
    path = './traffic-signs-data/new_test_images/'
    for image in os.listdir(path):
        img = cv2.imread(path + image)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_test_images.append(img)
    new_IDs = [13, 3, 14, 27, 17]
    print("Number of new testing examples: ", len(new_test_images))

    plt.figure(figsize=(15, 16))
    for i in range(len(new_test_images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(new_test_images[i])
        plt.xlabel(signs[new_IDs[i]])
        plt.ylabel("New testing image")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

    # New test data preprocessing
    showTestImagesWithLabels(new_test_images, new_IDs, kerasModel)


if __name__ == '__main__':
    main()
