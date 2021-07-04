import csv
import pickle
from sklearn.utils import shuffle
from keras.datasets import mnist, cifar10
import fiftyone as fo
import cv2
import numpy as np
from sklearn.utils import shuffle
import skimage.io


# Step 1, Load data
def load_data_traffic_signs():
    training_file = "datasets/traffic-signs-data/train.p"
    validation_file = "datasets/traffic-signs-data/valid.p"
    testing_file = "datasets/traffic-signs-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    signs = []
    with open('datasets/traffic-signs-data/signnames.csv', 'r') as csvfile:
        signnames = csv.reader(csvfile, delimiter=',')
        next(signnames, None)
        for row in signnames:
            signs.append(row[1])
        csvfile.close()

    # Step 2, dataset info

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    X_test, y_test = shuffle(X_test, y_test)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), signs


def load_data_cifar10():
    signs = []
    with open('datasets/cifar10/classnames.csv', 'r') as csvfile:
        signnames = csv.reader(csvfile, delimiter=',')
        next(signnames, None)
        for row in signnames:
            signs.append(row[1])
        csvfile.close()

    # Step 2, dataset info
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return (X_train, y_train), (X_test, y_test), (X_test, y_test), signs


def load_data_high_res():
    classes = []
    with open('datasets/HighRes/categories.csv', 'r') as csvfile:
        classnames = csv.reader(csvfile, delimiter=',')
        next(classnames, None)
        for row in classnames:
            classes.append(row[1])
        csvfile.close()
    X_train = []
    y_train = []
    with open("datasets/HighRes/images.csv", 'r') as csvfile:
        imageNames = csv.reader(csvfile, delimiter=',')
        next(imageNames, None)
        for row in imageNames:
            X_train.append(skimage.io.imread(f"datasets/HighRes/images/{row[0]}.png"))
            y_train.append(row[6])
        csvfile.close()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train, y_train = shuffle(X_train, y_train)
    return (X_train, y_train), (X_train, y_train), (X_train, y_train), classes


