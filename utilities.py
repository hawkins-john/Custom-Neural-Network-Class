"""
Utility Functions
Author: John Hawkins
"""

import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
import h5py


# load and preprocess red wine data
def load_wine_data():
    # load data and transform quality into binary label
    redwine = pd.read_csv('./datasets/wine-quality/winequality-red.csv')
    redwine.loc[redwine.quality < 5.5, 'quality'] = 0
    redwine.loc[redwine['quality'] > 5.5, 'quality'] = 1

    # partition data into samples and labels
    redwineX = redwine.drop('quality',1).copy().values
    redwineY = redwine['quality'].copy().values

    # scale all sample feature data
    scaler = StandardScaler()
    scaler.fit(redwineX)
    redwineX = scaler.transform(redwineX)

    # split data in train and test sets
    X_train, X_test, Y_train, Y_test = ms.train_test_split(redwineX, redwineY, test_size = 0.2, random_state = 0, stratify = redwineY)
    Y_train = np.reshape(Y_train, (1, Y_train.shape[0]))
    Y_test = np.reshape(Y_test, (1, Y_test.shape[0]))
    X_train = X_train.T
    X_test = X_test.T

    return X_train, Y_train, X_test, Y_test


# load and preprocess cat data set
def load_cat_data():
    # load training data and partition samples and labels
    train_dataset = h5py.File('datasets/cats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    # load test data and partition samples and labels
    test_dataset = h5py.File('datasets/cats/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    # identify classes
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    # reshape label data
    Y_train = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    Y_test = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # reshape the training and test samples
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # standardize data to have feature values between 0 and 1.
    X_train = train_x_flatten/255.
    X_test = test_x_flatten/255.

    return X_train, Y_train, X_test, Y_test, classes
