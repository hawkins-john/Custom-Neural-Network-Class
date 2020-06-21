"""
Benchmark Main
Author: John Hawkins
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import NeuralNetwork as nn
import tensorflow as tf
from utilities import *
import time

if __name__ == '__main__':

    np.random.seed(5)

    # load and preprocess redwine quality dataset
    X_train, Y_train, X_test, Y_test = load_wine_data()

    # load and preprocess cat image dataset
    #X_train, Y_train, X_test, Y_test, classes = load_cat_data()

    # build, train and predict using custom neural network model
    nn_start_time = time.time()
    network = nn.NeuralNetwork(hidden_layer_sizes=(20, 7, 5), learning_rate=0.0075, iterations=2500)
    network.train(X_train, Y_train, print_cost=False)
    nn_end_time = time.time()
    nn_predictions = network.predict(X_test)
    nn_accuracy = network.score(X_test, Y_test)

    # build, train and predict using Tensorflow Keras Sequential model
    tf.random.set_seed(1)
    seq_start_time = time.time()
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    network.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.0075),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics = ['accuracy'])
    network.fit(X_train.T, Y_train.T, epochs=2500, verbose=0)
    seq_end_time = time.time()
    seq_predictions = network.predict(X_test.T)
    seq_score = network.evaluate(X_test.T,  Y_test.T, verbose=0)

    # build, train and predict using scikit-learn MLPClassifier model
    mlp_start_time = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(20, 7, 5, 1), activation='relu', solver='sgd',
        alpha=0.0, learning_rate_init=0.0075, max_iter=2500, shuffle=False, random_state=1,
        verbose=False, momentum=0.0, n_iter_no_change=2500)
    clf.fit(X_train.T, np.ravel(Y_train.T))
    mlp_end_time = time.time()
    mlp_predictions = clf.predict(X_test.T)
    mlp_score = clf.score(X_test.T, np.ravel(Y_test.T))

    # output classifier results
    print("custom network confusion matrix:")
    print(confusion_matrix(Y_test[0], nn_predictions[0]))
    print("custom network classification report:")
    print(classification_report(Y_test[0], nn_predictions[0]))
    print("custom network training time (seconds):", nn_end_time - nn_start_time)

    print("tensorflow network confusion matrix:")
    print(confusion_matrix(Y_test[0], np.where(seq_predictions.T[0] > 0.5, 1, 0)))
    print("tensorflow network classification report:")
    print(classification_report(Y_test[0], np.where(seq_predictions.T[0] > 0.5, 1, 0)))
    print("tensorflow network training time (seconds):", seq_end_time - seq_start_time)

    print("scikit-learn network confusion matrix:")
    print(confusion_matrix(Y_test[0], mlp_predictions))
    print("scikit-learn network classification report:")
    print(classification_report(Y_test[0], mlp_predictions))
    print("scikit-learn network training time (seconds):", mlp_end_time - mlp_start_time)
