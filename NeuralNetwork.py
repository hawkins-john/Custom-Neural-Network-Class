"""
Neural Network Class
Author: John Hawkins
Description: Vanilla feed forward neural network classifier with standard back propagation and gradient descent weight update. "ReLu" activations are used
on the hidden layers and a "Sigmoid" activation is used on the output layer.
Citations: This code adapts and extends the core concepts and helper functions taught in the deeplearning.ai Coursera "Neural Networks and Deep Learning" Course.
"""

import numpy as np
import h5py
from utilities import *


class NeuralNetwork:

    def __init__(self, hidden_layer_sizes = (50,), activation = 'relu', learning_rate = 0.0001, iterations = 100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.parameters = {}

    def train(self, X_train, Y_train, print_cost = False):
        # call function to initialize weight and bias parameters for each layer
        self.parameters = self.initialize_parameters(X_train.shape[0], self.hidden_layer_sizes, 1)
        # step through each iteration up to self.iterations
        for i in range(0, self.iterations):
            # call function to perform forward propagation through each layer of network to calculate activations and store paramaters in cache
            A_output, caches = self.forward_propagation(X_train, self.parameters)
            # call function to compute the cross-entropy cost on loss function
            cost = self.compute_cost(A_output, Y_train)
            # call function to perform backward propagation through each layer of network to calculate gradients
            gradients = self.backward_propagation(A_output, Y_train, caches)
            # call function to update the weight and bias parameters using gradients and self.learning_rate
            self.parameters = self.update_parameters(self.parameters, gradients, self.learning_rate)
            # if print equals True then print the cost every iteration
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
        return

    def predict(self, X_test):
        probs, caches = self.forward_propagation(X_test, self.parameters)
        predictions = np.where(probs <= 0.5, 0, 1)
        return predictions

    def score(self, X_test, Y_test):
        probs, caches = self.forward_propagation(X_test, self.parameters)
        predictions = np.where(probs <= 0.5, 0, 1)
        accuracy = np.sum((predictions == Y_test) / X_test.shape[1])
        return accuracy

    @staticmethod
    def initialize_parameters(input_layer_size, hidden_layer_sizes, output_layer_size):
        layer_sizes = (input_layer_size,) + hidden_layer_sizes + (output_layer_size,)
        parameters = {}
        for i in range(1, len(layer_sizes)):
            parameters['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) / np.sqrt(layer_sizes[i-1])
            parameters['b' + str(i)] = np.zeros((layer_sizes[i], 1))
        return parameters

    @staticmethod
    def forward_propagation(X_train, parameters):
        caches = []
        layers = len(parameters) // 2
        A = X_train
        # calculate activations through hidden layers using "ReLu" activation function
        for i in range(layers - 1):
            A_prev = A
            W = parameters['W' + str(i + 1)]
            b = parameters['b' + str(i + 1)]
            Z = np.dot(W, A_prev) + b
            linear_cache = (A_prev, W, b)
            A = np.maximum(0,Z)
            activation_cache = Z
            caches.append((linear_cache, activation_cache))
        # calculate activation on output layer using "Sigmoid" activation function
        W = parameters['W' + str(layers)]
        b = parameters['b' + str(layers)]
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        A_output = 1 / (1 + np.exp(-Z))
        activation_cache = Z
        caches.append((linear_cache, activation_cache))
        return A_output, caches

    @staticmethod
    def compute_cost(A_output, Y_train):
        cost = -(1 / Y_train.shape[1]) * np.sum(Y_train * np.log(A_output)+(1 - Y_train) * np.log(1 - A_output))
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def backward_propagation(A_output, Y_train, caches):
        gradients = {}
        layers = len(caches)
        Y_train = Y_train.reshape(A_output.shape)
        # initialize back propagation
        dA_output = - (np.divide(Y_train, A_output) - np.divide(1 - Y_train, 1 - A_output))
        # calculate gradients on last layer
        current_cache = caches[layers - 1]
        s = 1 / (1 + np.exp(-current_cache[1]))
        dZ = dA_output * s * (1 - s)
        A_prev, W, b = current_cache[0]
        m = A_prev.shape[1]
        gradients["dW" + str(layers)] = (1 / m) * np.dot(dZ, A_prev.T)
        gradients["db" + str(layers)] = (1 / m) * np.sum(dZ, axis = -1, keepdims = True)
        gradients["dA" + str(layers - 1)] = np.dot(W.T, dZ)
        # loop through remaining layers and calculate gradients
        for i in reversed(range(layers - 1)):
            current_cache = caches[i]
            dZ = np.array(gradients["dA" + str(i + 1)], copy=True)
            dZ[current_cache[1] <= 0] = 0
            A_prev, W, b = current_cache[0]
            m = A_prev.shape[1]
            gradients["dW" + str(i + 1)] = (1 / m) * np.dot(dZ, A_prev.T)
            gradients["db" + str(i + 1)] = (1 / m) * np.sum(dZ, axis = -1, keepdims = True)
            gradients["dA" + str(i)] = np.dot(W.T, dZ)
        return gradients

    @staticmethod
    def update_parameters(parameters, gradients, learning_rate):
        layers = len(parameters) // 2
        for i in range(layers):
            parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * gradients["dW" + str(i + 1)]
            parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * gradients["db" + str(i + 1)]
        return parameters


if __name__ == '__main__':

    np.random.seed(1)
    train_x, train_y, test_x, test_y, classes = load_cat_data()
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    m_train = train_x.shape[0]
    num_px = train_x.shape[1]
    m_test = test_x.shape[0]
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x.shape))
    print ("test_y shape: " + str(test_y.shape))

    network = NeuralNetwork(hidden_layer_sizes = (20, 7, 5), learning_rate = 0.0075, iterations = 2500)
    #network = NeuralNetwork(hidden_layer_sizes = (7,), learning_rate = 0.0075, iterations = 2500)
    network.train(train_x, train_y, print_cost = 'True')
    accuracy = network.score(test_x, test_y)
    print(accuracy)
