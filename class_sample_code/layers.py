import sys, os
from lib.functions import sigmoid, softmax, cross_entropy_error, mean_squared_error
import numpy as np

class Dense:
    def __init__(self, input_size, output_size, initializer='random'):
        self.W = 0.1 * np.random.randn(input_size, output_size)
        self.b = 0.1 * np.zeros(output_size)
        self.x = None
        self.y = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b
        return self.y

    def backward(self, d_out, learning_rate):
        self.dW = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis = 0)
        d_x = np.dot(d_out, self.W.T)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return d_x

class SoftmaxWithLoss:
    def __init__(self):
        self.error = None
        