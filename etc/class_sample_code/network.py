# coding: utf-8
import sys, os
import math
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, os.pardir))
import numpy as np
from mnist02 import load_mnist
from layers import Dense, Relu, SoftmaxWithLoss
from matplotlib.pylab import plt

class Network1:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, t):
        return self.layers[-1].loss(t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)