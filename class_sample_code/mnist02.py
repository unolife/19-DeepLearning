import numpy as np
from mnist_data import load_mnist
from functions import sigmoid, softmax

layers = [784, 20, 10, 10]

def init_network(): # 가중치와 편향값(weight and bias) 설정
    network = {}
    network['W1'] = 0.01 * np.random.randn(layers[0], layers[1]) # 784행 20열 짜리 난수가 생김
    network['W2'] = 0.01 * np.random.randn(layers[1], layers[2]) # 20,10
    network['W3'] = 0.01 * np.random.randn(layers[2], layers[3]) # 10,10
    network['b1'] = np.zeros(layers[1])
    network['b2'] = np.zeros(layers[2])
    network['b3'] = np.zeros(layers[3])
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    x1 = sigmoid(np.dot(x, W1) + b1) # 각각 변수에 담은 후에, dot 연산하고 bias 더해서 sigmoid 돌리면 다음 층으로 이동함
    x2 = sigmoid(np.dot(x1, W2) + b2)
    x3 = np.dot(x2, W3) + b3
    y = softmax(x3)
    return y

def accuracy(network, x, t):
    y = predict(network, x) # 뉴럴넷 돌림
    y = np.argmax(y, axis=1) # 그 결과를 디코딩해서 숫자만 뽑음
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) /  float(x.shape[0])  # y랑 t랑 같은 애들만 더해서 행의 개수(x.shape[0])로 나눔
    return accuracy # 전체 행중에 둘이 같은게 몇개나 있는지 구함

(x_train, y_train), (x_test, y_test) = load_mnist()
network = init_network()
print(accuracy(network, x_train, y_train))