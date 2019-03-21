import numpy as np
import matplotlib.pylab as plt

# 시그모이드 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # np.exp() 함수는 밑(base)이 자연상수 e인 지수함수 y=e^x 로 변환

# test
print(sigmoid(1))
print(sigmoid(np.array([0,0.5,1])))

x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

# 단순 신경망 구조
def init_network():
    network = {}
    network['W'] = np.array([  # input 값들을 배열로 입력
        [0.2, 0.5, 0.3],
        [0.8, 0.6, 0.4]
    ])
    return network

def forward(network, x):
    y = sigmoid(np.dot(x, network['W'])) # 시그모이드까지 적용
    # y = np.dot(x, network['W'])   # NumPy를 이용해서 layer의 계산을 한번에 함
    return y

network = init_network()
y = forward(network, np.array([1.0, 2.0]))
print(y)