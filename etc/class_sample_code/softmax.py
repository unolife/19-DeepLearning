import numpy as np

def softmax(x):
    # e = np.exp(x)
    e = np.exp(x - np.max(x))  # 입력값의 범위를 조절함으로써 해결
    s = np.sum(e) # 이게 무한대로 커지면 NaN이 뜸
    return e / s

print(softmax(np.array([4, 0, 2, 9])))
print(softmax(np.array([10, 10, 10, 10])))
print(softmax(np.array([0, 0, 0, 10])))
print(softmax(np.array([200, 1000, 100, 200])))