import numpy as np
from mnist_data import load_mnist
from matplotlib.pylab import plt

(x_train, y_train), (x_test, y_test) = load_mnist() # mnist_data.py에서 load_mnist 함수를 불러와서 실행하고, 그 결과값을 튜플에 담음

for i in range(10):
    img = x_train[i]
    label = np.argmax(y_train[i]) # 파라미터 안에서 제일 큰 값을 return하는게 argmax임 / 원핫 인코딩 반대로 하는거임
    print(label, end=', ')
    img = img.reshape(28,28)
    plt.subplot(1, 10, i + 1)
    plt.imshow(img)

print()
plt.show()
