import gzip
import pickle
import os
import numpy as np

files = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

def _load_img(filename):
    with gzip.open(filename, 'rb') as f: # rb = 이진파일 읽기 전용 모드 / 헥사코드로 출력됨
        data = np.frombuffer(f.read(), np.uint8, offset=16) # np.frombuffer = binary를 numpy float list로 변환
    data = data.reshape(-1, 784) # 열부터 채우고, 열이 다 차면, 다음 행을 만듦 / 행의 갯수는 모르고 열만 알때 사용
    return data

def _load_label(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10)) # size 행 / 10열의 0으로 채워진 배열
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T
# enumerate: 반복문 사용 시 몇 번째 반복문인지 확인이 필요할 때 사용 / 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환 
# 원 핫 인코딩: 표현하고 싶은 단어에만 1을 부여하고, 나머지는 0을 부여하는 단어의 벡터 표현 방식

def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    dataset = {}
    # 이미지랑 라벨 로드에서 dataset에 넣기
    for key in ('train_img', 'test_img'):
        dataset[key] = _load_img(files[key])
   
    for key in ('train_label', 'test_label'):
        dataset[key] = _load_label(files[key])
   
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32) # type을 float으로 바꾸고, 255로 나눠서 저장
            dataset[key] /= 255.0
   
    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = _change_one_hot_label(dataset[key])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28) # x * 1 * 28 * 28 = 784 
            # batch_size, width, height, channel / 배치 사이즈를 -1로 두면 자동으로 배치 사이즈를 조정함 
            # 파이프라인을 변경해서 배치사이즈를 변경해야하더라도 reshape의 batch 사이즈 크기를 안 바꿔도 된다고 함

    return ((dataset['train_img'], dataset['train_label']),
                (dataset['test_img'], dataset['test_label']))
   