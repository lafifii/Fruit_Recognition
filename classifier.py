import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from os import walk


def loadImg(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def showImg(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def hog(img, B):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32((ang*B)/(2*np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), B)
             for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


def full_data(paths_ts, paths_tr):
    train = []
    target = []
    test = []

    for clase in range(1, 4):
        a, b, c = load_data(paths_ts[clase - 1], paths_tr[clase - 1], clase)
        train += a
        target += b
        test += c
        print(len(a), len(b))

    train = np.vstack(train)
    test = np.vstack(test)
    target = np.vstack(target)

    train = np.float32(train)
    test = np.float32(test)

    return train, test, target


def predictor(train, target):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    svm.train(train, cv2.ml.ROW_SAMPLE, target)

    return svm


def correctness_predictor(svm, test):
    result = svm.predict(test)[1]
    mask = (result == target)
    correct = np.count_nonzero(mask)
    return (correct*100.0/result.size)


def predict_class(img, svm, hist):
    hist = hog(img, 16)
    hist = np.float32(hist)
    return int(svm.predict(hist.reshape(1, 64))[1][0][0])


def get_imgs(mypath, arr):
    f = []
    imgs = []
    c = 0
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break

    for e in f:
        arr.append(loadImg(mypath + "/" + e))
        c += 1

    return arr, c


def load_dataset(d_train, d_test):
    train = []
    test = []
    target = []
    for i in range(3):
        train, c = get_imgs(d_train[i], train)
        for j in range(c):
            target.append(i)
        test, c = get_imgs(d_test[i], test)

    train = np.vstack(train)
    target = np.vstack(target)
    test = np.vstack(test)

    train = np.float32(train)
    test = np.float32(test)

    return train, test, target


d_train = ['FruitsDB/Apples/Test',
           'FruitsDB/Mangoes/Test', 'FruitsDB/Oranges/Test']

d_test = ['FruitsDB/Apples/Train',
          'FruitsDB/Mangoes/Train', 'FruitsDB/Oranges/Train']

train, test, target = load_dataset(d_train, d_test)
print(train.shape, test.shape, target.shape)
