import glob
import matplotlib.pyplot as plt
import random
import numpy as np


def loadImg(path):
    img = cv2.imread(path)
    img = cv2.resize(img, tuple((500, 500)))
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


def load_data(path, clase):
    train = [cv2.imread(file) for file in glob.glob(path + '/Train/*')]
    test = [cv2.imread(file) for file in glob.glob(path + '/Test/*')]

    target = [clase for i in range(len(train))]  # class

    return train, target, test


def full_data(paths):
    train = []
    target = []
    test = []
    clase = 1

    for path in paths:
        a, b, c = load_data(path, clase)
        train += a
        target += b
        test += c
        clase += 1
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


def predict_class(img):
    hist = hog(img, 16)
    hist = np.float32(hist)
    return int(svm.predict(hist.reshape(1, 64))[1][0][0])


paths = ['FuitsDB/Mangoes', 'FruitsDB/Apples', 'FuitsDB/Oranges']
a, b, c = full_data(paths)
