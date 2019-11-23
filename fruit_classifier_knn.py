import cv2
import numpy as np
import pandas as pd

from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean

from sklearn.model_selection import train_test_split
from skimage import data, color, feature
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity



import glob

def load_data(fruit, tipo):
    label=[]
    arr = []
    strr = "FruitsDB/"+fruit+"/" + tipo + "/*"
    for file_ in glob.glob(strr):
      img = cv2.imread(file_)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      arr.append(img)
      label.append(fruit)
   
    return arr,label

def whole_train_data(tipo):
  apples_data, apples_label = load_data('Apples', tipo)
  mangoes_data, mangoes_label = load_data('Mangoes', tipo)
  oranges_data, oranges_label = load_data('Oranges', tipo)
  data =np.concatenate((apples_data,mangoes_data,oranges_data))
  labels =np.concatenate((apples_label, mangoes_label, oranges_label))
  return data, labels

def preprocessing(arr):
    arr_prep=[]
    for i in range(arr.shape[0]):
        img=cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img=resize(img, (72, 72),anti_aliasing=True)
        arr_prep.append(img)
    return arr_prep


def ExtractHOG(img):
    ftr,_=hog(img, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, multichannel=False)
    return ftr
  
def preprocessing_part_two(arr):
    arr_feature=[]
    for i in range(np.shape(arr)[0]):
        arr_feature.append(ExtractHOG(arr[i])) 
    return arr_feature


def run_knn(k, flag):
    data_train, labels_train = whole_train_data('Train')
    data_test, labels_test = whole_train_data('Test')

    data_train_p = preprocessing(data_train)
    data_test_p = preprocessing(data_test)

    data_train_ftr = preprocessing_part_two(data_train_p)
    data_test_ftr= preprocessing_part_two(data_test_p)

    knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=k)
    knn_clf.fit(data_train_ftr, labels_train)

    y_knn_pred = knn_clf.predict(data_test_ftr)
    accuracy = accuracy_score(labels_test, y_knn_pred)*100
    
    ans = [accuracy, data_test, y_knn_pred]
    return ans

