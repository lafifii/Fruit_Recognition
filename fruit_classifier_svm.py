import cv2
import numpy as np
import pandas as pd

import glob
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

# tipo : type of fruit 
# clase : class [0,1,2..]  

def _hog(img, B):
  c = img.shape[0]//2
  f = img.shape[1]//2
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag,ang = cv2.cartToPolar(gx, gy)
  bins = np.int32((ang*B)/(2*np.pi))
  bin_cells = bins[:c,:c], bins[f:,:c], bins[:c,f:], bins[f:,f:]
  mag_cells = mag[:c,:c], mag[f:,:c], mag[:c,f:], mag[f:,f:]
  hists = [np.bincount(b.ravel(), m.ravel(), B) for b,m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)
  return hist

def hog(img, B):
  """ 
    instead of having one channel when using black and white, we need to have one for every channel rgb
    so that is what we are doing and then doing a hog for each channel and then concatenate every hog in one big histogram
  """
  r = img[:, :, 0]
  g = img[:, :, 1]
  b = img[:, :, 2]
  r = _hog(r, B)
  g = _hog(g, B)
  b = _hog(b, B)
  hist = np.concatenate((r,g,b))
  return hist

def load_data(fruit, tipo, B, clase, testing, flag):
    label=[]
    arr = []
    strr = "FruitsDB/"+fruit+"/" + tipo + "/*"
    for file_ in glob.glob(strr):
      img = cv2.imread(file_)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if(testing):
        if(flag):
          yen_threshold = threshold_yen(img)
          bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
          h = hog(bright, B)
          arr.append(h)
        else:
          h = hog(img, B)
          arr.append(h)
      else:
        arr.append(img)
      label.append(clase)
   
    return arr,label

def whole_train_data(tipo, B, flag):
  """ In order to add other fruits just do this for it: Example if you want strawberrys you will need to do:
     strawberrys_data, strawberrys_label = load_data
    ( Name of the Extension of the folder in FruitsDB, 
      type : Send either 'Test' or 'Train' for the folder name,
      B: number of pixels, we use 16,
      Class: 0, 1 , 2 ... the way you index the fruits,
      testing: Send true if this is the Test set or false if it is the Train set)
      flag: Send true if you want the image with tresholding applied and false if not)
      Just guide yourself from the main.py of this repo
      Of course you will need to add the images in a folder in FruitsDB following the structure that it has now
  """
  apples_data, apples_label = load_data('Apples', tipo, B, 0, 1, flag)
  mangoes_data, mangoes_label = load_data('Mangoes', tipo, B, 1, 1, flag)
  oranges_data, oranges_label = load_data('Oranges', tipo, B, 2, 1, flag)
  # then just concatenate the data and the labels
  data =np.concatenate((apples_data,mangoes_data,oranges_data))
  labels =np.concatenate((apples_label, mangoes_label, oranges_label))
  
  return data, labels


def train_model(data_train, labels_train):
  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_LINEAR)
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setC(2.67)
  svm.setGamma(5.383)

  svm.train(data_train, cv2.ml.ROW_SAMPLE, labels_train)
  return svm

def get_precission(svm, test_target, test):
  result = svm.predict(test)[1]   
  mask = (result==test_target)
  correct = np.count_nonzero(mask)
  return (correct*100.0/result.size)

def run_svm(flag):
  data_train, labels_train = whole_train_data('Train', 16, flag)
  data_test, labels_test = whole_train_data('Test', 16, flag)

  data_test = np.vstack(data_test)
  data_train = np.vstack(data_train)
  labels_train = np.vstack(labels_train)
  labels_test = np.vstack(labels_test)

  data_train = np.float32(data_train)
  data_test = np.float32(data_test)

  svm = train_model(data_train, labels_train)
  # more fruits only add more classes here
  classes = ["Apples", "Mangoes", "Oranges"]

  apples_data, apples_label = load_data('Apples', 'Test', 0, 0, 0, flag)
  mangoes_data, mangoes_label = load_data('Mangoes', 'Test', 0, 1, 0, flag)
  oranges_data, oranges_label = load_data('Oranges', 'Test', 0, 2, 0, flag)
  data =np.concatenate((apples_data,mangoes_data,oranges_data))
  
  # For knowing how precisse our model was
  ans = [get_precission(svm, labels_test, data_test) , data, data_test, svm]
  return ans , classes

