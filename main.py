from fruit_classifier_knn import run_knn
from fruit_classifier_svm import run_svm

import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib

from random import seed
from random import randint

def showImg(img, name, method, i):
    plt.subplot(2, 1, i)
    plt.imshow(img)
    plt.title(method + ": Result: " + name)
    plt.axis("off")

def try_predictors(data_knn, y_knn_pred, svm, data, data_test, classes, k):
    x_ = randint(0, data_knn.shape[0])
    showImg(data_knn[x_], y_knn_pred[x_], "K Nearest Neighbors (k = {})".format(k), 1)
    
    x_ = randint(0, data_test.shape[0])
    img = data[x_]
    result = int(svm.predict(data_test[x_].reshape(1,192))[1][0][0])
    result = classes[result]
    showImg(img, result, "Support Vector Machine" , 2)

    plt.show()

def gui(results_knn_no, results_svm_no, results_knn_yes, results_svm_yes, classes, k):
    root = tk.Tk()
    root.geometry("400x300")
    root.title("Fruit Classifier")

    l1 = tk.Label(root, text="Precission of KNN with no Preprocessing: {0:.2f}%".format(results_knn_no[0]))
    l1.pack()
    l2 = tk.Label(root, text="Precission of SVM with no Preprocessing: {0:.2f}%".format(results_svm_no[0]))
    l2.pack()
    button1 = tk.Button(root, text='Try Classifier - No Prepocessing',
                        command=lambda: try_predictors(results_knn_no[1], results_knn_no[2], results_svm_no[3], results_svm_no[1], results_svm_no[2], classes, k))
    button1.pack()

    l3 = tk.Label(root, text="Precission of KNN with Prepocessing - thresholding: {0:.2f}%".format(results_knn_yes[0]))
    l3.pack()
    l4 = tk.Label(root, text="Precission of SVM with Prepocessing - thresholding: {0:.2f}%".format(results_svm_yes[0]))
    l4.pack()
    button2 = tk.Button(root, text='Try Classifier - Prepocessing: thresholding',
                        command=lambda: try_predictors(results_knn_yes[1], results_knn_yes[2], results_svm_yes[3], results_svm_yes[1], results_svm_yes[2], classes, k))
    button2.pack()

    root.mainloop()

if __name__ == '__main__':
    k = 11
    results_knn_no = run_knn(k, False)
    results_svm_no, classes = run_svm(False)

    results_knn_yes = run_knn(k, True)
    results_svm_yes, classes = run_svm(True)

    gui(results_knn_no, results_svm_no, results_knn_yes, results_svm_yes, classes, k)