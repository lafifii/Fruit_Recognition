# Fruit Recognition 

Dataset: Train and test images splited  77%, 33% of Apples, Mangoes and Oranges
Two approaches for comparing results: KNN and Supporting Vector Machine for classifing the Fruits. Before that we used some image processing for making the results of the classification better. For that thresholding and Rescaling the Image Intensity were used.

The results were: 
- KNN: No preprocessing, precission of 88.33% and with Preprocessing 89.39%
- VSM: No preprocessing, precission of 92.42% and with Preprocessing 98.48%

<p align="center">
  <img width="40%" src="https://github.com/lafifii/Fruit_Classification/blob/master/images/no_pre.png">
  <img width="40%" src="https://github.com/lafifii/Fruit_Classification/blob/master/images/yes_pre.png">
</p> 


### Preprocessing: Thresholding and Rescaling Intensity

The input to a thresholding operation is typically a grayscale or color image. In the simplest implementation, the output is a binary image representing the segmentation. Black pixels correspond to background and white pixels correspond to foreground (or vice versa). In simple implementations, the segmentation is determined by a single parameter known as the intensity threshold. In a single pass, each pixel in the image is compared with this threshold. If the pixel's intensity is higher than the threshold, the pixel is set to, say, white in the output. If it is less than the threshold, it is set to black.

### Histogram of Oriented Gradients (HOG)

Histogram of oriented gradients (HOG) is a feature descriptor used to detect objects in computer vision and image processing. The HOG descriptor technique counts occurrences of gradient orientation in localized portions of an image - detection window, or region of interest (ROI).

### Supporting Vector Machine
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
Loss Function:
<p align="center">
  <img width="50%" src="https://miro.medium.com/max/1056/1*GQAd28bK8LKOL2kOOFY-tg.png">
</p> 

### K - Nearest Neighbors
KNN (K - Nearest Neighbors) is one of many (supervised learning) algorithms used in data mining and machine learning, it’s a classifier algorithm where the learning is based “how similar” is a data (a vector) from other .

## Installation

In order to run the scripts, you should perform the following steps:

### Install Python 3.x

You should install **Python** in your machine, to do so go to [download page](https://www.python.org/downloads/) and install the most recent version for your Operating System.

### Install VirtualEnv

VirtualEnv allows you to create isolated Python environments for the different projects you work in. This is useful when trying different version of packages or when wanting  to install same environment accross multiple developers.

You should install **virtualenv** in your machine. Once Python is installed, use pip (package manager) to achieve this by executing the following:

```bash
$ pip install virtualenv==16.1.0
```

### Create a virtual environment
Once **virtualenv** is installed, in the corresponding git repository folder, execute the command:

```bash
$ virtualenv .venv
```

It will create a folder called **.venv** (we use this name by convention) that contains all the python packages and dependencies out of the box.

To activate the virtual environment, you should run:

In macOS (within project folder):
```bash
$ source .venv/bin/activate
```

In Windows (within project folder):
```bash
$ .venv\Scripts\activate
```

### Install python packages

```bash
$ pip install -r requirements.txt
```

