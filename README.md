# Fruit_Classification

Dataset: Train and test images splited  77%, 33% of Apples, Mangoes and Oranges

## Preprocessing: thresholding and Rescaling Intensity

The input to a thresholding operation is typically a grayscale or color image. In the simplest implementation, the output is a binary image representing the segmentation. Black pixels correspond to background and white pixels correspond to foreground (or vice versa). In simple implementations, the segmentation is determined by a single parameter known as the intensity threshold. In a single pass, each pixel in the image is compared with this threshold. If the pixel's intensity is higher than the threshold, the pixel is set to, say, white in the output. If it is less than the threshold, it is set to black.

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
> We use version 16.1.0 since it exists some compatibility issues when packaging scripts with other versions of VirtualEnv

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

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install python packages. You can install them directly by executing:

```bash
$ pip install the_package
```

If you are importing an existing project, it must have a *requirements.txt* file from which you can install all dependencies directly by executing:

```bash
$ pip install -r requirements.txt
```
