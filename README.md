# CIFAR10 classification experiments

Here we present our experiments with classification of images from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) database consists of 60000 color images in 10 classes `['plane','auto','bird','cat','deer','dog','frog','horse','ship','truck']`.
Each image is of the size 32x32 with RGB features, so the single data sample has $32*32*3=3072$ features.
The dataset is partitioned to training (50000) and testing (10000) samples.

### Loading dataset
The script `myutils.py` provides very simple procedure for (down)loading the data.
```pythonimport myutilsdata_training, data_testing = myutils.load_CIFAR_dataset()
```
When loaded, `data_training` is a list of training images and its labels. For example, the $k$-th image is stored in the array `data_training[k][0]` (with shape `(32,32,3)`) and its label is `data_training[k][1]` (integer number in `range(10)`).

### Some example images
The `iPython` notebook
[CIFAR10-visualization.ipynb](CIFAR10-visualization.ipynb)
presents some examples images from each of 10 classes in CIFAR10.
<img src="img/cifar10-examples.png" width="450" alt="10 random examples from each class in CIFAR10 dataset">

## Classification

### Classification using HOG features
The original training data contains 50000 samples; each has 32*32*3 = **3072 features**.
We used **Histogram of Oriented Gradients** (HOG) feature descriptor in order to extract features.
The extraction process is presented in ipython notebook [Classification using HOG features.ipynb](Classification using HOG features.ipynb).
The *default* configuration extract **324 features** from each training image (preprocessed with grayscaling first).
The following figure shows the grayscalled image and its HOG visualization
<img src="img/cifar10-hog-features.png" width="450" alt="10 random examples from each class in CIFAR10 dataset">

Finally we used [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) classifier (from <tt>scikit-learn</tt> library) for the classification of CIFAR10 images.
The final score is about **49.13%**. Thus we correctly classify about the half of the testing images.We tried to <tt>SVC(kernel='linear', C=1)</tt> which is quite more complex than <tt>LinearSVC</tt>.
Our best result was about **51%**. Undoubtedly, one can get a little bit better score after tuning parameters for HOG descriptor and SVC classifier.See the additional remarks at the end of the notebook [Classification_using_HOG_features.ipynb](Classification_using_HOG_features.ipynb).# Trasnfer learning
