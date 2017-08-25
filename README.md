# cifar10-cnn
Some experiments with CIFAR-10 dataset

# About

# Loading CIFAR10 dataset
```python
import myutils
data_training, data_testing = myutils.load_CIFAR_dataset()
```
One can check that the data_training[:][0] contains 50000 images (each has 32x32x3 uint8 numbers)


# Visualization of some examples from each class
see [CIFAR10-visualization.ipynb](CIFAR10-visualization.ipynb)
<img src="img/cifar10-examples.png" width="450" alt="10 random examples from each class in CIFAR10 dataset">

# Classification using HOG features
The original training data contains 50000 samples; each has 32*32*3 = **3072 features**.
We used **Histogram of Oriented Gradients** (HOG) feature descriptor in order to extract features.
The extraction process is presented in ipython notebook [Classification using HOG features.ipynb](Classification using HOG features.ipynb).
The *default* configuration extract **324 features** from each training image (preprocessed with grayscaling first).
The following figure shows the grayscalled image and its HOG visualization
<img src="img/cifar10-hog-features.png" width="450" alt="10 random examples from each class in CIFAR10 dataset">

Finally we used [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) classifier (from <tt>scikit-learn</tt> library) for the classification of CIFAR10 images.
The final score is about **49.13%**. Thus we correctly classify about the half of the testing images.
We tried to <tt>SVC(kernel='linear', C=1)</tt> which is quite more complex than <tt>LinearSVC</tt>.
Our best result was about **51%**. Undoubtedly, one can get a little bit better score after tuning parameters for HOG descriptor and SVC classifier.
See the additional remarks at the end of the notebook [Classification_using_HOG_features.ipynb](Classification_using_HOG_features.ipynb).
