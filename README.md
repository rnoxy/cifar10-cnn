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
Finally we used [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) classifier (from <tt>scikit-learn</tt> library) for the classification of CIFAR10 images.
The final score is about **0.49** so we correctly classify about half of the testing images.
Probaly one can get little bit better results with tuning parameters for HOG descriptor and SVC classifier.

