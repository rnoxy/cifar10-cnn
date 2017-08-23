# -*- coding: utf-8 -*-
# File: myutils.py
# Author: Rafał Nowak <rafal.nowak@cs.uni.wroc.pl>

import os
import logging
import tarfile
import urllib.request
import pickle
import numpy as np

# pylint:

CIFAR_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_DIR_PATH = './cifar10_data/'
CIFAR_FOLDERNAME = 'cifar-10-batches-py'
CIFAR_BATCH_SIZE = 10000  # CIFAR10 data are split into blocks of 10000 images

CIFAR_TRAINING_FILENAMES = [
    os.path.join(CIFAR_DIR_PATH, CIFAR_FOLDERNAME, 'data_batch_%d' % i) for i in range(1, 6)
    ]
CIFAR_TESTING_FILENAMES = [os.path.join(CIFAR_DIR_PATH, CIFAR_FOLDERNAME, 'test_batch')]

def read_CIFAR_files(filenames):
    """
    Return the CIFAR dataset loaded from the bunch of files.
    
    Keyword arguments:
    filenames -- the list of filenames (strings)
    """
    dataSet = [] # dataset to be returned
    for file in filenames:
        with open(file, 'rb') as fo:
            _dict = pickle.load(fo, encoding='bytes')

        # Loaded in this way, each of the batch files contains a dictionary
        # with the following elements:
        data = _dict[b'data']
        labels = _dict[b'labels']
        #   data -- a 10000x3072 numpy array of uint8s.
        #           Each row of the array stores a 32x32 colour image.
        #           The first 1024 entries contain the red channel values,
        #           the next 1024 the green, and the final 1024 the blue.
        #           The image is stored in row-major order, so that the
        #           first 32 entries of the array are the red channel values
        #           of the first row of the image.
        #   labels -- a list of 10000 numbers in the range 0-9. The number
        #           at index i indicates the label of the ith image in the
        #           array data.

        # data[0] is the first image, data[1] is second and so on
        assert data[0].size == 3*32*32

        for k in range(CIFAR_BATCH_SIZE):
            # pick k-th image
            image = data[k].reshape(3, 32, 32)
            # image[ C, x, y ] where C means the color
            # image[ :, x, y ] is array [ R,G,B ]
            #        0  1  2
            # Since we want to transpose image to have image[x,y,:]
            #                                                1 2 0
            image = np.transpose(image, [1, 2, 0])
            # img[x,y,:] is array [R,G,B]
            dataSet.append([image, labels[k]])
    return dataSet


def load_CIFAR_dataset(shuffle=True):
    """
    Download (if necessary) CIFAR database file and extract it.
    Return the tuple of training and testing dataset.
    """
    logging.info("Loading dataset ...")
    # checking if the data is already in the folder
    if not os.path.isdir(os.path.join(CIFAR_DIR_PATH, CIFAR_FOLDERNAME)):
        # if not, we download the data
        os.makedirs(CIFAR_DIR_PATH, exist_ok=True) # create folder for the data
        filename = CIFAR_DATA_URL.split('/')[-1]
        filepath = os.path.join(CIFAR_DIR_PATH, filename)
        # try to download the file
        try:
            logging.info("Downloading file {f}. This may take some time.".format(f=CIFAR_DATA_URL))
            fpath, _ = urllib.request.urlretrieve(CIFAR_DATA_URL, filepath)
            statinfo = os.stat(fpath)
            size = statinfo.st_size
        except:
            logging.error("Failed to download {f}".format(f=CIFAR_DATA_URL))
            raise

        print('Succesfully downloaded {f} ({s} bytes)'.format(f=filename,s=size))
        tarfile.open(filepath, 'r:gz').extractall(CIFAR_DIR_PATH)

    trainingData = read_CIFAR_files(CIFAR_TRAINING_FILENAMES)
    testingData = read_CIFAR_files(CIFAR_TESTING_FILENAMES)

    if shuffle:
        logging.info("Shuffling data ...")
        import sklearn
        trainingData = sklearn.utils.shuffle(trainingData)
        testingData = sklearn.utils.shuffle(testingData)

    return trainingData, testingData


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    load_CIFAR_dataset()
