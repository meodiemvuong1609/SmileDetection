import os
import cv2
import numpy as np
from imutils import paths
import imutils
from const import *

DATASET = BASE_DIR + "dataset\SMILEsmileD\SMILEs"
# initialize the list of data and labels
data = []
labels = []


def convert_data():
    X = []
    # loop over the input images
    for imagePath in sorted(list(paths.list_images(DATASET))):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=28)
        T = np.zeros([28, 28, 1])
        T[:, :, 0] = image

        # extract the class label from the image path and update the label list
        label = imagePath.split(os.path.sep)[-3]
        label = "smiling" if label == "positives" else "not_smiling"
        X.append((image, label))

    for _ in range(10):
        np.random.shuffle(X)

    train_data, test_data = X[:10000], X[10000:]

    np.save(BASE_DIR + 'dataset/data/train.npy', train_data)
    np.save(BASE_DIR + 'dataset/data/test.npy', test_data)

convert_data()
