from sklearn import cluster
from validate import validate
import numpy as np


def kmeans3func(input_img):
    ''' Perform 3-fold K-means segmentation and return predicted labels. '''

    dims = input_img.shape[0:2]
    clusterer = cluster.KMeans(n_clusters=3)
    labels = np.array(clusterer.fit_predict(input_img.reshape(input_img.size, 1)))
    labels = labels.reshape(dims)

    return labels


def kmeans4func(input_img):
    ''' Perform 4-fold K-means segmentation and return predicted labels. '''

    dims = input_img.shape[0:2]
    clusterer = cluster.KMeans(n_clusters=4)
    labels = np.array(clusterer.fit_predict(input_img.reshape(input_img.size, 1)))
    labels = labels.reshape(dims)

    return labels


classes = 4
inpath = "../training_4final/drosophila_4classdata_avgweight/"

validate(kmeans4func, classes, inpath, "kmeans_validation/kmeans4/")
