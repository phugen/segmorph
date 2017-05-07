from sklearn import mixture
from validate import validate
import numpy as np
import time


def GMMfunc(input_img):
    ''' fit GMM and return predicted labels. '''

    dims = input_img.shape[0:2]
    rowimg = input_img.reshape((input_img.size), 1)
    gmm = mixture.GaussianMixture(4, covariance_type='full')
    gmm.fit(rowimg)
    labels = gmm.predict(rowimg)
    labels = labels.reshape(dims)

    return labels


classes = 4
inpath = "../training_4final/drosophila_4classdata_avgweight/"

validate(GMMfunc, classes, inpath, "GMM_validation/gmm4/")
