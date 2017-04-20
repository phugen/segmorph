from sklearn import mixture
from validate import validate
import numpy as np


def GMMfunc(input_img):
    ''' fit GMM and return predicted labels. '''

    input_img = input_img.reshape((input_img.size), 1)
    gmm = mixture.GaussianMixture(classes, max_iter=300, covariance_type='full')
    gmm.fit(input_img)
    labels = gmm.predict(input_img)

    return labels


classes = 3
inpath = "../archs/drosophila_0_validation_3.h5"

validate(GMMfunc, classes, inpath, "GMM_validation/")
