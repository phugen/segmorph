from otsu_val_impl import otsu3, otsu4
from validate import validate

import numpy as np



def otsu3func(input_img):
    ''' 3-way otsu thresholding. '''

    dims = input_img.shape[0:2]
    input_img = input_img.reshape((input_img.size), 1)
    threshs = otsu3(input_img)
    labels = input_img.copy()

    # apply thresh
    labels[input_img < threshs[0] / 255.] = 0
    labels[[input_img >= threshs[0] / 255.][0] & [input_img < threshs[1]/ 255.][0]] = 1
    labels[input_img >= threshs[1]/ 255.] = 2


    return labels.reshape(dims)


def otsu4func(input_img):
    ''' 4-way otsu thresholding. '''

    dims = input_img.shape[0:2]
    input_img = input_img.reshape((input_img.size), 1)
    threshs = otsu4(input_img)
    labels = input_img.copy()

    # apply thresh
    labels[input_img < threshs[0] / 255.] = 0
    labels[[input_img >= threshs[0] / 255.][0] & [input_img < threshs[1] / 255.][0]] = 1
    labels[[input_img >= threshs[1] / 255.][0] & [input_img < threshs[2] / 255.][0]] = 2
    labels[input_img >= threshs[2] / 255.] = 3


    return labels.reshape(dims)


classes = 4
inpath = "../training_4final/unaltered_4class_avgweight_validation/"

validate(otsu4func, classes, inpath, "otsu_validation/otsu4/")
