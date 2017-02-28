# This script uses a trained Caffe network to create segmentation maps for images
# pixel-by-pixel. To predict images larger than the network input 244x244,
# a overlapping tiling technique can be used (see U-Net paper).

import caffe
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import h5py
import progressbar
from PIL import Image


# predict using GPU
caffe.set_device(0)
caffe.set_mode_gpu()

# set paths to deployment net, trained model weights and the images to predict (in HDF5 format, e.g. validation set)
MODEL_FILE = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/unet_weighted_deploy.prototxt"
WEIGHTS = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/snapshots/unet_weighted_pseudo_iter_0.caffemodel"
INPUT_FILE = "G:/CDEF_2013/CF/F_GNAGNA/Stuff/Studium/Master/WS_2016/Masterarbeit/cell_quantifier/Network/cell_quantifier/training_4final/drosophila_validation.h5"

# initialize U-Net with trained weights
net = caffe.Net(MODEL_FILE, WEIGHTS, caffe.TEST)
#net = caffe.Classifier(MODEL_FILE, PRETRAINED,
#                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
#                       channel_swap=(2,1,0),
#                       raw_scale=255,
#                       image_dims=(428, 428))

# get mirrored image data from HDF5 file, ignore labels
with h5py.File(INPUT_FILE, "r") as f:
    input_images = np.array(f["data"])

# predict all images in the validation set file
bar = progressbar.ProgressBar()
for imgno in bar(range(input_images.shape[0])):

    # load input image
    #input_image = caffe.io.load_image(IMAGE_FILE) # switch from HxWxC to CxHxW layout
    #transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    #transformer.set_transpose('data', (2,0,1)) # swap axes
    net.blobs['data'].data[...] = input_images[imgno, ...] #transformer.preprocess('data', input_images[imgno, ...]) # assign data

    # make prediction
    prediction = net.forward()
    prediction = prediction['softmax']

    # set class with maximum probability to max value
    output = np.zeros((prediction.shape))

    for y in range(prediction.shape[2]):
        for x in range(prediction.shape[3]):
            maxchannel = prediction[0, :, y, x].argmax()
            output[0, maxchannel, y, x] = 1.0


    # interpret probabilities as RGB values, omit BG values
    scipy.misc.toimage(output[0, 1:4, :, :], cmin=0.0, cmax=1.0).save("./predictions/prediction_" + str(imgno) + ".png")

print "Predicted " + str(input_images.shape[0]) + " images."
