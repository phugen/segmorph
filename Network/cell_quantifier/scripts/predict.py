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

if len(sys.argv) < 5:
    print "Too few arguments!"
    print "Usage: python predict.py modelfile weightfile input output_folder"
    sys.exit(-1)

# set paths to deployment net, trained model weights and the images to predict (in HDF5 format, e.g. validation set)
model_file = sys.argv[1]
weights = sys.argv[2]
input_file = sys.argv[3]
output_path = sys.argv[4]

# initialize U-Net with trained weights
net = caffe.Net(model_file, 1, weights=weights)

# get mirrored image data from HDF5 file, ignore labels
with h5py.File(input_file, "r") as f:
    input_images = np.array(f["data"])

# predict all images in the validation set file
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

bar = progressbar.ProgressBar()
for imgno in bar(range(input_images.shape[0])):

    # load input image
    net.blobs['data'].data[...] = input_images[imgno, ...]

    # center / normalize data
    transformer.preprocess('data', input_images[imgno, ...])

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
    scipy.misc.toimage(output[0, 1:4, :, :], cmin=0.0, cmax=1.0).save(output_path + "prediction_" + str(imgno) + ".png")

print "Predicted " + str(input_images.shape[0]) + " images."



#---------------------------------- SINGLE IMAGE PREDICTION ---------------------------------------------
# read image data from file
'''imgfile = caffe.io.load_image(input_file)
imgfile = np.transpose(imgfile, (2,0,1)) # switch from HxWxC to CxHxW layout

# load input image into network
net.blobs['data'].data[...] = imgfile

# center / normalize data
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.preprocess('data', imgfile)

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
scipy.misc.toimage(output[0, 1:4, :, :], cmin=0.0, cmax=1.0).save(output_path + "prediction.png")

print "Predicted one image."'''
