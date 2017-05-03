# This script uses a trained Caffe network to create segmentation maps for images
# pixel-by-pixel. To predict images larger than the network input 244x244,
# a overlapping tiling technique can be used (see U-Net paper).

import caffe
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import h5py
import math
import progressbar
from PIL import Image



if len(sys.argv) < 6:
    print "Too few arguments!"
    print "Usage: python predict.py device_id batchsize modelfile weightfile input_folder output_folder"
    sys.exit(-1)

# set paths to deployment net, trained model weights and the images to predict (in HDF5 format, e.g. validation set)
device_id = int(sys.argv[1])
batchsize = int(sys.argv[2])
model_file = sys.argv[3]
weights = sys.argv[4]
input_path = sys.argv[5]
output_path = sys.argv[6]

if batchsize < 1:
    print "Batchsize was " + str(batchsize) + " but needs to be > 1!"
    exit(-1)

# predict using GPU
caffe.set_device(device_id)
caffe.set_mode_gpu()

# initialize U-Net with trained weights
net = caffe.Net(model_file, 1, weights=weights)

# find all validation files in folder
filenames = glob.glob(input_path + "*validation*.h5")

# predict all images in that file one after another
# without reading all of them into memory at once
imoffset = 0 # image number offset for saving outputs

bar = progressbar.ProgressBar()
for fi in bar(filenames):

    # get mirrored image and label data from HDF5 file
    input_images = None
    labels = None

    with h5py.File(fi, "r") as f:
        input_images = np.array(f["data"])
        labels = np.array(f["label"])

    # translate integer labels back to RGB
    tlabels = np.zeros((labels.shape[0], 3, labels.shape[1], labels.shape[2]))
    for index in range(labels.shape[0]):

        # get bool masks for each label
        bluemask = labels[index] == 3
        greenmask = labels[index] == 2
        redmask = labels[index] == 1

        # replace old labels
        tlabels[index, 2, ...][bluemask == 1] = 1.0
        tlabels[index, 1, ...][greenmask == 1] = 1.0
        tlabels[index, 0, ...][redmask == 1] = 1.0


    # predict all images in the validation set file
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    imgno = 0 # actual image count
    for batchno in range(0, int(math.ceil(input_images.shape[0] / float(batchsize)))):

        # deal with validation sets which have a number
        # of elements that isn't cleanly divisable by the batch size
        start = 0
        end = batchsize

        if (batchno * batchsize) + batchsize > input_images.shape[0]:
            end = input_images.shape[0] % batchsize

        # batch load input images
        for b in range(start, end):

            input_b = (batchno * batchsize) + b
            net.blobs['data'].data[b, ...] = input_images[input_b, ...]

            # center / normalize data # NOTE: not needed because mean is substracted manually before packing HDF files!
            #transformer.set_mean('data', np.repeat(np.mean(input_images[input_b, ...]), 3)) # "fake" rgb mean = [greymean, greymean, greymean]

        # make predictions
        prediction = net.forward()
        prediction = prediction['softmax']

        # set class with maximum probability to max value
        output = np.zeros((batchsize, 4, prediction.shape[2], prediction.shape[3]))

        for b in range(start, end):
            for y in range(prediction.shape[2]):
                for x in range(prediction.shape[3]):
                    maxchannel = prediction[b, :, y, x].argmax()
                    output[b, maxchannel, y, x] = 1.0

            # interpret probabilities as RGB values, omit BG values
            # and save prediction alongside GT image
            imgno = (batchno * batchsize) + b
            scipy.misc.toimage(output[b, 1:4, :, :], cmin=0.0, cmax=1.0).save(output_path + "prediction_" + str(imgno + imoffset) + ".png")
            scipy.misc.toimage(tlabels[imgno, ...], cmin=0.0, cmax=1.0).save(output_path + "prediction_" + str(imgno + imoffset) + "_GT.png")

    # save images of next HDF5 file with offset numbering
    imoffset += input_images.shape[0]


'''bar = progressbar.ProgressBar()
for fi in bar(filenames):

    # get mirrored image and label data from HDF5 file
    input_images = None
    labels = None

    with h5py.File(fi, "r") as f:
        input_images = np.array(f["data"])
        labels = np.array(f["label"])

    # translate integer labels back to RGB
    tlabels = np.zeros((labels.shape[0], 3, labels.shape[1], labels.shape[2]))
    for index in range(labels.shape[0]):

        # get bool masks for each label
        bluemask = labels[index] == 3
        greenmask = labels[index] == 2
        redmask = labels[index] == 1

        # replace old labels
        tlabels[index, 2, ...][bluemask == 1] = 1.0
        tlabels[index, 1, ...][greenmask == 1] = 1.0
        tlabels[index, 0, ...][redmask == 1] = 1.0


    # predict all images in the validation set file
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    for imgno in range(input_images.shape[0]):

        # load input image
        net.blobs['data'].data[...] = input_images[imgno, ...]

        # center / normalize data
        transformer.preprocess('data', input_images[imgno, ...])

        # make prediction
        prediction = net.forward()
        prediction = prediction['softmax']

        # set class with maximum probability to max value
        output = np.zeros((1, 4, prediction.shape[2], prediction.shape[3]))

        for y in range(prediction.shape[2]):
            for x in range(prediction.shape[3]):
                maxchannel = prediction[0, :, y, x].argmax()
                output[0, maxchannel, y, x] = 1.0

        # interpret probabilities as RGB values, omit BG values
        scipy.misc.toimage(output[0, 1:4, :, :], cmin=0.0, cmax=1.0).save(output_path + "prediction_" + str(imgno + imoffset) + ".png")
        scipy.misc.toimage(tlabels[imgno, ...], cmin=0.0, cmax=1.0).save(output_path + "prediction_" + str(imgno + imoffset) + "_GT.png")

    # save images of next HDF5 file with offset numbering
    imoffset += input_images.shape[0]'''


print "Predicted " + str(input_images.shape[0] + imoffset) + " images."
