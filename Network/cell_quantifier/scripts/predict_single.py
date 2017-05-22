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
import preparation_utils as preputils
import progressbar
from PIL import Image



if len(sys.argv) < 6:
    print "Too few arguments!"
    print "Usage: python predict_single.py device_id batchsize modelfile weightfile input_file output_folder"
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

# load image
largeimg = caffe.io.load_image(input_path)
largeimg = np.transpose(largeimg, (2,0,1))

# add extra "imgno" dimension because previous script expect 4 dimensions instead of 3
largeimg = np.expand_dims(largeimg, axis=0)
#print largeimg.shape

# add mirror padding to image
# NOTE: for explanation of these calculations, check out scripts/converthdf5.py
n_tiles = 1
downsampleFactor = 16
d4a_size = 0
padInput =   (((d4a_size * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2
padOutput = ((((d4a_size - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2
d4a_size = np.ceil(((np.array([largeimg.shape[2], np.ceil(largeimg.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor))) \
             .astype(dtype=np.int)
input_size = downsampleFactor * d4a_size + padInput
output_size = downsampleFactor * d4a_size + padOutput
border = np.around(np.subtract(input_size, output_size)) / 2
border = np.array(((border[0], border[0]), (border[1], border[1])))


# prevent broken tiles
input_size = 244 # actual input size without padding
magic_number = 428 # padded input size that fits the network architecture
cheight = largeimg.shape[2]
cwidth = largeimg.shape[3]

# add normal mirror border
largeimg_padded = preputils.enlargeByMirrorBorder(largeimg, border)

# subtract image mean
means = [np.mean(largeimg_padded[:, 0, ...]), \
         np.mean(largeimg_padded[:, 1, ...]), \
         np.mean(largeimg_padded[:, 2, ...])]

for i in range(3):
    largeimg_padded[:, i, ...] -= means[i]

# cut padded image into tiles
PAD_HEIGHT = largeimg_padded.shape[2]
PAD_WIDTH = largeimg_padded.shape[3]
tiles = np.array(preputils.tileMirrorImage (1, input_size, magic_number, largeimg_padded, \
                           [cheight, cwidth], [PAD_HEIGHT, PAD_WIDTH]))

# setup network
# predict using GPU
caffe.set_device(device_id)
caffe.set_mode_gpu()

# initialize U-Net with trained weights
net = caffe.Net(model_file, 1, weights=weights)

# predict all tiles
predictions = []
imgno = 0 # actual image count
bar = progressbar.ProgressBar()
for batchno in bar(range(0, int(math.ceil(tiles.shape[0] / float(batchsize))))):

    # deal with validation sets which have a number
    # of elements that isn't cleanly divisable by the batch size
    start = 0
    end = batchsize

    if (batchno * batchsize) + batchsize > tiles.shape[0]:
        end = tiles.shape[0] % batchsize

    # batch load input images
    for b in range(start, end):

        input_b = (batchno * batchsize) + b
        net.blobs['data'].data[b, ...] = tiles[input_b, ...]

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

        # save predicted tile for stitching later
        predictions.append(output[b, 1:4, :, :])


# stitch together output image from predicted tiles
# while using only the real part of the repaired
# broken tiles (if there were any to begin with)
stitched = np.zeros((3, cheight, cwidth))
numytiles = cheight / input_size
numxtiles = cwidth / input_size
brokenheight = cheight - (numytiles * input_size)
brokenwidth = cwidth - (numxtiles * input_size)


for y in range(numytiles):
    for x in range(numxtiles + 1):

        # calculate offsets for current tile
        offy = y * input_size
        offx = x * input_size

        # stitch normal tiles together
        if x != numxtiles:
            stitched[:, offy:offy + input_size, offx:offx + input_size] \
          = predictions[y * (numxtiles+1) + x][:, 0:input_size, 0:input_size]

        # add horizontally broken tile at the end of current row
        else:
            stitched[:, offy:offy + input_size, cwidth - brokenwidth:cwidth] \
          = predictions[y * (numxtiles+1) + x][:, 0:input_size, input_size - brokenwidth:]

# add row of vertically broken tiles
for x in range(numxtiles):

    offx = x * input_size
    offy = (y+1) * input_size

    stitched[:, offy:cheight, offx:offx + input_size] \
  = predictions[(y + 1) * (numxtiles+1) + x][:, input_size - brokenheight:, 0:input_size]

# add final vertically and horizontally broken tile
stitched[:, cheight - brokenheight:cheight, cwidth - brokenwidth:cwidth] \
= predictions[(y + 1) * numxtiles + (x + 1)][:, input_size - brokenheight:, input_size - brokenwidth:]



scipy.misc.toimage(stitched).save(output_path + "stitched.png")


print "Predicted the input image using " + str(len(tiles)) + " tiles."
