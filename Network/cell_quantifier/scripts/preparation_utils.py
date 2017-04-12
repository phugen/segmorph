# This file contains utility functions for deep learning data preparation etc.

import math
import os
import h5py
import numpy as np
import progressbar
import scipy.ndimage
import matplotlib.pyplot as plt


def enlargeByMirrorBorder(input_image, border):
    """Enlarge RGB images (imgno X channel X height x width) \
     by a certain border by mirroring their edges"""

    #create output array, enlarged by borders
    '''paddedFullVolume = np.zeros((input_image.shape[0], \
                                input_image.shape[1], \
                                input_image.shape[2] + 2 * border[0], \
                                input_image.shape[3] + 2 * border[1]), \
                                dtype=np.float32)'''

    paddedFullVolume = np.zeros((input_image.shape[0], \
                                input_image.shape[1], \
                                input_image.shape[2] + border[1, 0] + border[1, 1], \
                                input_image.shape[3] + border[0, 0] + border[0, 1]), \
                                dtype=np.float32)

    print "Original image size: " + str(input_image.shape)
    print "Enlarged image size: " + str(paddedFullVolume.shape) + "\n"

    # fill in original data in center, leaving the zero borders intact
    # image and channel indices have no need for padding, copy them as they are
    paddedFullVolume[:, \
                     :, \
                     border[1, 0]:border[1, 0] + input_image.shape[2], \
                     border[0, 0]:border[0, 0] + input_image.shape[3]] = input_image;


    # fill zero-padded areas with meaningful data
    xpad_l  = border[0, 0] - 1 # last padding element before image data
    xpad_r = border[0, 1] - 1
    xfrom = border[0, 0] # first image element
    xto   = border[0, 0] + input_image.shape[3] # first padding element after image data

    ypad_u  = border[1, 0] - 1
    ypad_d = border[1, 1] - 1
    yfrom = border[1, 0]
    yto   = border[1, 0] + input_image.shape[2]

    print "x: " + str(xfrom) + " to " + str(xto) + " with xpad_l = " + str(xpad_l) + ", xpad_r = " + str(xpad_r)
    print "y: " + str(yfrom) + " to " + str(yto) + " with ypad_u = " + str(ypad_u) + ", ypad_d = " + str(ypad_d)

    # do mirroring
    paddedFullVolume[:, :, yfrom:yto, 0:xfrom] = paddedFullVolume[:, :, yfrom:yto, xfrom + xpad_l:xfrom - 1:-1] # left
    paddedFullVolume[:, :, yfrom:yto, xto:xto + xpad_r + 1] = paddedFullVolume[:, :, yfrom:yto, xto - 1:xto - xpad_r - 2:-1] # right
    paddedFullVolume[:, :, 0:yfrom, xfrom:xto] = paddedFullVolume[:, :, yfrom + ypad_u:yfrom - 1:-1, xfrom:xto] # above
    paddedFullVolume[:, :, yto:yto + ypad_d + 1, xfrom:xto] = paddedFullVolume[:, :, yto - 1:yto - ypad_d - 2:-1, xfrom:xto] # below

    paddedFullVolume[:, :, 0:yfrom, 0:xfrom] = paddedFullVolume[:, :, yfrom + ypad_u:yfrom - 1:-1, xfrom + xpad_l:xfrom - 1:-1] # top left
    paddedFullVolume[:, :, 0:yfrom, xto:xto + xpad_r + 1] = paddedFullVolume[:, :, yfrom + ypad_u:yfrom - 1:-1, xto - 1:xto - xpad_r - 2:-1] # top right
    paddedFullVolume[:, :, yto:yto + ypad_d + 1, 0:xfrom] = paddedFullVolume[:, :, yto - 1:yto - ypad_d - 2:-1, xfrom + xpad_l:xfrom - 1:-1] # bottom left
    paddedFullVolume[:, :, yto:yto + ypad_d + 1, xto:xto + xpad_r + 1] = paddedFullVolume[:, :, yto - 1:yto - ypad_d - 2:-1, xto - 1:xto - xpad_r - 2:-1] # bottom right



    return paddedFullVolume




def tileMirrorImage (imgno, input_size, magic_number, input_image, img_dims, pad_dims):
    " Tile large, mirrored input images into smaller sub-images \
     while mirroring overlapping parts near the edges when image \
     dimensions can't be divided cleanly"

    # how many tiles are there in the x- and y-directions?
    # does the input size divide the dimensions cleanly?
    subPadRaw = [float(pad_dims[0]) / input_size, float(pad_dims[1]) / input_size]
    subPad = [int(math.floor(subPadRaw[0])), int(math.floor(subPadRaw[1]))]

    subLabRaw = [float(img_dims[0]) / input_size, float(img_dims[1]) / input_size]
    subLab = [int(math.floor(subLabRaw[0])), int(math.floor(subLabRaw[1]))]

    # output list
    paddedSubImages = []

    # tile images
    for index in range(imgno):
        for y in range(0, (subLab[0] + 1) * input_size, input_size):
            for x in range(0, (subLab[1] + 1) * input_size, input_size):

                # last element, both axes not cleanly divisable: add y- and x-overlapping sub-image
                if y == ((subLab[0]) * input_size) and not (subPadRaw[0]).is_integer() \
                and x == ((subLab[1]) * input_size) and not (subPadRaw[1]).is_integer():
                    paddedSubImages.append(input_image[index, :, pad_dims[0] - magic_number:pad_dims[0], \
                                                       pad_dims[1] - magic_number:pad_dims[1]])

                # last row and y-axis not cleanly divisable: add y-overlapping sub-image
                elif y == ((subLab[0]) * input_size) and not (subPadRaw[0]).is_integer():
                    paddedSubImages.append(input_image[index, :, pad_dims[0] - magic_number:pad_dims[0], \
                                                       x:x + magic_number])

                # last col and x-axis not cleanly divisable: add x-overlapping sub-image
                elif x == ((subLab[1]) * input_size) and not (subPadRaw[1]).is_integer():
                    paddedSubImages.append(input_image[index, :, y:y + magic_number, \
                                                       pad_dims[1] - magic_number:pad_dims[1]])

                else:
                    paddedSubImages.append(input_image[index, :, y:y + magic_number, \
                                                       x:x + magic_number])

    return paddedSubImages



def tileMirrorLabel (imgno, input_size, input_image, img_dims):
    " Tile large, mirrored input LABEL images into smaller sub-images \
     while mirroring overlapping parts near the edges when image \
     dimensions can't be divided cleanly"

    subLabRaw = [float(img_dims[0]) / input_size, float(img_dims[1]) / input_size]
    subLab = [int(math.floor(subLabRaw[0])), int(math.floor(subLabRaw[1]))]

    subLabels = []

    bar = progressbar.ProgressBar()
    print "Tiling and mirroring images/labels ..."

    for index in bar(range(imgno)):
        for y in range(0, (subLab[0] + 1) * input_size, input_size):
            for x in range(0, (subLab[1] + 1) * input_size, input_size):

                if y == ((subLab[0]) * input_size) and not (subLabRaw[0]).is_integer() \
                and x == ((subLab[1]) * input_size) and not (subLabRaw[1]).is_integer():
                    subLabels.append(input_image[index, img_dims[0] - input_size:img_dims[0], \
                                                     img_dims[1] - input_size:img_dims[1]])

                elif y == ((subLab[0]) * input_size) and not (subLabRaw[0]).is_integer():
                    subLabels.append(input_image[index, img_dims[0] - input_size:img_dims[0], \
                                                     x:x + input_size])

                elif x == ((subLab[1]) * input_size) and not (subLabRaw[1]).is_integer():
                    subLabels.append(input_image[index, y:y + input_size, \
                                                     img_dims[1] - input_size:img_dims[1]])

                else:
                    subLabels.append(input_image[index, y:y + input_size, \
                                                     x:x + input_size])

    print ""

    return subLabels



def getFlipRotationCombinations(input_images, numchannels):
    ''' Returns a numpy array that contains the original images \
    as well as those images created by flipping the originals vertically
    and rotating them by 90, 180 and 270 degrees.'''

    vertical = np.zeros((input_images.shape))
    bar = progressbar.ProgressBar()

    # flip vertically
    print "Flipping images/labels vertically ..."

    if numchannels > 1:
        for x in bar(range(input_images.shape[0])):
            for ch in range(numchannels):
                vertical[x, ch, ...] = np.flipud(input_images[x, ch, ...])

    else:
        for x in bar(range(input_images.shape[0])):
            vertical[x, ...] = np.flipud(input_images[x, ...])

    origAndVertical = np.concatenate((input_images,vertical), axis=0)
    print ""

    # rotate by 90, 180 and 270 degrees
    deg90 = np.zeros((origAndVertical.shape))
    deg180 = np.zeros((origAndVertical.shape))
    deg270 = np.zeros((origAndVertical.shape))

    bar = progressbar.ProgressBar()
    print "Rotating images/labels by 90, 180 and 270 degrees ..."

    if numchannels > 1:
        for x in bar(range(origAndVertical.shape[0])):
            for ch in range(numchannels):
                deg90[x, ch, ...] = np.rot90(origAndVertical[x, ch, ...], 1)
                deg180[x, ch, ...] = np.rot90(origAndVertical[x, ch, ...], 2)
                deg270[x, ch, ...] = np.rot90(origAndVertical[x, ch, ...], 3)

    else:
        for x in bar(range(origAndVertical.shape[0])):
            deg90[x, ...] = np.rot90(origAndVertical[x, ...], 1)
            deg180[x, ...] = np.rot90(origAndVertical[x, ...], 2)
            deg270[x, ...] = np.rot90(origAndVertical[x, ...], 3)

    print ""

    # return originals plus all variations
    return np.concatenate((origAndVertical, deg90, deg180, deg270), axis=0)



def writeOrAppendHDF5(filePath, training_images, training_labels, training_weights=None):
    ''' This function writes a resizable HDF5 dataset containing the passed data,
    or, if a file with the same name already exists, resizes the dataset
    and then appends the data.'''

    if os.path.exists(filePath):
        with h5py.File(filePath, "a", libver="latest") as f:

            print str(filePath) + " exists, appending ... "

            # get number of data points in file
            # (assume that #images = #labels!)
            startpos = f["data"].shape[0]

            # change depth dimension of datasets ("grow" them)
            f["data"].resize(((startpos + training_images.shape[0], ) + training_images.shape[1:4]))
            f["label"].resize(((startpos + training_labels.shape[0], ) + training_labels.shape[1:3]))

            # append new data
            f["data"][startpos:startpos + training_images.shape[0], ...] = training_images
            f["label"][startpos:startpos + training_labels.shape[0], ...] = training_labels

            if training_weights != None:
                f["weights"].resize(((startpos + training_weights.shape[0], ) + training_weights.shape[1:3]))
                f["weights"][startpos:startpos + training_weights.shape[0], ...] = training_weights

    # else, create extendable (chunked) HDF file
    else:
        with h5py.File(filePath, "w", libver="latest") as f:

            print str(filePath) + " doesn't exist, creating ... "

            f.create_dataset("data", dtype=np.float32, data=training_images, maxshape=((None, ) + training_images.shape[1:4]))
            f.create_dataset("label", dtype=np.uint8, data=training_labels, maxshape=((None, ) + training_labels.shape[1:4]))

            if training_weights != None:
                f.create_dataset("weights", dtype=np.float32, data=training_weights, maxshape=((None, ) + training_weights.shape[1:3]))


def getInverseAvgProbs(images):
    '''Returns the average inverse label probabilies
    over all images in the list.'''

    colorcount = np.zeros(4)

    bar = progressbar.ProgressBar()
    for filename in bar(images):
        if filename.endswith("_label.png"): # only look at label images

            labelfile = scipy.ndimage.imread(filename, mode='RGB')

            sizex = labelfile.shape[1]
            sizey = labelfile.shape[0]
            labelfile = labelfile.reshape(sizey, sizex, 3)

            # translate to integer labels
            intlabels = np.zeros((sizey, sizex))

            bluemask = labelfile[..., 2] == 255
            greenmask = labelfile[..., 1] == 255
            redmask = labelfile[..., 0] == 255

            intlabels[bluemask == 1] = 3
            intlabels[greenmask == 1] = 2
            intlabels[redmask == 1] = 1


            # count label colors
            for color in range(4):
                colorcount[color] += list(intlabels.reshape(intlabels.size)).count(color)

    # average over all images
    colorcount /= (len(images) / 2)
    probs = 1. - (colorcount / float(sizex * sizey))

    # return averaged inverse probabilities
    return probs
