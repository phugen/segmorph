# This file contains utility functions for deep learning data preparation etc.

import math
import numpy as np


def enlargeByMirrorBorder(input_image, border):
    " Enlarge RGB images (imgno X channel X height x width) \
     by a certain border by mirroring their edges \
     create output array, enlarged by borders"

    paddedFullVolume = np.zeros((input_image.shape[0], \
                                input_image.shape[1], \
                                input_image.shape[2] + 2 * border[0], \
                                input_image.shape[3] + 2 * border[1]), \
                                dtype=np.float32)

    print "Original image size: " + str(input_image.shape)
    print "Enlarged image size: " + str(paddedFullVolume.shape) + "\n"

    # fill in original data in center, leaving the zero borders intact
    # image and channel indices have no need for padding, copy them as they are
    paddedFullVolume[:, \
                     :, \
                     border[0]:border[0] + input_image.shape[2], \
                     border[1]:border[1] + input_image.shape[3]] = input_image;


    # fill zero-padded areas with meaningful data
    xpad  = border[0] - 1 # last padding element before image data
    xfrom = border[0] # first image element
    xto   = border[0] + input_image.shape[3] # first padding element after image data

    ypad  = border[1] - 1
    yfrom = border[1]
    yto   = border[1] + input_image.shape[2]

    # do mirroring
    paddedFullVolume[:, :, yfrom:yto, 0:xfrom] = paddedFullVolume[:, :, yfrom:yto, xfrom + xpad:xfrom - 1:-1] # left
    paddedFullVolume[:, :, yfrom:yto, xto:xto + xpad + 1] = paddedFullVolume[:, :, yfrom:yto, xto - 1:xto - xpad - 2:-1] # right
    paddedFullVolume[:, :, 0:yfrom, xfrom:xto] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xfrom:xto] # above
    paddedFullVolume[:, :, yto:yto + ypad + 1, xfrom:xto] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xfrom:xto] # below

    paddedFullVolume[:, :, 0:yfrom, 0:xfrom] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xfrom + xpad:xfrom - 1:-1] # top left
    paddedFullVolume[:, :, 0:yfrom, xto:xto + xpad + 1] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xto - 1:xto - xpad - 2:-1] # top right
    paddedFullVolume[:, :, yto:yto + ypad + 1, 0:xfrom] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xfrom + xpad:xfrom - 1:-1] # bottom left
    paddedFullVolume[:, :, yto:yto + ypad + 1, xto:xto + xpad + 1] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xto - 1:xto - xpad - 2:-1] # bottom right


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

    for index in range(imgno):
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

    return subLabels
