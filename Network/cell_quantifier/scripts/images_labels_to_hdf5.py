# This script takes all image and their label-image files in a folder and
# converts them into a HDF5 database. One dataset for images and one for labels
# is created.
# Labels are expected to have the same name as normal images,
# but with "_label" appended.

import sys
import os
import h5py
import numpy as np
import glob
import caffe
import timeit
import math
import scipy
from PIL import Image


FORMAT = "png" # set file extension of format to look for (without dot)
DEF_PIXELWEIGHTS = True # weigh pixels with weight map?

print "Format set as " + FORMAT + "."
print ""

if len(sys.argv) < 3:
    print "Too few arguments!"
    print "Usage: python images_labels_to_hdf5.py /path/to/input_folder /path/to/output_folder"
    sys.exit(-1)

# paths to the image folder containing images and their labels
# and the path to the output HDF5 file
path = sys.argv[1]
hdf5path = sys.argv[2]

# check how many images (minus labels) there are in the folder
# (assuming every image in the folder has a corresponding label image!)
all_images = glob.glob(path + "/*." + FORMAT)
IMG_NO = len(all_images)

if IMG_NO == 0:
    print "No images found in path" + path + " with format ." + FORMAT + "!"
    sys.exit(-1)

if (IMG_NO % 2) != 0:
    print("Number of images didn't match number of labels!") # not foolproof - could use #labels == #images check
    sys.exit(-1)

print "Found " + str(IMG_NO/2) + " image(s) with label(s) in folder."
print ""

# find image height and width (must be uniform across all images/labels!)
size = Image.open(all_images[0]).size
IMG_WIDTH = size[0]
IMG_HEIGHT = size[1]

# temporary arrays for image and label data
# TODO: does Caffe NEED float32? Input images are only 16-bit uint...
data_array = np.zeros((IMG_NO/2, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
label_array = np.zeros((IMG_NO/2, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)


# TODO: work with resizable HDF files so data can be written in chunks!
# TODO: split into training and validation test, e.g. 6:1 scale
# TODO: mean substraction?
# extract image data from all images and labels in the directory
# and store them in the HDF5 database
index = 0

for filename in all_images:
    if not filename.endswith("_label." + FORMAT): # treat label images inside the loop

        print "Packing image " + str(index+1) + "/" + str(IMG_NO/2) + ": " + filename

        # get data handle of image/label (caffe.io uses np.float32, returns vals in [0,1]!)
        imgfile = caffe.io.load_image(filename)
        imgfile = np.transpose(imgfile, (2,0,1)) # switch from HxWxC to CxHxW layout
        labelfile = caffe.io.load_image(filename[:-4] + "_label." + FORMAT)
        labelfile = np.transpose(labelfile, (2,0,1))

        # store current image in HDF5 output array by its index
        data_array[index,:,:,:] = imgfile
        label_array[index,:,:,:] = labelfile

        # number of image/label pairs
        index += 1

print ""

# translate label colors to integer labels in vectorized fashion
print "Translating colored label pixels to integers ..."
label_array_int = np.zeros((IMG_NO/2, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

num_labels = 3 # number of labels in the data
intmask = np.arange(num_labels + 1) # [0, 1, 2, 3]

for index in range(IMG_NO/2):
    print "    Processing label " + str(index + 1) + "/" + str(IMG_NO/2) + " ..."

    # get bool masks for each color component
    bluemask = label_array[index][2] == 1
    greenmask = label_array[index][1] == 1
    redmask = label_array[index][0] == 1

    # label_array_int is already initialized with background label values = 0
    # now simply use masks to place appropriate integer labels for each value
    label_array_int[index][bluemask == 1] = 3
    label_array_int[index][greenmask == 1] = 2
    label_array_int[index][redmask == 1] = 1

print ""




# path to network definition
train_model_def_file = "unet-train.prototxt"
# where to save trained parameters
model_file = "unet.caffemodel"
# adjust this depending on image size / GPU mem size
n_tiles = 1
# = applying 2x2 pooling four times
downsampleFactor = 16
# Size of the image when it reaches layer d4a
d4a_size = 0
# by how many pixels must the image be enlarged to reach original size when reaching layer d4a + 2 convs?
padInput =   (((d4a_size * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2
# by how many pixels will the image be shrunk when it reaches layer d4a + 2 convs?
padOutput = ((((d4a_size - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2
# use average of flipped tile results for classification
average_mirror = True

# calculate d4a_size (d4a_size is a misnomer - two convs follow the layer d4a, so it's layer d4c's image size!),
# and consequently, the padding needed for the convolutions to work
# (shape of data array: (numimages, channels, height, width))
d4a_size = np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor))) \
             .astype(dtype=np.int)
input_size = downsampleFactor * d4a_size + padInput # size the image needs to be padded to
output_size = downsampleFactor * d4a_size + padOutput # original size of the image == size before classification
border = np.around(np.subtract(input_size, output_size)) / 2; # equal padding on both sides

# TODO: (d4a_size + 124) % 16 == 0, dimension condition?


print "Image dims: " + str(data_array.shape[2:4])
print "padInput: " + str(padInput)
print "padOutput: " + str(padOutput)
print "d4a_size: " + str(d4a_size)
print "input_size: " + str(input_size)
print "output_size: " + str(output_size)
print "border_size: " + str(border)

print "border3: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor*4))) \
             .astype(dtype=np.int))
print "border2: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor*2))) \
             .astype(dtype=np.int))
print "border1: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor/4))) \
             .astype(dtype=np.int))
print "border0: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor/2))) \
             .astype(dtype=np.int))

# create zero-padded data array using
# padding values calculated above
paddedFullVolume = np.zeros((data_array.shape[0], \
                            data_array.shape[1], \
                            data_array.shape[2] + 2 * border[0], \
                            data_array.shape[3] + 2 * border[1]), \
                            dtype=np.float32)


# fill in data, leaving the zero borders intact
# image and channel indices have no need for padding, copy them as they are
paddedFullVolume[:, \
                 :, \
                 border[0]:border[0] + data_array.shape[2], \
                 border[1]:border[1] + data_array.shape[3]] = data_array;


# fill zero-padded areas with meaningful data

xpad  = border[0] - 1 # last padding element before image data
xfrom = border[0] # first image element
xto   = border[0] + data_array.shape[3] # first padding element after image data

ypad  = border[1] - 1
yfrom = border[1]
yto   = border[1] + data_array.shape[2]

# fill all 8 zero-padded rectangles adjacent to image with mirrored image data
paddedFullVolume[:, :, yfrom:yto, 0:xfrom] = paddedFullVolume[:, :, yfrom:yto, xfrom + xpad:xfrom - 1:-1] # left
paddedFullVolume[:, :, yfrom:yto, xto:xto + xpad + 1] = paddedFullVolume[:, :, yfrom:yto, xto - 1:xto - xpad - 2:-1] # right
paddedFullVolume[:, :, 0:yfrom, xfrom:xto] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xfrom:xto] # above
paddedFullVolume[:, :, yto:yto + ypad + 1, xfrom:xto] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xfrom:xto] # below

paddedFullVolume[:, :, 0:yfrom, 0:xfrom] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xfrom + xpad:xfrom - 1:-1] # top left
paddedFullVolume[:, :, 0:yfrom, xto:xto + xpad + 1] = paddedFullVolume[:, :, yfrom + ypad:yfrom - 1:-1, xto - 1:xto - xpad - 2:-1] # top right
paddedFullVolume[:, :, yto:yto + ypad + 1, 0:xfrom] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xfrom + xpad:xfrom - 1:-1] # bottom left
paddedFullVolume[:, :, yto:yto + ypad + 1, xto:xto + xpad + 1] = paddedFullVolume[:, :, yto - 1:yto - ypad - 2:-1, xto - 1:xto - xpad - 2:-1] # bottom right


if(DEF_PIXELWEIGHTS == True):
    print "Computing pixel weight image(s) ... "
    # TODO: base on average distribution of classes in training data

    w_c = (0.25, 0.5, 0.75, 1.0) # class weights to counter uneven label distributions
    w_0 = 10.0 # weight multiplicator, tune as necessary; def = 10
    sigma = 5.0 # distance influence dampening, tune as necessary; def = 5

    pixel_weights = np.zeros((int(IMG_NO/2), \
                               IMG_HEIGHT, \
                               IMG_WIDTH), \
                               dtype=np.float32)

    # get matrix of point coordinates
    gradient = np.array([[i, j] for i in range (0, IMG_HEIGHT) for j in range (0, IMG_WIDTH)])

    for index in range(0, IMG_NO/2):
        print "     Calculating pixel weight image " + str(index + 1) + "/" + str(IMG_NO/2)

        # find all label pixels in this image
        labelmask = np.reshape([(label_array_int[index,:,:] == 1) |
                                (label_array_int[index,:,:] == 2) |
                                (label_array_int[index,:,:] == 3)], (IMG_HEIGHT * IMG_WIDTH))

        # put label pixels into KD-tree for fast kNN-lookups
        print gradient[labelmask == True]
        tree = scipy.spatial.cKDTree(gradient[labelmask == True])

        for y in range(0, IMG_HEIGHT):
            print "         Row " + str(y+1) + "/" + str(IMG_HEIGHT)

            for x in range(0, IMG_WIDTH):

                val = label_array_int[index, y, x]

                # if pixel has a cell label ignore distance weighting
                #if val == 1 or val == 2 or val == 3:
                #    pixel_weights[index, y, x] = 0.5 #w_c[val]

                # pixel is labelled as background
                #else:
                # look for the two nearest neighbors of current pixel, using Manhattan distance
                closest, indices = tree.query(np.array([y, x]), k=2, p=1, eps=0.1)

                d1 = closest[0]
                d2 = closest[1]

                # pixel weight = class weight + distance modifier
                pixel_weights[index, y, x] = w_c[val] + w_0 * math.exp(-(((d1 + d2)**2) / (2*(sigma)**2)))





# assign data to according HDF dataset and write HDF file to HDD
print "Writing HDF5 file to " + hdf5path + " ..."
with h5py.File(hdf5path, "w", libver="latest") as f:

    f.create_dataset("data", dtype=np.float32, data=paddedFullVolume)
    f.create_dataset("label", dtype=np.uint8, data=label_array_int)

    if(DEF_PIXELWEIGHTS == True):
        f.create_dataset("weights", dtype=np.float32, data=pixel_weights)

    f.attrs["CLASS"] = "IMAGE"
    f.attrs["IMAGE_VERSION"] = "1.2"
    f.attrs["IMAGE_SUBCLASS"] =  "IMAGE_TRUECOLOR"


# Write test network definition to file
print "Writing test network definition to ../unet-test.prototxt ..."
with open("unet.prototxt", "r") as net, open("unet-test.prototxt", "w") as test_net:
    test_net.write("input: \"data\"\n"); # batch size
    test_net.write("input_dim: " + str(data_array.shape[1]) + "\n"); # channels
    test_net.write("input_dim: " + str(input_size[0]) + "\n"); # height
    test_net.write("input_dim: " + str(input_size[1]) + "\n"); # width
    test_net.write("state: { phase: TEST }" + "\n");
    test_net.write(net.read()); # the rest of the network definition


print "Done!"



# TODO: REMEMBER: even predictions need dummy labels.. apparently. See segmentAndTrack2.m
