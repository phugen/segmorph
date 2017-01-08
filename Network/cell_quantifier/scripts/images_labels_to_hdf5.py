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
from PIL import Image


FORMAT = "png" # set file extension of format to look for (without dot)
print "Format set as " + FORMAT + "."
print ""

if len(sys.argv) < 2:
    print "Too few arguments!"
    print "Usage: python images_labels_to_hdf5.py /path/to/image_folder"
    sys.exit(-1)

# path to the image folder containing images and their labels
# (Passed on the command line)
path = sys.argv[1]

# check how many images (minus labels) there are in the folder
# (assuming every image in the folder has a corresponding label image!)
all_images = glob.glob(path + "/*." + FORMAT)
IMG_NO = len(all_images)

if IMG_NO == 0:
    print "No images found in path" + path + " with format ." + FORMAT + "!"
    sys,exit(-1)

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

#print "img1:\n" + str(label_array[0,:,:,:])
#print "img2:\n" + str(data_array[1,:,:,:])
#print "img3:\n" + str(data_array[2,:,:,:])

# translate label colors to integer labels in vectorized fashion
print "Translating colored label pixels to integers ..."
start = timeit.default_timer()
label_array_int = np.zeros((IMG_NO/2, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
#label_array_hot = np.zeros((IMG_NO/2, IMG_HEIGHT, IMG_WIDTH, 4), dtype=np.uint8)

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

    # labels for every pixel must be BGR one-hot encoded: [0, 0, 0, 1] = red etc.
    #label_array_int[index] = (intmask == label_array_int[:,:,:,np.newaxis]).astype(int)
    #label_array_hot[index] = np.eye(num_labels + 1)[label_array_int[index]]

print "Translating labels took " + str(timeit.default_timer() - start) + " ms."
print ""




# padding parameters
train_model_def_file = "unet-train.prototxt"
model_file = "unet.caffemodel"
n_tiles = 1 # adjust this depending on image size / GPU mem size
downsampleFactor = 16
d4a_size = 0
padInput =   (((d4a_size * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2) * 2 + 2 + 2
padOutput = ((((d4a_size - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2) * 2 - 2 - 2
average_mirror = True

# calculate the image's size once it reaches layer d4a
# and consequently, the padding needed for the convolutions to work
# (shape of data array: (numimages, channels, height, width))
d4a_size = np.divide(np.subtract([data_array.shape[2], (data_array.shape[3] / n_tiles)], padOutput), downsampleFactor)
input_size = downsampleFactor * d4a_size + padInput # [height, width] vector
output_size = downsampleFactor * d4a_size + padOutput

print "d4a_size: " + str(d4a_size)
print "input_size: " + str(input_size)
print "output_size: " + str(output_size)

# create zero-padded data array using
# padding values calculated above
border = np.around(np.subtract(input_size, output_size)) / 2;
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

# fill zero-padded areas with meaningful data.
# in this case: mirror the present data
xpad  = border[0];
xfrom = border[0] + 1;
xto   = border[0] + size(data,1);
paddedFullVolume(1:xfrom - 1,:,:) = paddedFullVolume( xfrom + xpad:-1:xfrom + 1,:,:);
paddedFullVolume(xto + 1:end,:,:) = paddedFullVolume( xto - 1:-1:xto - xpad,:,:);

ypad  = border(2);
yfrom = border(2)+1;
yto   = border(2)+size(data,2);
paddedFullVolume(:, 1:yfrom-1,:) = paddedFullVolume( :, yfrom+ypad:-1:yfrom+1,:);
paddedFullVolume(:, yto+1:end,:) = paddedFullVolume( :, yto-1:-1:yto-ypad,    :);


start = timeit.default_timer()
# assign data to according HDF dataset and write HDF file to HDD
with h5py.File("../training/drosophila_training.h5", "w", libver="latest") as f:

    f.create_dataset("data", dtype=np.float32, data=paddedFullVolume)
    f.create_dataset("label", dtype=np.uint8, data=label_array_int)

    f.attrs["CLASS"] = "IMAGE"
    f.attrs["IMAGE_VERSION"] = "1.2"
    f.attrs["IMAGE_SUBCLASS"] =  "IMAGE_TRUECOLOR"
    #f.attrs["IMAGE_MINMAXRANGE"] = np.array([0,1], dtype=np.float32)
    #f.attrs["PALETTE"] =

    #f.close()

print "Writing to HDF5 took " + str(timeit.default_timer() - start) + " ms."
print "Done!"

with h5py.File("../training/drosophila_training.h5", "r", libver="latest") as f:
    print f["data"].shape
    print f["label"].shape

# TODO: REMEMBER: even predictions need dummy labels.. apparently. See segmentAndTrack2.m
