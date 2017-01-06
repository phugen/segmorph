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


FORMAT = "bmp" # set file extension of format to look for (without dot)
print "Format set as " + FORMAT + "."

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
# extract image data from all images and labels in the directory
# and store them in the HDF5 database
index = 0

for filename in all_images:
    if not filename.endswith("_label." + FORMAT): # treat label images inside the loop

        print "Packing image-label pair " + str(index+1) + "/" + str(IMG_NO/2) + " ..."

        # get data handle of image/label (caffe.io uses np.float32, returns vals in [0,1]!)
        imgfile = caffe.io.load_image(filename)
        imgfile = np.transpose(imgfile, (2,0,1)) # switch from RGB to BGR, not sure if needed
        labelfile = caffe.io.load_image(filename[:-4] + "_label." + FORMAT)
        labelfile = np.transpose(labelfile, (2,0,1))

        # store current image in temp array
        data_array[index] = imgfile
        label_array[index] = labelfile

        # number of image/label pairs
        index =+ 1


# translate label colors to integer labels in vectorized fashion
print "Translating colored label pixels to integers ..."
label_array_int = np.zeros((IMG_NO/2, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

start = timeit.default_timer()

for index in range(IMG_NO/2):
    print "    Processing label " + str(index + 1) + "/" + str(IMG_NO/2) + " ..."

    # get bool masks for each color component
    bluemask = label_array[index][0] == 1
    greenmask = label_array[index][1] == 1
    redmask = label_array[index][2] == 1

    # label_array_int is already initialized with background label values = 0
    # now simply use masks to place appropriate integer labels for each value
    label_array_int[index][bluemask == 1] = 3
    label_array_int[index][greenmask == 1] = 2
    label_array_int[index][redmask == 1] = 1

# labels must be in Nx1x1x1 format
label_array_int.reshape(IMG_NO/2 * IMG_HEIGHT * IMG_WIDTH, 1, 1, 1)


print "Translating labels took " + str(timeit.default_timer() - start) + " ms."


start = timeit.default_timer()

# assign data to according HDF dataset and write HDF file to HDD
with h5py.File("../training/drosophila_training.h5", "w", libver="latest") as f:

    f.create_dataset("data", data=data_array)
    f.create_dataset("label", data=label_array_int)

    f.attrs["CLASS"] = "IMAGE"
    f.attrs["IMAGE_VERSION"] = "1.2"
    f.attrs["IMAGE_SUBCLASS"] =  "IMAGE_TRUECOLOR"
    f.attrs["IMAGE_MINMAXRANGE"] = np.array([0,1], dtype=np.float32)
    #f.attrs["PALETTE"] =

    f.close()

print "Writing to HDF5 took " + str(timeit.default_timer() - start) + " ms."
print "Done!"
