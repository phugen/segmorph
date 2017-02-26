# This script takes all image and their label-image files in a folder and
# converts them into a HDF5 database. One h5 file for training and one for testing is created.
# Labels are expected to have the same name as their corresponding images, but with "_label" appended.

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

import progressbar # loading bar for weightmap calc
import preparation_utils as preputils # utility functions like mirroring and tiling


FORMAT = "png" # set file extension of format to look for (without dot)
WEIGHT_MODE = "individual" # possible modes: "none", "individual", "average"
CREATE_UNLABELED = True # create extra HDF5 container for unlabelled data to use for pseudo-labelling?

print "Format set as " + FORMAT + "."
print ""

if len(sys.argv) < 4:
    print "Too few arguments!"
    print "Usage: python images_labels_to_hdf5.py /input_folder /unlabeled_input_folder /output_folder"
    sys.exit(-1)

# paths to the image folder containing images and their labels
# and the path to the output HDF5 file
path = sys.argv[1]
un_path = sys.argv[2]
hdf5path = sys.argv[3]


# check how many images (minus labels) there are in the folder
# (assuming every image in the folder has a corresponding label image!)
all_images = glob.glob(path + "/*." + FORMAT)
IMG_NO = len(all_images)

if IMG_NO == 0:
    print "No images found in path " + path + " with format ." + FORMAT + "!"
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
data_array = np.zeros((IMG_NO/2, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
label_array = np.zeros((IMG_NO/2, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

# extract image data from all images and labels in the directory
# and store them in the HDF5 database
index = 0

bar = progressbar.ProgressBar()
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

        index += 1

print ""

# translate label colors to integer labels in vectorized fashion
print "Translating colored label pixels to integers ..."
label_array_int = np.zeros((IMG_NO/2, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

num_labels = 3 # number of labels in the data
intmask = np.arange(num_labels + 1) # [0, 1, 2, 3]

for index in bar(range(IMG_NO/2)):

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

# (d4a_size + 124) % 16 == 0, dimension condition


#print "Image dims: " + str(data_array.shape[2:4])
#print "padInput: " + str(padInput)
#print "padOutput: " + str(padOutput)
#print "d4a_size: " + str(d4a_size)
#print "input_size: " + str(input_size)
#print "output_size: " + str(output_size)
#print "border_size: " + str(border)

#print "border3: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor*4))) \
#             .astype(dtype=np.int))
#print "border2: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor*2))) \
#             .astype(dtype=np.int))
#print "border1: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor/4))) \
#             .astype(dtype=np.int))
#print "border0: " + str(np.ceil(((np.array([data_array.shape[2], np.ceil(data_array.shape[3] / n_tiles)]) - padOutput) / float(downsampleFactor/2))) \
#             .astype(dtype=np.int))


# create enlarged input image by mirroring edges
paddedFullVolume = preputils.enlargeByMirrorBorder(data_array, border)


input_size = 244 # actual input size without padding
magic_number = 428 # padded input size that fits the network architecture
PAD_HEIGHT = paddedFullVolume.shape[2]
PAD_WIDTH = paddedFullVolume.shape[3]


# tile images and labels so they fit network input size
paddedSubImages = preputils.tileMirrorImage (IMG_NO/2, input_size, magic_number, paddedFullVolume, \
                                             [IMG_HEIGHT, IMG_WIDTH], [PAD_HEIGHT, PAD_WIDTH])


subLabels = preputils.tileMirrorLabel (IMG_NO/2, input_size, label_array_int, \
                                       [IMG_HEIGHT, IMG_WIDTH])


# total number of sub-images and labels
SUB_NO = len(paddedSubImages)
SUB_L_NO = len(subLabels)

# convert lists to numpy array representation
paddedSubImages = np.array(paddedSubImages)
subLabels = np.array(subLabels)






# TODO: add ROTATION and MEAN SUBSTRACTIOn etc






# set class weights for weighting
class_probs = np.zeros((SUB_NO, 4)) # BG, R, G, B
if WEIGHT_MODE == "individual":

    # calculate inverse color probability for each color in each image
    print "Calculating INDIVIDUAL class weights:"

    bar = progressbar.ProgressBar()
    for index in bar(range(SUB_NO)):
        for color in range(4):
            class_probs[index, color] = 1. - float(list(subLabels[index].reshape(input_size * input_size)).count(color)) / (input_size * input_size)

    print ""


elif WEIGHT_MODE == "average":
    temp_probs = np.zeros(4)

    # calculate inverse average color probability over all images
    print "Calculating AVERAGE class weights:"

    bar = progressbar.ProgressBar()
    for index in bar(range(SUB_NO)):
        for color in range(4):
            temp_probs[color] += 1. - float(list(subLabels[index].reshape(input_size * input_size)).count(color)) / (input_size * input_size)

    temp_probs /= SUB_NO
    for i in range(4):
        class_probs[:, i] = temp_probs[i] # fill probs for all images with the same, averaged values

    print ""



if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    print "Computing pixel weight image(s) ... "
    # TODO: base on average distribution of classes in training data

    #w_c = (0.1, 0.75, 0.25, 1.0) # class weights to counter uneven label distributions
    w_0 = 10.0 # weight multiplicator, tune as necessary; def = 10
    sigma = 5.0 # distance influence dampening, tune as necessary; def = 5

    pixel_weights = np.zeros((SUB_NO, \
                               input_size, \
                               input_size), \
                               dtype=np.float32)

    # get matrix of point coordinates
    gradient = np.array([[i, j] for i in range (0, input_size) for j in range (0, input_size)])

    bar = progressbar.ProgressBar()
    for index in bar(range(SUB_NO)):

        w_c = class_probs[index, :]

        # find all label pixels in this image
        labelmask = np.reshape([(subLabels[index,:,:] == 1) |
                                (subLabels[index,:,:] == 2) |
                                (subLabels[index,:,:] == 3)], (input_size * input_size))

        # put label pixels into KD-tree for fast kNN-lookups
        tree = scipy.spatial.cKDTree(gradient[labelmask == True])

        # Calculate weight map
        for y in range(0, input_size):
            for x in range(0, input_size):

                val = subLabels[index, y, x]

                # look for the two nearest neighbors of current pixel, using Manhattan distance
                # TODO: enable tree lookups
                #closest, indices = tree.query(np.array([y, x]), k=2, p=1, eps=0.1)

                d1 = 0.00001#closest[0]
                d2 = 0.00001#closest[1]

                # pixel weight = class weight + distance modifier
                pixel_weights[index, y, x] = w_c[val] + w_0 * math.exp(-(((d1 + d2)**2) / (2*(sigma)**2)))




# split images and labels into training and validation set
ratio = 5 # split ratio, each x-th training sample used for validation

# TRAINING set:
training_images = [paddedSubImages[index, ...] for index in range(paddedSubImages.shape[0]) if index % ratio != 0]
training_labels = [subLabels[index, ...] for index in range(subLabels.shape[0]) if index % ratio != 0]

if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    training_weights = [pixel_weights[index, ...] for index in range(pixel_weights.shape[0]) if index % ratio != 0]

# VALIDATION set:
validation_images = paddedSubImages[::ratio, ...]
validation_labels = subLabels[::ratio, ...]

if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    validation_weights = pixel_weights[::ratio, ...] # actually not needed...





print "Writing HDF5 files to " + hdf5path + " ..."

# write training HDF5 file
with h5py.File(hdf5path + "/drosophila_training.h5", "w", libver="latest") as f:

    f.create_dataset("data", dtype=np.float32, data=training_images)
    f.create_dataset("label", dtype=np.uint8, data=training_labels)

    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
        f.create_dataset("weights", dtype=np.float32, data=training_weights)



# write validation set HDF5 file
with h5py.File(hdf5path + "/drosophila_validation.h5", "w", libver="latest") as f:

    f.create_dataset("data", dtype=np.float32, data=validation_images)
    f.create_dataset("label", dtype=np.uint8, data=validation_labels)

    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
        f.create_dataset("weights", dtype=np.float32, data=validation_weights)



# write unlabelled data HDF5 file
if CREATE_UNLABELED == True:
    with h5py.File(hdf5path + "/drosophila_unlabelled.h5", "w", libver="latest") as f:

        # pack unlabelled images into numpy array
        all_un = glob.glob(un_path + "/*." + FORMAT)
        UN_NUM = len(all_un)
        un_data_array = np.zeros((UN_NUM, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

        # get raw image data
        for index, filename in enumerate(all_un):
            imgfile = caffe.io.load_image(filename)
            imgfile = np.transpose(imgfile, (2,0,1))
            un_data_array[index, ...] = imgfile

        # mirror and tile label images
        paddedUnlabeled = preputils.enlargeByMirrorBorder(un_data_array, border)
        subUnlabeled = preputils.tileMirrorImage (UN_NUM, input_size, magic_number, paddedUnlabeled, \
                                                  [IMG_HEIGHT, IMG_WIDTH], [PAD_HEIGHT, PAD_WIDTH])


        f.create_dataset("data", dtype=np.float32, data=subUnlabeled)

        if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
            pseudo_weights = np.zeros((len(subUnlabeled), input_size, input_size), dtype=np.float32)
            pseudo_weights[...] = 1.0
            f.create_dataset("weights", dtype=np.float32, data=pseudo_weights)




print "Written " + str(len(training_images)) + " training image(s), " \
                 + str(len(validation_images)) + " validation image(s) and " \
                 + str(len(subUnlabeled)) + " unlabelled images to " \
                 + str(hdf5path) + "!\n"

# TODO: (del) Write mirrored images out for testing
#scipy.misc.toimage(paddedFullVolume[0]).save("./training_4final/mirrored.png")


print "Done!"
