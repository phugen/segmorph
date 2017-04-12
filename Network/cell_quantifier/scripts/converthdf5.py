# This script takes all images and their label-images in a folder and
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
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt

import progressbar # loading bar for weightmap calc
import preparation_utils as preputils # utility functions like mirroring and tiling

def convert2HDF5(path, hdf5path, weight_mode="individual", un_path=None, override=None):
    FORMAT = "png" # set file extension of format to look for (without dot)
    WEIGHT_MODE = weight_mode # possible modes: "none", "individual", "average", or "override" and set override array

    print "Format set as " + FORMAT + "."
    print ""

    '''if len(sys.argv) < 3:
        print "Too few arguments!"
        print "Usage: python converthdf5.py /input_folder /output_folder/filename [/unlabeled_input_folder]"
        sys.exit(-1)


    # set paths from args
    path = sys.argv[1]
    hdf5path = sys.argv[2]

    if len(sys.argv) < 5:
        un_path = None
    else:
        sys.argv[3]'''


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

    for index in bar(range(IMG_NO/2)):

        # get bool masks for each color component
        bluemask = label_array[index][2] == 1
        greenmask = label_array[index][1] == 1
        redmask = label_array[index][0] == 1

        # label_array_int is already initialized with background label values = 0
        # now simply use masks to place appropriate integer labels for each value
        label_array_int[index][bluemask == 1] = 1 # TODO: change back to 3 if needed
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
    border = np.array(((border[0], border[0]), (border[1], border[1]))) # two dimensional border array: ((up, down), (left, right))

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




    # check if input image is cleanly tileable. If not, enlarge it by mirroring
    # at the bottom and/or right sides to make use of "broken tile" data instead
    # of discarding it.

    input_size = 244 # actual input size without padding
    magic_number = 428 # padded input size that fits the network architecture
    cheight = data_array.shape[3]
    cwidth = data_array.shape[2]

    # check height
    if cheight % magic_number != 0:
        htilenum = cheight / magic_number # get number of complete tiles
        hdiff = ((htilenum + 1) * magic_number) - cwidth # get #pixels by which to enlarge vertically
        print "HDIFF = " + str(hdiff)
        cborder = np.array(((0, 0), (0, hdiff))) # only bottom border
        data_array = preputils.enlargeByMirrorBorder(data_array, cborder) # enlarge bottom side by mirroring

    # check width
    if cwidth % magic_number != 0:
        wtilenum = cwidth / magic_number
        wdiff = ((wtilenum + 1) * magic_number) - cheight
        print "WDIFF = " + str(wdiff)
        cborder = np.array(((0, wdiff), (0, 0)))
        data_array = preputils.enlargeByMirrorBorder(data_array, cborder) # enlarge right side by mirroring

    scipy.misc.toimage(data_array[0, ...], cmin=0, cmax=1).save("hide/hide/mirrored.png")

    # create enlarged input image by mirroring edges
    paddedFullVolume = preputils.enlargeByMirrorBorder(data_array, border)

    #TODO remove this
    scipy.misc.toimage(paddedFullVolume[0, ...], cmin=0, cmax=1).save("hide/hide/mirrored_REAL.png")
    exit(-1)



    PAD_HEIGHT = paddedFullVolume.shape[2]
    PAD_WIDTH = paddedFullVolume.shape[3]


    # tile images and labels so they fit network input size
    paddedSubImages = preputils.tileMirrorImage (IMG_NO/2, input_size, magic_number, paddedFullVolume, \
                                                 [IMG_HEIGHT, IMG_WIDTH], [PAD_HEIGHT, PAD_WIDTH])


    subLabels = preputils.tileMirrorLabel (IMG_NO/2, input_size, label_array_int, \
                                           [IMG_HEIGHT, IMG_WIDTH])



    # throw out subimages and their labels
    # that possess only samples of one class
    print "\nDeleting images with only one class..."
    todelete = [] # find indices of samples to delete
    bar = progressbar.ProgressBar()
    for index, sub in enumerate(subLabels):

        justone = True
        class1 = sub[0, 0]

        for y in range(sub.shape[0]):
            for x in range(sub.shape[1]):
                if sub[y, x] != class1:
                    justone = False
                    break
            if not justone:
                break

        if justone:
            todelete.append(index)

    # delete all marked images and labels in sorted
    # reverse order so indices prior indices
    # don't change and throw off deletion
    for index in sorted(todelete, reverse=True):
        del paddedSubImages[index]
        del subLabels[index]

    print "DELETED " + str(len(todelete)) + " images with one only class!\n"

    # convert lists to numpy array representation
    paddedSubImages = np.array(paddedSubImages)
    subLabels = np.array(subLabels)

    # total number of sub-images and labels
    SUB_NO = paddedSubImages.shape[0]
    SUB_L_NO = subLabels.shape[0]


    # set class weights for weighting
    class_probs = np.zeros((SUB_NO, 4)) # BG, R, G, B
    if WEIGHT_MODE == "individual":

        # calculate inverse color probability for each color in each image
        print "Calculating INDIVIDUAL class weights:"

        bar = progressbar.ProgressBar()
        for index in bar(range(SUB_NO)):
            for color in range(4):
                class_probs[index, color] = 1. - float(list(subLabels[index].reshape(input_size * input_size)).count(color)) / (input_size * input_size)

                # if an image only contains one class,
                # weigh that class with a very small factor but not 0
                # to prevent "sum of pixel losses is zero" error
                if (abs(0. - class_probs[index, color]) < 0.00001):
                    class_probs[index, color] = 0.00001
        print ""


    # use external class probabilities, for instance global average probs
    elif WEIGHT_MODE == "override":

        if override is None:
            raise ValueError("WEIGHT_MODE was override, but no override array was supplied!")

        class_probs = np.zeros((SUB_NO, 4))
        class_probs[:] = override



    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":
        print "Computing pixel weight image(s) ... "
        # TODO: base on average distribution of classes in training data

        #w_c = (0.1, 0.75, 0.25, 1.0) # class weights to counter uneven label distributions
        w_0 = 10.0 # weight multiplicator, tune as necessary; def = 10
        sigma = 5.0 # distance influence dampening, tune as necessary; def = 5

        pixel_weights = np.zeros((SUB_NO, \
                                   input_size, \
                                   input_size), \
                                   dtype=np.float32)

        # get matrix of point coordinate tuples
        #gradient = np.array([[i, j] for i in range (0, input_size) for j in range (0, input_size)])
        gradient = np.array(list(np.ndindex(input_size, input_size)))

        bar = progressbar.ProgressBar()
        for index in bar(range(SUB_NO)):

            w_c = class_probs[index, :]

            # find all label pixels in this image
            labelmask = np.reshape([(subLabels[index,:,:] == 1) |
                                    (subLabels[index,:,:] == 2) |
                                    (subLabels[index,:,:] == 3)], (input_size * input_size))

            label_pixels = gradient[labelmask == True]

            # put label pixels into KD-tree for fast kNN-lookups
            if label_pixels.size != 0:
                tree = scipy.spatial.cKDTree(label_pixels)
            else:
                tree = None # in case the input image only has BG values

            # Calculate weight map
            for y in range(0, input_size):
                for x in range(0, input_size):

                    val = subLabels[index, y, x]

                    # look for the two nearest neighbors of current pixel, using Manhattan distance
                    if tree != None:
                        #closest, indices = tree.query(np.array([y, x]), k=2, p=1, eps=0.1)

                        #d1 = closest[0]
                        #d2 = closest[1]

                        # TODO: RE-ENABLE WEIGHTMAP
                        # pixel weight = class weight + distance modifier
                        # pixel_weights[index, y, x] = w_c[val] + w_0 * math.exp(-(((d1 + d2)**2) / (2*(sigma)**2)))
                        pixel_weights[index, y, x] = w_c[val]

                    else:
                        pixel_weights[index, y, x] = w_c[val]


    # Data augmentation: Flip and rotate images
    # to get more training samples
    flippedRotatedImages = preputils.getFlipRotationCombinations(paddedSubImages, 3)
    flippedRotatedLabels = preputils.getFlipRotationCombinations(subLabels, 1)

    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":
        flippedRotatedWeights = preputils.getFlipRotationCombinations(pixel_weights, 1)



    # split images and labels into training and validation set
    ratio = 5 # split ratio, each x-th training sample used for validation

    # TRAINING set:
    training_images = [flippedRotatedImages[index, ...] for index in range(flippedRotatedImages.shape[0]) if index % ratio != 0]
    training_labels = [flippedRotatedLabels[index, ...] for index in range(flippedRotatedLabels.shape[0]) if index % ratio != 0]

    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":
        training_weights = [flippedRotatedWeights[index, ...] for index in range(flippedRotatedWeights.shape[0]) if index % ratio != 0]

    # VALIDATION set:
    validation_images = flippedRotatedImages[::ratio, ...]
    validation_labels = flippedRotatedLabels[::ratio, ...]

    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":
        validation_weights = flippedRotatedWeights[::ratio, ...] # actually not needed...



    # Preprocessing:
    # Compute mean and standard deviation over training set ONLY
    # for each RGB channel independently and then substract from
    # validation set to prevent validation set information from biasing the network
    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    training_weights = np.array(training_weights)

    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)
    validation_weights = np.array(validation_weights)

    training_means = [np.mean(training_images[:, 0, ...]), \
                      np.mean(training_images[:, 1, ...]), \
                      np.mean(training_images[:, 2, ...])]

    training_stds = [np.std(training_images[:, 0, ...]), \
                    np.std(training_images[:, 1, ...]), \
                    np.std(training_images[:, 2, ...])]


    # center data by substracting training mean
    for i in range(3):
        training_images[:, i, ...] -= training_means[i]
        validation_images[:, i, ...] -= training_means[i]

    # normalize data by dividing by training standard deviation
    for i in range(3):
        training_images[:, i, ...] /= training_stds[i]
        validation_images[:, i, ...] /= training_stds[i]


    # TODO: (del) write preprocessed images out
    #for x in range(training_images.shape[0]):
    #    scipy.misc.toimage(training_images[x]).save("./training_4final/debug/train_" + str(x) + ".png")

    #for x in range(validation_images.shape[0]):
    #    scipy.misc.toimage(validation_images[x]).save("./training_4final/debug/valid_" + str(x) + ".png")



    print "Writing HDF5 files to " + hdf5path + " ..."

    # Create HDF5 datasets WITH weightmaps
    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":

        preputils.writeOrAppendHDF5(hdf5path + "_training.h5", \
                                    training_images, training_labels, training_weights)

        preputils.writeOrAppendHDF5(hdf5path + "_validation.h5", \
                                    validation_images, validation_labels, validation_weights)

    else:
        preputils.writeOrAppendHDF5(hdf5path + "_training.h5", \
                                    training_images, training_labels)

        preputils.writeOrAppendHDF5(hdf5path + "_validation.h5", \
                                    validation_images, validation_labels)


    # create unlabelled data HDF5 file for
    # pseudo-labelling if needed
    if un_path != None:

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

        if WEIGHT_MODE == "individual" or WEIGHT_MODE == "override":
            pseudo_weights = np.zeros((len(subUnlabeled), input_size, input_size), dtype=np.float32)
            pseudo_weights[...] = 1.0

            preputils.writeOrAppendHDF5(hdf5path + "_unlabelled.h5", \
                                            subUnlabeled, None, pseudo_weights)
        else:
            preputils.writeOrAppendHDF5(hdf5path + "_unlabelled.h5", \
                                            subUnlabeled, None)


    # If training file already exists, add data to existing file
    #if os.path.exists(hdf5path + "/drosophila_training.h5"):
    #    with h5py.File(hdf5path + "/drosophila_training.h5", "a", libver="latest") as f:
    #
            # get number of data points in file
            # (assume that #images = #labels!)
    #        startpos = f["data"].shape[0]
    #
            # change depth dimension of datasets ("grow" them)
    #        f["data"].resize((startpos + training_images.shape[0], training_images.shape[1:4]))
    #        f["label"].resize((startpos + training_labels.shape[0], training_labels.shape[1:3]))
    #
            # append new data
    #        f["data"][startpos:startpos + training_images.shape[0], ...] = training_images
    #        f["label"][startpos:startpos + training_labels.shape[0], ...] = training_labels
    #
    #        if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    #            f["weights"].resize((startpos + training_weights.shape[0], training_weights.shape[1:3]))
    #            f["data"][startpos:startpos + training_weights.shape[0], ...] = training_weights

    # else, create extendable (chunked) HDF file
    #else:
    #    with h5py.File(hdf5path + "/drosophila_training.h5", "w", libver="latest") as f:
    #
    #        f.create_dataset("data", dtype=np.float32, data=training_images, maxshape=(None, training_images.shape[1:4]))
    #        f.create_dataset("label", dtype=np.uint8, data=training_labels, maxshape=(None, training_labels.shape[1:4]))
    #
    #        if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    #            f.create_dataset("weights", dtype=np.float32, data=training_weights, maxshape=(None, training_weights.shape[1:3]))
    #


    # write validation set HDF5 file
    #with h5py.File(hdf5path + "/drosophila_validation.h5", "w", libver="latest") as f:
    #
    #    f.create_dataset("data", dtype=np.float32, data=validation_images)
    #    f.create_dataset("label", dtype=np.uint8, data=validation_labels)
    #
    #    if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    #        f.create_dataset("weights", dtype=np.float32, data=validation_weights)



    # write unlabelled data HDF5 file
    #if un_path != None:
    #    with h5py.File(hdf5path + "/drosophila_unlabelled.h5", "w", libver="latest") as f:
    #
            # pack unlabelled images into numpy array
    #        all_un = glob.glob(un_path + "/*." + FORMAT)
    #        UN_NUM = len(all_un)
    #        un_data_array = np.zeros((UN_NUM, 3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    #
            # get raw image data
    #        for index, filename in enumerate(all_un):
    #            imgfile = caffe.io.load_image(filename)
    #            imgfile = np.transpose(imgfile, (2,0,1))
    #            un_data_array[index, ...] = imgfile
    #
            # mirror and tile label images
    #        paddedUnlabeled = preputils.enlargeByMirrorBorder(un_data_array, border)
    #        subUnlabeled = preputils.tileMirrorImage (UN_NUM, input_size, magic_number, paddedUnlabeled, \
    #                                                  [IMG_HEIGHT, IMG_WIDTH], [PAD_HEIGHT, PAD_WIDTH])
    #
    #
    #        f.create_dataset("data", dtype=np.float32, data=subUnlabeled)
    #
    #        if WEIGHT_MODE == "individual" or WEIGHT_MODE == "average":
    #            pseudo_weights = np.zeros((len(subUnlabeled), input_size, input_size), dtype=np.float32)
    #            pseudo_weights[...] = 1.0
    #            f.create_dataset("weights", dtype=np.float16, data=pseudo_weights)


    tr_no = len(training_images)
    val_no = len(validation_images)

    if 'subUnlabeled' in globals():
        unlab_no = len(subUnlabeled)
    else:
        unlab_no = 0

    print "Written " + str(tr_no) + " training image(s), " \
                     + str(val_no) + " validation image(s) and " \
                     + str(unlab_no) + " unlabelled images to " \
                     + str(hdf5path) + "!\n"



    print "Done!"
