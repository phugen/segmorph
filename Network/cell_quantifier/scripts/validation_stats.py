# Collects raw validation statistics (TP, FP, FN)
# by comparing classification results to ground truth images
# and saves the resulting values by image in a CSV file.

import numpy as np
import glob
import sys
import scipy.ndimage
import progressbar


def RGB_image_to_labels(image, numclasses):
    ''' Transforms a 3- or 4- label image into
    integer labels. '''

    if numclasses < 3 or numclasses > 4:
        raise ValueError("numclasses has to be = 3 or = 4!")

    # init label image with zeroes
    label_array_int = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # get bool masks for each color component
    if numclasses == 4:
        bluemask = image[..., 2] == 255
    greenmask = image[..., 1] == 255
    redmask = image[..., 0] == 255

    # label_array_int is already initialized with background label values = 0
    # now simply use masks to place appropriate integer labels for each value
    if numclasses == 4:
        label_array_int[bluemask == True] = 3
    label_array_int[greenmask == True] = 2
    label_array_int[redmask == True] = 1

    # return int labels
    return label_array_int



def getstats(labels, gt, numclasses):
    ''' Calculates multi-class information-retrieval stats. '''

    # generate sublists for each classes'
    # individual F-Measure scores
    classwise = []

    for c in range(numclasses):

        # calculate multi-class per-image stats by alternatingly
        # treating each class as "true" and all others as "false"
        #
        # NOTE: use "hack": Python counts "true" as "1"
        # and "false" as "0", so np.sum(arr) is the same
        # as  list(arr).count(True), but much faster!

        # true positives (TP):
        # ground truth is class "c" and
        # labels indicate this class as well
        tp = np.sum((gt == c) & (labels == c))

        # false positives (FP):
        # ground truth is not class "c" but
        # labels indicates "i" nonetheless
        fp = np.sum((gt != c) & (labels == c))


        # false negatives (FN):
        # ground truth is class "c" but
        # label indicates another class
        fn = np.sum((gt == c) & (labels != c))

        # store results for current class in sublist
        classwise.append([tp, fp, fn])


    return classwise



if len(sys.argv) < 4:
    print "Usage: python validation_stats.py numclasses input_path output_file"
    exit(-1)

numclasses = int(sys.argv[1])
input_path = sys.argv[2]
output_file = sys.argv[3]

print "Finding files ..."

# get all input files
input_files = glob.glob(str(input_path) + "/*.png")

print str(len(input_files)) + " files found in " + input_path + " !"
print "Processing images ... "

# get stats for all files and write them to a CSV file
with open(output_file, "w+") as csv:

    # write header
    print >> csv, "# F-Measure stats for segmented images. Each block corresponds to the stats of each individual class."
    print >> csv, "# Image no, TP, FP, FN"

    imgno = 0 # count label + GT as pair instead of individually
    bar = progressbar.ProgressBar()
    for fi in bar(input_files):
        if not fi.endswith("_GT.png"): # handle labels inside loop

            # translate segmented image and GT image to integer labels
            labels = RGB_image_to_labels(scipy.ndimage.imread(fi), numclasses)
            ground_truth = RGB_image_to_labels(scipy.ndimage.imread(fi[:-4] + "_GT.png"), numclasses)

            # get stats
            tmp = getstats(labels, ground_truth, numclasses)

            # write stats to file as block of stats for each class,
            # one class per line
            for i in range(len(tmp)):
                print >> csv, str(imgno) + "," + str(tmp[i][0]) + "," + str(tmp[i][1]) + "," + str(tmp[i][2])
            print >> csv, ""

            imgno += 1


print "Saved stats for " + str(imgno) + " images in" + output_file + "!"
