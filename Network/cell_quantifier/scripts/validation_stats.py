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

    # cumulative stats over all classes of the image
    alltp = 0
    allfn = 0
    allfp = 0

    # get 1D representation of labels and GT
    # since this is easier to manage
    labels = labels.reshape(labels.size)
    gt = gt.reshape(labels.size)

    # calculate multi-class per-image stats by alternatingly
    # treating each class as "true" and all others as "false"
    numlabels = labels.size
    for c in range(numclasses):

        # true positives (TP):
        # ground truth is class "i" and
        # labels indiciate this class as well
        tp = [gt[i] == c and labels[i] == c for i in range(numlabels)] \
             .count(True)

        # false positives (FP):
        # ground truth is not class "i" but
        # labels indicates "i" nonetheless
        fp = [gt[i] != c and labels[i] == c for i in range(numlabels)] \
             .count(True)

        # false negatives (FN):
        # ground truth is class "i" but
        # label indicates another class
        fn = [gt[i] == c and labels[i] != c for i in range(numlabels)] \
             .count(True)

        # update overall stats
        alltp += tp
        allfp += fp
        allfn += fn

    return [alltp, allfp, allfn]



if len(sys.argv) < 4:
    print "Usage: python numclasses validation_stats.py input_path output_file"

numclasses = int(sys.argv[1])
input_path = sys.argv[2]
output_file = sys.argv[3]

# get all input files
input_files = glob.glob(str(input_path) + "/*.png")

# get stats for all files and write them to a CSV file
with open(output_file, "w+") as csv:

    # write header
    print >> csv, "#, TP, FP, FN"

    imgno = 0 # count label + GT as pair instead of individually
    bar = progressbar.ProgressBar()
    for fi in bar(input_files):
        if not fi.endswith("_GT.png"): # handle labels inside loop

            # translate segmented image and GT image to integer labels
            labels = RGB_image_to_labels(scipy.ndimage.imread(fi), numclasses)
            ground_truth = RGB_image_to_labels(scipy.ndimage.imread(fi[:-4] + "_GT.png"), numclasses)

            # get stats
            tmp = getstats(labels, ground_truth, numclasses)

            # write stats to file as line
            print >> csv, str(imgno) + "," + str(tmp[0]) + "," + str(tmp[1]) + "," + str(tmp[2])

            imgno += 1


print "Saved stats for " + str(imgno) + " images in" + output_file + "!"
