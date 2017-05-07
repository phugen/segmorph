# Measure how well a classificator segments an image, measured by
# the F-Measure score of the best class permutation.
# For F-Measure formulas, see https://en.wikipedia.org/wiki/F1_score

import PIL.Image as Image
import numpy as np
import h5py
import progressbar
import glob
import time

from permute_validation import labels_as_RGB_image
from permute_validation import fmeasure

def validate(func, numclasses, inpath, outpath):
    ''' Validates a function func using a HDF5 file
    at <inpath> and saves the results in <outpath>. '''

    # find all validation files in inpath
    valfiles = glob.glob(inpath + "*validation.h5")

    # predict all files
    imoffset = 0 # image offset per file
    for idx,h5file in enumerate(valfiles):

        # load validation data from a HDF5 file
        with h5py.File(h5file, "r", libver="latest") as f:

            # predict all validation images
            no_samples = f["data"].shape[0]

            # TODO: remove
            if idx < 9:
                imoffset +=no_samples
                continue

            print "Predicting file number " + str(idx) + "/" + str(len(valfiles)-1) + ":"
            bar = progressbar.ProgressBar()
            for no, i in enumerate(bar(range(no_samples))):

                # image is grayscale anyway, reduce to 1 channel
                # and shave off mirror padding because it is not needed
                input_img = f["data"][i, 0, 90:334, 90:334]
                ground_truth = f["label"][i, ...]

                # get labels from classificator
                labels = func(input_img)

                # calculate labels with best fmeasure
                fscore, bestlabels = fmeasure(labels, ground_truth, numclasses)

                # save output image and GT image with the usual RGB color scheme
                labelimage = Image.fromarray(labels_as_RGB_image(bestlabels, numclasses))
                gtimage = Image.fromarray(labels_as_RGB_image(ground_truth, numclasses))
                labelimage.save(outpath + func.__name__ + str(imoffset + no) + ".png")
                gtimage.save(outpath + func.__name__ + str(imoffset + no) + "_GT.png")

            imoffset += no_samples
