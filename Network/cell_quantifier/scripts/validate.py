# Measure how well a classificator segments an image, measured by
# the F-Measure score of the best class permutation.
# For F-Measure formulas, see https://en.wikipedia.org/wiki/F1_score

import PIL.Image as Image
import numpy as np
import h5py

from fmeasure_validation import labels_as_RGB_image
from fmeasure_validation import fmeasure

def validate(func, numclasses, inpath, outpath):
    ''' Validates a function func using a HDF5 file
    at <inpath> and saves the results in <outpath>. '''

    # load validation data from a HDF5 file
    with file(outpath + "log.txt", "a") as log:
        with h5py.File(inpath, "r", libver="latest") as f:

            # predict all validation images
            no_samples = f["data"].shape[0]
            for no, i in enumerate(range(no_samples)):

                # image is grayscale anyway, reduce to 1 channel
                # and shave off mirror padding because GMMs doesn't need it
                input_img = f["data"][i, 0, 90:334, 90:334]
                ground_truth = f["label"][i, ...].reshape((input_img.size), 1)

                # get labels from classificator
                labels = func(input_img)

                # calculate labels that give best fmeasure
                fscore, bestlabels = fmeasure(labels, ground_truth, numclasses)

                # log performance
                log.write(str(no) + " " + str(fscore) + "\n")

                # save output image and GT image with the usual RGB color scheme
                labelimage = Image.fromarray(labels_as_RGB_image(bestlabels, numclasses))
                gtimage = Image.fromarray(labels_as_RGB_image(ground_truth, numclasses))
                labelimage.save(outpath + "GMM_output_" + str(no) + ".png")
                gtimage.save(outpath + "GMM_output_" + str(no) + "_GT.png")
