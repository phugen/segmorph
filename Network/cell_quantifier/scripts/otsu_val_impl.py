# Implements a multi-class variant of Otsu's
# thresholding algorithm.
#
# See: http://www.iis.sinica.edu.tw/page/jise/2001/200109_01.pdf
# and http://stackoverflow.com/questions/22706742/multi-otsumulti-thresholding-with-opencv


import numpy as np
import scipy.ndimage
from skimage import exposure
from skimage import filters
import matplotlib.pyplot as plt


# Normal Otsu for two classes, vectorized implementation.
# Returns one threshold.
def otsu_2(in_img):

    # calculate probability histogram with 256 bins (8-bit grayscale image)
    numpx = in_img.size
    numbins = 256
    histogram = np.histogram(in_img, numbins, density=True)[0].astype(float) # get hist values
    numbins = len(counts)

    # get weights for all possible thresholds
    # with thresholds in [1, numbins]
    w_1 = np.cumsum(counts)
    w_2 = np.cumsum(counts[::-1]) # reverse to get other class

    # get means for all possible thresholds
    mu_1 = np.cumsum(counts * np.array(range(numbins))) / w_1
    mu_2 = (np.cumsum(counts * np.array(range(numbins))[::-1]) / w_2[::-1])[::-1]

    # find optimum threshold using Otsu's algorithm
    variances = w_1[:-1] * w_2[1:] * (mu_1[:-1] - mu_2[1:])**2
    thresh = np.argmax(variances)


    return thresh



# Otsu algorithm for three classes.
# Returns two thresholds.
def otsu3(image):

    # calculate histogram with 256 bins (8-bit grayscale image)
    numpx = image.size
    numbins = 256
    histogram = np.histogram(image, numbins)[0].astype(float) # get hist values
    numbins = len(histogram) # np.histogram omits bins with zero items!
    eps = 1e-5 # prevent division by zero

    # weights and means for each class
    W0K = 0
    M0K = 0
    W1K = 0
    M1K = 0

    mu_total = 0 # total mean of the image
    maxvar = 0 # current maximum variance

    # get total mean
    mu_total = np.sum([k * (histogram[k] / float(numpx)) for k in range(numbins)])

    for t1 in range(numbins):

        W0K += histogram[t1] / float(numpx) + eps # weight class 1
        M0K += t1 * (histogram[t1] / float(numpx)) # mean class 1
        M0 = M0K / W0K

        # reset vars
        W1K = 0
        M1K = 0

        for t2 in range(t1+1, numbins):

            W1K += histogram[t2] / float(numpx) + eps
            M1K += t2 * (histogram[t2] / float(numpx))
            M1 = M1K / W1K

            # shortcut: weight/mean of last class can be
            # calculated without using actual formula
            W2K = 1 - (W0K + W1K)
            M2K = mu_total - (M0K + M1K)

            if W2K <= 0:
                break

            M2 = M2K / W2K

            # calculate multiclass variance
            variance = W0K * (M0 - mu_total)**2 \
                     + W1K * (M1 - mu_total)**2 \
                     + W2K * (M2 - mu_total)**2


            # update thresholds if variance got larger
            if maxvar < variance:
                maxvar = variance
                thresh1 = t1
                thresh2 = t2


    threshs = [thresh1, thresh2]

    return threshs




# Otsu algorithm for four classes.
# Returns three thresholds.
def otsu4(image):

    # calculate histogram with 256 bins (8-bit grayscale image)
    numpx = image.size
    numbins = 256
    histogram = np.histogram(image, numbins)[0].astype(float) # get hist values
    numbins = len(histogram) # np.histogram omits bins with zero items!
    eps = 1e-5 # prevent division by zero

    # weights and means for each class
    W1K = 0; M1K = 0
    W2K = 0; M2K = 0
    W3K = 0; M3K = 0
    W4K = 0; M4K = 0

    mu_total = 0 # total mean of the image
    maxvar = 0 # current maximum variance

    # get total mean
    mu_total = np.sum([k * (histogram[k] / float(numpx)) for k in range(numbins)])

    for t1 in range(numbins):

        W1K += histogram[t1] / float(numpx) + eps # weight class 1
        M1K += t1 * (histogram[t1] / float(numpx)) # mean class 1
        M1 = M1K / W1K

        # reset vars
        W2K = 0
        M2K = 0

        for t2 in range(t1+1, numbins):

            W2K += histogram[t2] / float(numpx) + eps
            M2K += t2 * (histogram[t2] / float(numpx))
            M2 = M2K / W2K

            # reset vars
            W3K = 0
            M3K = 0

            for t3 in range(t2+1, numbins):

                W3K += histogram[t3] / float(numpx) + eps
                M3K += t3 * (histogram[t3] / float(numpx))
                M3 = M3K / W3K

                # shortcut: weight/mean of last class can be
                # calculated without using actual formula
                W4K = 1 - (W1K + W2K + W3K)
                M4K = mu_total - (M1K + M2K + M3K)

                if W4K <= 0:
                    break

                M4 = M4K / W4K

                # calculate multiclass variance
                variance = W1K * (M1 - mu_total)**2 \
                         + W2K * (M2 - mu_total)**2 \
                         + W3K * (M3 - mu_total)**2 \
                         + W4K * (M4 - mu_total)**2


                # update thresholds if variance got larger
                if maxvar < variance:
                    maxvar = variance
                    thresh1 = t1
                    thresh2 = t2
                    thresh3 = t3


    threshs = [thresh1, thresh2, thresh3]


    return threshs
