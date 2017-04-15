# Implements a multi-class variant of Otsu's
# thresholding algorithm.
#
# See: http://www.iis.sinica.edu.tw/page/jise/2001/200109_01.pdf
# and http://stackoverflow.com/questions/22706742/multi-otsumulti-thresholding-with-opencv

import sys
import os
import numpy as np
import scipy.ndimage
from skimage import exposure
from skimage import filters
import matplotlib.pyplot as plt



# Otsu algorithm for three classes.
# Returns two thresholds.
def otsu3(image):

    # calculate probability histogram with 256 bins (8-bit grayscale image)
    numpx = image.size
    numbins = 256
    histogram = np.histogram(image, numbins, density=True)[0].astype(float) # get hist values
    numbins = len(histogram) # np.histogram omits bins with zero items!
    eps = 1e-5

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

        W0K += histogram[t1] / float(numpx) + eps # Pi
        M0K += t1 * (histogram[t1] / float(numpx)) # i * Pi
        M0 = M0K / W0K # (i * Pi)/Pi

        W1K = 0
        M1K = 0

        for t2 in range(t1+1, numbins):

            W1K += histogram[t2] / float(numpx) + eps # Pi
            M1K += t2 * (histogram[t2] / float(numpx)) # i * Pi
            M1 = M1K / W1K # (i * Pi)/Pi

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

    # show histogram and result
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(histogram)), histogram)

     # mark thresholds with vertical lines
    for i, thr in enumerate(threshs):
        ax1.axvline(thr, color="orange")
        ax1.text(thr+10, 0.05 ,"t" + str(i) + "=" + str(thr), rotation=0)

    plt.show()
    return threshs




if len(sys.argv) < 1:
    print "Usage: python otsu_multi.py path/to/image"

# load image
in_img = scipy.ndimage.imread(sys.argv[1])
in_img = in_img.reshape(in_img.shape[0] * in_img.shape[1]) # 1D represenation

'''# calculate probability histogram with 256 bins (8-bit grayscale image)
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

# apply thresholding
out_img = in_img.copy()
thresh = filters.threshold_otsu(out_img)
out_img[in_img < thresh] = 0
out_img[in_img >= thresh] = 255'''

# apply 3-class Otsu algorithm to image
threshs = otsu3(in_img)

# class color values
tvals = [0, 130, 255]

# apply thresholds
out_img = in_img.copy()

out_img[in_img < threshs[0]] = tvals[0]
out_img[[in_img >= threshs[0]][0] & [in_img < threshs[1]][0]] = tvals[1]
out_img[in_img >= threshs[1]] = tvals[2]


# new figure: original and otsu result
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.axis("off")
ax1.imshow(scipy.ndimage.imread("latex_test_REAL.png"))

ax2 = fig.add_subplot(212)
ax2.axis("off")
ax2.imshow(out_img.reshape(160, 150), cmap=plt.cm.gray)


#plt.show()
