# Applies mutli-otsu algorithm to images.
# Implementation in otsu_multi.py

import sys
import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import otsu_multi_impl as oimpl


if len(sys.argv) < 2:
    print "Usage: python otsu_multi.py path/to/image"

# load image
in_img = scipy.ndimage.imread(sys.argv[1])
in_img = in_img.reshape(in_img.shape[0] * in_img.shape[1]) # 1D represenation

'''# apply thresholding for two classes
out_img = in_img.copy()
thresh = filters.threshold_otsu(out_img)
out_img[in_img < thresh] = 0
out_img[in_img >= thresh] = 255'''

'''# apply 3-class Otsu algorithm to image
threshs = oimpl.otsu3(in_img)

# class color values
tvals = [0, 130, 255]

# apply thresholds
out_img = in_img.copy()

out_img[in_img < threshs[0]] = tvals[0]
out_img[[in_img >= threshs[0]][0] & [in_img < threshs[1]][0]] = tvals[1]
out_img[in_img >= threshs[1]] = tvals[2]'''

# apply 4-class otsu
threshs = oimpl.otsu4(in_img)

# class color values
tvals = [0, 70, 150, 255]

# apply thresholds
out_img = in_img.copy()

out_img[in_img < threshs[0]] = tvals[0]
out_img[[in_img >= threshs[0]][0] & [in_img < threshs[1]][0]] = tvals[1]
out_img[[in_img >= threshs[1]][0] & [in_img < threshs[2]][0]] = tvals[2]
out_img[in_img >= threshs[2]] = tvals[3]


# new figure: original and otsu result
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.axis("off")
ax1.imshow(scipy.ndimage.imread("latex_test_REAL.png"))

ax2 = fig.add_subplot(212)
ax2.axis("off")
ax2.imshow(out_img.reshape(160, 150), cmap=plt.cm.gray)


plt.show()
