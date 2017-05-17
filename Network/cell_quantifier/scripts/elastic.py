# Applies elastic deformation to all images in a folder
# and saves the result as a copy with appended "_AUGMENTED" suffix.

import numpy as np
import sys
import glob
import os.path
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import progressbar
import scipy.ndimage
import scipy.misc


def reduc(colors, thresh):
    """ Maximize maximum value and set all others
    to 0, or all to 0 if max < thresh. """

    # find maximum channel
    maxindex, maxval = np.argmax(colors), np.max(colors)

    # "truncate" to black if needed
    if maxval < thresh:
        return [0, 0, 0]

    else:
        reduced = [0, 0, 0]
        reduced[maxindex] = 255
        return reduced


def reduce_to_rgbk(image, thresh=20):
    """ Does primitive thresholding by setting max channel to
    255, others to 0, unless the maximum channel is below a threshold.
    If that is the case, the pixel is set to black = (0, 0, 0). """

    h, w, c = image.shape
    temp = image.reshape(h*w, c)
    reduced = np.array([reduc(temp[x, :], thresh) for x in range(h*w)])

    return reduced.reshape(h, w, c)




def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3 and image.shape[2] == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape


    # create random displacement vectors for each pixel
    # from a Gaussian distribution and amplify them by alpha
    dx = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="reflect") * alpha

    # apply displacement vectors to coordinate mesh to get new pixel positions
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    # calculate distortion in each channel using bicubic interpolation
    r = map_coordinates(image[..., 0], indices, order=3).reshape(shape[0:2])
    g = map_coordinates(image[..., 1], indices, order=3).reshape(shape[0:2])
    b = map_coordinates(image[..., 2], indices, order=3).reshape(shape[0:2])

    # reassemble image
    trans = np.zeros((image.shape))
    trans[..., 0] = r
    trans[..., 1] = g
    trans[..., 2] = b

    return trans




if len(sys.argv) < 2:
    print "Too few arguments!"
    print "Usage: python elastic.py inputfolder outputfolder"
    sys.exit(-1)

inpath = sys.argv[1]
outpath = sys.argv[2]

# transform all images on the inpath
imagepaths = glob.glob(inpath + "\*.png")
IMG_NO = len(imagepaths)

if(IMG_NO == 0):
    print "No images found in " + str(inpath) + " !"
    exit(-1)


bar = progressbar.ProgressBar()
for imgno in bar(range(len(imagepaths))):

    # generate new distortion params every second image
    if imgno % 2 == 0:
        elastic_params = np.random.RandomState(None)
        start_state = elastic_params.get_state()

    else:
        elastic_params.set_state(start_state) # reset random generator to pos 0

    # read image and get filename
    image = scipy.ndimage.imread(imagepaths[imgno], mode="RGB")
    pathto, fname = os.path.split(imagepaths[imgno])

    # image is a label, distort with previous params to match associated image
    # and quantize colors because of interpolation
    if imagepaths[imgno].endswith("_label.png"):
        outfile = outpath + fname[:-10] + "_AUGMENTED_label.png"
        trans = elastic_transform(image, 200, 10, random_state=elastic_params)
        trans = reduce_to_rgbk(trans, thresh=80)

    # image is not a label, deform normally
    else:
        trans = elastic_transform(image, 200, 10, random_state=elastic_params)
        outfile = outpath + fname[:-4] + "_AUGMENTED.png"

    # save distorted image
    scipy.misc.toimage(trans).save(outfile)


print "Augmented " + str(IMG_NO/2) + " images and their labels!\n"
