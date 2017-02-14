# Prepares PNG data for use in CNN training by scaling to
# zero mean and unit variance, random cropping to a proper input size
# TODO: Rotation after cropping?

import sys
import os
import glob
import numpy as np
import random
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
from sklearn import preprocessing
from PIL import Image

if len(sys.argv) < 5:
    print "Too few arguments!"
    print "Usage: python preprocess.py /path/to/input_folder /path/to/output_folder new_height new_width startcount"
    print ""

    # display magic dimensions
    # (d4a_size + 124) % 16 == 0, dimension condition
    print "Valid U-Net input dimensions up to 1000px:"
    print ""
    for i in range (0, 1000):
        if (i + 124) % 16 == 0:
            print i

    sys.exit(-1)

# paths to the image folder containing images and their labels
# and a path where to save the processed images to
path = sys.argv[1]
outpath = sys.argv[2]

# desired dims after processing
magic_height = int(sys.argv[3])
magic_width = int(sys.argv[4])

# training samples are saved as "sample_NUMBER[_label].png"
# the startcount parameter determines at which number the
# algorithm starts when naming all samples.
startcount = int(sys.argv[5])

# check how many images (minus labels) there are in the folder
# (assuming every image in the folder has a corresponding label image!)
all_images = glob.glob(path + "/*." + "png")
IMG_NO = len(all_images)

if IMG_NO == 0:
    print "No PNG images found in path " + os.path.abspath(path)
    sys.exit(-1)

print "Found " + str(IMG_NO/2) + " image(s) with label(s) in folder."
print ""


# preprocess images
curcount = startcount

for filename in all_images:
    if not filename.endswith("_label.png"): # treat label images inside the loop
        with Image.open(filename) as png_img, Image.open(filename[:-4] + "_label.png") as png_label:

            height = png_img.size[1]
            width = png_img.size[0]

            if(height > magic_height and width > magic_width):
                newheight = random.randrange(0, height - magic_height)
                newwidth = random.randrange(0, width - magic_width)

                # random crop to "magic dimensions"
                png_img = png_img.crop((newheight, newwidth, newheight + magic_height, newwidth + magic_width))
                png_label = png_label.crop((newheight, newwidth, newheight + magic_height, newwidth + magic_width))

            # apply ZMUV scaling for each channel independently
            png_img = np.array(png_img.convert("RGB"), dtype=np.float64)

            for i in range (0, 3):
                png_img[:,:,i] = preprocessing.scale(png_img[:,:,i])

            # rescale values to [0, 255]
            for i in range (0, 3):
                oldMin = png_img[:,:,i].min()
                oldMax = png_img[:,:,i].max()
                oldRange = oldMax - oldMin

                newMin = 0
                newMax = 255
                newRange = 255

                png_img[:,:,i] = [(((val - oldMin) * newRange) / oldRange) + newMin for val in png_img[:,:,i]]

            # save processed image as PNG
            png_img = scipy.misc.toimage(png_img, cmin=0, cmax=255)

            png_img.save(outpath + "sample" + "_" + str(curcount) + ".png", "PNG")
            png_label.save(outpath + "sample" + "_" + str(curcount) + "_label.png", "PNG")

            print "    Preprocessed image " + str(curcount+1) + " / " + str(startcount + IMG_NO/2) + "..."

            curcount += 1


print "Wrote " + str(IMG_NO/2) + " preprocessed image(s) with label(s) to " + os.path.abspath(outpath)
