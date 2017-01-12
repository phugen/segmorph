# This script converts all images and their labels on the given input path
# from BMP to PNG and saves them to the given output path.
# Additionally, data processing is

import sys
import os
import glob
import numpy as np
import scipy.misc
from PIL import Image

if len(sys.argv) < 3:
    print "Too few arguments!"
    print "Usage: python bmp_to_png.py /path/to/input_folder /path/to/output_folder"
    sys.exit(-1)

# paths to the image folder containing images and their labels
# and the path to the output HDF5 file
path = sys.argv[1]
outpath = sys.argv[2]

# check how many images (minus labels) there are in the folder
# (assuming every image in the folder has a corresponding label image!)
all_images = glob.glob(path + "/*." + "bmp")
IMG_NO = len(all_images)

if IMG_NO == 0:
    print "No BMP images found in path " + os.path.abspath(path)
    sys.exit(-1)

print "Found " + str(IMG_NO/2) + " image(s) with label(s) in folder."
print ""

# display magic dimensions
for i in range (0, 960):
    if (i + 124) % 16 == 0:
        print i

for filename in all_images:
    if not filename.endswith("_label.bmp"): # treat label images inside the loop
        with Image.open(filename) as bmp_img, Image.open(filename[:-4] + "_label.bmp") as bmp_label:

            # resize to "magic dimensions" given by data dims and U-Net conditions
            bmp_img = bmp_img.crop((0, 0, 244, 244))
            bmp_label = bmp_label.crop((0, 0, 244, 244))

            # substract mean from image
            bmp_img = bmp_img.convert("RGB")
            mean_img = np.mean(bmp_img)
            bmp_img -= mean_img
            bmp_img = scipy.misc.toimage(bmp_img, cmin=0, cmax=255)

            # save processed image as PNG
            bmp_img.save(filename[:-4] + ".png", "PNG")
            bmp_label.save(outpath + filename[:-4].split("\\")[-1] + "_label.png", "PNG")


print "Wrote " + str(IMG_NO/2) + " PNG image(s) with label(s) to " + os.path.abspath(outpath)
