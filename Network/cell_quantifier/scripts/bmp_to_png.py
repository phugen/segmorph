# This script converts all images and their labels on the given input path
# from BMP to PNG and saves them to the given output path.

import sys
import os
import glob
from PIL import Image

if len(sys.argv) < 3:
    print "Too few arguments!"
    print "Usage: python bmp_to_png.py /path/to/input_folder /path/to/output_folder"
    sys.exit(-1)

# paths to the image folder containing images and their labels
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


for filename in all_images:
    if not filename.endswith("_label.bmp"): # treat label images inside the loop
        with Image.open(filename) as bmp_img, Image.open(filename[:-4] + "_label.bmp") as bmp_label:

            # save image as PNG
            bmp_img.save(outpath + filename[:-4].split("\\")[-1] + ".png", "PNG")
            bmp_label.save(outpath + filename[:-4].split("\\")[-1] + "_label.png", "PNG")


print "Wrote " + str(IMG_NO/2) + " PNG image(s) with label(s) to " + os.path.abspath(outpath)
