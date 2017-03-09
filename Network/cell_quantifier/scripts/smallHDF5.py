# Uses the "convertHDF5.py" script to split the Drosophila data
# into HDF5 files that are less than 2GB large because this is a
# file limit imposed by Caffe.

import sys
import os
import glob
import shutil

# Number of (unaugmented/input) training samples per HDF5 file
PERSPLIT = 9 # 1 Image (+ Label + weights) ~ 194MB training HDF5 file size => 9 * 194MB = 1746MB < 2000MB



# check args
if len(sys.argv) < 3:
    print "Too few arguments!"
    print "Usage: python smallHDF5.py input_folder/ output_folder/"
    sys.exit(-1)

# set paths from args
inpath = sys.argv[1]
outpath = sys.argv[2]

# get file names in input folder
imagepaths = glob.glob(inpath + "/*.png")
img_no = len(imagepaths)

if img_no < 2:
    print "Found " + str(len(imagepaths)) + " images in " + inpath + ", quitting."
    exit(-1)

if (img_no % 2) != 0:
    print "Found uneven number (" + str(img_no) + ") of images in " + inpath + "."
    print "At least either one image or label image are missing, quitting."
    exit(-1)



# create temporary folder if needed
path_pro = inpath + "/processing"

if not os.path.exists(path_pro):
    os.makedirs(path_pro)


# write data to HDF5 files as long as
# there are files to write left in the list
fileindex = 0 # current file number
maxindex = int(math.ceil(img_no / (2 * PERSPLIT)))

for fileindex in range(maxindex):

    # if # of unprocessed files is not divisable by (2 * PERSPLIT)
    # just take the rest of the files
    if img_no < 2 * PERSPLIT:
        PERSPLIT = img_no

    # copy data for current file to temporary folder
    to_copy = imagepaths[0:PERSPLIT * 2]

    print "PATHPRO: " + pathpro
    print "tocopy: " + to_copy
    wait = input("PRESS ENTER TO CONTINUE.")

    for f in to_copy:
        shutil.copy(f, path_pro)

    # remove copied data from list of file names
    imagepaths = imagepaths[PERSPLIT * 2:]
    img_no = len(imagepaths)

    # run converthdf5.py script with copied data
    # TODO: pseudo labels (third argument)
    convert_inpath = path_pro
    convert_outpath = outpath + "_" + str(fileindex) + "_"
    input_vars = {convert_inpath, convert_outpath}

    print "starting script with args: " + str(input_vars)
    wait = input("PRESS ENTER TO CONTINUE.")

    execfile("converthdf5.py", input_vars)

    # delete copied data
    to_remove = glob.glob(path_pro + "/*.png")

    print "to_remove: " + to_remove
    wait = input("PRESS ENTER TO CONTINUE.")

    for copied in to_remove:
        os.rm(copied)
