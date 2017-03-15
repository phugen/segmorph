# Uses the "convertHDF5.py" script to split the Drosophila data
# into HDF5 files that are less than 2GB large because this is a
# file limit imposed by Caffe.
# Also creates input list text files for Caffes HDF5 data layers.

import sys
import os
import glob
import shutil
import math
import subprocess # for starting python scripts from this script
import progressbar

# Number of (unaugmented/input) training samples per HDF5 file
PERSPLIT = 9 # 1 Image (+ Label + weights) ~ 194MB training HDF5 file size => 9 * 194MB = 1746MB < 2000MB



# check args
if len(sys.argv) < 4:
    print "Too few arguments!"
    print "Usage: python smallHDF5.py path/input_folder/ path/output_file path/converthdf5.py"
    print ""
    print "NOTE: first argument is a folder ending with /, the second one is a FILE!"
    sys.exit(-1)

# set paths from args
inpath = sys.argv[1]
outpath = sys.argv[2]
script_path = sys.argv[3]

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
maxindex = int(math.ceil(img_no / (2 * float(PERSPLIT))))

print ""
print "Writing small HDF5 files using " + str(PERSPLIT) + " images per file ..."
print "This may take a while. The progress is only updated once per finished HDF5 file set!"

bar = progressbar.ProgressBar()
for fileindex in bar(range(maxindex)):

    # if # of unprocessed files is not divisable by (2 * PERSPLIT)
    # just take the rest of the files
    if img_no < 2 * PERSPLIT:
        PERSPLIT = img_no

    # copy data for current file to temporary folder
    to_copy = imagepaths[0:PERSPLIT * 2]

    for f in to_copy:
        shutil.copy(f, path_pro)

    # remove copied data from list of file names
    imagepaths = imagepaths[PERSPLIT * 2:]
    img_no = len(imagepaths)

    # run converthdf5.py script with copied data
    # TODO: pseudo labels (third argument)
    convert_inpath = path_pro
    convert_outpath = outpath + "_" + str(fileindex)

    subprocess.call(["python", script_path, convert_inpath, convert_outpath], \
                     stdout=open(os.devnull, 'wb'), \
                     stderr=open(os.devnull, 'wb')) # silence conversion script output



    # delete copied data
    to_remove = glob.glob(path_pro + "/*.png")

    for copied in to_remove:
        os.remove(copied)


# remove processing folder
os.rmdir(path_pro)

print ""
print ""
print "Created " + str(maxindex * 2) + " (or " + str(maxindex * 3) + \
      " if pseudo-labels) HDF5 files as " + outpath + "_* !"


# bonus: create Caffe HDF5 input location files automatically
slashpos = outpath.rfind("/")
outdir = outpath + outpath[0:slashpos + 1] # cut off filename prefix to get path

# if outpath is filename only and contains no folders
if slashpos == -1:
    outdir = "./"

outfile = outpath[outpath.rfind("/") + 2:]

training_paths = glob.glob(outpath + "*_training.h5")
validation_paths = glob.glob(outpath + "*_validation.h5")

with open(outdir + "caffeHDF5.txt", "a") as f:
    for h5path in training_paths:
        f.write("../" + h5path + "\n")

with open(outdir + "caffeHDF5_validation.txt", "a") as f:
    for h5path in validation_paths:
        f.write("../" + h5path + "\n")


print "Created HDF5 Caffe input files as " + outdir + "caffeHDF5*.txt"
