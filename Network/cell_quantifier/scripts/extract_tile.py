# Extracts the non-mirrored tile indicated by the input
# number from a stack of all validation files
# and saves it as a .png.

import h5py
import sys
import glob
import numpy as np
import scipy.misc

valpath = sys.argv[1]
imgno = int(sys.argv[2])


# get validation files and stack them in array
filenames = glob.glob(valpath + "*validation.h5")

# will hold all trimmed tiles from all files
tiles = np.zeros((0, 3, 244, 244))

for name in filenames:

    # extract all tiles from file
    with h5py.File(name, "r", libver="latest") as f:

        # trim mirrored border off tiles
        data = f["data"][:, :, 92:428-92, 92:428-92]
        tiles = np.concatenate((tiles, data))


# write out image to script folder
scipy.misc.toimage(tiles[imgno, ...]).save("tile_" + str(imgno) + ".png")
