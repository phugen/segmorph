# This script uses a trained Caffe network and a number
# of weight snapshots to find the snapshot with the best
# validation loss performance (according to the lowest Cross-Entropy loss / highest Fmeasure score).

import caffe
import os
import glob
import numpy as np
import h5py
import math
import progressbar



def find_by_fmeasure(batchsize, modelfile, validation_path, log_path, interval, snapshotnames):
    ''' Finds best snapshot for networks trained by F-Measure loss function.'''

    # test each snapshot for its minimum validation loss
    scores = [] # scores for each snapshot

    input_images = None
    labels = None

    # get mirrored image and label data from HDF5 file
    numimages = 0

    for fi in glob.glob(validation_path + "*validation*.h5"):
        with h5py.File(fi, "r") as f:

            if input_images is None:
                input_images = np.array(f["data"])
                labels = np.array(f["label"])

            else:
                input_images = np.concatenate((input_images, np.array(f["data"])))
                labels = np.concatenate((labels, np.array(f["label"])))

    numimages = input_images.shape[0] # count number of images


    bar = progressbar.ProgressBar()
    for snapshot in bar(snapshotnames):

        # initialize network with snapshot in TEST mode
        net = caffe.Net(modelfile, snapshot, caffe.TEST)

        # find all validation files in folder
        filenames = glob.glob(validation_path + "*validation*.h5")

        # predict all images in the validation set file
        # use batches of images
        score = 0. # total score for one sample

        batches = int(math.ceil(input_images.shape[0] / float(batchsize)))
        for batchno in range(batches):

            # deal with validation sets which have a number
            # of elements that isn't cleanly divisable by the batch size
            start = 0
            end = batchsize

            if (batchno * batchsize) + batchsize > input_images.shape[0]:
                end = input_images.shape[0] % batchsize

            # batch load input images
            for b in range(start, end):
                input_b = (batchno * batchsize) + b
                net.blobs['data'].data[b, ...] = input_images[input_b, ...]

            # get fmeasure score for this batch
            prediction = net.forward()
            fmeasure = prediction['loss']

            # accumulate overall fmeasure scores over all batches
            for i in range(len(fmeasure)):
                score += fmeasure[i]

        # save score for this snapshot
        # normalize with num classes to get overall fmeasure, then by num batches
        scores.append((score / len(fmeasure)) / float(batches))


    # write "real" fmeasure losses to file for plotting validation loss
    suffix = "_F1Score.txt"

    with open(log_path + os.path.basename(modelfile)[:-9] + suffix, "w") as f:
        for idx,sc in enumerate(scores):
            f.write(str((idx+1) * interval) + "," + str(sc) + "\n")

    # find best snapshot and write its name into logfile
    bestnum = np.argmax(scores) # fmeasure: higher = better!

    # return best actual best snapshot fmeasure and its snapshot number
    return (scores[bestnum], bestnum)




def find_by_ce(batchsize, modelfile, validation_path, log_path, interval, snapshotnames):
    '''Finds best snapshot for networks trained by Cross-Entropy loss function.'''


    scores = [] # scores for each snapshot
    bar = progressbar.ProgressBar()

    input_images = None
    labels = None
    weights = None

    # get mirrored image and label data from HDF5 file
    numimages = 0

    for fi in glob.glob(validation_path + "*validation*.h5"):
        with h5py.File(fi, "r") as f:

            if input_images is None:
                input_images = np.array(f["data"])
                labels = np.array(f["label"])
                weights = np.array(f["weights"])

            else:
                input_images = np.concatenate((input_images, np.array(f["data"])))
                labels = np.concatenate((labels, np.array(f["label"])))
                weights = np.concatenate((weights, np.array(f["weights"])))

    numimages = input_images.shape[0] # count number of images


    # test each snapshot for its minimum validation loss
    for snapshot in bar(snapshotnames):

        # initialize network with snapshot in TEST mode
        net = caffe.Net(modelfile, snapshot, caffe.TEST)

        score = 0.
        numallbatches = 0

        # predict all images in the validation set
        # use batches of images
        batches = int(math.ceil(input_images.shape[0] / float(batchsize))) # how many batches for this file?
        numallbatches += batches # keep track of overall number of batches

        for batchno in range(batches):

            # deal with validation sets which have a number
            # of elements that isn't cleanly divisable by the batch size
            start = 0
            end = batchsize

            if (batchno * batchsize) + batchsize > input_images.shape[0]:
                end = input_images.shape[0] % batchsize

            # batch load input images
            for b in range(start, end):
                input_b = (batchno * batchsize) + b
                net.blobs['data'].data[b, ...] = input_images[input_b, ...]

            # get loss for this batch
            prediction = net.forward()
            score += prediction['loss']

        # save batch-average score for this snapshot
        scores.append(score / float(batches))


    # write "real" cross entropy losses to file for plotting validation loss
    suffix = "_CEloss.txt"


    with open(log_path + os.path.basename(modelfile)[:-9] + suffix, "w") as f:
        for idx,sc in enumerate(scores):
            f.write(str((idx+1) * interval) + "," + str(sc) + "\n")

    # find best snapshot and write its name into logfile
    bestnum = np.argmin(scores) # cross-entropy: lower = better!

    # return actual best score and its entry number in snapshot name list
    return (scores[bestnum], bestnum)



def find_snapshot(device_id, batchsize, modelfile, validation_path, snapshot_path, log_path, use_fmeasure, interval):

    # predict using GPU
    caffe.set_device(device_id)
    caffe.set_mode_gpu()

    # find all available snapshots
    print "Checking " + snapshot_path + " for snapshot files ..."
    snapshotnames = glob.glob(snapshot_path + "*.caffemodel")

    # sort snapshot names by number extracted from their filename
    # to prevent misordering by lexicographical sorting
    iternums = []
    for name in snapshotnames:
        iternums.append(int(name.split("_")[-1][:-11]))

    sortby = zip(iternums, snapshotnames)
    snapshotnames = [snapshotnames for iternums,snapshotnames in sorted(sortby)]


    if len(snapshotnames) < 1:
        print "find_best_snapshot.py: No snapshots found!"
        return -1

    print "Testing " + str(len(snapshotnames)) + " snapshots."

    # find best snapshot
    if use_fmeasure == "True":
        bestscore, bestnum = find_by_fmeasure(batchsize, modelfile, validation_path, log_path, interval, snapshotnames)
    else:
        bestscore, bestnum = find_by_ce(batchsize, modelfile, validation_path, log_path, interval, snapshotnames)

    # log name of best snapshot
    with open(log_path + "bestsnaps.txt", "a") as f:
        f.write(os.path.basename(modelfile) + ": " + snapshotnames[bestnum] + " with loss=" + str(bestscore) + "\n")

    print "Wrote loss values of snapshots of " + os.path.basename(modelfile) + " to a logfile in " + log_path + "."
    print "Wrote the name of the best snapshot of " + os.path.basename(modelfile) + " to a logfile in " + log_path + "."
    print


    return 0
