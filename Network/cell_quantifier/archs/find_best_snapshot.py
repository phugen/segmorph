# This script uses a trained Caffe network and a number
# of weight snapshots to find the snapshot with the best
# validation loss performance (according to the lowest Cross-Entropy loss).

import caffe
import os
import glob
import numpy as np
import h5py
import math
import progressbar


def fmeasure(labels, gt, numclasses):
    ''' Calculate the per-class F-Measure and overall F-Measure. '''

    # calculate multi-class F-Measure by alternatingly
    # treating each class as "true" and all others as "false"
    # and then taking the average score over all classes
    temp_fscore = 0
    eps = 1e-10 # prevent division by zero
    fscores = [] # [score_c0 ... score_c(numclasses-1), score_overall]
    for c in range(numclasses):

        # calculate multi-class per-image stats by alternatingly
        # treating each class as "true" and all others as "false"
        #
        # NOTE: use "hack": Python counts "true" as "1"
        # and "false" as "0", so np.sum(arr) is the same
        # as  list(arr).count(True), but much faster!

        # true positives (TP):
        # ground truth is class "c" and
        # labels indicate this class as well
        tp = np.sum((gt == c) & (labels == c)) + eps

        # false positives (FP):
        # ground truth is not class "c" but
        # labels indicates "i" nonetheless
        fp = np.sum((gt != c) & (labels == c)) + eps


        # false negatives (FN):
        # ground truth is class "c" but
        # label indicates another class
        fn = np.sum((gt == c) & (labels != c)) + eps

        # precision
        prec = tp / float(tp + fp)

        # recall
        rec = tp / float(tp + fn)

        # calculate F-Measure for current class
        class_score = 2 * ((prec * rec) / (prec + rec))
        fscores.append(class_score)

        # accumulate F-Measure over all classes
        temp_fscore += class_score

    # get avergage F-Measure over all classes
    fscores.append(temp_fscore / float(numclasses))


    return fscores



def find_snapshot(device_id, batchsize, modelfile, validation_path, snapshot_path, log_path, use_fmeasure, interval):

    # predict using GPU
    caffe.set_device(device_id)
    caffe.set_mode_gpu()

    # find all available snapshots
    print "Checking " + snapshot_path + " for snapshot files ..."
    snapshotnames = glob.glob(snapshot_path + "*.caffemodel")

    if len(snapshotnames) < 1:
        print "find_best_snapshot.py: No snapshots found!"
        return -1

    print "Testing " + str(len(snapshotnames)) + " snapshots."

    # test each snapshot for its minimum validation loss
    scores = [] # scores for each snapshot
    bar = progressbar.ProgressBar()
    for snapshot in bar(snapshotnames):

        # initialize network with snapshot
        net = caffe.Net(modelfile, 1, weights=snapshot)

        # find all validation files in folder
        filenames = glob.glob(validation_path + "*validation*.h5")

        # test all images and calculate overall score across all images
        currscore = 0.
        numimages = 0
        for fi in filenames:

            # get mirrored image and label data from HDF5 file
            input_images = None
            labels = None

            with h5py.File(fi, "r") as f:
                input_images = np.array(f["data"])
                labels = np.array(f["label"])

            numimages += input_images.shape[0] # count number of images for normalizing F-Measure

            # predict all images in the validation set file
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

            # use batches of images
            for batchno in range(0, int(math.ceil(input_images.shape[0] / float(batchsize)))):

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

                # make predictions
                prediction = net.forward()
                prediction = prediction['softmax']

                if not use_fmeasure:
                    # calculate Cross-Entropy loss between predicted
                    # image and its label for current image
                    imgloss = 0.

                    for b in range(batchsize):
                        for y in range(prediction.shape[1]):
                            for x in range(prediction.shape[2]):
                                high = np.argmax(prediction[b, ..., y, x]) # find predicted class
                                imgloss += -np.log(prediction[b, ..., y, x][high]) # cross-entropy for current pixel

                    imgloss /= (prediction.shape[1]*prediction.shape[2]) # normalize loss with number of pixels
                    currscore += imgloss

                else:
                    # calculate F-Measure score
                    for b in range(batchsize):
                        # use overall score only
                        currscore += fmeasure(prediction[b, ...], labels[b, ...], prediction.shape[1])[-1]

        currscore /= numimages # normalize score by number of images
        scores.append(currscore) # save score for this snapshot


    # write "real" cross entropy losses to file for plotting validation loss
    suffix = None
    if not use_fmeasure:
        suffix = "_CEloss.txt"
    else:
        suffix = "_F1Score.txt"

    with open(log_path + os.path.basename(modelfile)[:-9] + suffix, "w") as f:
        for idx,sc in enumerate(scores):
            f.write(str((idx+1) * interval) + "," + str(sc) + "\n")

    # find best snapshot and write its name into logfile
    bestnum = None
    if not use_fmeasure:
        bestnum = np.argmin(scores) # cross-entropy: lower = better!

    else:
        bestnum = np.argmax(scores) # fmeasure: higher = better!


    with open(log_path + "bestsnaps.txt", "a") as f:
        f.write(os.path.basename(modelfile) + ": " + snapshotnames[bestnum] + " with loss=" + str(scores[bestnum]) + "\n")

    print "Wrote loss values of snapshots of " + os.path.basename(modelfile) + " to a logfile in " + log_path + "."
    print "Wrote the name of the best snapshot of " + os.path.basename(modelfile) + " to a logfile in " + log_path + "."
    print


    return 0
