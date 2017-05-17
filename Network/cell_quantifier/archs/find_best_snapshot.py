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

    # calculate F-Measure stats by alternatingly
    # treating each class as "true" and all others as "false"
    # and then taking the average score over all classes
    eps = 1e-10 # prevent division by zero
    stats = [0, 0, 0] # raw stats accumulated over all classes
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

        # accumulate stats
        stats[0] += tp
        stats[1] += fp
        stats[2] += fn

    return stats



def find_by_fmeasure(batchsize, modelfile, validation_path, log_path, interval, snapshotnames):
    ''' Finds best snapshot for networks trained by F-Measure loss function.'''

    # test each snapshot for its minimum validation loss
    scores = [] # scores for each snapshot
    classes = [0, 1, 2, 3] # convenience table for label "names"
    bar = progressbar.ProgressBar()
    for snapshot in bar(snapshotnames):

        # initialize network with snapshot
        net = caffe.Net(modelfile, 1, weights=snapshot)

        # find all validation files in folder
        filenames = glob.glob(validation_path + "*validation*.h5")

        # test all images and calculate overall score across all images
        stats = [0, 0, 0] # TP, FP, FN
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

                # accumulate stats over batch
                for b in range(start, end):

                    # create actual labels from softmax probabilities
                    pred_labels = np.zeros((prediction.shape[0], prediction.shape[2], prediction.shape[3]))

                    for y in range(prediction.shape[2]):
                        for x in range(prediction.shape[3]):
                            high = np.argmax(prediction[b, ..., y, x]) # find predicted class
                            pred_labels[b, y, x] = classes[high] # use label of that class

                    # calculate F-Measure using predicted labels
                    tmp_stats = fmeasure(pred_labels[b, ...], labels[b, ...], prediction.shape[1])
                    stats[0] += tmp_stats[0]
                    stats[1] += tmp_stats[1]
                    stats[2] += tmp_stats[2]


        # calculate F-Measure using stats of all images
        prec = stats[0] / float(stats[0] + stats[1]) # precision
        rec = stats[0] / float(stats[0] + stats[2]) # recall
        fscore = 2 * ((prec * rec) / (prec + rec))

        print fscore

        # save score for this snapshot
        scores.append(fscore / float(prediction.shape[1])) # average over all classes


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
    print input_images.shape


    # test each snapshot for its minimum validation loss
    for snapshot in snapshotnames: #TODO bar

        # initialize network with snapshot
        net = caffe.Net(modelfile, 1, weights=snapshot)

        # find all validation files in folder
        filenames = glob.glob(validation_path + "*validation*.h5")

        score = 0.
        numallbatches = 0

        # predict all images in the validation set
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        # use batches of images
        batches = range(0, int(math.ceil(input_images.shape[0] / float(batchsize)))) # how many batches for this file?
        numallbatches += len(batches) # keep track of overall number of batches

        for batchno in bar(batches):

            batchloss = 0. # sum of pixel CE losses

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

            # calculate Cross-Entropy loss between predicted
            # image and its label for current image
            for b in range(start, end):
                imgloss = 0.
                for y in range(prediction.shape[2]):
                    for x in range(prediction.shape[3]):
                        high = np.argmax(prediction[b, ..., y, x]) # find predicted class
                        imgloss += weights[b, y, x] * -np.log(prediction[b, ..., y, x][high]) # weighted cross-entropy for current pixel

                imgloss /= prediction.shape[2] * prediction.shape[3]
                batchloss += imgloss

            print

            # normalize with batchsize
            print "BATCHLOSS", batchloss
            score += batchloss

        # normalize score for all images
        # by the number of batches and save it
        print "SCORE: ", score / float(numimages)
        scores.append(score / float(numimages))


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
