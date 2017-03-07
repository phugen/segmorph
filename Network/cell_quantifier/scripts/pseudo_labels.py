# Starts a training with pseudo-labels, that is, in addition to
# normal image+label pairs, additional pairs are created for training by running
# them through the network and using the predicted image as a label.
#
# Idea from paper "Pseudo-Label: The Simple and Efficient Semi-Supervised
# Learning Method for Deep Neural Networks" by Dong-Hyun Lee

import sys
import os
import numpy as np
import h5py
import scipy.misc
from PIL import Image
import caffe



if(len(sys.argv)) < 2:
    print "Too few arguments (" + str(len(sys.argv) - 1) + ")!"
    print "Usage: python pseudo_labels.py hdf5_path [solverstate_path]"
    exit(-1)

else:
    hdf5path = sys.argv[1]


# As step size increases, make unlabelled data
# loss gradually more influential regarding the combined loss
alpha_f = 3.0
t1 = 100.0
t2 = 600.0

def alpha (t):
    if t < t1:
        return 0.0

    elif t1 <= t < t2:
        return (t - t1) / (t2 - t1) * alpha_f

    else:
        return alpha_f


# number of classes to segment by (possible labels + 1)
numclasses = 4

# params for manual SGD (TODO: can this be done better?)
weight_decay = 0.0005
lr_w_mult = 1
lr_b_mult = 2

base_lr = 0.001 # base learning rate, originally 0.001
lr_policy = "step" # drop learning rate in steps by a factor gamma
gamma = 0.1 # see above
stepsize = 20000 # drop learning rate every <stepsize> steps

iter_size = 1 # how many images are processed simultaneously in one learning step (batch size)
max_iter = 300000 # total number of training iterations
momentum_weight = 0.99 # weight of previous update

snapshot = 500


caffe.set_device(0)
caffe.set_mode_gpu()
print "Loading Solver... "
solver = caffe.get_solver("unet_solver_weighted_pseudo.prototxt")

if len(sys.argv) > 2:
    if sys.argv[1].endswith(".solverstate"):
        solver.restore(sys.argv[2])

print "Training with Pseudo-Labels ..."

# batch sizes of labelled and unlabelled data
batch_normal = solver.net.blobs['data'].data.shape[0]
batch_pseudo = solver.net.blobs['data'].data.shape[0]

# get raw labelled and unlabelled data
images = None
labels = None
weights = None
unlabelled = None
pseudo_weights = None


with h5py.File(hdf5path + "/drosophila_training.h5", "r", libver="latest") as f:

    images = f['data'].value
    labels = f['label'].value
    weights = f['weights'].value


with h5py.File(hdf5path + "/drosophila_unlabelled.h5", "r", libver="latest") as f:

    unlabelled = f['data'].value
    pseudo_weights = f['weights'].value  # calc those for each image on the fly or just set to 1.0 / approx value?



# TODO: add batchsize > 1 compability, which includes prototxt  on-the-fly writing for data layers!
# TODO: IMPORTANT: Backward doesn't actually update weights (+ momentum) but just calculates diffs.
# for more info, see: https://github.com/BVLC/caffe/issues/1855
max_iter = 300000
for step in range(max_iter):

    # pass normal labels through network and get loss
    solver.net.blobs['data'].data[...] = images[step % images.shape[0], ...]
    solver.net.blobs['label'].data[...] = labels[step % labels.shape[0], ...]
    solver.net.blobs['weights'].data[...] = weights[step % weights.shape[0], ...]

    '''solver.net.forward()
    normal_loss = solver.net.blobs['loss'].data[...]

    # pass unlabelled data through network to get prediction
    solver.net.blobs['data'].data[...] = unlabelled[step % unlabelled.shape[0], ...]
    solver.net.blobs['label'].data[...] = np.zeros(weights.shape[1:3]) # use dummy labels because we don't care about the loss at this point
    solver.net.forward(start=None, end='loss')

    # extract pseudo-labels
    pseudo_scores = solver.net.blobs['score'].data[...]
    pseudo_labels = np.zeros((batch_pseudo, 1, pseudo_scores.shape[2], pseudo_scores.shape[3]), dtype=np.uint8)

    for y in range(pseudo_scores.shape[2]):
        for x in range(pseudo_scores.shape[3]):
            #one_hot = np.zeros((numclasses))
            #one_hot[np.argmax(pseudo_scores[..., y, x])] #score with maximum probability becomes 1, others 0
            pseudo_labels[..., y, x] = np.argmax(pseudo_scores[..., y, x]) #one_hot

    # compare prediction for unlabelled image with pseudo-labels of the same image to get pseudo-loss
    solver.net.blobs['label'].data[...] = pseudo_labels[...]
    solver.net.blobs['weights'].data[...] = weights[step % weights.shape[0], ...] # TODO real weights
    solver.net.forward(start='rawscores', end=None)
    pseudo_loss = alpha(step) * solver.net.blobs['loss'].data[...]

    # update loss as partially alpha-weighted sum of both losses
    combined_loss = (1 / batch_normal) * normal_loss + (1 / batch_pseudo) * pseudo_loss
    solver.net.blobs['loss'].data[...] = combined_loss

    print "Iteration=" + str(step) + " alpha=" + str(alpha(step)) + " Loss=" + str(combined_loss)

    # perform backpropagation with combined loss
    solver.net.backward()'''

    solver.net.forward()
    solver.net.backward()

    print "Iteration=" + str(step) + " Loss=" + str(solver.net.blobs['loss'].data[...])


    # create arrays for saving the (step - 1)th update
    # to use for the "momentum" SGD optimization
    velocity = {}

    for layer in solver.net.params:
        v_w = np.zeros(solver.net.params[layer][0].shape)
        v_b = np.zeros(solver.net.params[layer][1].shape)
        velocity[layer] = [v_w, v_b]

    # manually update layers according
    # to backward-generated diffs
    for layer in solver.net.params:

        # calculate updates and save as current momentum.
        # In the very first iteration, momentum will be 0 and thus
        # not contribute to the update.
        velocity[layer][0][...] = (momentum_weight * velocity[layer][0]) + base_lr * solver.net.params[layer][0].diff[...]
        velocity[layer][1][...] = (momentum_weight * velocity[layer][1]) + base_lr * solver.net.params[layer][1].diff[...]

        # apply updates to weights and biases
        solver.net.params[layer][0].data[...] -= velocity[layer][0]
        solver.net.params[layer][1].data[...] -= velocity[layer][1]

        # reset diffs; don't reset if gradient accumulation is needed
        solver.net.params[layer][0].diff[...] *= 0
        solver.net.params[layer][1].diff[...] *= 0

    # lower learning rate if adequate
    base_lr = base_lr * np.power(gamma, (np.floor(step / stepsize)))

    #solver.step(1)


    # extract forward pass output image and write to file
    if step % 10 == 0:
        features_out = solver.net.blobs['visualize_out'].data[0, ...] # plt needs WxHxC
        features_out = features_out[1:4,:,:] # omit BG probability layer - pixel will become dark if other probabilities are low

        minval = features_out.min()
        maxval = features_out.max()
        scipy.misc.toimage(features_out, cmin=minval, cmax=maxval).save("./visualizations/weighted_pseudo_" + str(step) + ".png")

    # save solver state
    if step % snapshot == 0 and step != 0:
        solver.snapshot()

        # hack: because solver.step() is not being used, so rename
        # solverstate and model file to match actual iteration count
        # TODO


print 'Pseudo-Label training done'
