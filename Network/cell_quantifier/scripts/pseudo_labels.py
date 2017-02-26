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
t1 = 100.0 * 10
t2 = 600.0 * 10

def alpha (t):
    if t < t1:
        return 0.0

    elif t1 <= t < t2:
        return (t - t1) / (t2 - t1) * alpha_f

    else:
        return alpha_f


# params for manual SGD (TODO: can this be done better?)
weight_decay = 0.0005
lr_w_mult = 1
lr_b_mult = 1

base_lr = 0.001 # base learning rate, originally 0.001
lr_policy = "step" # drop learning rate in steps by a factor gamma
gamma = 0.1 # see above
stepsize = 20000 # drop learning rate every <stepsize> steps

iter_size = 1 # how many images are processed simultaneously in one learning step (batch size)
max_iter = 300000 # total number of training iterations
momentum = 0.99 # weight of previous update

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

    solver.net.forward()
    normal_loss = solver.net.blobs['loss'].data[...]

    # extract forward pass output image and write to file
    if step % 50 == 0:
        features_out = solver.net.blobs['visualize_out'].data[0, ...] # plt needs WxHxC
        features_out = features_out[1:4,:,:] # omit BG probability layer - pixel will become dark if other probabilities are low

        minval = features_out.min()
        maxval = features_out.max()
        scipy.misc.toimage(features_out, cmin=minval, cmax=maxval).save("./visualizations/visualize_out_" + str(step) + ".png")

    # pass unlabelled data through network to get pseudo-labels
    solver.net.blobs['data'].data[...] = unlabelled[step % unlabelled.shape[0], ...]
    solver.net.blobs['label'].data[...] = np.zeros(weights.shape[1:3]) # use dummy labels because we don't care about the loss at this point
    solver.net.forward()

    # extract pseudo-labels
    pseudo_raw = solver.net.blobs['score'].data[...]
    pseudo_labels = np.zeros((1, 1, pseudo_raw.shape[2], pseudo_raw.shape[3]), dtype=np.uint8)

    for y in range(pseudo_raw.shape[2]):
        for x in range(pseudo_raw.shape[3]):
            pseudo_labels[..., y, x] = np.argmax(pseudo_raw[..., :, y, x])


    # pass unlabelled data through the network again to get pseudo loss
    solver.net.blobs['label'].data[...] = pseudo_labels[...]
    solver.net.blobs['weights'].data[...] = weights[step % weights.shape[0], ...] # TODO real weights
    solver.net.forward()
    pseudo_loss = alpha(step) * solver.net.blobs['loss'].data[...]

    # update loss as partially alpha-weighted sum of both losses
    combined_loss = (1 / batch_normal) * normal_loss + (1 / batch_pseudo) * pseudo_loss
    solver.net.blobs['loss'].data[...] = combined_loss

    print "Iteration=" + str(step) + " alpha=" + str(alpha(step)) + " Loss=" + str(combined_loss)

    # perform backpropagation with combined loss
    solver.net.backward()

    # manually update layer weights according
    # to backward-generated diffs
    momentum_hist = {}
    for layer in solver.net.params:
        m_w = np.zeros_like(solver.net.params[layer][0].data)
        m_b = np.zeros_like(solver.net.params[layer][1].data)
        momentum_hist[layer] = [m_w, m_b]

    for layer in solver.net.params:
        momentum_hist[layer][0] = momentum_hist[layer][0] * momentum + (solver.net.params[layer][0].diff + weight_decay * \
                                                       solver.net.params[layer][0].data) * base_lr * lr_w_mult
        momentum_hist[layer][1] = momentum_hist[layer][1] * momentum + (solver.net.params[layer][1].diff + weight_decay * \
                                                       solver.net.params[layer][1].data) * base_lr * lr_b_mult

        solver.net.params[layer][0].data[...] -= momentum_hist[layer][0]
        solver.net.params[layer][1].data[...] -= momentum_hist[layer][1]

        # reset diffs; don't reset if gradient accumulation is needed
        solver.net.params[layer][0].diff[...] *= 0
        solver.net.params[layer][1].diff[...] *= 0

    # lower learning rate if necessary
    base_lr = base_lr * np.power(gamma, (np.floor(step / stepsize)))

    # save solver state
    if step % snapshot == 0 and step != 0:
        solver.snapshot()

        # hack: because solver.step() is not being used, so rename
        # solverstate and model file to match actual iteration count
        # TODO


print 'Pseudo-Label training done'
