# Starts a training with pseudo-labels, that is, in addition to
# normal image+label pairs, additional pairs are created for training by running
# them through the network and using the predicted image as a label.
#
# Idea from paper "Pseudo-Label: The Simple and Efficient Semi-Supervised
# Learning Method for Deep Neural Networks" by Dong-Hyun Lee

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import caffe

# As step size increases, make unlabelled data
# loss gradually more influential regarding the combined loss
alpha_f = 3
t1 = 100
t2 = 600

def alpha (t):
    if t < t1:
        return 0

    elif t1 <= t < t2:
        return (t - t1) / (t2 - t1) * alpha_f

    else:
        return alpha_f



# start training
caffe.set_device(0)
caffe.set_mode_gpu()
print "Loading Solver... "
solver = caffe.get_solver("unet_solver.prototxt")
if len(sys.argv) > 1:
    if sys.argv[1].endswith(".solverstate"):
        solver.restore(sys.argv[1])

print "Training with Pseudo-Labels ..."
#solver.step(1)

# batch sizes of labelled and unlabelled data
batch_normal = net.blobs['data'].data.shape[0]
batch_pseudo = net.blobs['data'].data.shape[0]

# get raw labelled and unlabelled data
images = None
labels = None
weights = None
unlabelled = None
pseudo = None
pseudo_weights = np.full(net.blobs['labels'].data.shape, 1.0) # calc those for each image on the fly or just set to 1.0 / approx value??


with h5py.File(hdf5path + "/drosophila_training.h5", "r", libver="latest") as f:

    images = f['data']
    labels = f['label']
    weights = f['weights']


with h5py.File(hdf5path + "/drosophila_unlabelled.h5", "r", libver="latest") as f:

    unlabelled = f['data']



# TODO: add batchsize > 1 compability
max_iter = 300000
for step in range(max_iter):

    # pass normal labels through network and get loss
    net.blobs['data'].data[...] = images[step % images.shape[0], ...]
    net.blobs['label'].data[...] = labels[step % labels.shape[0], ...]
    net.forward()
    normal_loss = net.blobs['loss'].data[...]

    # extract forward pass output image and write to file
    if step % 50 == 0:
        features_out = net.blobs['visualize_out'].data[0,:,:,:] # plt needs WxHxC
        features_out = features_out[1:4,:,:] # omit BG probability layer - pixel will become dark if other probabilities are low

        minval = features_out.min()
        maxval = features_out.max()
        scipy.misc.toimage(features_out, cmin=minval, cmax=maxval).save("./visualizations/visualize_out_" + str(step) + ".png")

    # pass unlabelled data through network to get pseudo-labels
    net.blobs['data'].data[...] = unlabelled[step % unlabelled.shape[0], ...]
    net.forward()

    # pass unlabelled data through the network again to get pseudo loss
    net.blobs['pseudo'].data[...] = net.blobs['score']
    net.forward()
    pseudo_loss = alpha(step) * net.blobs['loss']

    # update loss as sum of both losses
    net.blobs['loss'].data[...] = (1 / batch_normal) * normal_loss \
                                + (1 / batch_pseudo) * pseudo_loss

    # perform backpropagation with combined loss
    net.backward()


print 'Pseudo-Label training done'
