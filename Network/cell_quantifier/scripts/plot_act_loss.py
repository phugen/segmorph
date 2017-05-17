# This script is like plot_log_ce.py, but plots
# the training and validation progress of
# the ELU, PReLU and LReLU networks at once.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


train_log = ["../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_elu_4/unet_weighted_batchnorm_shuffle_msra_elu_4_solver.log.train",\
             "../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_lrelu_4/unet_weighted_batchnorm_shuffle_msra_lrelu_4_solver.log.train",\
             "../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_prelu_4/unet_weighted_batchnorm_shuffle_msra_prelu_4_solver.log.train"]

val_log = ["../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_elu_4/unet_weighted_batchnorm_shuffle_msra_elu_4_solver.log.test",\
             "../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_lrelu_4/unet_weighted_batchnorm_shuffle_msra_lrelu_4_solver.log.test",\
             "../results/16_05_2017/unet_weighted_batchnorm_shuffle_msra_prelu_4/unet_weighted_batchnorm_shuffle_msra_prelu_4_solver.log.test"]


fig, ax = plt.subplots()

# colors for different activation functions
traincolors = ["mistyrose", "honeydew", "lavender"]
valcolors = ["red", "green", "blue"]


for ir in range(3):

    # Read raw lines
    with open(train_log[ir]) as f:
        lines = f.readlines()
        train_lines = [x.strip("\r\n") for x in lines] # remove line breaks for further parsing
        train_lines = train_lines[2:] # skip first two lines

    with open(val_log[ir]) as f:
        lines = f.readlines()
        val_lines = [x.strip("\r\n") for x in lines]
        val_lines = val_lines[2:]


    tloss = []
    vloss = []

    # extract iter/loss pairs per class from
    # training and test values
    for c,l in enumerate([train_lines, val_lines]):
        for x in range(0, len(l)):

            # get iters/loss from lines
            pair = l[x].split(",")[0:4:3]
            iters = pair[0]
            loss = pair[1]

            # training loss
            if c == 0:
                tloss.append((int(float(iters)), float(loss)))

            # validation loss
            else:
                vloss.append((int(float(iters)), float(loss)))


    #Iters x (Iter, Loss)
    train = np.array(tloss)
    val = np.array(vloss)

    ns = 1 # plot every "ns"th sample

    # plot training error
    ax.plot(train[::ns, 0], train[::ns, 1], color=traincolors[ir], zorder=1) # total error over all classes

    # plot validation error
    ax.plot(val[::ns, 0], val[::ns, 1], color=valcolors[ir], zorder=2)

#plt.yscale("log")

ax.set_xlabel("Epochs")
#ax.set_ylim([0, 1.3])
iterations = 40000
epochs = 6
every = 1 # show every 5th epoch

plt.xticks(np.arange(0, iterations, iterations / (epochs / every))) # uniform xtics
plt.yticks(np.arange(0, 1.1, 0.1)) # uniform ytics

plt.gca().set_xticklabels(range(0, epochs+1, every)) # iters -> epochs
ax.set_ylabel("Cross-Entropy loss")

fig.set_size_inches(5, 4)
plt.show()
