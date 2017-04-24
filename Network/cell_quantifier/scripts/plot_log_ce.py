# Plots training and validation loss
# against the number of training iterations.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 3:
    print "Usage: python plot_log.py path/to/train_log path/to/val_log"
    exit(-1)

train_log = sys.argv[1]
val_log = sys.argv[2]


# Read raw lines
with open(train_log) as f:
    lines = f.readlines()
    train_lines = [x.strip("\r\n") for x in lines] # remove line breaks for further parsing
    train_lines = train_lines[2:] # skip first two lines

with open(val_log) as f:
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

# find and show iteration with lowest validation loss
minloss = 1e10
minpair = None
for i in range(0, len(vloss)):
    if vloss[i][1] <= minloss and vloss[i][0] % 5000 == 0: # TODO: delete step for data that has proper snapshots!
        minloss = vloss[i][1]
        minpair = vloss[i]

print minpair

# get total loss over all classes
#train_sum = (train[0, ...] + train[1, ...] + train[2, ...]) / numclasses
#val_sum = (val[0, ...] + val[1, ...] + val[2, ...]) / numclasses

fig, ax = plt.subplots()
ns = 1 # display every "ns"th sample
talpha = 1.0

# plot all classes for training error
#ax.plot(train[0,::ns,0], train[0,::ns,1], color="black", alpha=talpha) # class 0: background
#ax.plot(train[1,::ns,0], train[1,::ns,1], color="red", alpha=talpha) # class 1: filopodia + lamellopodum
#ax.plot(train[2,::ns,0], train[2,::ns,1], color="green", alpha=talpha) # class 2: cell proper
ax.plot(train[::ns, 0], train[::ns, 1]) # total error over all classes

# plot all classes for validation error
#ax.plot(val[0,::ns,0], val[0,::ns,1], color="black") # class 0: background
#ax.plot(val[1,::ns,0], val[1,::ns,1], color="red") # class 1: filopodia + lamellopodum
#ax.plot(val[2,::ns,0], val[2,::ns,1], color="green") # class 2: cell proper
ax.plot(val[::ns, 0], val[::ns, 1])

#plt.yscale("log")

ax.set_xlabel("Iterations")
ax.set_ylabel("Cross-Entropy loss")
plt.show()