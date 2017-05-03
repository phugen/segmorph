# Plots training and validation F-measure score
# against the number of training iterations.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 4:
    print "Usage: python plot_log.py numclasses path/to/train_log path/to/val_log"
    exit(-1)

numclasses = int(sys.argv[1])
train_log = sys.argv[2]
val_log = sys.argv[3]


# Read raw lines
with open(train_log) as f:
    lines = f.readlines()
    train_lines = [x.strip("\r\n") for x in lines] # remove line breaks for further parsing
    train_lines = train_lines[2:] # skip first two lines

with open(val_log) as f:
    lines = f.readlines()
    val_lines = [x.strip("\r\n") for x in lines]
    val_lines = val_lines[2:]


train_classes = [[] for x in range(numclasses)]
val_classes = [[] for x in range(numclasses)]
classes = [train_classes, val_classes]

# extract iter/score pairs per class from
# training and test values
for c,l in enumerate([train_lines, val_lines]):

    cla = classes[c]
    for x in range(0, len(l)):

        pair = l[x].split(",")[0:4:3]
        iters = pair[0]
        loss = pair[1]

        # sort by classes
        cindex = x % numclasses
        cla[cindex].append((int(float(iters)), float(loss)))

# numclasses x #Iters x (Iter, Loss)
train = np.array((classes[0]))
val = np.array((classes[1]))

# get total score over all classes
train_sum = (train[0, ...] + train[1, ...] + train[2, ...]) / numclasses
val_sum = (val[0, ...] + val[1, ...] + val[2, ...]) / numclasses

fig, ax = plt.subplots()
ns = 1 # number of samples to skip - ns=1 makes the plot virtually unreadable
talpha = 0.2

# find an show iteration with highest validation F-Measure
np.set_printoptions(suppress=True)

maxscore = 0
maxpair = None
for i in range(0, len(val_sum)):
    if val_sum[i][1] > maxscore and val_sum[i][0] % 1000 == 0:
        maxscore = val_sum[i][1]
        maxpair = val_sum[i]

print maxpair

# plot all classes for training error
ax.plot(train[0,::ns,0], train[0,::ns,1], color="silver") # class 0: background
ax.plot(train[1,::ns,0], train[1,::ns,1], color="mistyrose") # class 1: lamellopodum (+ filopodia)
ax.plot(train[2,::ns,0], train[2,::ns,1], color="honeydew") # class 2: cell proper
if numclasses == 4:
    ax.plot(train[3,::ns,0], train[3,::ns,1], color="lavender") # class 3: filopodia
#ax.plot(train_sum[::ns, 0], train_sum[::ns, 1]) # total error over all classes

# plot all classes for validation error
ax.plot(val[0,::ns,0], val[0,::ns,1], color="black") # class 0: background
ax.plot(val[1,::ns,0], val[1,::ns,1], color="red") # class 1: lamellopodum (+ filopodia)
ax.plot(val[2,::ns,0], val[2,::ns,1], color="green") # class 2: cell proper
if numclasses == 4:
    ax.plot(val[3,::ns,0], val[3,::ns,1], color="blue") # class 3: filopodia
#ax.plot(val[::ns, 0], val[::ns, 1], color="orange")
#ax.plot(val_sum[::ns, 0], val_sum[::ns, 1])


#plt.yscale("log")
ax.set_xlabel("Epochs")
epochs = 30
every = 5
plt.xticks(np.arange(0, 80000, 80000/(epochs / every))) # show every 5th epoch
plt.gca().set_xticklabels(range(0, epochs+1, every))
ax.set_ylabel("F-Measure score")
fig.set_size_inches(5, 4)
plt.show()
