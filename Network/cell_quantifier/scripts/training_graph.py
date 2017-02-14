import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import time

filepath = sys.argv[1]
setsize = sys.argv[2] # number of images in the training set


cnt = 0
with open(filepath) as logfile:
    # find line where actual training starts
    for line in logfile:
        cnt += 1
        if "Solver scaffolding done" in line:
            break

losses = []

with open(filepath) as logfile:
    itercnt = 0
    lines = iter(logfile.read().splitlines()[cnt:])
    for line in lines:
        if itercnt % 2 != 0:
            line = next(lines)
            itercnt += 1
            continue

        try:
            val = re.split(" ", line)[9]
            loss = float(val)

        except(ValueError, IndexError):
            print line
            time.sleep(10)

        losses.append(loss)
        itercnt += 1

plt.plot(losses)
plt.xlabel("Iterations (Base LR = 0.001), Epoch = " + setsize)
plt.ylabel ("Cross-entropy loss")
plt.show()
plt.savefig("./visualizations/training_loss.png", bbox_inches='tight', pad_inches=0) # save plot
