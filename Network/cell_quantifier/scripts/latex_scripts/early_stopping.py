# Visualizes early stopping techniques
# "Early Stopping - But when?" by Lutz Prechel
# applied to artificial training statistic data.

# Clamping function.
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

import numpy as np
import math
import matplotlib.pyplot as plt

# create idealized training loss / validation loss data for some epochs
epochs = 120
tr = np.array([1 - math.log(x) for x in range(1, epochs)])
E_val = np.array([0.00012*(x - 80)**2 + 0.25 for x in range(1, epochs)])
E_tr = (tr - np.min(tr)) / (np.max(tr) - np.min(tr))

# add noise to both functions
noise = np.vectorize(lambda v: clamp((v + np.random.randn() * 0.1 * v), 0.0, 1.0))
E_val_n = noise(E_val)
E_tr_n = noise(E_tr)

# find stopping point according to class 1 criterion
# (generalization loss > alpha)
GL = np.full(shape=(epochs), fill_value=0.0)
alpha = 30

stop = epochs
for index, e in enumerate(range(0, epochs - 1)):
    curr_min = np.min(E_val_n[0:e+1])
    GL[e] = 100.0 * ((E_val_n[e] / float(curr_min)) - 1.0)

    # set first found stop
    # and ignore others past that
    if (stop == epochs) and (GL[e] > alpha):
        stop = index


fig, ax = plt.subplots()
ax.plot(E_val_n)
ax.plot(E_tr_n)

# mark criterion 1 stopping point
ax.annotate("a)", \
            xy=(stop - 1, E_val_n[stop - 1]), xycoords='data', \
            xytext=(stop - 1 + 30, E_val_n[stop - 1] + 30), textcoords='offset points', \
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8))


# find stopping point according to class 2 criterion
# (generalization loss / training process in interval k) > alpha

k = 10 # average losses over 10 epochs
P_k = np.zeros(epochs, dtype=np.float32) # progression values
stop2 = epochs-1
alpha2=10

for index, e in enumerate(range(k, epochs - 1)):
    k_tr = np.sum(E_tr[e-k+1:e]) # training error over last k epochs
    min_tr = k * np.min(E_tr[e-k+1:e]) # minimum training error over last k epochs

    P_k[e] = 1000.0 * ((k_tr / float(min_tr)) - 1.0) # "jitter" measure

    if (GL[e] / P_k[e]) > alpha2:
        stop2 = index
        break

# mark criterion 2 stopping point
ax.annotate("b)", \
            xy=(stop2 - 1, E_val_n[stop2 - 1]), xycoords='data', \
            xytext=(stop2 - 1 - 50, E_val_n[stop2 - 1] + 50), textcoords='offset points', \
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=8))

ax.set_xlabel("Epochs trained")
ax.set_ylabel("Loss")
plt.show()
