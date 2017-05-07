# Plots various activation functions and
# their derivatives.

import math
import numpy as np
import matplotlib.pyplot as plt


# helper functions
def ReLU(x, alpha=0):
    if x <= 0:
        return alpha * x
    else:
        return x

def dReLU(x, alpha=0):
    if x <= 0:
        return alpha
    else:
        return 1

def ELU(x, alpha):
    if x < 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x

def dELU(x, alpha):
    if x < 0:
        return ELU(x, alpha) + alpha
    else:
        return 1


xvars = np.arange(-5, 5, 0.01)

# Sigmoid function + derivative:
sig = [ (1 / (1 + math.exp(-x))) for x in xvars ]
dsig = [ (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x)))) for x in xvars ]

# tanh + derivative:
tanh = [ math.tanh(x) for x in xvars ]
dtanh = [ 1 - (math.tanh(x))**2 for x in xvars ]

# ReLU + derivative:
relu = [ ReLU(x) for x in xvars ]
drelu = [ dReLU(x) for x in xvars ]

# LReLU + derivative:
lrelu = [ ReLU(x, alpha=0.3) for x in xvars ]
dlrelu = [ dReLU(x, alpha=0.3) for x in xvars ]

# ELU + derivative:
elu = [ ELU(x, alpha=0.5) for x in xvars ]
delu = [ dELU(x, alpha=0.5) for x in xvars ]

vals = [sig, tanh, relu, lrelu, elu]
derivs = [dsig, dtanh, drelu, dlrelu, delu]

# set explicit yticks so tight_layout doesn't
# toy with the ytick frequency
ytix = [np.arange(0, 1.0+0.2, 0.2), \
        np.arange(-1.0, 1.0+0.5, 0.5), \
        np.arange(0.0, 5.0+1, 1.0), \
        np.arange(-1.0, 5.0+1, 1.0), \
        np.arange(0.0, 5.0+1, 1.0)]

# plot all functions separately
for i in range(5):
    plt.figure(figsize=(2.7,2.7))

    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.axhline(y=0, color='k', alpha=0.25)
    plt.axvline(x=0, color='k', alpha=0.25)

    plt.plot(xvars, vals[i], color="steelblue")
    plt.plot(xvars, derivs[i], color="orange")

    plt.tight_layout()
    plt.xticks(np.arange(-5.0, 5.0+1, 2.5))
    plt.yticks(ytix[i])

    plt.show()
