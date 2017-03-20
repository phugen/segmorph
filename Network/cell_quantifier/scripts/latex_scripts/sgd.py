# Implements the Gradient Descent algorithm
# and visualizes the minimization progress
# of the target function.

import math
import numpy as np
from random import random
import matplotlib.pyplot as plt

# return the gradient vector of the function
# f(a, b) = sin(a) + cos^2(b)
def nabla(theta):
    a = theta[0]
    b = theta[1]

    return np.array((math.cos(a), -2 * math.sin(b) * math.cos(b)))

# choose random theta
theta = np.array((random(), random()))

# prepare plot
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_title("Gradient Descent optimization")

# gradient descent optimization
lr = 0.1
thetas = []

for i in range(100):
    theta -= lr * nabla(theta)
    thetas.append((theta[0], theta[1])) # save for plotting

thetas = np.array(thetas)

# calculate f(a, b) for "all" a, b
gridsize = 100
levels = 15
X, Y = np.meshgrid(np.linspace(-2, 2, gridsize), np.linspace(-2, 4, gridsize))
Z = np.vectorize(lambda a, b: math.sin(a) + math.cos(b)**2)(X, Y)

# plot contours of f and route of the
# optimization process
conts = ax.contour(X, Y, Z, levels, cmap=plt.cm.gnuplot, alpha=0.45)
ax.clabel(conts, conts.levels[::3], fontsize="6", colors="k")
ax.plot(thetas[:,0], thetas[:,1])

# plot start and arrowhead to show direction
pos = (thetas[0,0], thetas[0,1])
circle = plt.Circle(pos, 0.060, color="steelblue")
ax.add_artist(circle)
pos = (thetas[-1,0], thetas[-1,1])
ax.annotate("", (pos[0]-0.05, pos[1]), pos, arrowprops=dict(fc="steelblue", width=0.1, edgecolor="steelblue"))

plt.show()
