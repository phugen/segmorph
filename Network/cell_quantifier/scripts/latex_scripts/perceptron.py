# Implements the perceptron algorithm for a dataset
# and visualizes the final classification border
# and a second, not-linearly separable dataset.

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


# Heaviside activation function.
def act(score):
    if score >= 0:
        return 1
    else:
        return -1

# Plots the segmentation boundary and the initial
# data set classified by it (+ unseparable set)
def plot_boundary(data, weights, d1, d2):

    # get boundary line
    x1 = 0
    y1 = (-x1 / weights[1]) * weights[0] - (weights[2] / weights[1])
    x2 = 1
    y2 = (-x2 / weights[1]) * weights[0] - (weights[2] / weights[1])

    p1 = [x1, y1]
    p2 = [x2, y2]


    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True)
    #ax1.set_title("Perceptron classification boundary")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    #ax2.set_title("Not-linearly separable dataset")
    ax2.set_xlabel("X")

    # plot dataset
    for xi in range(data.shape[0]):
        if(act(np.dot(w, data[xi])) < 0):
            ax1.scatter(data[xi,0], data[xi,1], c="#ff8c00")
        if(act(np.dot(w, data[xi])) >= 0):
            ax1.scatter(data[xi,0], data[xi,1], c="#6699CC")

    xmin, xmax = ax1.get_xbound()

    # find coordinates within line should be plotted
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax1.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax], c="k")
    ax1.add_line(l)


    # plot inseparable dataset
    ax2.scatter(d1[:,0], d1[:,1], c="#ff8c00")
    ax2.scatter(d2[:,0], d2[:,1], c="#6699CC")

    ax1.set_ylim(-4, 4)

    plt.show()



# create some data to separate
means = [[0, 0], [7, 0]]
covs = [[[1, 0],
         [0, 1]],
         [[1, 0],
         [0, 1]]]

setsize = 50
dimension = 2
c1 = np.random.multivariate_normal(means[0], covs[0], size=(setsize))
c2 = np.random.multivariate_normal(means[1], covs[1], size=(setsize))

# create not-linearly separable data
d1 = np.random.multivariate_normal(means[0], covs[0], size=(setsize))
d2 = np.array([[3, 0], [2, 2.24], [1, 2.83], [0, 3], \
      [-1, 2.83], [-2, 2.24], [-3, 0], \
      [-1, -2.83], [-2, -2.24], \
      [1, -2.83], [2, -2.24]])

# add 1 in third dim for bias weights
temp = np.concatenate((c1, c2))
x = np.array([np.concatenate((xi, [1])) for xi in temp])

# create label vector
t1 = np.array([1 for xi in range(setsize)])
t2 = np.array([-1 for xi in range(setsize)])
t = np.concatenate((t1, t2))

# initialize weights randomly
w = np.random.random_sample(size=(dimension + 1))
#print "Initial weights:\n" + str(w)

# show initial line
plot_boundary(x, w, d1, d2)

# Perform Stochastic Gradient Descent (1-sample updates)
lr = 1 # learning rate

for e in range(100): # 100 epochs
    for xi in range(x.shape[0]):

        # update weights if sample misclassified
        if act(np.dot(w, x[xi])) * t[xi] < 0:
            w += lr * x[xi] * t[xi]

            # if all predictions are correct for the updated
            # weights, the algorithm has converged and stops
        stop = True
        for xii in range(x.shape[0]):
            score = act(np.dot(w, x[xii])) * t[xii]
            if score < 0:
                stop = False

        if stop:
            print "Algorithm converged at iteration " + str(xi) + " in epoch " + str(e) + "!"
            print "Classifier weights: " + str(w)
            break

    if stop:
        break

# show new separation border
plot_boundary(x, w, d1, d2)
