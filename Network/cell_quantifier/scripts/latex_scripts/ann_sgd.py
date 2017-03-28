# Implements a neural network with one hidden layer
# that is trained to separate a not-linearly separable
# dataset using backpropagation and gradient descent.
#
# Based on http://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html .

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create not-linearly separable dataset
means = [[0, 0], [7, 0]]
covs = [[[1, 0],
         [0, 1]],
         [[1, 0],
         [0, 1]]]

setsize = 50
d1 = np.random.multivariate_normal(means[0], covs[0], size=(setsize))
d2 = np.array([[3, 0], [2, 2.24], [1, 2.83], [0, 3], \
      [-1, 2.83], [-2, 2.24], [-3, 0], \
      [-1, -2.83], [-2, -2.24], \
      [1, -2.83], [2, -2.24]])

X = np.concatenate((d1, d2))

# labels for dataset
y1 = np.array([0 for xi in range(setsize)])
y2 = np.array([1 for xi in range(11)])
y = np.concatenate((y1, y2))


h = 5 # number of neurons in layer 1 (hidden layer)
D = 2 # input dimension
K = 2 # number of output classes

# random weights and biases
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

# learning rate
step_size = 1e-1

# Training loop
num_examples = X.shape[0]
maxloss = 1e-3

for i in range(10000):

    # calculate ReLU activations of hidden layer
    hidden_layer = np.maximum(0, np.dot(X, W) + b)

    # calculate outputs
    scores = np.dot(hidden_layer, W2) + b2

    # get softmax probs and cross-entropy loss
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_logprobs = -np.log(probs[range(num_examples), y])
    loss = np.sum(correct_logprobs) / num_examples

    if loss <= maxloss:
        print "Training finished!"
        break

    if i % 250 == 0:
        print "Iteration %d: Loss %f" % (i, loss)

    # output layer deltas
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropagation for W2/b2 [output layer weights]
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    # get hidden layer deltas
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0 # apply ReLU derivative

    # backpropagation for W1/b2 [hidden layer weights]
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # All partial derivatives are known; optimize using
    # gradient descent step: go into negative gradient direction
    W += - step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2



# Classifying dataset by applying learned weights

# create coordinates
h = 0.01 # mesh resolution
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# pass mesh through neural network (forward pass)
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2 # get probabilities
Z = np.argmax(Z, axis=1) # choose class with highest probability
Z = Z.reshape(xx.shape)

# plot original dataset on top of mesh classification
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)

ax1.contourf(xx, yy, Z, cmap=plt.cm.spectral_r, alpha=0.3)
ax1.contour(xx, yy, Z, cmap=plt.cm.Greys, alpha=1)

ax1.scatter(d1[:,0], d1[:,1], c="#ff8c00")
ax1.scatter(d2[:,0], d2[:,1], c="#6699CC")

ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# plot output transformation for one of the output units
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
Z = np.dot(np.maximum(0, np.dot(X, W) + b), W2) + b2 # get transformed outputs
td1 = np.c_[(d1, Z[range(setsize), 1])] # add hidden layer transformation as 3rd dim
td2 = np.c_[(d2, Z[range(setsize, setsize + 11), 1])]

ax2.scatter(td1[:,0], td1[:, 1], td1[:, 2], c="#ff8c00")
ax2.scatter(td2[:,0], td2[:, 1], td2[:, 2], c="#6699CC")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z = net(x, y)")

plt.show()
