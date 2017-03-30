# Variation of ann_sgd.py that overfits massively on a pathological dataset.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create not-linearly separable dataset
means = [[1, 1], [6, 1]]
covs = [[[1, 0],
         [0, 1]],
         [[1, 0],
         [0, 1]]]

d1size = 60
d2size = 30
d1 = np.random.multivariate_normal(means[0], covs[0], size=(d1size))
d2 = np.random.multivariate_normal(means[1], covs[1], size=(d2size))
overfitme = [[0.2, 1.2], [0.67, -0.2], [1.5, 1.6]]
d2 = np.concatenate((d2, overfitme))

X = np.concatenate((d1, d2))

# labels for dataset
y1 = np.array([0 for xi in range(d1size)])
y2 = np.array([1 for xi in range(d2size+3)])
y = np.concatenate((y1, y2))


h = 100 # number of neurons in layer 1 (hidden layer)
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

for i in range(100000):

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
ax1 = fig.add_subplot(1, 1, 1)

ax1.contourf(xx, yy, Z, cmap=plt.cm.spectral_r, alpha=0.3)
ax1.contour(xx, yy, Z, cmap=plt.cm.Greys, alpha=1)

ax1.scatter(d1[:,0], d1[:,1], c="#ff8c00")
ax1.scatter(d2[:,0], d2[:,1], c="#6699CC")

ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())
ax1.set_xlabel("x")
ax1.set_ylabel("y")

plt.show()
