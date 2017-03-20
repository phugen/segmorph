# This script visualizes how a normal Gaussian and a
# GMM with k=3 fit a data set randomly sampled from
# three Gaussian distributions.

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

import progressbar
import itertools as IT
import math

from scipy import ndimage
from scipy import misc
from sklearn import mixture
from sklearn import preprocessing

# define transparent colors, according to pdf(x, y)
def map_s(val, max):
    ''' maps val from [0,max] to [0,1]'''
    return val / float(max)


# create data from three gaussians
numsamples = 100

mean = [8, 20]
cov = [[0, 1], [1, 0]]
data1 = np.random.multivariate_normal(mean, cov, numsamples)

mean = [14, 18]
cov = [[0, 1], [1, 0]]
data2 = np.random.multivariate_normal(mean, cov, numsamples)

mean = [13, 25]
cov = [[4, 2], [2, 4]]
data3 = np.random.multivariate_normal(mean, cov, numsamples)

data = np.concatenate((data1, data2, data3))

# fit data with one gaussian
n = 1
model = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
model.fit(data)

g_means = model.means_
g_covs = model.covariances_

# plot single gaussian probabilities
f, ax = plt.subplots(2, sharex=True)

yend = 35
xend = 30
probs = np.zeros((xend, yend))

for xi in range(xend):
    for yi in range(yend):
        probs[xi, yi] = scipy.stats.multivariate_normal(g_means[0], g_covs[0]).pdf((xi, yi))


colorlevels = 100
blackmap = np.array([(0, 0, 0, map_s(i, np.max(probs))) for i in np.linspace(0, np.max(probs), colorlevels)], dtype=np.float32)

ax[0].contourf(range(yend), range(xend), probs.reshape(xend, yend), colorlevels-1, cmap=plt.cm.gist_heat)
ax[0].set_xlim((13, xend))
ax[0].set_ylim((5, 20))

# plot data
ax[0].scatter(data[:,1], data[:,0], color="b", marker="o", s=2)


plt.rcParams['axes.facecolor'] = 'white'
ax[0].set_title("Probability map for single Gaussian fit")

# fit with 3 gaussians
n = 3
model = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
model.fit(data)

g_means = model.means_
g_covs = model.covariances_
g_weights = model.weights_

# plot mixed gaussians
probs = np.zeros((xend, yend))
for g in range(n):
    for xi in range(xend):
        for yi in range(yend):
            probs[xi, yi] += g_weights[0] * scipy.stats.multivariate_normal(g_means[g], g_covs[g]).pdf((xi, yi))

colorlevels = 100
blackmap = np.array([(0, 0, 0, map_s(i, np.max(probs))) for i in np.linspace(0, np.max(probs), colorlevels)], dtype=np.float32)

plotted = ax[1].contourf(range(yend), range(xend), probs.reshape(xend, yend), colorlevels-1, cmap=plt.cm.gist_heat)
#plt.colorbar()

# plot data
ax[1].scatter(data[:,1], data[:,0], color="b", marker="o", s=2)
ax[1].set_title("Probability map for GMM fit with K=3")
ax[1].set_ylim((5, 20))

plt.xlabel("X")
plt.ylabel("Y")

# add one color bar for both plots
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
plotted = f.colorbar(plotted, cax=cbar_ax)
plt.show()

#---------------------------------------------------------------------------------

# Fit K=4 GMM to real test image and show it next to real label
input_img = ndimage.imread("latex_test_REAL.png")[:,:,0] # load b/w image
ground_truth = ndimage.imread("latex_test_label.png") # load color GT image
imgdims = input_img.shape[0:2]
samples = input_img.reshape((input_img.size, 1))

# Do a GMM fit of the image to obtain
# estimated Gaussian parameters
n = 4
model = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
model.fit(samples)
labels = model.predict(samples).reshape(imgdims)

# get posterior probabilities for each samples
# according to each gaussian
manual = np.argmax(model.predict_proba(samples), axis=1)

g_means = model.means_
g_covs = model.covariances_
g_weights = model.weights_

g0 = np.random.multivariate_normal(g_means[0], g_covs[0], (100))
g1 = np.random.multivariate_normal(g_means[1], g_covs[1], (100))
g2 = np.random.multivariate_normal(g_means[2], g_covs[2], (100))
g3 = np.random.multivariate_normal(g_means[3], g_covs[3], (100))

probs = model.predict_proba(samples).reshape((imgdims + (4,)))

colorlevels = 8
blackmap = np.array([(0, 0, 0, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
redmap = np.array([(1, 1, 0, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
greenmap = np.array([(0, 1, 1, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
bluemap = np.array([(1, 0, 1, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)

maps = [blackmap, redmap, greenmap, bluemap]

print probs.shape
#for g in range(n):
plt.contourf(range(imgdims[1]), range(imgdims[0]), probs[..., 0], colors=blackmap)

plt.show()

# Show original, GMM result and ground truth next to each other
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(input_img, cmap=plt.cm.gray)
ax2.imshow(labels, cmap=plt.cm.viridis)
ax3.imshow(ground_truth)

# remove axis because images are interpretable
# without them
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
f.subplots_adjust(hspace=0)

plt.show()
