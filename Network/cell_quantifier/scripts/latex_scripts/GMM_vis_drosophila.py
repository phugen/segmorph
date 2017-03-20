# This script visualizes Gaussian Mixture Model parameters
# as calculated for an image with n=4 by the EM algorithm.

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

import progressbar
import itertools as IT

from scipy import ndimage
from scipy import misc
from sklearn import mixture
from sklearn import preprocessing


# load input image and
# prepare for GMM fitting
input_img = ndimage.imread("latex_test_REAL.png")[:,:,0] # load b/w image
print input_img.shape
imgdims = input_img.shape[0:2]

# get sparse version of input image (sampled at "meshsteps" only)
meshstep = 1
xcoords = range(0, imgdims[1], meshstep)
ycoords = range(0, imgdims[0], meshstep)
X, Y = np.meshgrid(xcoords, ycoords)
coords = list(IT.product(ycoords, xcoords))


samples = input_img.reshape((input_img.size, 1))
print "Samples: " + str(samples.shape)

# Do a GMM fit of the image to obtain
# estimated Gaussian parameters
n = 4
model = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
model.fit(samples)

plt.gca().invert_yaxis() # reverse y-axis: image convention
plt.rcParams['axes.facecolor'] = 'white'
plt.title("Original image (color mapped)")
plt.xlabel("X")
plt.ylabel("Y")
blackmap = np.array([(0, 0, 0, i) for i in np.linspace(0, 1, 50)], dtype=np.float32)
#plt.contourf(X, Y, input_img, colors=blackmap)
#plt.show()

means = model.means_
covs = model.covariances_
weights = model.weights_

print "Means: " + str(means.shape)
print "Covs: " + str(covs.shape)

labels = model.predict(samples).reshape(imgdims)
misc.imsave("gmm_labels.png", labels)




# get gaussian pdfs for all pixels of the image
'''d2means = np.zeros((n, 2))
d2covs = np.zeros((n, 2, 2))
indv_probs = np.zeros((n,) + imgdims)

for g in range(n):
    d2means[g] = [means[g], means[g]]
    d2covs[g, ...] = [[covs[g], 0], [0, covs[g]]]

# Create mixture model:
for g in range(n):
    indv_probs[g, ...] = np.array([scipy.stats.multivariate_normal(d2means[g], d2covs[g]).pdf([y, x]) \
                          for y in ycoords for x in xcoords]).reshape(imgdims)
    print str(np.min(indv_probs[g])) + ", " + str(np.max(indv_probs[g]))



indv_probs[0, ...] = np.sum([weights[g] * indv_probs[g, ...] for g in range(n)])

plt.contourf(xcoords, ycoords, np.array(indv_probs[0]).reshape(imgdims), \
                 colorlevels, cm=plt.cm.gray)'''

indv_probs = np.zeros((n, input_img.size))
mixture_probs = np.zeros((input_img.size))

#for g in range(n):
    #indv_probs[g, ...] = np.array([scipy.stats.multivariate_normal(means[g], covs[g]).pdf(x) \
    #                      for x in np.arange(input_img.size)])
    #mixture_probs[...] += weights[g] * indv_probs[g, ...]


# plot gaussian probabilites
#for g in range(n):
    #plt.plot(range(input_img.size), indv_probs[g])

# plot mixture of gaussians
#plt.plot(range(input_img.size), mixture_probs, color="m")


# custom colormaps: color -> transparent as the probability decreases
colorlevels = 8
blackmap = np.array([(0, 0, 0, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
redmap = np.array([(1, 1, 0, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
greenmap = np.array([(0, 1, 1, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
bluemap = np.array([(1, 0, 1, i) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)

maps = [blackmap, redmap, greenmap, bluemap]


# get probabilities of all pixels
# considering all gaussians
log_probs = model.score_samples(samples)
indv_probs = model.predict_proba(samples)

# scale data to [0, 1]
scaler = preprocessing.MinMaxScaler((0, 1))
log_probs = scaler.fit_transform(log_probs.reshape(-1, 1))

for g in range(n):
    scaler = preprocessing.MinMaxScaler((0, 1))
    indv_probs[:,g] = scaler.fit_transform(indv_probs[:,g].reshape(-1, 1)).reshape(-1)

# interpolate data for smoother plot
interp_fac = 1
log_probs = log_probs.reshape(imgdims[0], imgdims[1])
#log_probs = scipy.ndimage.zoom(log_probs, interp_fac)

#plt.gca().invert_yaxis() # reverse y-axis: image convention
plt.rcParams['axes.facecolor'] = 'white'

for g in range(n):
    plt.contourf(indv_probs[:, g].reshape(imgdims), colors=maps[g])

plt.colorbar()


#con_probs = plt.contourf(log_probs, cmap=plt.cm.afmhot)
#plt.clabel(con_probs, con_probs.levels[0::2], colors="black", inline=0, fontsize=8)


plt.xlabel("X")
plt.ylabel("Y")
plt.title("Individual GMM probabilities for  n=" + str(n))
plt.show()

# plot GMM labels
labelmap = np.array([(i, i, i, 1) for i in np.linspace(0, 1, colorlevels)], dtype=np.float32)
plt.contourf(labels.reshape(imgdims), colors=labelmap)

plt.gca().invert_yaxis() # reverse y-axis: image convention
plt.xlabel("X")
plt.ylabel("Y")
plt.title("GMM labels for n=" + str(n))
plt.show()




#con_labels = plt.contourf(ycoord, xcoord, gmm, n, colors=blackmap)

#con_black = plt.contourf(ycoord, xcoord, black, n, colors=blackmap)
#plt.colorbar(format='%.0e')
#plt.clabel(con_black, levels[1::2], colors="white", inline=0, fontsize=8)

#con_red = plt.contourf(ycoord, xcoord, red, n, colors=redmap)
#plt.colorbar(format='%.0e')
#plt.clabel(con_red, levels[1::2], colors="black", inline=0, fontsize=8)

#con_green = plt.contourf(ycoord, xcoord, green, n, colors=greenmap)
#plt.colorbar(format='%.0e')

#con_blue = plt.contourf(ycoord, xcoord, blue, n, colors=bluemap)
#plt.colorbar(format='%.0e')

plt.show()
