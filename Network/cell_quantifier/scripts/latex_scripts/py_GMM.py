# Do a GMM fit using the pixels of an image as input.

from scipy import ndimage
from scipy import misc
from sklearn import mixture
import numpy as np
import h5py
import matplotlib.pyplot as plt

n = 4
feature_no = 1
hdf5path = "../../archs/drosophila_0_training_4.h5"

# load training data from a HDF5 file
#samples = None
#with h5py.File(hdf5path, "r", libver="latest") as f:
#     samples = f["data"][0:4, 0, ...] # image is grayscale anyway, reduce to 1 dim
#print samples.shape

#samples = samples.reshape(samples.shape[0], samples.shape[1] * samples.shape[2])

# bring data in GMM format: n_samples x n-dimensional data points
#pro_samples = []
#for samp in range(samples.shape[0]):
#    samppx = []
#    for px in range(samples.shape[1]):
#        samppx.append(samples[samp, px])
#
#    pro_samples.append(samppx)

#print "Reordering done"

# image to be classified after training
input_img = ndimage.imread("latex_test_REAL.png")[:,:,0]
dims = input_img.shape[0:2]
pro_samples = input_img.reshape((input_img.size, 1))

# train GMM
gmm = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
gmm.fit(pro_samples)

np.set_printoptions(suppress=True)

print "Means:"
print gmm.means_
print ""

print "Covariances:"
print gmm.covariances_
print ""

print "Weights:"
print gmm.weights_
print ""

labels = gmm.predict(pro_samples).reshape(dims)
print labels.shape

# plot input image and result
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.axis("off")
ax1.imshow(input_img, plt.cm.gray)

ax2 = fig.add_subplot(122)
ax2.axis("off")
ax2.imshow(labels, cmap=plt.cm.viridis)

plt.show()

#misc.imsave("gmm_labels.png", labels)
