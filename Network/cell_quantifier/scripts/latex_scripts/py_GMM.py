# Do a GMM fit using the pixels of an image as input.

from scipy import ndimage
from scipy import misc
from sklearn import mixture
import numpy as np

n = 4
feature_no = 1

input_img = ndimage.imread("latex_test.png")[:,:,0]
dims = input_img.shape[0:2]

samples = input_img.reshape(dims[0] * dims[1], feature_no)
print samples

gmm = mixture.GaussianMixture(n, max_iter=300, covariance_type='full')
gmm.fit(samples)

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

labels = gmm.predict(samples).reshape(dims)
print labels.shape

misc.imsave("gmm_labels.png", labels)
