from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc
from sklearn import cluster

import numpy as np

img = ndimage.imread("latex_test_REAL.png")[:,:,0] # load b/w image
ground_truth = ndimage.imread("latex_test_label.png") # load color GT image
imgdims = img.shape[0:2]

# do K-Means clustering for image
# TODO: talk about SLIC
labels = segmentation.slic(img, compactness=0.1, n_segments=20)
kmeans_1 = color.label2rgb(labels, img, kind="avg")

# create Region Adjacency Graph from K-means image
# and use graph cut algorithm on it recursively
g = graph.rag_mean_color(img, labels, mode='similarity')
labels_out = graph.cut_normalized(labels, g)
out = color.label2rgb(labels_out, img)

# plot original, graph cut and ground truth
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(labels_out, cmap=plt.cm.gnuplot)
ax3.imshow(ground_truth)

for a in (ax1, ax2, ax3):
    a.axis('off')

plt.subplots_adjust(top=0.88, bottom=0.11, left=0.12, right=0.90, hspace=0.00, wspace=0.05)
plt.show()

# -------------------------------------------------------------------------------
# plot original, k-means segmentation (2 params) and ground truth

#labels = segmentation.slic(img, compactness=0.1, n_segments=4)
clusterer = cluster.KMeans(n_clusters=4)
labels_1 = np.array(clusterer.fit_predict(img.reshape(img.size, 1))).reshape(imgdims[0], imgdims[1])
kmeans_1 = color.label2rgb(labels_1, img, kind="avg")

clusterer = cluster.KMeans(n_clusters=6)
#labels2 = segmentation.slic(img, compactness=0.1, n_segments=20)
labels_2 = np.array(clusterer.fit_predict(img.reshape(img.size, 1))).reshape(imgdims[0], imgdims[1])
kmeans_2 = color.label2rgb(labels_2, img, kind="avg")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax1[0].imshow(img, cmap=plt.cm.gray)
ax1[1].imshow(ground_truth)
ax2[0].imshow(kmeans_1, cmap=plt.cm.gnuplot)
ax2[1].imshow(kmeans_2, cmap=plt.cm.gnuplot)

for a in (ax1, ax2):
    for i in range(2):
        a[i].axis('off')

plt.subplots_adjust(top=0.99, bottom=0.01, left=0.17, right=0.87, hspace=0.05, wspace=0.05)
plt.show()
