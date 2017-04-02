# Creates a figure showing the effects of
# Elastic Deformation side-by-side.

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    assert len(image.shape) == 3 and image.shape[2] == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # create random shifting vectors for each pixel
    dx = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # shift pixels using linear interpolation
    # (interpolate each channel in the same way)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    # use nearest neighbor interpolation to preserve labels
    r = map_coordinates(image[..., 0], indices, order=3).reshape(shape[0:2])
    g = map_coordinates(image[..., 1], indices, order=3).reshape(shape[0:2])
    b = map_coordinates(image[..., 2], indices, order=3).reshape(shape[0:2])

    # reassemble image
    trans = np.zeros((image.shape))
    trans[..., 0] = r
    trans[..., 1] = g
    trans[..., 2] = b

    return trans



image = scipy.ndimage.imread("elastic2.png", mode="RGB")

# normal image
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(image)

# deformed image
elastic = elastic_transform(image, 100, 10)
elastic = -elastic # invert colors for matplotlib
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(elastic)

ax1.axis('off')
ax2.axis('off')
plt.show()
