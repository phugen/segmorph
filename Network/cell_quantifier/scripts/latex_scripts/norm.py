# Visualizes zero-mean / unit standard deviation
# normalization for ANNs using 2D data.

import numpy as np
from random import random
import matplotlib.pyplot as plt

# create random 2D data
data_mean = [7, 3]
data_cov = [[8, 0],[0, 1]]

setsize = 500
data = np.random.multivariate_normal(data_mean, data_cov, size=(setsize))

# get mean/std
mu = np.mean(data, axis=0)
sigma = np.std(data, axis=0)
norm_data = data.copy()

print mu
print sigma

# apply normalization
norm_data[:,0] = (norm_data[:,0] - mu[0]) / sigma[0]
norm_data[:,1] = (norm_data[:,1] - mu[1]) / sigma[1]

# plot difference
fig = plt.figure(figsize=(4,2.5))
plt.xlabel("x")
plt.ylabel("y")
ax1 = fig.add_subplot(111)
ax1.scatter(data[:,0], data[:,1], s=4)
ax1.scatter(norm_data[:,0], norm_data[:,1], s=4)
ax1.set_ylim(-6, 6)

# show origin axes for emphasis
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')

plt.tight_layout()
plt.show()
