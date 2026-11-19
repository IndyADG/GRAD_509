# Load the dataset of MNIST images and keep the first 60000
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
#X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
print(X_train.shape)

import PIL
from PIL import Image
# Plot the first image. Here we reshape the image from a 784 long vector to a 28-by-28 grid
# Since this is quite small to look at, we scale it up to 280-by-280 by interpolating. This is only
# needed for visualization
Image.fromarray((X_train[0,:]).astype('uint8').reshape((28,28))).resize((280, 280)).show()

import numpy as np
# This is the "average image". Try visualizing it
mu = np.mean(X_train, axis=0)
# Subtract out the mean image from each image
X_train = X_train - mu

# To view the first image, we now must add back the mean
Image.fromarray((X_train[0,:]+mu).astype('uint8').reshape((28,28))).resize((280, 280)).show()

from sklearn.decomposition import PCA
# Use PCA to reduce the dimensionality from 784 to 75 (about 10%)
pca = PCA(n_components=75)
X_reduced = pca.fit_transform(X_train)
print(X_reduced.shape)

# Project the 75-d data back to 784-d space
X_recovered = pca.inverse_transform(X_reduced)

# Plot the first image. We first shift and rescale it to lie between 0 and 255

#Add back mean
X_rec_plt = X_recovered[0,:]+mu

# Rescale
X_rec_plt = X_rec_plt - X_rec_plt.min()
X_rec_plt = X_rec_plt*255 / X_rec_plt.max()
Image.fromarray((X_rec_plt).astype('uint8').reshape((28,28))).resize((280, 280)).show()

# Project the data down to 2d and plot a scatterplot of the images
import matplotlib.pyplot as plt
plt.scatter(X_reduced[:,0], X_reduced[:,1],c=y_train.astype('int'), alpha=0.2)
plt.show()