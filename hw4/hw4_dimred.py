# Load the dataset of MNIST images and keep the first 60000
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
#X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]
print(X_train.shape)

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
#Image.fromarray((X_train[0,:]+mu).astype('uint8').reshape((28,28))).resize((280, 280)).show()

from sklearn.decomposition import PCA
# Use PCA to reduce the dimensionality from 784 to 75 (about 10%)
pca = PCA(n_components=75)
X_reduced = pca.fit_transform(X_train)
print(X_reduced.shape)

# What is the explained variance ratio for the first several principle components?
print(pca.explained_variance_ratio_)

# Project the 75-d data back to 784-d space
X_recovered = pca.inverse_transform(X_reduced)

# Plot the first image. We first shift and rescale it to lie between 0 and 255

#Add back mean
X_rec_plt = X_recovered[0,:]+mu

# Rescale
X_rec_plt = X_rec_plt - X_rec_plt.min()
X_rec_plt = X_rec_plt*255 / X_rec_plt.max()
#Image.fromarray((X_rec_plt).astype('uint8').reshape((28,28))).resize((280, 280)).show()

# What is the explained variance ratio for the first several principle components?
print(pca.explained_variance_ratio_)

# Project the data down to 3d and plot a scatterplot of the images
import matplotlib.pyplot as plt
import plotly.express as pltx
#plt.scatter(X_reduced[:,0], X_reduced[:,1],c=y_train.astype('int'), alpha=0.2)
#add 3D and take sample of total dataset for visualization
n = 4000
#plt.scatter(X_reduced[:n,0], X_reduced[:n,1], X_reduced[:n, 2],c=y_train[:n].astype('int'), alpha=0.4)
threeD_plot = pltx.scatter_3d(x=X_reduced[:n, 0], y=X_reduced[:n, 1], z=X_reduced[:n, 2],
    color=y_train[:n].astype(str), labels={'x': 'PC 1', 'y': 'PC 2', 'z': 'PC 3', 'color': 'Digit'},
    opacity=0.5, title='3D PCA Projection of MNIST')

threeD_plot.show()

# Use isomap to perform nonlinear dimensionality reduction
'''
from sklearn.manifold import Isomap
embedding = Isomap(n_components=2)
# Since this is slower, use a smaller dataset
X_isomap = embedding.fit_transform(X_train[:5000])
print(X_isomap.shape)

plt.scatter(X_isomap[:,0], X_isomap[:,1],c=y_train[:5000].astype('int'),alpha=0.4)
plt.show()
'''