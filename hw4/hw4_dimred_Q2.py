from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]

# Compute the mean image and mean-center the dataset
mu = np.mean(X_train, axis=0)
X_train = X_train - mu

# Visualize mean image
Image.fromarray(mu.astype('uint8').reshape((28,28))).resize((280, 280)).show()

# Select any single image and display
original_img = X_train[0:1]

# Have to add the mu back before visualizing image
X_original_plt = original_img[0, :] + mu
#Image.fromarray(X_original_plt.astype('uint8').reshape((28,28))).resize((280, 280)).show()

range_of_dims = [1, 10, 25, 50, 75, 125, 175, 300, 500, 784]
mse_vals = []

for k in range_of_dims:
    pca = PCA(n_components=k)
    pca.fit(X_train)

    # Compress and reconstruct the single selected image
    img_reduced = pca.transform(original_img)
    img_recovered = pca.inverse_transform(img_reduced)

    # Calculate MSE for this image, using mean of (original pixels minus approximated pixels, element-wise squared)
    mse = np.mean((original_img - img_recovered) ** 2)
    mse_vals.append(mse)

    # Plot the recovered image, first adding back the mean image then rescaling it to lie between 0 and 255
    X_recovered_plt = img_recovered[0, :] + mu
    X_recovered_plt = X_recovered_plt - X_recovered_plt.min()
    X_recovered_plt = X_recovered_plt * 255 / X_recovered_plt.max()

    Image.fromarray(X_recovered_plt.astype('uint8').reshape((28, 28))).resize((280, 280)).show()

plt.plot(range_of_dims, mse_vals, marker='o')
plt.xlabel('Number of Dimensions')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('PCA Reconstruction Error vs. Dimensions - Single Image X_train[0:1]')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(75))
plt.show()

'''
# Total reconstruction error of dataset vs dimension
total_errors = []

for k in range_of_dims:
    pca = PCA(n_components=k)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)

    mse_per_image = np.mean((X_train - X_recovered) ** 2, axis=1)
    total_errors.append(np.sum(mse_per_image))

plt.plot(range_of_dims, total_errors, marker='o')
plt.xlabel('Number of Dimensions')
plt.ylabel('Total Reconstruction Error (Sum of MSEs)')
plt.title('PCA Reconstruction Error vs. Dimensions - Full Dataset')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(75))
plt.show()
'''