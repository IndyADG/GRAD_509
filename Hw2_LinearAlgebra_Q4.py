import numpy as np
import matplotlib.pyplot as plt

lims = (-5,5)

np.random.seed(1)

data2 = np.random.multivariate_normal([0,0], [[2,2],[2,2]], 1000)
data2 = data2.T
print(data2.shape)

#Plot 1: Create a scatterplot, each point representing 1 of the 1000 2d Gaussians
plt.scatter(data2[0,:], data2[1,:], alpha=0.5)
plt.title("Original Dataset w/ Feature Variance Equal & 2.0 Covariance")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Feature x")
plt.ylabel("Feature y")
plt.show()
print(lims)

data_cv = np.cov(data2)
print("data_cv=\n", data_cv)

U, V = np.linalg.eigh(data_cv)
print("U=\n", U)
print("V=\n", V)

#Rotate original dataset by left-multiplying it by V, a rotation matrix;
#This aligns the dataset's directions of maximum variance with the standard x and y coordinate axes
data_rot = V @ data2

#Plot 2: Plot the rotated data
plt.scatter(data_rot[0,:], data_rot[1,:], alpha=0.5)
plt.title("Axis-Aligned Data w/ Feature x Variance Higher")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

#The x-axis is now aligned in the direction of maximum variance, w/ less variance along the y-axis,
#so we'll perform dimensionality reduction by setting the row with least variance to 0
# UPDATE: Switching to ordered function .eigh() means that largest principal component is always last
#data_rot[1] = 0
data_rot[0] = 0

#Plot 3: Visualize the dimensionality-reduced rotated dataset
plt.scatter(data_rot[0,:], data_rot[1,:], alpha=0.5)
plt.title("PCA Applied w/ Feature x Variance Higher")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.show()

#Plot 4: Rotate back the matrix to see the original dataset after the dimensionality reduction
#To do that, multiply the dataset by the inverse of V, which is also the transpose of V since V is a rotation matrix)
data2_1d = V.T @ data_rot
plt.scatter(data2_1d[0,:], data2_1d[1,:], alpha=0.5)
plt.title("PCA-Reduced Data w/ Feature Variance Equal & 2.0 Covariance")
plt.xlim(lims)
plt.ylim(lims)
plt.xlabel("Feature x")
plt.ylabel("Feature y")
plt.show()








