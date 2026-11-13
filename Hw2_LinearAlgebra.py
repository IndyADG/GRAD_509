import numpy as np
import matplotlib.pyplot as plt

lims = (-5,5)

np.random.seed(1)

data2 = np.random.multivariate_normal([0,0], [[2,1.5],[1.5,2]], 1000)
data2 = data2.T
print(data2.shape)

#Create a scatterplot, each point representing 1 of the 1000 2d Gaussians
plt.scatter(data2[0,:], data2[1,:], alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)
plt.show()
print(lims)

data_cv = np.cov(data2)
print("data_cv=\n", data_cv)

U, V = np.linalg.eig(data_cv)
print("U=\n", U)
print("V=\n", V)

#Rotate original dataset by left-multiplying it by V, a rotation matrix
data_rot = V @ data2

#Plot the rotate data to see that it is indeed rotated to be axis aligned
plt.scatter(data_rot[0,:], data_rot[1,:], alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)
plt.show()

#The x-axis is now aligned in the direction of maximum variance, w/ less variance along the y-axis,
#so we'll perform dimensionality reduction by setting the second row to 0
data_rot[1] = 0

#Visualize the dimensionality-reduced rotated dataset
plt.scatter(data_rot[0,:], data_rot[1,:], alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)
plt.show()

#Rotate back the matrix to see the original dataset after the dimensionality reduction
#To do that, multiply the dataset by the inverse of V, which is also the transpose of V since V is a rotation matrix)
data2_1d = V.T @ data_rot
plt.scatter(data2_1d[0,:], data2_1d[1,:], alpha=0.5)
plt.xlim(lims)
plt.ylim(lims)
plt.show()

## Questions:
# Q2).Supporting code
V_transpose = V.T
print("V_transpose:\n", V_transpose.real)
V_inv = np.linalg.inv(V)
print("V_inv:\n", V_inv.real)
# A matrix whose transpose equals its inverse is orthogonal
print("\nV_transpose = V_inv is", np.allclose(V_transpose, V_inv))
# An orthogonal matrix is a rotational matrix if determinant = 1
determinant = np.linalg.det(V).real
print("The determinant of V is", determinant)

#Q3
#Matrix form of Eigenvalue equation (where D = diagonalized eigenvalues matrix): AV = VD -> A = VDV^T
# Create the diagonal eigenvalue matrix
D = np.diag(U)
print("\nDiagonalized Eigenvalues D:\n",D.real)
VD = V @ D
VD_VT = VD @ V.T

print("\nVDV^T:\n",VD_VT.real)
print("\nA:\n", data_cv)
print("\nTherefore, A = VDV^T is :", np.allclose(VD_VT,data_cv))

#Q4






