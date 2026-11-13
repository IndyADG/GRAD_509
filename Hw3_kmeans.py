# Read and display an image using Python Imaging Library
from PIL import Image
import numpy as np
image = Image.open("./ladybug.jpg")
#image.show()

# Convert the image to an array
X = np.asarray(image)

# Downsample every 4th pixel
X = X[::4, ::4]
shp = X.shape
print(shp)

# Display the downscaled image
image = (Image.fromarray(X.astype(np.uint8), 'RGB'))
#image.show()

# Convert X to a long array of RGB values
X = X.reshape(-1, 3)
print(X.shape)

# Run k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=1).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
#print("segmented_img", segmented_img)
# Convert image back to a grid of RGB values
img_t = segmented_img.reshape(shp)

# Convert from array back to image (the RGB values must be 8-bit integers)
image = (Image.fromarray(img_t.astype(np.uint8), 'RGB'))
#image.show()
#print(kmeans.inertia_) #Print the k-means inertia score

inertia = np.zeros(20)
for ii in range(0,20):
    kmeans = KMeans(n_clusters=ii+1, random_state=1).fit(X)
    inertia[ii] = kmeans.inertia_

# Test keeping k clusters same value
result = []
for k in range(0,10):
    kmeans = KMeans(n_clusters=5, random_state=k).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    #result[k] = kmeans.inertia_
    result.append([kmeans.inertia_, segmented_img])

# mean & spread
inertia_vals = np.array([entry[0] for entry in result])
print(inertia_vals)
inertia_mean = inertia_vals.mean()
inertia_std = inertia_vals.std()
print("inertia_mean", inertia_mean)
print("inertia_std", inertia_std)

# plot best & worst inertia scores
from operator import itemgetter
min_inertia_img = min(result, key=itemgetter(0))[1]
max_inertia_img = max(result, key=itemgetter(0))[1]

# Convert image back to a grid of RGB values
img_best = min_inertia_img.reshape(shp)
img_worst = max_inertia_img.reshape(shp)

# Convert from array back to image (the RGB values must be 8-bit integers)
image_best = (Image.fromarray(img_best.astype(np.uint8), 'RGB'))
image_best.show()

image_worst = (Image.fromarray(img_worst.astype(np.uint8), 'RGB'))
image_worst.show()