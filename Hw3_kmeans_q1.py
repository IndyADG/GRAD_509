from PIL import Image
import numpy as np
image = Image.open("./ladybug.jpg")

# Convert image to an array, downsample every 4th pixel, then reshape to an array of RGB values
X = np.asarray(image)
X = X[::4, ::4]
shp = X.shape
X = X.reshape(-1, 3)

# Run k-means clustering w/ keeping k clusters same value
from sklearn.cluster import KMeans
result = []
for j in range(0,10):
    kmeans = KMeans(n_clusters=5, random_state=j).fit(X)
    # store fitted model and its inertia score, for easy access of min & max
    result.append([kmeans.inertia_, kmeans])

# Get inertia mean & spread
inertia_vals = np.array([entry[0] for entry in result])
inertia_mean = inertia_vals.mean()
inertia_std = inertia_vals.std()
print("inertia score mean:", inertia_mean)
print("inertia score standard deviation: ", inertia_std)

# Find then plot best & worst inertia scores
from operator import itemgetter
min_inertia_model = min(result, key=itemgetter(0))[1]
max_inertia_model = max(result, key=itemgetter(0))[1]
print("minimum inertia score: ", min_inertia_model.inertia_)
print("maximum inertia score: ", max_inertia_model.inertia_)

# For best & worst scores, convert segmented images to a grid of RGB values, then back to an image
segmented_img_best = min_inertia_model.cluster_centers_[min_inertia_model.labels_]
img_grid_best = segmented_img_best.reshape(shp)
image_best = (Image.fromarray(img_grid_best.astype(np.uint8), 'RGB'))
image_best.show()

segmented_img_worst = max_inertia_model.cluster_centers_[max_inertia_model.labels_]
img_grid_worst = segmented_img_worst.reshape(shp)
image_worst = (Image.fromarray(img_grid_worst.astype(np.uint8), 'RGB'))
image_worst.show()
