from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
image = Image.open("color_fun_crop.jpg")

# Convert image to an array, downsample every 4th pixel, then reshape to an array of RGB values
X = np.asarray(image)
X = X[::4, ::4]
shp = X.shape
X = X.reshape(-1, 3)

# Run k-means clustering w/ keeping k clusters same value
from sklearn.cluster import KMeans
results = []
max_cluster_boundary = 21
k_range = range(1,max_cluster_boundary)

for k in k_range:
    k_inertia_vals = []
    for j in range(0,10):
        kmeans = KMeans(n_clusters=k, random_state=j).fit(X)
        # store fitted model and its inertia score, for easy access of min & max
        k_inertia_vals.append(kmeans.inertia_)

    # Get mean, min, and max inertia scores
    k_inertia_array = np.array(k_inertia_vals)
    mean_inertia = k_inertia_array.mean()
    print(f"For k={k}, average inertia score: {mean_inertia},")
    min_inertia = min(k_inertia_array)
    max_inertia = max(k_inertia_array)
    print(f" min inertia score: {min_inertia}, max inertia score: {max_inertia}")
    results.append((k, mean_inertia, min_inertia, max_inertia))

plt.xlabel("Number of Clusters k")
plt.ylabel("Inertia Score")

mean_inertia_vals = np.array([entry[1] for entry in results])
mean_line = plt.plot(k_range, mean_inertia_vals, 'b-', label="Mean Inertia")

min_inertia_vals = np.array([entry[2] for entry in results])
max_inertia_vals = np.array([entry[3] for entry in results])
range_band = plt.fill_between(k_range, min_inertia_vals, max_inertia_vals, color='b',
                              alpha=0.2, label="Inertia Range")

plt.title("Inertia Score Range by k")
plt.margins(x=0)
plt.legend()
plt.show()

